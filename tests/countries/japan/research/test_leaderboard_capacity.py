"""Ticket #491: the pool-depth capacity beside each leaderboard row (synthetic data only).

Contract pinned here (the coder implements what these tests call):

- ``core.eval.leaderboard.LeaderboardRow`` gains the optional fields ``class_group`` and
  ``odds_band`` (str or None, default None: the row's depth cell beside its ``pool``) and
  ``capacity_note`` (str or None, default None), plus whatever fields carry the capacity (their
  layout is the coder's). ``row.capacity_at(edge, quantile=0.5) -> float | None`` is the capacity
  stake in yen the row carries at that edge and quantile (edges and quantiles matched as floats),
  None when it carries none there. ``to_json``/``from_json`` round-trip every new field
  (``from_json(to_json(rows)) == rows``), and a row file written before #491 (no new keys) still
  reads, with no capacity.
- ``countries.japan.research.leaderboard_build.with_capacity(rows, depth) -> list[LeaderboardRow]``:
  the rows in their order, every other field as it was, each joined to the cell (row.pool,
  row.class_group, row.odds_band) of ``depth``, #432's long depth table (``pool_depth.depth_table``
  columns: pool, class_group, band, quantile, ``capacity_{e:g}`` per edge, others ignored), or
  None when there is no table. A row whose cell is in the table with finite capacities carries
  them at every edge and quantile of the table (``capacity_at(e, q)`` is the table's
  ``capacity_{e:g}`` at q). Every other row carries no capacity and a non-blank
  ``capacity_note`` saying why; at least these reasons differ from one another: no depth table;
  a co-placer pool (PLACE, QUINELLA_PLACE: the depth table has no single-winner capacity); a row
  naming a pool, class group and band that the table has no cell for; a row naming no class
  group or band (a model, a registry rule). The capacity is never a reason to withhold the row:
  ``row.withheld`` is unchanged.
- ``render_markdown``: the private text shows a row's capacity figures (at least the median at
  each of the three edges) on a line holding the row's option id; the public text shows
  ``pool_depth.WITHHELD`` ('withheld (D21)') on such a line instead, and no capacity figure
  anywhere (the depth table is JRA-VAN, D21), while the public row's own return still shows.
- ``pool_depth.write_depth_table(table, directory, *, spec_name=SPEC_NAME) -> Path``: the depth
  table of a spec's latest run, written under ``directory`` through ``write_report`` with sources
  ('jravan',) (inside the repository: ``LicensedOutputInRepo``, nothing written);
  ``pool_depth.read_depth_tables(directory) -> DataFrame | None``: every spec's table found there,
  concatenated (the same cells and capacities, NaN kept), None when there is none or the directory
  does not exist. ``run_depth`` writes its table under its ``out_dir`` (both specs: #432's and
  #490's exotics).
- ``leaderboard_build.build_rows(*, registry=None, candidate_dir=None, depth_dir=None)``: the rows
  as before, candidate rows carrying their cell's ``class_group`` and ``odds_band``, all of them
  passed through ``with_capacity`` with ``read_depth_tables(depth_dir)``, ``depth_dir`` defaulting
  to the reports dir; ``horsetrack japan leaderboard`` does the same with the defaults.

Every figure here is invented; the capacities are five-digit integers so that any rendering of
them (12345, 12,345, 12345.00) is found.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import re
import shutil
import uuid

import pandas as pd
import pytest
import test_leaderboard_build as tlb
import test_pool_depth as tpd
import test_pool_depth_exotics as tpe

# isort: split
from horsetrack.core import env
from horsetrack.core.eval.leaderboard import LeaderboardRow, from_json, render_markdown, to_json
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.countries.japan.research import pool_depth as depth

# #452's fixtures: a temp HORSETRACK_DATA_DIR (``data``), an empty temp registry (``reg``) and a
# stand-in repository root for the command (``stand_in_repo``)
data = tlb.data
reg = tlb.reg
stand_in_repo = tlb.stand_in_repo

REPO = env.repo_root()
WITHHELD = "withheld (D21)"
EDGES = (0.05, 0.1, 0.2)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
MEDIAN = 0.5

# The depth cells with a single-winner capacity, each with its own base: no two figures agree.
CAPACITY_CELLS = {
    ("WIN", "maiden", "5-10"): 41_000,
    ("WIN", "newcomer", "5-10"): 52_000,  # the class differs
    ("WIN", "maiden", "10-20"): 63_000,  # the band differs
    ("QUINELLA", "maiden", "5-10"): 74_000,  # the pool differs
    ("TIERCE", "maiden", "100-300"): 85_000,  # an exotics cell (#490)
}
PLACE_CELL = ("PLACE", "maiden", "1.5-2")  # a co-placer cell: depth only, no capacity


def capacity_value(cell: tuple[str, str, str], edge_index: int, q_index: int) -> int:
    """The made-up capacity of a cell at an edge and quantile: rising with both, unique."""
    return CAPACITY_CELLS[cell] + 3_000 * edge_index + 400 * q_index + 7


def depth_frame(*, seed: int = 491) -> pd.DataFrame:
    """A depth table in ``pool_depth.depth_table``'s long format, its rows shuffled."""
    out = []
    for cell in [*CAPACITY_CELLS, PLACE_CELL]:
        pool, group, band = cell
        for qi, q in enumerate(QUANTILES):
            row = {
                "pool": pool,
                "class_group": group,
                "band": band,
                "quantile": q,
                "n_combinations": 120 + qi,
                "n_races": 30,
                "combo_yen": 3_100.0 + qi,
                "pool_yen": 9_900_000.0 + qi,
            }
            for ei, e in enumerate(EDGES):
                row[depth.capacity_column(e)] = (
                    math.nan if cell == PLACE_CELL else float(capacity_value(cell, ei, qi))
                )
            out.append(row)
    random.Random(seed).shuffle(out)
    return pd.DataFrame(out)


def all_figures() -> set[str]:
    return {
        str(capacity_value(cell, ei, qi))
        for cell in CAPACITY_CELLS
        for ei in range(len(EDGES))
        for qi in range(len(QUANTILES))
    }


def make_row(option_id: str, **overrides) -> LeaderboardRow:
    fields = {
        "option_id": option_id,
        "family_id": "japan-n491-synthetic",
        "ticket": 491,
        "kind": "cell",
        "pool": "WIN",
        "segment": option_id,
        "rule": f"flat bets in {option_id}",
        "selection": "decision_time",
        "sources": ("netkeiba",),
        "variants_searched": 3,
        "status": "candidate",
        "return_per_100": 92.5,
        "se_per_100": 3.0,
        "n": 500,
        "n_unit": "bets",
        "payout_rate": 0.8,
    }
    fields.update(overrides)
    return LeaderboardRow(**fields)


def cell_row(cell: tuple[str, str, str], **overrides) -> LeaderboardRow:
    pool, group, band = cell
    return make_row(
        f"japan-n491-synthetic/{pool}|class:{group}|band:{band}",
        **{"pool": pool, "class_group": group, "odds_band": band, **overrides},
    )


def lb():
    from horsetrack.countries.japan.research import leaderboard_build

    return leaderboard_build


def assert_carries(row: LeaderboardRow, cell: tuple[str, str, str]) -> None:
    for ei, e in enumerate(EDGES):
        for qi, q in enumerate(QUANTILES):
            got = row.capacity_at(e, q)
            want = capacity_value(cell, ei, qi)
            assert got is not None and float(got) == pytest.approx(want), (
                row.option_id,
                e,
                q,
                got,
                want,
            )
        assert float(row.capacity_at(e)) == pytest.approx(capacity_value(cell, ei, 2)), (
            "capacity_at's default quantile is the median",
            row.option_id,
            e,
        )


def assert_states_a_reason(row: LeaderboardRow) -> str:
    for e in EDGES:
        for q in QUANTILES:
            got = row.capacity_at(e, q)
            assert got is None, f"{row.option_id} carries {got!r} at {e}, {q} without a cell"
    note = row.capacity_note
    assert isinstance(note, str) and note.strip(), f"{row.option_id} states no reason: {note!r}"
    return note


def unescaped_lines(text: str) -> list[str]:
    return [line.replace("\\|", "|") for line in text.splitlines()]


def figure_tokens(text: str) -> set[str]:
    """Every run of digits once thousands separators are dropped."""
    return set(re.findall(r"\d+", re.sub(r"(?<=\d),(?=\d{3})", "", text)))


KEPT_FIELDS = (
    "option_id",
    "family_id",
    "kind",
    "pool",
    "segment",
    "rule",
    "selection",
    "sources",
    "variants_searched",
    "status",
    "return_per_100",
    "se_per_100",
    "n",
    "licensed",
    "note",
)


# --------------------------------------------------------------------------- the join


def test_a_leaderboard_row_gets_its_cells_capacity_at_each_edge():
    """DoD 'every leaderboard row carries its cell's capacity at edges 5, 10 and 20%': a row in
    each cell (pool, class group and band each varied) carries exactly that cell's capacity stake
    at every edge and quantile of the table, whatever the table's row order; every other field and
    the rows' order are kept, and the JRA-VAN capacity withholds no public row."""
    rows = [cell_row(cell, return_per_100=80.0 + i) for i, cell in enumerate(CAPACITY_CELLS)]
    rows.reverse()

    for seed in (1, 2):
        joined = lb().with_capacity(rows, depth_frame(seed=seed))

        assert [r.option_id for r in joined] == [r.option_id for r in rows]
        for before, after, cell in zip(rows, joined, reversed(list(CAPACITY_CELLS)), strict=True):
            assert_carries(after, cell)
            for name in KEPT_FIELDS:
                assert getattr(after, name) == getattr(before, name), (before.option_id, name)
            assert (after.class_group, after.odds_band) == cell[1:]
            assert not after.withheld


def test_a_row_without_a_cell_states_why():
    """DoD 'or a stated reason it has none': a co-placer pool row (its cell has no capacity), a
    row whose band the table lacks, a model row and a row of every class (no class group) carry
    no figure and a reason; with no depth table every row states one. The reasons tell the causes
    apart."""
    place = cell_row(PLACE_CELL, pool="PLACE")
    no_cell = cell_row(("WIN", "maiden", "ge50"))
    model = make_row("japan-c4-meta-synthetic/meta-a", kind="model", segment="all")
    all_classes = make_row(
        "japan-n491-synthetic/WIN|class:all|band:5-10", class_group=None, odds_band="5-10"
    )
    matched = cell_row(("WIN", "maiden", "5-10"))
    rows = [place, no_cell, model, all_classes, matched]

    joined = {r.option_id: r for r in lb().with_capacity(rows, depth_frame())}

    assert_carries(joined[matched.option_id], ("WIN", "maiden", "5-10"))
    reasons = {
        "co-placer": assert_states_a_reason(joined[place.option_id]),
        "no cell": assert_states_a_reason(joined[no_cell.option_id]),
        "no class or band": assert_states_a_reason(joined[model.option_id]),
    }
    assert_states_a_reason(joined[all_classes.option_id])

    bare = lb().with_capacity(rows, None)
    assert [r.option_id for r in bare] == [r.option_id for r in rows]
    for row in bare:
        reasons.setdefault("no table", assert_states_a_reason(row))
    assert len(set(reasons.values())) == 4, f"the reasons do not tell the causes apart: {reasons}"


def test_the_capacity_survives_the_json_row_file():
    """The private JSON keeps each row's capacity and reason; a row file from before #491 still
    reads, with no capacity."""
    rows = lb().with_capacity(
        [
            cell_row(("WIN", "maiden", "5-10")),
            cell_row(("TIERCE", "maiden", "100-300"), pool="TIERCE"),
            cell_row(PLACE_CELL, pool="PLACE"),
            make_row("japan-c4-meta-synthetic/meta-a", kind="model"),
        ],
        depth_frame(),
    )

    back = from_json(to_json(rows))

    assert back == rows
    assert_carries(back[1], ("TIERCE", "maiden", "100-300"))
    assert back[2].capacity_note == rows[2].capacity_note

    old_fields = {f.name for f in dataclasses.fields(LeaderboardRow)} - {
        "class_group",
        "odds_band",
        "capacity_note",
    }
    old = [
        {k: v for k, v in item.items() if k in old_fields and not k.startswith("capacity")}
        for item in json.loads(to_json([make_row("japan-n491-synthetic/old")]))
    ]
    (read,) = from_json(json.dumps(old))
    assert all(read.capacity_at(e) is None for e in EDGES)


# --------------------------------------------------------------------------- the two texts


def test_the_private_text_shows_the_capacity_and_the_public_text_withholds_it():
    """DoD 'no JRA-VAN figure in public': the depth table is JRA-VAN, so the public text marks a
    public row's capacity 'withheld (D21)' and holds none of its figures, while the row itself
    and its return still show; the private text shows the median capacity at each edge on the
    row's line."""
    cell = ("WIN", "maiden", "5-10")
    public_row = cell_row(cell, return_per_100=93.75)
    rows = lb().with_capacity(
        [public_row, cell_row(("QUINELLA", "maiden", "5-10"), pool="QUINELLA")], depth_frame()
    )
    medians = [str(capacity_value(cell, ei, 2)) for ei in range(len(EDGES))]

    private = render_markdown(rows, title="Japan leaderboard", public=False)
    public = render_markdown(rows, title="Japan leaderboard", public=True)

    mine = "\n".join(line for line in unescaped_lines(private) if public_row.option_id in line)
    assert set(medians) <= figure_tokens(mine), (
        f"the private lines of {public_row.option_id} do not hold its medians {medians}"
    )
    shown = [line for line in unescaped_lines(public) if public_row.option_id in line]
    assert shown and any("93.75" in line for line in shown), "the public row no longer shows"
    assert any(WITHHELD in line for line in shown), (
        f"the public text does not mark {public_row.option_id}'s capacity {WITHHELD!r}"
    )
    leaked = all_figures() & figure_tokens(public)
    assert not leaked, f"JRA-VAN capacity figures {sorted(leaked)} are in the public text"
    assert depth.WITHHELD == WITHHELD


# --------------------------------------------------------------------------- the depth table file


def test_depth_tables_are_written_privately_and_read_back(data):
    """The depth tables of #432's spec and of the exotics spec, written beside the reports, read
    back together with the same cells and capacities (Place's NaN kept); a directory without one
    gives None; the repository is refused and nothing lands there."""
    frame = depth_frame()
    exotic = frame["pool"].to_numpy() == "TIERCE"
    main, exotics = frame.loc[~exotic], frame.loc[exotic]
    folder = data / "reports"

    assert depth.read_depth_tables(folder / "missing") is None
    folder.mkdir(parents=True)
    assert depth.read_depth_tables(folder) is None
    depth.write_depth_table(main, folder)
    depth.write_depth_table(exotics, folder, spec_name=depth.EXOTICS)

    read = depth.read_depth_tables(folder)

    columns = [*depth.CELL, "quantile", *(depth.capacity_column(e) for e in EDGES)]
    key = [*depth.CELL, "quantile"]
    want = frame[columns].sort_values(key).reset_index(drop=True)
    got = read[columns].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False, check_categorical=False)

    inside = REPO / f".depth-table-test-{uuid.uuid4().hex[:12]}"
    try:
        with pytest.raises(LicensedOutputInRepo):
            depth.write_depth_table(main, inside)
        assert not inside.exists() or not any(p.is_file() for p in inside.rglob("*"))
    finally:
        shutil.rmtree(inside, ignore_errors=True)


def test_run_depth_writes_its_table_for_the_leaderboard(tmp_path, data):
    """The entry the leaderboard reads is fed by the runs: #432's depth run and #490's exotics run
    into one out_dir, and the tables read back are the two runs' tables."""
    out_dir = data / "reports"
    registry, _ = tpd.frozen_registry(tmp_path)
    tpd.write_store(data, tpd.run_races(tpd.JUNE, range(9)))
    main = depth.run_depth(
        date_from=tpd.JUNE[0], date_to=tpd.JUNE[-1], out_dir=out_dir, registry=registry
    )
    exotics_registry, _ = tpe.exotics_frozen(tmp_path, "exotics")
    tpe.write_exotic_store(data, tpe.exotic_races(tpe.YEAR_END, range(9)))
    exotics = depth.run_depth(
        **tpe.YEAR_END_SPAN, out_dir=out_dir, registry=exotics_registry, spec_name=depth.EXOTICS
    )

    read = depth.read_depth_tables(out_dir)

    assert read is not None, "run_depth wrote no depth table for the leaderboard"
    columns = [*depth.CELL, "quantile", *(depth.capacity_column(e) for e in EDGES)]
    key = [*depth.CELL, "quantile"]
    want = pd.concat([main.table[columns], exotics.table[columns]], ignore_index=True)
    want = want.astype({c: str for c in depth.CELL}).sort_values(key).reset_index(drop=True)
    got = read[columns].astype({c: str for c in depth.CELL}).sort_values(key)
    got = got.reset_index(drop=True)
    assert {"WIN", "TIERCE"} <= set(got["pool"])
    pd.testing.assert_frame_equal(got, want, check_dtype=False)


# --------------------------------------------------------------------------- the build and command

PUBLIC_WIN_CELL = {
    **tlb.TICKET_CELL,
    "study": "N9",
    "family_id": "japan-n491-public-scan",
    "cell_id": "WIN|class:maiden|band:5-10",
    "mean_return": 0.9375,
    "se_mean_return": 0.025,
    "n_bets": 876,
    "sources": ["netkeiba"],
}
PUBLIC_WIN_ID = "japan-n491-public-scan/WIN|class:maiden|band:5-10"
NEWCOMER_CELL = {
    **PUBLIC_WIN_CELL,
    "cell_id": "WIN|class:newcomer|band:5-10",
    "class_group": "newcomer",
}
NEWCOMER_ID = "japan-n491-public-scan/WIN|class:newcomer|band:5-10"


def test_build_rows_joins_the_depth_table(reg, tmp_path):
    """The build wires the capacity in: candidate rows carry their cell's class group and band
    and, from the depth table under depth_dir, their cell's capacity; a registry row (no class
    group or band) and a cell with no class group state why they have none; every row carries
    one or the other. Without a depth table every row states a reason."""
    reg.append(tlb.convention("flat:all", [tlb.dev("return_per_100", 78.5, 74.0, 83.0, 2400)]))
    cells_dir = tmp_path / "cells"
    tlb.write_cells(
        cells_dir, "cells.json", [PUBLIC_WIN_CELL, NEWCOMER_CELL, dict(tlb.QUINELLA_CELL)]
    )
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    depth.write_depth_table(depth_frame(), depth_dir)

    rows = tlb.unique(lb().build_rows(registry=reg, candidate_dir=cells_dir, depth_dir=depth_dir))

    assert (rows[PUBLIC_WIN_ID].class_group, rows[PUBLIC_WIN_ID].odds_band) == ("maiden", "5-10")
    assert_carries(rows[PUBLIC_WIN_ID], ("WIN", "maiden", "5-10"))
    assert_carries(rows[NEWCOMER_ID], ("WIN", "newcomer", "5-10"))
    assert_states_a_reason(rows[f"{tlb.CONVENTION}/flat:all"])
    assert_states_a_reason(rows[tlb.QUINELLA_ID])  # no class group, band 'all'
    for row in rows.values():
        has = all(row.capacity_at(e) is not None for e in EDGES)
        assert has or (row.capacity_note or "").strip(), row.option_id

    empty = tmp_path / "no-depth"
    empty.mkdir()
    bare = lb().build_rows(registry=reg, candidate_dir=cells_dir, depth_dir=empty)
    assert {r.option_id for r in bare} == set(rows)
    for row in bare:
        assert_states_a_reason(row)


def test_the_command_shows_the_capacity_privately_and_withholds_it_publicly(
    data, tmp_path, stand_in_repo
):
    """`horsetrack japan leaderboard --public` reads the depth table under the reports dir: the
    private JSON and text carry each row's capacity or its reason, the public file marks the
    capacity withheld and holds none of its figures, and neither stream prints one."""
    tlb.seed_command_data(data, tmp_path)
    tlb.write_cells(data / "reports" / "candidate-cells", "public.json", [PUBLIC_WIN_CELL])
    depth.write_depth_table(depth_frame(), data / "reports")

    result = tlb.invoke("--public")

    assert result.exit_code == 0, result.output
    rows = tlb.unique(from_json((data / "reports" / "leaderboard-japan.json").read_text("utf-8")))
    assert_carries(rows[PUBLIC_WIN_ID], ("WIN", "maiden", "5-10"))
    assert_carries(rows[tlb.TICKET_ID], ("WIN", "maiden", "5-10"))  # withheld, private figures
    for row in rows.values():
        has = all(row.capacity_at(e) is not None for e in EDGES)
        assert has or (row.capacity_note or "").strip(), row.option_id

    medians = {str(capacity_value(("WIN", "maiden", "5-10"), ei, 2)) for ei in range(3)}
    private = (data / "reports" / "leaderboard-japan.md").read_text(encoding="utf-8")
    assert medians <= figure_tokens(private)
    public = (stand_in_repo / "docs" / "leaderboard" / "japan.md").read_text(encoding="utf-8")
    shown = [line for line in unescaped_lines(public) if PUBLIC_WIN_ID in line]
    assert shown and any(WITHHELD in line for line in shown)
    leaked = all_figures() & figure_tokens(public)
    assert not leaked, f"JRA-VAN capacity figures {sorted(leaked)} are in the public file"

    out, err = tlb.streams(result, tmp_path)
    figures = tlb.SEEDED_FIGURES | all_figures()
    tlb.assert_no_figure(out, figures)
    tlb.assert_no_figure(err, figures)


# --------------------------------------------------------------------------- the pool scan's runs

# #429/#450's pool-scan cell runs as the registry holds them: one family per scan version and pool,
# 'japan-n1-pool-scan[-v<N>]-cells-<pool, lower case>', one run per cell, its variant (and
# params['cell']) the cell id, a class:<group>&band:<band> cell naming one depth cell, its
# params['pool'] the pool as the depth table spells it. No leaderboard block: an unmeasured row
# '<family>/<cell id>', segment 'all', pool None, which only its family and variant place in a cell.
SCAN_SPAN = "from-2014-06-07"


def pool_scan_family(pool: str, version: int = 1) -> str:
    return f"japan-n1-pool-scan{'' if version == 1 else f'-v{version}'}-cells-{pool.lower()}"


def pool_scan_run(pool: str, cell_id: str, *, version: int = 1, seed: int = 0):
    """A pool-scan cell run shaped as the real registry's (every figure made up)."""

    def metric(name: str, value: float):
        return tlb.FoldMetric(
            fold=SCAN_SPAN, metric=name, value=value, lo=None, hi=None, n=640 + seed
        )

    return tlb.record(
        hypothesis_id="E8",
        family_id=pool_scan_family(pool, version),
        variant=cell_id,
        ticket=429,
        sources=("jravan",),
        data_builds={"final_odds": tlb.BUILD_HASH},
        fold_scheme="in-sample:development-days",
        components={"bet": "flat_box_final_board"},
        params={
            "cell": cell_id,
            "end": "2026-11-27",
            "pool": pool,
            "seed": 0,
            "span": SCAN_SPAN,
            "start": "2014-06-07",
        },
        metrics=(
            metric("mean_return", 0.71 + 0.01 * seed),
            metric("flat_rho", 0.93 + 0.001 * seed),
            metric("n_meetings", 88.0 + seed),
            metric("se_mean_return", 0.02),
        ),
    )


def pool_scan_id(pool: str, cell_id: str, version: int = 1) -> str:
    return f"{pool_scan_family(pool, version)}/{cell_id}"


# (pool, cell id, scan version) -> the depth cell it names; each dimension varied against the first
POOL_SCAN_JOINS = {
    ("WIN", "class:maiden&band:5-10", 1): ("WIN", "maiden", "5-10"),
    ("WIN", "class:maiden&band:5-10", 2): ("WIN", "maiden", "5-10"),  # v2: no candidate copy
    ("WIN", "class:newcomer&band:5-10", 3): ("WIN", "newcomer", "5-10"),  # the class differs
    ("WIN", "class:maiden&band:10-20", 4): ("WIN", "maiden", "10-20"),  # the band differs
    ("QUINELLA", "class:maiden&band:5-10", 5): ("QUINELLA", "maiden", "5-10"),  # the pool differs
}


def test_build_rows_joins_the_pool_scans_registry_cells(reg, tmp_path):
    """Review of #491 (outline item 1, DoD 'its cell's capacity at 5, 10 and 20%'): a registry
    row of the pool scan (#450) is a pool x class x band depth cell, named by its family's
    '-cells-<pool>' suffix and its 'class:<group>&band:<band>' cell id, so with a depth table
    under depth_dir it carries that cell's capacity at every edge and quantile, in every scan
    version (v1/v2 have no candidate copy), with no candidate file at all."""
    for i, (pool, cell_id, version) in enumerate(POOL_SCAN_JOINS):
        reg.append(pool_scan_run(pool, cell_id, version=version, seed=i))
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    depth.write_depth_table(depth_frame(), depth_dir)

    rows = tlb.unique(
        lb().build_rows(registry=reg, candidate_dir=tmp_path / "no-cells", depth_dir=depth_dir)
    )

    for (pool, cell_id, version), cell in POOL_SCAN_JOINS.items():
        option = pool_scan_id(pool, cell_id, version)
        assert option in rows, f"{option} is not a leaderboard row: {sorted(rows)}"
        assert_carries(rows[option], cell)
        assert not (rows[option].capacity_note or "").strip(), (
            f"{option} carries its capacity and a reason it has none: {rows[option].capacity_note}"
        )


def test_a_pool_scan_cell_without_a_depth_cell_states_the_right_reason(reg, tmp_path):
    """A pool-scan registry cell names its depth cell, so when the table lacks it the reason is
    the one a candidate row in the same missing cell gets (no such cell), never that it names no
    class group or odds band; a Wide or Place scan cell gets the co-placer reason (Wide's
    'quinella_place' suffix is QUINELLA_PLACE, not Quinella, whose cell has a capacity); a scan
    cell that names no class or no band (all, band only, class only, class x favourite or pool
    size) gets the reason a candidate row naming as little gets."""
    frame = depth_frame()
    missing = {
        ("WIN", "class:graded&band:ge50"): ("WIN", "graded", "ge50"),
        ("WIN", "class:maiden&band:lt2"): ("WIN", "maiden", "lt2"),
        ("TRIO", "class:maiden&band:10-30"): ("TRIO", "maiden", "10-30"),  # no TRIO cells at all
    }
    co_placer = {
        ("QUINELLA_PLACE", "class:maiden&band:5-10"): ("QUINELLA_PLACE", "maiden", "5-10"),
        ("PLACE", "class:maiden&band:1.5-2"): PLACE_CELL,
    }
    unnamed = {  # cell id -> the (class group, band) it names
        "all": (None, None),
        "band:5-10": (None, "5-10"),
        "class:maiden": ("maiden", None),
        "class:maiden&fav:le1.9": ("maiden", None),
        "class:maiden&pool_size:high": ("maiden", None),
    }
    for i, (pool, cell_id) in enumerate([*missing, *co_placer]):
        reg.append(pool_scan_run(pool, cell_id, version=1 + i % 5, seed=i))
    for i, cell_id in enumerate(unnamed):
        reg.append(pool_scan_run("WIN", cell_id, version=3, seed=10 + i))
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    depth.write_depth_table(frame, depth_dir)

    rows = tlb.unique(
        lb().build_rows(registry=reg, candidate_dir=tmp_path / "no-cells", depth_dir=depth_dir)
    )

    # the reasons a candidate row gets for the same cause, from the join already pinned above
    def reason_for(**fields) -> str:
        (joined,) = lb().with_capacity([make_row("japan-n491-synthetic/probe", **fields)], frame)
        return assert_states_a_reason(joined)

    no_class_or_band = reason_for(class_group=None, odds_band=None)
    for i, ((pool, cell_id), cell) in enumerate(missing.items()):
        option = pool_scan_id(pool, cell_id, 1 + i % 5)
        want = reason_for(pool=cell[0], class_group=cell[1], odds_band=cell[2])
        assert want != no_class_or_band
        assert assert_states_a_reason(rows[option]) == want, option
    for i, ((pool, cell_id), cell) in enumerate(co_placer.items(), start=len(missing)):
        option = pool_scan_id(pool, cell_id, 1 + i % 5)
        want = reason_for(pool=cell[0], class_group=cell[1], odds_band=cell[2])
        assert want != no_class_or_band
        assert assert_states_a_reason(rows[option]) == want, option
    for cell_id, (group, band) in unnamed.items():
        option = pool_scan_id("WIN", cell_id, 3)
        want = reason_for(pool="WIN", class_group=group, odds_band=band)
        assert assert_states_a_reason(rows[option]) == want, option
