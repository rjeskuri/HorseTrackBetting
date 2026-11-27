"""Ticket #450: leaderboard rows read from the run registry.

Contract pinned here (the coder implements what these tests call):

- ``rows_from_registry(registry, *, adapters=None) -> list[LeaderboardRow]`` reads
  ``Registry.records()`` only and never writes. Every (family, variant) gets rows: one measured
  row per development-fold ``return_per_100`` / ``return_per_100[<segment>]`` metric of a record
  whose ``params['leaderboard']`` block follows the convention, else an adapter's rows, else one
  unmeasured row ``'<family>/<variant>'`` (kind from the block if any, else 'other'; pool from
  the block or None; rule the variant text; selection 'unknown').
- A measured row: option id ``'<family>/<variant>'`` for 'all', ``'<family>/<variant>[<seg>]'``
  otherwise; return, bounds, n from the metric; level, kind, pool, rule, selection, payout rate,
  n unit from the block; ``se_per_100[...]`` and ``n_meetings[...]`` of the same segment beside
  it; variants searched the family size; ticket, sources and status the record's; licensed
  ``is_licensed_record(record)``.
- The latest run of an option in append order wins; a run with ``holdout_open`` gives no row;
  a malformed block lists the variant unmeasured with the reason in ``note``.
- ``core.eval.registry.is_licensed_record(record)`` returns what ``_licensed`` returns.

Every registry is a temp file under ``tmp_path`` built as ``tests/core/eval/test_registry.py``
builds them; HORSETRACK_DATA_DIR points into ``tmp_path``. Every value is synthetic, and every
expected value is written out here, never computed by the code under test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from horsetrack.core.eval import registry as registry_module
from horsetrack.core.eval.leaderboard import (
    LeaderboardRow,
    public_view,
    rank,
    render_markdown,
    rows_from_registry,
)
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord, is_licensed_record

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"
GIT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
OTHER_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4"
BUILD_HASH = hashlib.sha256(b"synthetic build").hexdigest()
SEED_SHA = hashlib.sha256(b"synthetic seed").hexdigest()

WIN_BLOCK = {
    "kind": "cell",
    "pool": "WIN",
    "rule": "flat Win bets",
    "selection": "look_ahead",
    "payout_rate": 0.8,
    "ci_level": 0.95,
    "n_unit": "bets",
}
PLACE_BLOCK = {
    "kind": "rule",
    "pool": "PLACE",
    "rule": "flat Place on the T-2 favourite",
    "selection": "decision_time",
    "payout_rate": 0.8,
    "ci_level": 0.9,
    "n_unit": "races",
}


# --------------------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No test reads the real data directory or changes a real holdout log."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "guard-data"))

    def snapshot() -> dict[Path, bytes]:
        return {p: p.read_bytes() for p in REAL_COUNTRIES.glob("*/holdout_log.jsonl")}

    before = snapshot()
    yield
    assert snapshot() == before, "a test wrote to a real holdout_log.jsonl"


def fold(name: str, metric: str, value: float, lo=None, hi=None, n: int = 0) -> FoldMetric:
    return FoldMetric(fold=name, metric=metric, value=value, lo=lo, hi=hi, n=n)


def dev(metric: str, value: float, lo=None, hi=None, n: int = 0) -> FoldMetric:
    return fold("development", metric, value, lo, hi, n)


def make_record(**overrides) -> RunRecord:
    fields = {
        "hypothesis_id": "E1",
        "family_id": "F1",
        "variant": "baseline",
        "ticket": 450,
        "git_sha": GIT_SHA,
        "data_builds": {"results": BUILD_HASH},
        "sources": ("netkeiba",),
        "fold_scheme": "walk-forward-6",
        "components": {"model": "logit-1.0"},
        "params": {},
        "metrics": (),
        "status": "scored",
    }
    fields.update(overrides)
    return RunRecord(**fields)


def open_registry(tmp_path: Path) -> Registry:
    path = tmp_path / "data" / "registry" / "japan.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "countries" / "japan" / "holdout_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    return Registry(path, holdout_log=log)


def append_all(reg: Registry, records: list[RunRecord]) -> None:
    for record in records:
        reg.append(record)


def holdout_open_hash(reg: Registry) -> str:
    """One synthetic #71 log line, written the way ``LockedHoldout.open`` writes it."""
    line = json.dumps(
        {
            "utc": "2026-12-01T09:00:00+00:00",
            "ticket": 450,
            "purpose": "synthetic check",
            "git_sha": GIT_SHA,
            "n_rows": 1200,
            "period": {"start": "2026-11-28", "seed_sha256": SEED_SHA},
        }
    )
    with open(reg.holdout_log, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def read_rows(reg: Registry, **kwargs) -> dict[str, LeaderboardRow]:
    """The rows by option id; every option id appears once and the set ranks."""
    rows = rows_from_registry(reg, **kwargs)
    assert isinstance(rows, list)
    assert all(isinstance(r, LeaderboardRow) for r in rows)
    ids = [r.option_id for r in rows]
    assert len(ids) == len(set(ids)), f"duplicate option ids: {sorted(ids)}"
    rank(rows)  # a leaderboard the ranking accepts
    return {r.option_id: r for r in rows}


def variant_of(row: LeaderboardRow) -> str:
    rest = row.option_id[len(row.family_id) + 1 :]
    assert row.option_id.startswith(row.family_id + "/"), row.option_id
    return rest.split("[", 1)[0]


def assert_unmeasured(row: LeaderboardRow, *, variant: str, kind="other", pool=None) -> None:
    assert row.return_per_100 is None and not row.measured
    assert (row.ci_lo, row.ci_hi, row.shrunk_per_100) == (None, None, None)
    assert row.kind == kind
    assert row.pool == pool
    assert row.rule == variant
    assert row.selection == "unknown"


# --------------------------------------------------------------------------- every variant


def test_every_family_and_variant_is_on_the_leaderboard(tmp_path):
    reg = open_registry(tmp_path)
    records = [
        make_record(
            hypothesis_id="E10",
            family_id="N1-cells",
            variant="maiden-win",
            ticket=429,
            params={"leaderboard": WIN_BLOCK},
            metrics=(
                dev("return_per_100[class:maiden]", 87.5, 80.0, 95.0, 400),
                dev("return_per_100[class:open]", 74.25, 70.0, 78.5, 900),
            ),
        ),
        make_record(
            hypothesis_id="E10",
            family_id="N1-cells",
            variant="all-place",
            ticket=429,
            params={"leaderboard": PLACE_BLOCK},
            metrics=(dev("return_per_100", 81.0, 77.0, 85.0, 1500),),
        ),
        make_record(
            hypothesis_id="E20",
            family_id="C2-logit",
            variant="logit-a",
            ticket=88,
            params={"C": 1.0},
            metrics=(
                dev("log_loss_gain", 0.0031, 0.0010, 0.0052, 3000),
                fold("wf-1", "log_loss_gain", 0.0040, None, None, 500),
            ),
        ),
        make_record(
            hypothesis_id="E20",
            family_id="C2-logit",
            variant="logit-b",
            ticket=88,
            params={"C": 0.1},
            metrics=(dev("log_loss_gain", -0.0007, -0.0021, 0.0006, 3000),),
        ),
        make_record(
            hypothesis_id="E30",
            family_id="M3-gbm",
            variant="gbm-1",
            ticket=97,
            params={"depth": 6},
            metrics=(),
            status="failed",
        ),
    ]
    append_all(reg, records)
    before = reg.path.read_bytes()

    rows = read_rows(reg)

    assert reg.path.read_bytes() == before, "reading the leaderboard changed the registry"
    expected = {(r.family_id, r.variant) for r in records}
    assert {(row.family_id, variant_of(row)) for row in rows.values()} == expected

    assert set(rows) == {
        "N1-cells/maiden-win[class:maiden]",
        "N1-cells/maiden-win[class:open]",
        "N1-cells/all-place",
        "C2-logit/logit-a",
        "C2-logit/logit-b",
        "M3-gbm/gbm-1",
    }
    for variant in ("logit-a", "logit-b"):
        row = rows[f"C2-logit/{variant}"]
        assert_unmeasured(row, variant=variant)
        assert row.status == "scored"
        assert row.ticket == 88
    failed = rows["M3-gbm/gbm-1"]
    assert failed.status == "failed"
    assert_unmeasured(failed, variant="gbm-1")
    assert failed.ticket == 97
    assert rows["N1-cells/maiden-win[class:open]"].return_per_100 == 74.25
    assert rows["N1-cells/all-place"].return_per_100 == 81.0


# --------------------------------------------------------------------------- the convention


MEASURED_CASES = [
    # (block, segment, value, lo, hi, n, ticket, sources)
    (WIN_BLOCK, "class:maiden", 87.5, 80.0, 95.0, 400, 450, ("netkeiba",)),
    (PLACE_BLOCK, "surface:dirt", 112.25, 101.5, 123.0, 61, 432, ("jra", "netkeiba")),
    (
        {**WIN_BLOCK, "kind": "member", "pool": "EXACTA", "rule": "fav-first exacta"},
        "odds:10-20",
        64.0,
        41.0,
        87.0,
        9,
        503,
        ("jra",),
    ),
]


@pytest.mark.parametrize(
    ("block", "segment", "value", "lo", "hi", "n", "ticket", "sources"), MEASURED_CASES
)
def test_a_return_metric_becomes_a_measured_row(
    tmp_path, block, segment, value, lo, hi, n, ticket, sources
):
    reg = open_registry(tmp_path)
    reg.append(
        make_record(
            family_id="N5-study",
            variant="v-one",
            ticket=ticket,
            sources=sources,
            params={"leaderboard": block},
            metrics=(dev(f"return_per_100[{segment}]", value, lo, hi, n),),
        )
    )

    rows = read_rows(reg)

    option = f"N5-study/v-one[{segment}]"
    assert set(rows) == {option}
    row = rows[option]
    assert row.family_id == "N5-study"
    assert row.segment == segment
    assert (row.return_per_100, row.ci_lo, row.ci_hi) == (value, lo, hi)
    assert row.ci_level == block["ci_level"]
    assert (row.n, row.n_unit) == (n, block["n_unit"])
    assert row.kind == block["kind"]
    assert row.pool == block["pool"]
    assert row.rule == block["rule"]
    assert row.selection == block["selection"]
    assert row.payout_rate == block["payout_rate"]
    assert row.ticket == ticket
    assert row.sources == sources
    assert row.status == "scored"
    assert row.licensed is False
    assert row.variants_searched == 1


def test_a_plain_return_metric_is_the_all_segment(tmp_path):
    reg = open_registry(tmp_path)
    reg.append(
        make_record(
            family_id="N6-place",
            variant="fav-place",
            params={"leaderboard": PLACE_BLOCK},
            metrics=(
                dev("return_per_100", 79.5, 76.0, 83.0, 2400),
                dev("return_per_100[class:maiden]", 84.0, 75.0, 93.0, 310),
            ),
        )
    )

    rows = read_rows(reg)

    assert set(rows) == {"N6-place/fav-place", "N6-place/fav-place[class:maiden]"}
    plain = rows["N6-place/fav-place"]
    assert plain.segment == "all"
    assert (plain.return_per_100, plain.ci_lo, plain.ci_hi, plain.n) == (79.5, 76.0, 83.0, 2400)
    seg = rows["N6-place/fav-place[class:maiden]"]
    assert seg.segment == "class:maiden"
    assert (seg.return_per_100, seg.n) == (84.0, 310)


def test_se_and_meetings_metrics_are_read_beside_the_return(tmp_path):
    reg = open_registry(tmp_path)
    reg.append(
        make_record(
            family_id="N7-win",
            variant="maiden-win",
            params={"leaderboard": WIN_BLOCK},
            metrics=(
                dev("return_per_100[class:maiden]", 87.5, 80.0, 95.0, 400),
                dev("se_per_100[class:maiden]", 3.2, None, None, 400),
                dev("n_meetings[class:maiden]", 37.0, None, None, 400),
                dev("return_per_100[class:open]", 76.0, 73.0, 79.0, 1200),
                dev("se_per_100[class:open]", 1.75, None, None, 1200),
                dev("n_meetings[class:open]", 52.0, None, None, 1200),
                dev("return_per_100", 78.0, 75.5, 80.5, 1600),
            ),
        )
    )

    rows = read_rows(reg)

    assert set(rows) == {
        "N7-win/maiden-win[class:maiden]",
        "N7-win/maiden-win[class:open]",
        "N7-win/maiden-win",
    }
    maiden = rows["N7-win/maiden-win[class:maiden]"]
    assert maiden.se_per_100 == 3.2
    assert maiden.n_meetings == 37
    assert maiden.n == 400
    other = rows["N7-win/maiden-win[class:open]"]
    assert other.se_per_100 == 1.75
    assert other.n_meetings == 52
    # no se or meetings metric for 'all': none borrowed from a segment
    plain = rows["N7-win/maiden-win"]
    assert plain.se_per_100 is None
    assert plain.n_meetings is None


def test_variants_searched_is_the_family_size(tmp_path):
    reg = open_registry(tmp_path)
    block = {"leaderboard": WIN_BLOCK}
    append_all(
        reg,
        [
            make_record(
                family_id="N8-spread",
                variant="box-3",
                params=block,
                metrics=(
                    dev("return_per_100", 71.0, 65.0, 77.0, 800),
                    dev("return_per_100[class:maiden]", 69.0, 60.0, 78.0, 200),
                ),
            ),
            # a rerun of box-3 at another git sha: another run, not another variant
            make_record(
                family_id="N8-spread",
                variant="box-3",
                git_sha=OTHER_SHA,
                params=block,
                metrics=(
                    dev("return_per_100", 72.0, 66.0, 78.0, 800),
                    dev("return_per_100[class:maiden]", 70.0, 61.0, 79.0, 200),
                ),
            ),
            make_record(
                family_id="N8-spread",
                variant="wheel-1",
                params=block,
                metrics=(dev("return_per_100", 66.0, 60.0, 72.0, 700),),
            ),
            make_record(family_id="N8-spread", variant="formation", status="failed"),
            make_record(
                family_id="P9-pair",
                variant="a",
                params={"leaderboard": PLACE_BLOCK},
                metrics=(dev("return_per_100", 90.0, 85.0, 95.0, 300),),
            ),
            make_record(family_id="P9-pair", variant="b", params={"x": 1}),
        ],
    )

    rows = read_rows(reg)

    spread = [r for r in rows.values() if r.family_id == "N8-spread"]
    assert len(spread) == 4  # box-3 'all' and maiden, wheel-1, formation
    assert {r.variants_searched for r in spread} == {3}
    pair = [r for r in rows.values() if r.family_id == "P9-pair"]
    assert len(pair) == 2
    assert {r.variants_searched for r in pair} == {2}


def test_the_latest_run_of_an_option_wins(tmp_path):
    reg = open_registry(tmp_path)
    block = {"leaderboard": WIN_BLOCK}
    append_all(
        reg,
        [
            make_record(
                family_id="N10-up",
                variant="v",
                ticket=101,
                params=block,
                metrics=(dev("return_per_100", 90.0, 85.0, 95.0, 100),),
            ),
            make_record(
                family_id="N11-down",
                variant="v",
                ticket=201,
                params=block,
                metrics=(dev("return_per_100[class:maiden]", 110.0, 100.0, 120.0, 50),),
            ),
            make_record(
                family_id="N10-up",
                variant="v",
                ticket=102,
                git_sha=OTHER_SHA,
                params=block,
                metrics=(dev("return_per_100", 104.0, 99.0, 109.0, 180),),
            ),
            make_record(
                family_id="N11-down",
                variant="v",
                ticket=202,
                git_sha=OTHER_SHA,
                params=block,
                metrics=(dev("return_per_100[class:maiden]", 95.5, 86.0, 105.0, 75),),
            ),
            make_record(
                family_id="N11-down",
                variant="v",
                ticket=203,
                git_sha="c" * 40,
                params=block,
                metrics=(dev("return_per_100[class:maiden]", 99.0, 90.0, 108.0, 90),),
            ),
        ],
    )

    rows = read_rows(reg)

    assert set(rows) == {"N10-up/v", "N11-down/v[class:maiden]"}
    up = rows["N10-up/v"]
    assert (up.return_per_100, up.ci_lo, up.ci_hi) == (104.0, 99.0, 109.0)
    assert (up.n, up.ticket) == (180, 102)
    down = rows["N11-down/v[class:maiden]"]
    assert (down.return_per_100, down.n, down.ticket) == (99.0, 90, 203)


@pytest.mark.parametrize("status", ["rejected", "adopted"])
def test_old_statuses_read_like_scored(tmp_path, status):
    reg = open_registry(tmp_path)
    reg.append(
        make_record(
            family_id="L12-old",
            variant="legacy-rule",
            params={"leaderboard": PLACE_BLOCK},
            metrics=(dev("return_per_100", 83.0, 79.0, 87.0, 640),),
            status=status,
            owner_signoff=99 if status == "adopted" else None,
        )
    )

    rows = read_rows(reg)

    assert set(rows) == {"L12-old/legacy-rule"}
    row = rows["L12-old/legacy-rule"]
    assert (row.return_per_100, row.ci_lo, row.ci_hi, row.n) == (83.0, 79.0, 87.0, 640)
    assert row.status in (status, "scored")


def test_other_folds_give_no_row(tmp_path):
    reg = open_registry(tmp_path)
    append_all(
        reg,
        [
            make_record(
                family_id="N13-folds",
                variant="per-fold-only",
                params={"leaderboard": WIN_BLOCK},
                metrics=(
                    fold("wf-1", "return_per_100", 120.0, 100.0, 140.0, 80),
                    fold("wf-2", "return_per_100[class:maiden]", 60.0, 40.0, 80.0, 30),
                ),
            ),
            make_record(
                family_id="N13-folds",
                variant="mixed",
                params={"leaderboard": PLACE_BLOCK},
                metrics=(
                    fold("wf-1", "return_per_100", 130.0, 110.0, 150.0, 70),
                    dev("return_per_100", 88.0, 84.0, 92.0, 560),
                ),
            ),
        ],
    )

    rows = read_rows(reg)

    assert set(rows) == {"N13-folds/per-fold-only", "N13-folds/mixed"}
    assert_unmeasured(
        rows["N13-folds/per-fold-only"], variant="per-fold-only", kind="cell", pool="WIN"
    )
    assert rows["N13-folds/mixed"].return_per_100 == 88.0
    assert rows["N13-folds/mixed"].n == 560


def test_an_unmeasured_variant_takes_kind_and_pool_from_its_block(tmp_path):
    reg = open_registry(tmp_path)
    append_all(
        reg,
        [
            make_record(
                family_id="N14-pending",
                variant="exacta-around-fav",
                params={"leaderboard": {**WIN_BLOCK, "kind": "rule", "pool": "EXACTA"}},
                status="failed",
            ),
            make_record(
                family_id="N14-pending",
                variant="member-x",
                params={"leaderboard": {**PLACE_BLOCK, "kind": "member"}},
                metrics=(dev("log_loss_gain", 0.002, 0.001, 0.003, 900),),
            ),
            make_record(family_id="N14-pending", variant="no-block", params={"seed": 3}),
        ],
    )

    rows = read_rows(reg)

    assert set(rows) == {
        "N14-pending/exacta-around-fav",
        "N14-pending/member-x",
        "N14-pending/no-block",
    }
    exacta = rows["N14-pending/exacta-around-fav"]
    assert_unmeasured(exacta, variant="exacta-around-fav", kind="rule", pool="EXACTA")
    assert exacta.status == "failed"
    member = rows["N14-pending/member-x"]
    assert_unmeasured(member, variant="member-x", kind="member", pool="PLACE")
    assert member.status == "scored"
    assert_unmeasured(rows["N14-pending/no-block"], variant="no-block")
    assert {r.variants_searched for r in rows.values()} == {3}


# --------------------------------------------------------------------------- withholding


LICENSED_CASES = [
    # (sources, data_builds, components, expected licensed)
    (("netkeiba",), {"results": BUILD_HASH}, {"model": "logit-1.0"}, False),
    (("jra", "netkeiba"), {"results": BUILD_HASH}, {"model": "gbm-2"}, False),
    (("jravan",), {"results": BUILD_HASH}, {"model": "logit-1.0"}, True),
    (("JRA-VAN",), {"results": BUILD_HASH}, {"model": "logit-1.0"}, True),
    (("jra",), {"results": BUILD_HASH}, {"features": "jrdb-idm-v1"}, True),
    (("jra",), {"results": BUILD_HASH}, {"JRDB speed": "v2"}, True),
    (("jra",), {"jrdb_kyi": BUILD_HASH}, {"model": "logit-1.0"}, True),
]


def test_a_licensed_run_gives_withheld_rows(tmp_path):
    reg = open_registry(tmp_path)
    block = {"leaderboard": WIN_BLOCK}
    append_all(
        reg,
        [
            make_record(
                family_id="J15-jravan",
                variant="odds-cell",
                sources=("jravan",),
                params=block,
                metrics=(
                    dev("return_per_100", 93.0, 88.0, 98.0, 700),
                    dev("return_per_100[class:maiden]", 97.0, 85.0, 109.0, 120),
                ),
            ),
            make_record(
                family_id="J16-jrdb",
                variant="idm-cell",
                sources=("jra",),
                components={"features": "jrdb-idm-v1"},
                params=block,
                metrics=(dev("return_per_100", 91.0, 86.0, 96.0, 650),),
            ),
            make_record(
                family_id="J16-jrdb",
                variant="idm-failed",
                sources=("jra",),
                components={"features": "jrdb-idm-v1"},
                status="failed",
            ),
            make_record(
                family_id="J17-public",
                variant="plain-cell",
                sources=("jra",),
                params=block,
                metrics=(dev("return_per_100", 80.0, 76.0, 84.0, 900),),
            ),
        ],
    )

    rows = read_rows(reg)

    assert set(rows) == {
        "J15-jravan/odds-cell",
        "J15-jravan/odds-cell[class:maiden]",
        "J16-jrdb/idm-cell",
        "J16-jrdb/idm-failed",
        "J17-public/plain-cell",
    }
    for option in (
        "J15-jravan/odds-cell",
        "J15-jravan/odds-cell[class:maiden]",
        "J16-jrdb/idm-cell",
        "J16-jrdb/idm-failed",
    ):
        assert rows[option].licensed is True, option
        assert rows[option].withheld, option
    assert rows["J16-jrdb/idm-cell"].sources == ("jra",)
    public = rows["J17-public/plain-cell"]
    assert public.licensed is False
    assert not public.withheld


@pytest.mark.parametrize(("sources", "builds", "components", "expected"), LICENSED_CASES)
def test_is_licensed_record_is_what_the_summary_withholds(sources, builds, components, expected):
    record = make_record(sources=sources, data_builds=builds, components=components)
    assert is_licensed_record(record) is expected
    assert is_licensed_record(record) == registry_module._licensed(record)


# --------------------------------------------------------------------------- holdout


def test_holdout_runs_give_no_row(tmp_path):
    reg = open_registry(tmp_path)
    opened = holdout_open_hash(reg)
    block = {"leaderboard": WIN_BLOCK}
    append_all(
        reg,
        [
            make_record(
                family_id="N18-hold",
                variant="dev-then-holdout",
                ticket=300,
                params=block,
                metrics=(dev("return_per_100", 92.0, 87.0, 97.0, 500),),
            ),
            # appended later, so the latest run in append order, but a holdout run
            make_record(
                family_id="N18-hold",
                variant="dev-then-holdout",
                ticket=301,
                git_sha=OTHER_SHA,
                params=block,
                metrics=(dev("return_per_100", 140.0, 120.0, 160.0, 90),),
                holdout_open=opened,
            ),
            make_record(
                family_id="N18-hold",
                variant="holdout-only",
                ticket=302,
                params=block,
                metrics=(dev("return_per_100[class:maiden]", 150.0, 130.0, 170.0, 40),),
                holdout_open=opened,
            ),
            make_record(
                family_id="N19-holdout-family",
                variant="only",
                ticket=303,
                params=block,
                metrics=(dev("return_per_100", 125.0, 110.0, 140.0, 60),),
                holdout_open=opened,
            ),
        ],
    )

    rows = read_rows(reg)

    assert set(rows) == {"N18-hold/dev-then-holdout"}
    row = rows["N18-hold/dev-then-holdout"]
    assert (row.return_per_100, row.n, row.ticket) == (92.0, 500, 300)
    assert all(r.return_per_100 not in (140.0, 150.0, 125.0) for r in rows.values())


# --------------------------------------------------------------------------- adapters


def test_an_adapter_replaces_the_default_rows_for_its_family(tmp_path):
    reg = open_registry(tmp_path)
    append_all(
        reg,
        [
            make_record(
                family_id="C20-legacy",
                variant="gain-a",
                ticket=81,
                params={"C": 1.0},
                metrics=(dev("roi_pct", 0.912, 0.88, 0.94, 1000),),
            ),
            make_record(
                family_id="C20-legacy",
                variant="gain-b",
                ticket=81,
                params={"C": 0.5},
                metrics=(dev("roi_pct", 0.871, 0.83, 0.91, 800),),
            ),
            make_record(
                family_id="C20-legacy",
                variant="gain-c",
                ticket=81,
                params={"C": 0.25},
                metrics=(dev("log_loss_gain", 0.001, 0.0, 0.002, 800),),
            ),
            # a convention family under an adapter: the adapter's rows replace its own
            make_record(
                family_id="N21-conv",
                variant="cell",
                ticket=82,
                params={"leaderboard": WIN_BLOCK},
                metrics=(dev("return_per_100", 70.0, 65.0, 75.0, 300),),
            ),
            make_record(
                family_id="C22-untouched",
                variant="plain",
                ticket=83,
                params={"C": 2.0},
                metrics=(dev("log_loss_gain", 0.004, 0.002, 0.006, 900),),
            ),
        ],
    )
    seen: list[tuple[str, str, int]] = []

    def legacy(record: RunRecord, size: int) -> list[LeaderboardRow]:
        seen.append((record.family_id, record.variant, size))
        roi = [m for m in record.metrics if m.metric == "roi_pct"]
        return [
            LeaderboardRow(
                option_id=f"{record.family_id}/{record.variant}[adapted]",
                family_id=record.family_id,
                ticket=record.ticket,
                kind="comparator",
                pool="WIN",
                segment="adapted",
                rule=f"adapted {record.variant}",
                selection="card_only",
                sources=record.sources,
                variants_searched=size,
                status=record.status,
                return_per_100=m.value * 100,
                n=m.n,
                n_unit="bets",
            )
            for m in roi
        ]

    def replaced(record: RunRecord, size: int) -> list[LeaderboardRow]:
        seen.append((record.family_id, record.variant, size))
        return [
            LeaderboardRow(
                option_id="N21-conv/replaced",
                family_id="N21-conv",
                ticket=record.ticket,
                kind="model",
                pool="PLACE",
                segment="all",
                rule="replaced",
                selection="decision_time",
                sources=record.sources,
                variants_searched=size,
                status=record.status,
                return_per_100=55.0,
            )
        ]

    rows = read_rows(reg, adapters={"C20-legacy": legacy, "N21-conv": replaced})

    assert set(rows) == {
        "C20-legacy/gain-a[adapted]",
        "C20-legacy/gain-b[adapted]",
        "C20-legacy/gain-c",
        "N21-conv/replaced",
        "C22-untouched/plain",
    }
    assert rows["C20-legacy/gain-a[adapted]"].return_per_100 == pytest.approx(91.2)
    assert rows["C20-legacy/gain-b[adapted]"].return_per_100 == pytest.approx(87.1)
    assert rows["C20-legacy/gain-b[adapted]"].kind == "comparator"
    assert {rows[o].variants_searched for o in rows if o.startswith("C20-legacy/")} == {3}
    # the adapter returned [] for gain-c: the default unmeasured row
    fallback = rows["C20-legacy/gain-c"]
    assert_unmeasured(fallback, variant="gain-c")
    assert fallback.variants_searched == 3
    assert rows["N21-conv/replaced"].return_per_100 == 55.0
    untouched = rows["C22-untouched/plain"]
    assert_unmeasured(untouched, variant="plain")
    # each adapter saw its own family's records only, with the family size
    assert {(f, s) for f, _, s in seen} == {("C20-legacy", 3), ("N21-conv", 1)}
    assert {v for f, v, _ in seen if f == "C20-legacy"} == {"gain-a", "gain-b", "gain-c"}


# --------------------------------------------------------------------------- malformed blocks


MALFORMED_CASES = [
    # (block, words of which the note names at least one)
    ({**WIN_BLOCK, "kind": "bogus"}, ("kind", "bogus")),
    ({**WIN_BLOCK, "selection": "psychic"}, ("selection", "psychic")),
    ({**WIN_BLOCK, "n_unit": "furlongs"}, ("n_unit", "furlongs")),
    ({**WIN_BLOCK, "ci_level": 1.5}, ("ci_level", "1.5")),
    ("flat Win bets", ("leaderboard", "dict", "mapping", "object", "block")),
]


@pytest.mark.parametrize(("block", "words"), MALFORMED_CASES)
def test_a_malformed_block_is_listed_not_raised(tmp_path, block, words):
    reg = open_registry(tmp_path)
    append_all(
        reg,
        [
            make_record(
                family_id="N23-good",
                variant="ok",
                params={"leaderboard": WIN_BLOCK},
                metrics=(dev("return_per_100", 85.0, 81.0, 89.0, 450),),
            ),
            make_record(
                family_id="N24-bad",
                variant="broken",
                ticket=451,
                params={"leaderboard": block},
                metrics=(dev("return_per_100[class:maiden]", 87.5, 80.0, 95.0, 400),),
            ),
            make_record(
                family_id="N24-bad",
                variant="fine",
                params={"leaderboard": PLACE_BLOCK},
                metrics=(dev("return_per_100", 79.0, 75.0, 83.0, 500),),
            ),
            make_record(
                family_id="N25-after",
                variant="later",
                params={"leaderboard": PLACE_BLOCK},
                metrics=(dev("return_per_100", 77.0, 73.0, 81.0, 520),),
            ),
        ],
    )

    rows = read_rows(reg)

    assert set(rows) == {"N23-good/ok", "N24-bad/broken", "N24-bad/fine", "N25-after/later"}
    broken = rows["N24-bad/broken"]
    assert broken.return_per_100 is None and not broken.measured
    assert broken.kind in ("cell", "other")
    assert broken.rule == "broken"
    assert broken.selection == "unknown"
    assert broken.ticket == 451
    assert broken.variants_searched == 2
    assert broken.note, "the malformed block's reason is not in note"
    assert any(word.lower() in broken.note.lower() for word in words), broken.note
    assert rows["N23-good/ok"].return_per_100 == 85.0
    assert rows["N24-bad/fine"].return_per_100 == 79.0
    assert rows["N25-after/later"].return_per_100 == 77.0


# --------------------------------------------------------------------------- review fixes (#588)


def adapted_row(record: RunRecord, size: int, value: float, **extra) -> LeaderboardRow:
    """An adapter's row that copies the record's sources and leaves ``licensed`` as given."""
    return LeaderboardRow(
        option_id=f"{record.family_id}/{record.variant}[adapted]",
        family_id=record.family_id,
        ticket=record.ticket,
        kind="comparator",
        pool="WIN",
        segment="adapted",
        rule=f"adapted {record.variant}",
        selection="card_only",
        sources=record.sources,
        variants_searched=size,
        status=record.status,
        return_per_100=value,
        se_per_100=2.0,
        n=500,
        n_unit="bets",
        **extra,
    )


def test_adapter_rows_of_a_licensed_run_are_withheld(tmp_path):
    reg = open_registry(tmp_path)
    append_all(
        reg,
        [
            # licensed by a component naming JRDB, under a public source name
            make_record(
                family_id="A30-adapted",
                variant="jrdb-comp",
                sources=("jra",),
                components={"features": "jrdb-idm-v1"},
                metrics=(dev("roi_pct", 1.2345, None, None, 500),),
            ),
            # licensed by its source
            make_record(
                family_id="A30-adapted",
                variant="jravan-src",
                sources=("jravan",),
                metrics=(dev("roi_pct", 1.1789, None, None, 500),),
            ),
            # a public run whose row the adapter itself marks licensed
            make_record(
                family_id="A30-adapted",
                variant="marked",
                sources=("netkeiba",),
                metrics=(dev("roi_pct", 1.0617, None, None, 500),),
            ),
            # a public run the adapter leaves alone: not over-withheld
            make_record(
                family_id="A30-adapted",
                variant="public",
                sources=("jra", "netkeiba"),
                metrics=(dev("roi_pct", 0.8432, None, None, 500),),
            ),
        ],
    )

    def adapter(record: RunRecord, size: int) -> list[LeaderboardRow]:
        value = round(record.metrics[0].value * 100, 2)
        extra = {"licensed": True} if record.variant == "marked" else {}
        return [adapted_row(record, size, value, **extra)]

    rows = read_rows(reg, adapters={"A30-adapted": adapter})

    withheld = {
        "A30-adapted/jrdb-comp[adapted]": "123.45",
        "A30-adapted/jravan-src[adapted]": "117.89",
        "A30-adapted/marked[adapted]": "106.17",
    }
    assert set(rows) == {*withheld, "A30-adapted/public[adapted]"}
    for option in withheld:
        assert rows[option].licensed is True, option
        assert rows[option].withheld, option
    public_row = rows["A30-adapted/public[adapted]"]
    assert public_row.licensed is False
    assert not public_row.withheld

    shown = public_view(rows.values())
    assert [r.option_id for r in shown] == ["A30-adapted/public[adapted]"]
    text = render_markdown(list(rows.values()), title="Japan leaderboard", public=True)
    assert "84.32" in text  # the public row's return is shown
    for option, figure in withheld.items():
        assert figure not in text, f"{option}'s return {figure} is in the public text"


LATEST_RUN_CASES = [
    # (id, later runs of R40-latest/v after one measured run with 'all' and 'class:maiden',
    #  expected {option id: (status, return or None, ticket)} for variant v)
    (
        "failed-rerun",
        [{"status": "failed", "metrics": (), "ticket": 611}],
        {"R40-latest/v": ("failed", None, 611)},
    ),
    (
        "rerun-drops-a-segment",
        [{"metrics": (dev("return_per_100", 96.5, 91.0, 102.0, 820),), "ticket": 612}],
        {"R40-latest/v": ("scored", 96.5, 612)},
    ),
    (
        "rerun-keeps-only-a-new-segment",
        [
            {
                "metrics": (dev("return_per_100[class:open]", 73.25, 69.0, 77.5, 640),),
                "ticket": 613,
            }
        ],
        {"R40-latest/v[class:open]": ("scored", 73.25, 613)},
    ),
    (
        "rerun-with-no-convention-metric",
        [{"metrics": (dev("log_loss_gain", 0.002, 0.001, 0.003, 900),), "ticket": 614}],
        {"R40-latest/v": ("scored", None, 614)},
    ),
    (
        "failed-then-measured-again",
        [
            {"status": "failed", "metrics": (), "ticket": 615},
            {
                "metrics": (dev("return_per_100[class:maiden]", 101.0, 94.0, 108.0, 210),),
                "ticket": 616,
            },
        ],
        {"R40-latest/v[class:maiden]": ("scored", 101.0, 616)},
    ),
]


@pytest.mark.parametrize(
    ("later", "expected"),
    [case[1:] for case in LATEST_RUN_CASES],
    ids=[case[0] for case in LATEST_RUN_CASES],
)
def test_the_latest_run_of_a_variant_replaces_all_its_rows(tmp_path, later, expected):
    reg = open_registry(tmp_path)
    block = {"leaderboard": WIN_BLOCK}
    records = [
        make_record(
            family_id="R40-latest",
            variant="v",
            ticket=610,
            params=block,
            metrics=(
                dev("return_per_100", 88.75, 84.0, 93.5, 800),
                dev("return_per_100[class:maiden]", 92.5, 85.0, 100.0, 200),
            ),
        ),
        make_record(
            family_id="R40-latest",
            variant="sibling",
            ticket=620,
            params=block,
            metrics=(dev("return_per_100", 81.5, 77.0, 86.0, 750),),
        ),
    ]
    shas = ["b" * 40, "c" * 40]
    for sha, overrides in zip(shas, later, strict=False):
        records.append(
            make_record(family_id="R40-latest", variant="v", git_sha=sha, params=block, **overrides)
        )
    append_all(reg, records)

    rows = read_rows(reg)

    assert set(rows) == {*expected, "R40-latest/sibling"}
    for option, (status, value, ticket) in expected.items():
        row = rows[option]
        assert row.status == status, option
        assert row.return_per_100 == value, option
        assert row.measured is (value is not None), option
        assert row.ticket == ticket, option
    sibling = rows["R40-latest/sibling"]
    assert (sibling.return_per_100, sibling.ticket) == (81.5, 620)
    # no figure of the superseded run ranks
    assert all(r.return_per_100 not in (88.75, 92.5) for r in rows.values())


def test_a_later_malformed_run_supersedes_a_measured_one(tmp_path):
    reg = open_registry(tmp_path)
    append_all(
        reg,
        [
            make_record(
                family_id="R41-broken-later",
                variant="v",
                params={"leaderboard": WIN_BLOCK},
                metrics=(dev("return_per_100", 89.25, 85.0, 93.5, 600),),
            ),
            make_record(
                family_id="R41-broken-later",
                variant="v",
                git_sha=OTHER_SHA,
                ticket=617,
                params={"leaderboard": {**WIN_BLOCK, "kind": "bogus"}},
                metrics=(dev("return_per_100", 97.0, 92.0, 102.0, 650),),
            ),
        ],
    )

    rows = read_rows(reg)

    assert set(rows) == {"R41-broken-later/v"}
    row = rows["R41-broken-later/v"]
    assert row.return_per_100 is None and not row.measured
    assert row.ticket == 617
    assert row.note and ("kind" in row.note.lower() or "bogus" in row.note.lower()), row.note


@pytest.mark.parametrize(("block", "words"), MALFORMED_CASES[:2] + MALFORMED_CASES[-1:])
def test_a_malformed_block_under_an_empty_adapter_names_the_problem(tmp_path, block, words):
    reg = open_registry(tmp_path)
    append_all(
        reg,
        [
            make_record(
                family_id="A31-empty",
                variant="broken",
                ticket=452,
                params={"leaderboard": block},
                metrics=(dev("return_per_100", 87.5, 80.0, 95.0, 400),),
            ),
            make_record(
                family_id="A31-empty",
                variant="fine",
                params={"leaderboard": PLACE_BLOCK},
                metrics=(dev("return_per_100", 79.0, 75.0, 83.0, 500),),
            ),
        ],
    )

    rows = read_rows(reg, adapters={"A31-empty": lambda record, size: []})

    assert set(rows) == {"A31-empty/broken", "A31-empty/fine"}
    broken = rows["A31-empty/broken"]
    assert broken.return_per_100 is None and not broken.measured
    assert broken.ticket == 452
    assert broken.note, "the malformed block's reason is not in note"
    assert any(word.lower() in broken.note.lower() for word in words), broken.note
    # the adapter, not the convention, gives this family's rows
    assert not rows["A31-empty/fine"].measured
