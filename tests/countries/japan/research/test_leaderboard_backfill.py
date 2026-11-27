"""Ticket #451: the leaderboard backfilled from the Japan registry's legacy families.

Contract pinned here, module ``horsetrack.countries.japan.research.leaderboard_backfill``:

- ``LEGACY_ADAPTERS``: a mapping family id -> adapter (``core.eval.leaderboard.Adapter``) with
  one entry for each of ``japan-family-l-exotic-pool-choice`` (#81), ``japan-coplacer-place-e15``
  (#83) and ``win5-carryover-v1`` (#135), for ``rows_from_registry(registry, adapters=...)``.
- ``LEGACY_KINDS``: the kind of an unmeasured legacy row, by family.
- ``legacy_rows(registry) -> list[LeaderboardRow]``: ``rows_from_registry(registry,
  adapters=LEGACY_ADAPTERS)`` with the kinds applied and Family L's comparators deduplicated on
  (rule, value, n), keeping the first in append order. It reads the registry and never writes it.

Family L: the rule row is ``'<family>/<variant>'`` (kind 'rule', return 100 x rule_mean_return,
no CI, n races from rule_mean_return, n_meetings from ``meetings``, rule the variant text,
payout_rate None); the comparator row is ``'<family>/<variant>/comparator'`` (kind 'comparator',
rule params['comparator'] or 'always the <pool>'). Pool: params['pool'] for L1, 'BRACKET_QUINELLA'
for L4:same_bracket_vs_quinellas, 'TIERCE' for L4:market_order_trifecta_vs_{trio,box}. Selection
from the variant suffix. A failed variant is its own ``'<family>/<variant>'`` row, unmeasured.

#83: one row per 'raw[<label>]' metric of a 'step1:<fields>:raw' variant, option
``'<family>/<variant>[<label>]'``, segment ``'<fields>|<label>'``; other variants unmeasured.
#135: one row per 'mean_day_value' fold, segment ``'carryover <fold>'``.

Every registry is a temp file under ``tmp_path``; every figure is made up. The real registry is
read only by the ``local`` test, which prints no figure.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import pytest

from horsetrack.core import paths
from horsetrack.core.eval.leaderboard import (
    LeaderboardRow,
    rank,
    render_markdown,
    rows_from_registry,
)
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord

REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"
GIT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
BUILD_HASH = hashlib.sha256(b"synthetic build").hexdigest()

FAMILY_L = "japan-family-l-exotic-pool-choice"
COPLACER = "japan-coplacer-place-e15"
WIN5 = "win5-carryover-v1"


# --------------------------------------------------------------------------- helpers


@pytest.fixture
def reg(tmp_path, monkeypatch) -> Registry:
    """An empty temp registry; the real data directory and holdout logs stay untouched."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "guard-data"))
    logs = {p: p.read_bytes() for p in REAL_COUNTRIES.glob("*/holdout_log.jsonl")}
    path = tmp_path / "data" / "registry" / "japan.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "countries" / "japan" / "holdout_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    yield Registry(path, holdout_log=log)
    assert {p: p.read_bytes() for p in REAL_COUNTRIES.glob("*/holdout_log.jsonl")} == logs


def backfill():
    from horsetrack.countries.japan.research import leaderboard_backfill

    return leaderboard_backfill


def dev(metric: str, value: float, lo=None, hi=None, n: int = 0) -> FoldMetric:
    return FoldMetric(fold="development", metric=metric, value=value, lo=lo, hi=hi, n=n)


def record(**overrides) -> RunRecord:
    fields = {
        "hypothesis_id": "E1",
        "family_id": "F1",
        "variant": "v1",
        "ticket": 451,
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


def l_metrics(rule: float, comparator: float, *, n: int, meetings: int, comp_n=None):
    """Family L's development metrics, shaped like the real ones (made-up values)."""
    comp_n = n if comp_n is None else comp_n
    diff = rule - comparator
    return (
        dev("paired_return_difference", diff, diff - 0.01, diff + 0.01, n),
        dev("rule_mean_return", rule, None, None, n),
        dev("comparator_mean_return", comparator, None, None, comp_n),
        dev("dropped_races", 3.0, None, None, n + 3),
        dev("meetings", float(meetings), None, None, n),
    )


def l_record(variant: str, *, params=None, status="scored", metrics=None, **kw) -> RunRecord:
    if params is None:
        params = {"stake_yen": 100}
        if variant.startswith("L1:"):
            _, pool, threshold, _ = variant.split(":")
            params = {"pool": pool, "stake_yen": 100, "threshold": int(threshold.split("_")[1])}
    if metrics is None:
        metrics = l_metrics(0.6, 0.55, n=1000, meetings=80) if status == "scored" else ()
    return record(
        hypothesis_id=variant.split(":")[0],
        family_id=FAMILY_L,
        variant=variant,
        ticket=81,
        params=params,
        metrics=metrics,
        status=status,
        **kw,
    )


def coplacer_record(variant: str, metrics=(), status="scored", **kw) -> RunRecord:
    return record(
        hypothesis_id="E15",
        family_id=COPLACER,
        variant=variant,
        ticket=83,
        params={"variant": variant},
        metrics=tuple(metrics),
        status=status,
        **kw,
    )


def win5_record(metrics, *, level: float = 0.95) -> RunRecord:
    return record(
        hypothesis_id="P1",
        family_id=WIN5,
        variant="pool-level",
        ticket=135,
        sources=("jra", "netkeiba"),
        params={"one_sided_level": level, "takeout": 0.3},
        metrics=tuple(metrics),
    )


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def seal_legacy(registry: Registry, run: RunRecord) -> None:
    """Append ``run`` as one sealed chain line, bypassing ``append``'s checks: how runs of the
    legacy 'japan/131-exacta-*' families (slash ids, refused by ``append`` since #525) sit in the
    real log (``tests/core/eval/test_registry.py::seal_lines``)."""
    data = asdict(run)
    path = registry.path
    raw = path.read_bytes() if path.exists() else b""
    head = hashlib.sha256(raw[:-1].split(b"\n")[-1]).hexdigest() if raw else "0" * 64
    body = {
        "kind": "run",
        "run_id": hashlib.sha256(_canonical(data).encode("ascii")).hexdigest(),
        "record": data,
        "prev": head,
        "utc": "2026-11-20T08:00:00+00:00",
    }
    body["seal"] = hashlib.sha256(_canonical(body).encode("ascii")).hexdigest()
    with open(path, "ab") as handle:
        handle.write(_canonical(body).encode("ascii") + b"\n")


def read(registry: Registry) -> dict[str, LeaderboardRow]:
    """``legacy_rows`` by option id; ids are unique and the set ranks."""
    rows = backfill().legacy_rows(registry)
    assert isinstance(rows, list)
    assert all(isinstance(r, LeaderboardRow) for r in rows)
    ids = [r.option_id for r in rows]
    assert len(ids) == len(set(ids)), f"duplicate option ids: {sorted(ids)}"
    rank(rows)
    return {r.option_id: r for r in rows}


def family_rows(rows: dict[str, LeaderboardRow], family: str) -> list[LeaderboardRow]:
    return [r for r in rows.values() if r.family_id == family]


def under(rows: dict[str, LeaderboardRow], family: str, variant: str) -> list[str]:
    """The option ids of one variant: '<family>/<variant>' and anything after it."""
    base = f"{family}/{variant}"
    return sorted(
        i for i in rows if i == base or i.startswith(base + "[") or i == base + "/comparator"
    )


def assert_unmeasured(row: LeaderboardRow) -> None:
    assert row.return_per_100 is None and not row.measured
    assert (row.ci_lo, row.ci_hi, row.se_per_100) == (None, None, None)


# --------------------------------------------------------------------------- Family L (#81)


def test_family_l_rules_and_comparators_become_rows(reg):
    variant = "L1:TRIO:threshold_20:final_look_ahead"
    reg.append(l_record(variant, metrics=l_metrics(0.6, 0.55, n=1000, meetings=80)))

    rows = read(reg)

    rule = rows[f"{FAMILY_L}/{variant}"]
    assert rule.kind == "rule"
    assert rule.return_per_100 == pytest.approx(60.0)
    assert (rule.ci_lo, rule.ci_hi, rule.se_per_100) == (None, None, None)
    assert (rule.n, rule.n_unit, rule.n_meetings) == (1000, "races", 80)
    assert rule.pool == "TRIO"
    assert rule.selection == "look_ahead"
    assert rule.rule == variant
    assert rule.payout_rate is None
    assert rule.status == "scored"
    assert rule.ticket == 81
    assert rule.licensed is False

    comp = rows[f"{FAMILY_L}/{variant}/comparator"]
    assert comp.kind == "comparator"
    assert comp.return_per_100 == pytest.approx(55.0)
    assert (comp.ci_lo, comp.ci_hi) == (None, None)
    assert (comp.n, comp.n_unit) == (1000, "races")
    assert comp.rule == "always the TRIO"
    assert comp.payout_rate is None
    assert comp.family_id == FAMILY_L

    assert sorted(r.option_id for r in family_rows(rows, FAMILY_L)) == [
        f"{FAMILY_L}/{variant}",
        f"{FAMILY_L}/{variant}/comparator",
    ]


L_VALUES = [
    # (variant, rule_mean_return, comparator_mean_return, races, comparator races, meetings)
    ("L1:QUINELLA:threshold_5:final_look_ahead", 0.62, 0.64, 2500, 2500, 210),
    ("L1:TRIO:threshold_100:final_look_ahead", 0.41, 0.37, 1800, 1800, 150),
    ("L4:same_bracket_vs_quinellas:card_only", 0.48, 0.53, 1700, 1700, 140),
    ("L4:market_order_trifecta_vs_trio:final_look_ahead", 0.66, 0.79, 900, 900, 75),
    ("L4:market_order_trifecta_vs_box:final_look_ahead", 0.71, 0.69, 640, 610, 52),
]


@pytest.mark.parametrize(("variant", "rule", "comp", "n", "comp_n", "meetings"), L_VALUES)
def test_family_l_rows_carry_each_variants_own_figures(
    reg, variant, rule, comp, n, comp_n, meetings
):
    reg.append(
        l_record(variant, metrics=l_metrics(rule, comp, n=n, meetings=meetings, comp_n=comp_n))
    )

    rows = read(reg)

    rule_row = rows[f"{FAMILY_L}/{variant}"]
    assert rule_row.return_per_100 == pytest.approx(100 * rule)
    assert (rule_row.n, rule_row.n_meetings) == (n, meetings)
    assert rule_row.rule == variant
    comp_row = rows[f"{FAMILY_L}/{variant}/comparator"]
    assert comp_row.return_per_100 == pytest.approx(100 * comp)
    assert comp_row.n == comp_n


L_POOLS = [
    ("L1:TRIO:threshold_20:final_look_ahead", "TRIO"),
    ("L1:QUINELLA:threshold_10:final_look_ahead", "QUINELLA"),
    ("L1:FORECAST:threshold_5:card_only", "FORECAST"),
    ("L4:same_bracket_vs_quinellas:card_only", "BRACKET_QUINELLA"),
    ("L4:market_order_trifecta_vs_trio:final_look_ahead", "TIERCE"),
    ("L4:market_order_trifecta_vs_box:final_look_ahead", "TIERCE"),
]


@pytest.mark.parametrize(("variant", "pool"), L_POOLS)
def test_family_l_pool_comes_from_params_or_the_l4_name(reg, variant, pool):
    reg.append(l_record(variant))

    rule = read(reg)[f"{FAMILY_L}/{variant}"]

    assert rule.pool == pool


def test_family_l_comparator_rule_is_named_by_params_or_its_pool(reg):
    named = "L4:market_order_trifecta_vs_box:final_look_ahead"
    reg.append(l_record(named, params={"stake_yen": 100, "comparator": "the synthetic box"}))
    reg.append(l_record("L1:QUINELLA:threshold_5:final_look_ahead"))

    rows = read(reg)

    assert rows[f"{FAMILY_L}/{named}/comparator"].rule == "the synthetic box"
    q = f"{FAMILY_L}/L1:QUINELLA:threshold_5:final_look_ahead/comparator"
    assert rows[q].rule == "always the QUINELLA"


@pytest.mark.parametrize("first", [20, 50])
def test_family_l_comparators_are_deduplicated(reg, first):
    second = 50 if first == 20 else 20
    shared = {"n": 1000, "meetings": 80}
    reg.append(
        l_record(
            f"L1:TRIO:threshold_{first}:final_look_ahead", metrics=l_metrics(0.6, 0.55, **shared)
        )
    )
    reg.append(
        l_record(
            f"L1:TRIO:threshold_{second}:final_look_ahead", metrics=l_metrics(0.62, 0.55, **shared)
        )
    )
    # the same comparator value on other races, another value, another pool: rows of their own
    reg.append(
        l_record(
            "L1:TRIO:threshold_100:final_look_ahead",
            metrics=l_metrics(0.63, 0.55, n=999, meetings=80),
        )
    )
    reg.append(
        l_record("L1:TRIO:threshold_200:final_look_ahead", metrics=l_metrics(0.64, 0.56, **shared))
    )
    reg.append(
        l_record(
            "L1:QUINELLA:threshold_5:final_look_ahead", metrics=l_metrics(0.61, 0.55, **shared)
        )
    )

    rows = read(reg)

    comparators = sorted(r.option_id for r in family_rows(rows, FAMILY_L) if r.kind == "comparator")
    assert comparators == sorted(
        [
            f"{FAMILY_L}/L1:TRIO:threshold_{first}:final_look_ahead/comparator",
            f"{FAMILY_L}/L1:TRIO:threshold_100:final_look_ahead/comparator",
            f"{FAMILY_L}/L1:TRIO:threshold_200:final_look_ahead/comparator",
            f"{FAMILY_L}/L1:QUINELLA:threshold_5:final_look_ahead/comparator",
        ]
    )
    kept = rows[f"{FAMILY_L}/L1:TRIO:threshold_{first}:final_look_ahead/comparator"]
    assert kept.return_per_100 == pytest.approx(55.0)
    assert kept.n == 1000
    rules = [r for r in family_rows(rows, FAMILY_L) if r.kind == "rule"]
    assert len(rules) == 5, "deduplication drops comparators only, never a rule row"
    assert rows[f"{FAMILY_L}/L1:TRIO:threshold_{second}:final_look_ahead"].return_per_100 == (
        pytest.approx(62.0)
    )


def test_family_l_failed_variants_are_listed_as_failed(reg):
    reg.append(l_record("L1:TRIO:threshold_20:final_look_ahead"))
    reg.append(l_record("L1:TRIO:threshold_20:board_at_t", status="failed"))
    reg.append(l_record("L4:market_order_trifecta_vs_trio:board_at_t", status="failed"))

    rows = read(reg)

    for variant, pool in (
        ("L1:TRIO:threshold_20:board_at_t", "TRIO"),
        ("L4:market_order_trifecta_vs_trio:board_at_t", "TIERCE"),
    ):
        option = f"{FAMILY_L}/{variant}"
        assert under(rows, FAMILY_L, variant) == [option], (
            "a failed variant is one row, no comparator"
        )
        row = rows[option]
        assert row.status == "failed"
        assert row.selection == "decision_time"
        assert row.pool == pool
        assert_unmeasured(row)
    assert rows[f"{FAMILY_L}/L1:TRIO:threshold_20:final_look_ahead"].measured


SUFFIXES = [
    ("L1:TRIO:threshold_20:final_look_ahead", "look_ahead"),
    ("L1:TRIO:threshold_20:board_at_t", "decision_time"),
    ("L4:same_bracket_vs_quinellas:card_only", "card_only"),
    ("L4:market_order_trifecta_vs_box:board_at_t", "decision_time"),
    ("L1:QUINELLA:threshold_5:card_only", "card_only"),
]


@pytest.mark.parametrize(("variant", "selection"), SUFFIXES)
def test_family_l_selection_follows_the_variant_suffix(reg, variant, selection):
    reg.append(l_record(variant, metrics=l_metrics(0.5, 0.45, n=700, meetings=60)))

    rows = read(reg)

    rule = rows[f"{FAMILY_L}/{variant}"]
    assert rule.selection == selection
    assert rule.return_per_100 == pytest.approx(50.0)


def test_family_l_calendar_report_is_listed_unmeasured_as_other(reg):
    reg.append(
        l_record(
            "L5:calendar_report",
            params={"report": "calendar"},
            metrics=(
                dev("boosted_races[TRIO]", 120.0, None, None, 120),
                dev("mean_rate_gain_boosted[TRIO]", 0.05, None, None, 120),
            ),
        )
    )

    rows = read(reg)

    option = f"{FAMILY_L}/L5:calendar_report"
    assert under(rows, FAMILY_L, "L5:calendar_report") == [option]
    assert rows[option].kind == "other"
    assert_unmeasured(rows[option])


# --------------------------------------------------------------------------- co-placer (#83)


COPLACER_RAW = [
    # (variant, label, value, lo, hi, n)
    ("step1:8_plus:raw", "fav ≤ 1.3|rank 2", 0.9, 0.85, 0.95, 500),
    ("step1:8_plus:raw", "fav 1.6–2.0|rank 3", 0.7, 0.66, 0.74, 1200),
    ("step1:all_fields:raw", "fav ≤ 1.3|rank 2", 1.1, 1.0, 1.2, 800),
]


def test_coplacer_raw_returns_become_place_rows(reg):
    by_variant: dict[str, list[FoldMetric]] = {}
    for variant, label, value, lo, hi, n in COPLACER_RAW:
        by_variant.setdefault(variant, []).append(dev(f"raw[{label}]", value, lo, hi, n))
    for variant, metrics in by_variant.items():
        reg.append(coplacer_record(variant, metrics))

    rows = read(reg)

    measured = sorted(r.option_id for r in family_rows(rows, COPLACER) if r.measured)
    assert measured == sorted(f"{COPLACER}/{v}[{label}]" for v, label, *_ in COPLACER_RAW)
    for variant, label, value, lo, hi, n in COPLACER_RAW:
        row = rows[f"{COPLACER}/{variant}[{label}]"]
        fields = variant.split(":")[1]
        assert row.kind == "rule"
        assert row.segment == f"{fields}|{label}"
        assert row.return_per_100 == pytest.approx(100 * value)
        assert row.ci_lo == pytest.approx(100 * lo)
        assert row.ci_hi == pytest.approx(100 * hi)
        assert row.ci_level == pytest.approx(0.95)
        assert row.pool == "PLACE"
        assert row.payout_rate == pytest.approx(0.80)
        assert row.selection == "look_ahead"
        assert (row.n, row.n_unit) == (n, "bets")
        assert row.ticket == 83
        assert row.status == "scored"


def test_coplacer_odds_matched_and_failed_variants_are_unmeasured(reg):
    reg.append(
        coplacer_record("step1:8_plus:raw", [dev("raw[fav ≤ 1.3|rank 2]", 0.9, 0.85, 0.95, 500)])
    )
    unmeasured = {
        "step1:8_plus:odds_matched": (
            "scored",
            [dev("odds_matched[fav ≤ 1.3|rank 2]", 0.05, 0.01, 0.09, 500)],
        ),
        "step1:all_fields:odds_matched": (
            "scored",
            [dev("odds_matched[fav > 2.0|rank 2]", -0.02, -0.04, 0.0, 900)],
        ),
        "step2:decision_time:T-2:ranks_2_3": (
            "failed",
            [
                FoldMetric(
                    fold="logged_days", metric="selections", value=12.0, lo=None, hi=None, n=12
                )
            ],
        ),
        "step3:mechanism:stake_back_share": (
            "failed",
            [
                FoldMetric(
                    fold="logged_days",
                    metric="races_with_place_total",
                    value=30.0,
                    lo=None,
                    hi=None,
                    n=30,
                )
            ],
        ),
    }
    for variant, (status, metrics) in unmeasured.items():
        reg.append(coplacer_record(variant, metrics, status=status))

    rows = read(reg)

    for variant, (status, _) in unmeasured.items():
        option = f"{COPLACER}/{variant}"
        assert under(rows, COPLACER, variant) == [option], f"{variant} is one unmeasured row"
        assert_unmeasured(rows[option])
        assert rows[option].status == status
    assert rows[f"{COPLACER}/step1:8_plus:raw[fav ≤ 1.3|rank 2]"].measured


def test_legacy_adapters_plug_into_rows_from_registry(reg):
    adapters = backfill().LEGACY_ADAPTERS
    assert isinstance(adapters, Mapping)
    assert {FAMILY_L, COPLACER, WIN5} <= set(adapters)
    reg.append(
        coplacer_record(
            "step1:all_fields:raw", [dev("raw[fav ≤ 1.5|rank 2]", 0.88, 0.8, 0.96, 300)]
        )
    )

    rows = {r.option_id: r for r in rows_from_registry(reg, adapters=adapters)}

    row = rows[f"{COPLACER}/step1:all_fields:raw[fav ≤ 1.5|rank 2]"]
    assert row.return_per_100 == pytest.approx(88.0)
    assert row.pool == "PLACE"


# --------------------------------------------------------------------------- WIN5 (#135)


WIN5_BINS = [
    # (fold, value, lo, n)
    ("ratio 1..2", 1.05, 1.02, 4),
    ("ratio 0..0.5", 0.8, 0.77, 7),
    ("ratio 0.5..1", 0.93, 0.9, 6),
]


@pytest.mark.parametrize("level", [0.95, 0.9])
def test_win5_bins_become_rows_with_a_lower_bound(reg, level):
    metrics = [
        FoldMetric(fold=fold, metric="mean_day_value", value=v, lo=lo, hi=None, n=n)
        for fold, v, lo, n in WIN5_BINS
    ]
    metrics.append(
        FoldMetric(
            fold="carryover days before 2014-06-07",
            metric="gross_pool_carryover_days",
            value=8.0,
            lo=None,
            hi=None,
            n=8,
        )
    )
    reg.append(win5_record(metrics, level=level))

    rows = read(reg)

    win5 = {r.segment: r for r in family_rows(rows, WIN5)}
    assert set(win5) == {f"carryover {fold}" for fold, *_ in WIN5_BINS}, (
        "one row per mean_day_value bin; gross_pool_carryover_days gives none"
    )
    for fold, value, lo, n in WIN5_BINS:
        row = win5[f"carryover {fold}"]
        assert row.return_per_100 == pytest.approx(100 * value)
        assert row.ci_lo == pytest.approx(100 * lo)
        assert row.ci_hi is None
        assert row.ci_level == pytest.approx(level)
        assert row.pool == "WIN5"
        assert row.payout_rate == pytest.approx(0.70)
        assert (row.n, row.n_unit) == (n, "days")
        assert row.kind == "cell"
        assert row.rule == "every WIN5 combination, pool-level value"
        assert row.selection == "look_ahead"
        assert row.ticket == 135


# --------------------------------------------------------------------------- the other families


KINDS = [
    ("japan-calibrated-market-final-win", "model"),
    ("japan-crowd-calibration-final-win", "model"),
    ("japan-tail-beta-final-win", "model"),
    ("japan-tail-hinge-final-win", "model"),
    ("japan-beta-clustered-final-win", "model"),
    ("japan-entropy-gradient-tail-conditioned", "model"),
    ("japan-c4-meta-final-win", "model"),
    ("japan-c1-nowcast-final-shares", "model"),
    ("japan-c1-nowcast-final-shares-trial", "model"),
    ("japan-c1-nowcast-final-shares-v2", "model"),
    ("japan-v0-win-residual", "model"),
    ("japan-board-quality-by-horizon", "model"),
    ("japan-c2-rating-final-win", "member"),
    ("japan-speed-top3-c3", "member"),
    ("japan-same-day-bias-final-win", "member"),
    ("japan-track-bias-day-sd", "member"),
    ("japan-track-bias-null-diagnostic", "member"),
    ("japan-late-path-jravan-trial", "member"),
    ("japan/131-exacta-category-v1", "cell"),
    ("japan/131-exacta-logscore-v1", "cell"),
    ("japan/131-exacta-breakage", "cell"),
    ("exacta-own-impact", "other"),
    ("japan-some-family-nobody-named", "other"),
]


@pytest.mark.parametrize("status", ["scored", "failed"])
def test_legacy_families_without_a_return_get_their_kind(reg, status):
    for family, _ in KINDS:
        sources = ("jra", "jrdb") if family == "japan-c4-meta-final-win" else ("netkeiba",)
        metrics = () if status == "failed" else (dev("log_loss_gain", 0.0042, 0.001, 0.007, 3000),)
        run = record(
            family_id=family, variant="v-a", sources=sources, metrics=metrics, status=status
        )
        if "/" in family:  # a legacy slash id: in the real log from before #525, refused today
            seal_legacy(reg, run)
        else:
            reg.append(run)

    rows = read(reg)

    for family, kind in KINDS:
        option = f"{family}/v-a"
        assert [r.option_id for r in family_rows(rows, family)] == [option]
        row = rows[option]
        assert row.kind == kind, family
        assert row.status == status
        assert_unmeasured(row)
    assert rows["japan-c4-meta-final-win/v-a"].withheld
    assert rows["japan-c4-meta-final-win/v-a"].licensed
    assert not rows["japan-c2-rating-final-win/v-a"].withheld
    assert not rows["japan-some-family-nobody-named/v-a"].withheld


def test_legacy_kinds_name_every_listed_family():
    kinds = backfill().LEGACY_KINDS
    assert isinstance(kinds, Mapping)
    for family, kind in KINDS:
        if "/" not in family and family not in (
            "exacta-own-impact",
            "japan-some-family-nobody-named",
        ):
            assert kinds[family] == kind, family


def test_withheld_legacy_rows_never_reach_the_public_text(reg):
    reg.append(
        record(
            family_id="japan-c1-nowcast-final-shares-v2",
            variant="nowcast-a",
            sources=("jravan",),
            metrics=(dev("log_loss_gain", 0.004321, 0.001234, 0.007654, 3000),),
        )
    )
    reg.append(
        record(
            family_id="japan-c4-meta-final-win",
            variant="meta-a",
            sources=("jra", "jrdb"),
            metrics=(dev("log_loss_gain", 0.006789, 0.002468, 0.009753, 3000),),
        )
    )
    # a return-bearing legacy variant built on a licensed source is withheld as well
    reg.append(
        coplacer_record(
            "step1:all_fields:raw",
            [dev("raw[fav ≤ 1.3|rank 2]", 0.9371, 0.8813, 0.9929, 777)],
            sources=("jrdb", "netkeiba"),
        )
    )
    reg.append(
        l_record(
            "L1:TRIO:threshold_20:final_look_ahead",
            metrics=l_metrics(0.6125, 0.55, n=1000, meetings=80),
        )
    )
    rows = backfill().legacy_rows(reg)
    withheld_ids = {
        "japan-c1-nowcast-final-shares-v2/nowcast-a",
        "japan-c4-meta-final-win/meta-a",
        f"{COPLACER}/step1:all_fields:raw[fav ≤ 1.3|rank 2]",
    }
    by_id = {r.option_id: r for r in rows}
    assert withheld_ids <= set(by_id)
    assert all(by_id[i].withheld for i in withheld_ids)

    private = render_markdown(rows, title="Japan leaderboard", public=False)
    public = render_markdown(rows, title="Japan leaderboard", public=True)

    assert "93.71" in private, "the licensed row is measured, so the public check means something"
    head, sep, tail = public.lower().partition("withheld (d21)\n")
    assert sep, "the public text has a withheld (D21) section"
    for option in withheld_ids:
        assert option not in head, f"{option} is in the public table"
        assert option.lower() in tail, f"{option} is not listed as withheld"
    for figure in ("93.71", "88.13", "99.29", "777", "0.0043", "0.0068"):
        assert figure not in public, "a withheld figure reached the public text"
    assert "61.25" in public, "the public Family L row is still shown"


# --------------------------------------------------------------------------- read only


def test_backfill_writes_nothing_to_the_registry(reg, tmp_path):
    reg.append(l_record("L1:TRIO:threshold_20:final_look_ahead"))
    reg.append(l_record("L1:TRIO:threshold_50:final_look_ahead"))
    reg.append(l_record("L1:TRIO:threshold_20:board_at_t", status="failed"))
    reg.append(
        coplacer_record("step1:8_plus:raw", [dev("raw[fav ≤ 1.3|rank 2]", 0.9, 0.85, 0.95, 500)])
    )
    reg.append(win5_record([FoldMetric("ratio 1..2", "mean_day_value", 1.05, 1.02, None, 4)]))
    reg.append(record(family_id="japan-c4-meta-final-win", sources=("jra", "jrdb")))
    before_bytes = reg.path.read_bytes()
    before_records = reg.records()
    before_files = sorted(p for p in tmp_path.rglob("*") if p.is_file())

    first = backfill().legacy_rows(reg)
    second = backfill().legacy_rows(reg)

    assert reg.path.read_bytes() == before_bytes, "the backfill changed the registry file"
    assert reg.records() == before_records
    assert sorted(p for p in tmp_path.rglob("*") if p.is_file()) == before_files
    assert first == second
    assert len(first) >= 6


# --------------------------------------------------------------------------- the real registry


@pytest.mark.local
def test_the_real_registry_backfills(tmp_path):
    try:
        real = paths.data_dir() / "registry" / "japan.jsonl"
    except paths.DataDirNotSet:
        pytest.skip(f"{paths.DATA_DIR_VAR} is not set: the real registry is unavailable")
    if not real.is_file():
        pytest.skip(f"no real Japan registry at {real}")
    # Other sessions append to the real registry concurrently, so run against a snapshot of it,
    # cut at the last complete line. The read-only guarantee on the registry file itself is
    # pinned by test_backfill_writes_nothing_to_the_registry.
    data = real.read_bytes()
    path = tmp_path / "japan.jsonl"
    path.write_bytes(data[: data.rfind(b"\n") + 1])
    registry = Registry(path, holdout_log=REAL_COUNTRIES / "japan" / "holdout_log.jsonl")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    rows = backfill().legacy_rows(registry)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
        "the backfill wrote the registry"
    )
    ids = [r.option_id for r in rows]
    assert len(ids) == len(set(ids)), "an option id appears twice"
    rank(rows)
    on_board = {r.family_id for r in rows}
    families = {r.family_id for r in registry.records() if r.holdout_open is None}
    missing = sorted(families - on_board)
    assert not missing, f"families missing from the board: {missing}"
    render_markdown(rows, title="Japan leaderboard", public=True)
