"""The leaderboard backfilled from the Japan registry's legacy families (ticket #451).

The leaderboard is a view: nothing here re-runs a study or appends a record (the owner's answer
7 on #447). Three legacy families carry returns in shapes of their own, read by adapters for
``core.eval.leaderboard.rows_from_registry``:

- ``japan-family-l-exotic-pool-choice`` (#81): per variant a rule row (100 x
  ``rule_mean_return``, no CI: the bounds belong to the paired difference) and a comparator row
  '<family>/<variant>/comparator' (100 x ``comparator_mean_return``) with its own pool and
  selection (``_l_comparator_side``), n in races; a failed
  variant is its own unmeasured row; 'L5:calendar_report' is unmeasured, kind 'other'. Pools mix
  within the family, so no payout rate.
- ``japan-coplacer-place-e15`` (#83): one Place row per 'raw[<label>]' metric of a
  'step1:<fields>:raw' variant, with its meeting-clustered one-sided 95% bounds, n in bets; the
  odds-matched variants (a difference from an expectation, not a return) and the failed step2
  and step3 variants are left to the core's unmeasured row.
- ``win5-carryover-v1`` (#135): one WIN5 cell per ``mean_day_value`` carryover bin, a lower
  bound only, n in days; a bin of params['ratio_edges'] with no days is an unmeasured cell.

Every other legacy family is unmeasured and takes its kind from ``LEGACY_KINDS``. Withheld rows
(JRDB, JRA-VAN) are marked by the core through ``is_licensed_record``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from datetime import date
from functools import cache

from horsetrack.core import config
from horsetrack.core.eval.leaderboard import Adapter, LeaderboardRow, rows_from_registry
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord

FAMILY_L = "japan-family-l-exotic-pool-choice"
COPLACER = "japan-coplacer-place-e15"
WIN5 = "win5-carryover-v1"

# The day the payout rates are read on: the rates in force when the backfill was specified.
RATE_DAY = date(2026, 11, 23)

_DEVELOPMENT = "development"

LEGACY_KINDS: dict[str, str] = {
    **dict.fromkeys(
        (
            "japan-calibrated-market-final-win",
            "japan-crowd-calibration-final-win",
            "japan-tail-beta-final-win",
            "japan-tail-hinge-final-win",
            "japan-beta-clustered-final-win",
            "japan-entropy-gradient-tail-conditioned",
            "japan-c4-meta-final-win",
            "japan-c1-nowcast-final-shares",
            "japan-c1-nowcast-final-shares-trial",
            "japan-c1-nowcast-final-shares-v2",
            "japan-v0-win-residual",
            "japan-board-quality-by-horizon",
            "japan-cross-pool-t2-v1",
            "japan-cross-pool-t2-trial",
        ),
        "model",
    ),
    **dict.fromkeys(
        (
            "japan-c2-rating-final-win",
            "japan-speed-top3-c3",
            "japan-same-day-bias-final-win",
            "japan-track-bias-day-sd",
            "japan-track-bias-null-diagnostic",
            "japan-late-path-jravan-trial",
            "japan-c2-ability-final-win",
            "japan-c2-ability-final-win-v2",
            "japan-speed-top3-c3-timing",
            "japan-debut-maiden-members-v1",
            "japan-debut-maiden-members-v2",
        ),
        "member",
    ),
    "exacta-own-impact": "other",
}

# #131's exacta families, the slash ids of the legacy log and the hyphen ids of PR #280: cells.
_CELL_PREFIXES = ("japan/131-exacta-", "japan-131-exacta-")


def legacy_kind(family_id: str) -> str | None:
    """The kind of an unmeasured row of ``family_id``, or None when no legacy kind is set."""
    if family_id in LEGACY_KINDS:
        return LEGACY_KINDS[family_id]
    if family_id.startswith(_CELL_PREFIXES):
        return "cell"
    return None


@cache
def payout_rate(pool: str) -> float | None:
    """1 - the pool's takeout on ``RATE_DAY``, from the jra source config."""
    takeout = config.load_source("japan", "jra").pools[pool].takeout_on(RATE_DAY)
    return None if takeout is None else 1.0 - takeout


def _scale(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def _whole(metric: FoldMetric | None) -> int | None:
    """A count metric's value as an int; None when absent or not a whole number."""
    if metric is None or not math.isfinite(metric.value) or not float(metric.value).is_integer():
        return None
    return int(metric.value)


def _development(record: RunRecord) -> dict[str, FoldMetric]:
    return {m.metric: m for m in record.metrics if m.fold == _DEVELOPMENT}


# --------------------------------------------------------------------------- Family L (#81)

_L4_POOLS = {
    "same_bracket_vs_quinellas": "BRACKET_QUINELLA",
    "market_order_trifecta_vs_trio": "TIERCE",
    "market_order_trifecta_vs_box": "TIERCE",
}
# The comparator's own pool, and whether it keeps the rule's selection (#81's avoidance study):
# L1's comparator is the direct pool on every combination of the card's starters and the
# same-bracket comparator the QUINELLAs within the bracket, neither reading a price (card_only);
# the market-order comparators bet the same market order as the rule, so take its selection.
_L4_COMPARATORS = {
    "same_bracket_vs_quinellas": ("QUINELLA", False),
    "market_order_trifecta_vs_trio": ("TRIO", True),
    "market_order_trifecta_vs_box": ("TIERCE", True),
}
_L_SELECTIONS = {
    "final_look_ahead": "look_ahead",
    "board_at_t": "decision_time",
    "card_only": "card_only",
}


def _l_pool(record: RunRecord) -> str | None:
    parts = record.variant.split(":")
    if parts[0] == "L1":
        pool = record.params.get("pool")
        return pool if isinstance(pool, str) and pool.strip() else None
    if parts[0] == "L4" and len(parts) > 1:
        return _L4_POOLS.get(parts[1])
    return None


def _l_comparator_side(record: RunRecord, selection: str) -> tuple[str | None, str]:
    """The comparator's (pool, selection): L1's the rule's pool, card_only; L4's per comparator."""
    parts = record.variant.split(":")
    if parts[0] == "L1":
        return _l_pool(record), "card_only"
    pool, same_selection = _L4_COMPARATORS.get(parts[1] if len(parts) > 1 else "", (None, False))
    return pool, selection if same_selection else "card_only"


def _family_l_rows(record: RunRecord, size: int) -> list[LeaderboardRow]:
    hypothesis = record.variant.split(":")[0]
    if hypothesis not in ("L1", "L4"):  # L5:calendar_report: no return, the core's 'other' row
        return []
    pool = _l_pool(record)
    selection = _L_SELECTIONS.get(record.variant.rsplit(":", 1)[-1], "unknown")
    base = dict(
        family_id=record.family_id,
        ticket=record.ticket,
        pool=pool,
        segment="all",
        selection=selection,
        sources=tuple(record.sources),
        variants_searched=size,
        status=record.status,
        payout_rate=None,
    )
    option = f"{record.family_id}/{record.variant}"
    found = _development(record)
    rule = found.get("rule_mean_return")
    if rule is None:  # failed (board_at_t) or no return: one unmeasured rule row
        return [LeaderboardRow(option_id=option, kind="rule", rule=record.variant, **base)]
    meetings = found.get("meetings")
    rows = [
        LeaderboardRow(
            option_id=option,
            kind="rule",
            rule=record.variant,
            return_per_100=_scale(rule.value),
            n=rule.n,
            n_unit="races",
            n_meetings=_whole(meetings),
            **base,
        )
    ]
    comparator = found.get("comparator_mean_return")
    if comparator is not None:
        named = record.params.get("comparator")
        comp_pool, comp_selection = _l_comparator_side(record, selection)
        rows.append(
            LeaderboardRow(
                option_id=f"{option}/comparator",
                kind="comparator",
                rule=named if isinstance(named, str) and named else f"always the {comp_pool}",
                return_per_100=_scale(comparator.value),
                n=comparator.n,
                n_unit="races",
                **(base | {"pool": comp_pool, "selection": comp_selection}),
            )
        )
    return rows


# --------------------------------------------------------------------------- co-placer (#83)

_COPLACER_LEVEL = 0.95


def _coplacer_rows(record: RunRecord, size: int) -> list[LeaderboardRow]:
    parts = record.variant.split(":")
    if len(parts) != 3 or parts[0] != "step1" or parts[2] != "raw":
        return []  # odds-matched, step2, step3: the core's unmeasured row
    fields = parts[1]
    rows = []
    for metric in record.metrics:
        if not (metric.metric.startswith("raw[") and metric.metric.endswith("]")):
            continue
        label = metric.metric[len("raw[") : -1]
        rows.append(
            LeaderboardRow(
                option_id=f"{record.family_id}/{record.variant}[{label}]",
                family_id=record.family_id,
                ticket=record.ticket,
                kind="rule",
                pool="PLACE",
                segment=f"{fields}|{label}",
                rule=f"{record.variant}[{label}]",
                selection="look_ahead",
                sources=tuple(record.sources),
                variants_searched=size,
                status=record.status,
                return_per_100=_scale(metric.value),
                ci_lo=_scale(metric.lo),
                ci_hi=_scale(metric.hi),
                ci_level=_COPLACER_LEVEL,
                n=metric.n,
                n_unit="bets",
                payout_rate=payout_rate("PLACE"),
            )
        )
    return rows


# --------------------------------------------------------------------------- WIN5 (#135)


_WIN5_METRIC = "mean_day_value"


def _win5_folds(record: RunRecord) -> list[str]:
    """The fold of every bin of params['ratio_edges'], named as ``win5_study._metrics`` names
    it ('ratio <lo>..<hi>', the last 'ratio <lo>..inf'); none without a list of edges."""
    edges = record.params.get("ratio_edges")
    if not isinstance(edges, list):
        return []
    his = [f"{hi}" for hi in edges[1:]] + ["inf"]
    return [f"ratio {lo}..{hi}" for lo, hi in zip(edges, his, strict=True)]


def _win5_rows(record: RunRecord, size: int) -> list[LeaderboardRow]:
    """A row per measured bin; a scored run's bins without days are unmeasured cells."""
    measured = {m.fold: m for m in record.metrics if m.metric == _WIN5_METRIC}
    folds = list(measured)
    if record.status != "failed":
        folds += [fold for fold in _win5_folds(record) if fold not in measured]
    level = record.params.get("one_sided_level")
    rows = []
    for fold in folds:
        segment = f"carryover {fold}"
        metric = measured.get(fold)
        figures = {}
        if metric is not None:
            figures = dict(
                return_per_100=_scale(metric.value),
                ci_lo=_scale(metric.lo),
                ci_hi=_scale(metric.hi),
                ci_level=level,
                n=metric.n,
                n_unit="days",
            )
        rows.append(
            LeaderboardRow(
                option_id=f"{record.family_id}/{record.variant}[{segment}]",
                family_id=record.family_id,
                ticket=record.ticket,
                kind="cell",
                pool="WIN5",
                segment=segment,
                rule="every WIN5 combination, pool-level value",
                selection="look_ahead",
                sources=tuple(record.sources),
                variants_searched=size,
                status=record.status,
                payout_rate=payout_rate("WIN5"),
                **figures,
            )
        )
    return rows


LEGACY_ADAPTERS: Mapping[str, Adapter] = {
    FAMILY_L: _family_l_rows,
    COPLACER: _coplacer_rows,
    WIN5: _win5_rows,
}


# --------------------------------------------------------------------------- the rows


def _dedupe_comparators(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    """Family L's comparators once per (rule, pool, selection, value, n), the first in append
    order kept: the L1 thresholds of one pool share one comparator."""
    seen: set[tuple] = set()
    kept = []
    for row in rows:
        if row.family_id == FAMILY_L and row.kind == "comparator":
            key = (row.rule, row.pool, row.selection, row.return_per_100, row.n)
            if key in seen:
                continue
            seen.add(key)
        kept.append(row)
    return kept


def _with_kind(row: LeaderboardRow) -> LeaderboardRow:
    kind = None if row.measured else legacy_kind(row.family_id)
    return row if kind is None or kind == row.kind else dataclasses.replace(row, kind=kind)


def legacy_rows(registry: Registry) -> list[LeaderboardRow]:
    """``rows_from_registry(registry, adapters=LEGACY_ADAPTERS)``, Family L's comparators
    deduplicated and each unmeasured row of a legacy family given its ``LEGACY_KINDS`` kind; a
    measured row keeps its kind. Reads the registry, never writes it."""
    rows = rows_from_registry(registry, adapters=LEGACY_ADAPTERS)
    return [_with_kind(row) for row in _dedupe_comparators(rows)]
