"""Exotic pool-choice rules L1, L4 and L5 on the same races (#81; 08 §8 A4, §5.1, §4 E11, E13, E16).

Family L (D18): each rule is compared with its stated comparator on the same races, by the paired
difference in realised return at final dividends, one-sided. A rule only has to lose less; nothing
here claims profit, and nothing here adopts a rule (adoption is the owner's, 08 §9.11).

The library (synthetic tests: tests/countries/japan/test_avoidance.py):
- Pool codes and combinations are the store's (docs/specs/store.md): TIERCE is the trifecta,
  FORECAST the exacta; horse numbers joined by ",", ascending in an unordered pool, in finishing
  order in an ordered one, bracket numbers in BRACKET_QUINELLA.
- `Ticket(pool, combination, stake)`, stake in yen. Dividends are per unit stake (jra.yaml, 100
  yen): a ticket on a paying row pays dividend * stake / unit; on an `is_refund` row, its stake.
- `trio_as_trifectas`, `quinella_as_exactas`, `same_bracket_to_quinellas`: the same event in
  another pool. `settle` settles tickets against one race's dividend rows.
- `RaceView(race_id, board, selection)`: `view_at` is the board at t (odds_timeseries through
  `read_asof`); a view of the final price on history carries LOOK_AHEAD.
- `express_set` (L1): a set priced under the threshold goes to the direct pool, at or above it to
  the ordered synthetic. `compare`: the paired per-race difference over the races both sides
  cover, with #69's `paired_bootstrap`. `neutral_ratio` (L5): the ordered pool's payout rate over
  the unordered pool's on the shipped boost calendar.

The study (`run_study`; `python -m horsetrack.countries.japan.studies.avoidance --run [--out DIR]`)
reads the owner's pre-registered inputs from config/avoidance.yaml (no defaults: a missing file or
key raises `MissingOwnerDecision` before anything is read), freezes the family in the run registry,
then reads development rows only (holdout days dropped, nothing before 2014-06-07), scores every
variant, appends one registry record per variant (failures and 'data pending' included) and writes
the report through `write_report`. On history only the winning combination's final price exists,
so the every-set comparisons are computed from the paying rows and the number of sets bet: a
losing set stakes the same and pays nothing on both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.bootstrap import Interval, paired_bootstrap
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day, split_off
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.pricing import clean_odds
from horsetrack.core.store.asof import load_availability, read_asof
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.odds_log import POOL_TOTAL
from horsetrack.countries.japan.studies import calibration
from horsetrack.countries.japan.studies.calibration import (
    MissingOwnerDecision,
    Study,
    Variant,
    _markdown_table,
    _metric,
    _real,
)
from horsetrack.countries.japan.takeout import (
    BoostCalendar,
    RaceConditionsNeeded,
    load_boosts,
    race_takeout,
)

COUNTRY = "japan"
SOURCE = "jra"
LOOK_AHEAD = "final (look-ahead)"
BOARD_AT_T = "board at t"
CARD_ONLY = "card only (no price read)"
CALENDAR = "takeout calendar (no price read)"
BOARD_TABLE = "odds_timeseries"
SYNTHETIC = {"TRIO": "TIERCE", "QUINELLA": "FORECAST"}  # the direct pool -> its ordered synthetic
NEUTRAL_PAIRS = {"TIERCE": "TRIO", "FORECAST": "QUINELLA"}
LEGS = {"TRIO": 3, "TIERCE": 3, "QUINELLA": 2, "FORECAST": 2, "BRACKET_QUINELLA": 2}

__all__ = [
    "MissingOwnerDecision",
    "RaceView",
    "RuleComparison",
    "Ticket",
    "compare",
    "express_set",
    "neutral_ratio",
    "quinella_as_exactas",
    "same_bracket_to_quinellas",
    "settle",
    "trio_as_trifectas",
    "view_at",
]


@dataclass(frozen=True)
class Ticket:
    pool: str
    combination: str
    stake: int  # yen


@dataclass(frozen=True, eq=False)
class RaceView:
    """What a strategy sees of one race: its board (pool, combination, odds decimal with the
    stake) and how that board was selected (LOOK_AHEAD, the board at t, ...)."""

    race_id: str
    board: pd.DataFrame
    selection: str


@dataclass(frozen=True, eq=False)
class RuleComparison:
    """The paired comparison: per paired race, rule minus comparator return (payout / stake)."""

    diff: pd.Series
    interval: Interval | None  # None when no race is paired
    n_races: int
    dropped: list[str]
    selection: str
    reasons: dict[str, str] = field(default_factory=dict)  # dropped race -> why
    rule_return: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    comparator_return: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# --------------------------------------------------------------------------- tickets


def _unit() -> int:
    return int(load_source(COUNTRY, SOURCE).unit_stake)


def _horses(combination: str) -> list[int]:
    try:
        return [int(h) for h in str(combination).split(",")]
    except ValueError:
        raise ValueError(f"{combination!r} is not horse numbers joined by ','") from None


def _joined(horses: Iterable[int]) -> str:
    return ",".join(str(int(h)) for h in horses)


def _orders(combination: str, legs: int) -> list[str]:
    horses = _horses(combination)
    if len(horses) != legs or len(set(horses)) != legs:
        raise ValueError(f"{combination!r} is not {legs} distinct horse numbers")
    return [_joined(p) for p in itertools.permutations(horses)]


def trio_as_trifectas(trio: str) -> list[str]:
    """The six TIERCE finishing orders that make up the TRIO set `trio`."""
    return _orders(trio, 3)


def quinella_as_exactas(pair: str) -> list[str]:
    """The two FORECAST orders that make up the QUINELLA pair `pair`."""
    return _orders(pair, 2)


def same_bracket_to_quinellas(bracket_pair: str, brackets: Mapping[int, int]) -> list[str]:
    """Every QUINELLA on two horses of the bracket of the same-bracket pair `bracket_pair` ("3,3"),
    `brackets` mapping horse number to bracket number."""
    first, *rest = _horses(bracket_pair)
    if rest != [first]:
        raise ValueError(f"{bracket_pair!r} is not a same-bracket pair")
    members = sorted(int(h) for h, b in brackets.items() if int(b) == first)
    return [_joined(p) for p in itertools.combinations(members, 2)]


Paying = dict[tuple[str, str], tuple[float, bool]]


def _paying(rows: pd.DataFrame) -> dict[str, Paying]:
    """race -> {(pool, combination): (dividend, is_refund)}; a key twice is a ValueError (as in
    logger.compact)."""
    out: dict[str, Paying] = {}
    columns = ["race_id", "pool", "combination", "dividend", "is_refund"]
    for race, pool, combination, dividend, refund in rows[columns].itertuples(index=False):
        race_rows = out.setdefault(race, {})
        if (pool, combination) in race_rows:
            raise ValueError(f"{race} {pool} {combination} appears twice in the dividends")
        race_rows[(pool, combination)] = (float(dividend), bool(refund))
    return out


def _settle(tickets: Iterable[Ticket], paying: Paying) -> tuple[int, float, str | None]:
    """(stake, payout, why the race cannot be settled or None)."""
    unit, pools = _unit(), {pool for pool, _ in paying}
    stake, payout, why = 0, 0.0, None if paying else "no dividend rows"
    for ticket in tickets:
        stake += int(ticket.stake)
        if why is None and ticket.pool not in pools:
            why = f"no {ticket.pool} dividend row"
        dividend, refund = paying.get((ticket.pool, ticket.combination), (0.0, False))
        if refund:
            payout += ticket.stake
        elif math.isfinite(dividend):
            payout += dividend * ticket.stake / unit
        else:
            why = f"a paying {ticket.pool} row has no dividend"
    return stake, payout, why


def settle(tickets: Iterable[Ticket], dividends: pd.DataFrame, race_id: str) -> tuple[int, float]:
    """(stake, payout) of `tickets` against `race_id`'s rows of the dividends table. A ticket
    without a row loses; a paying row without a dividend is a ValueError."""
    rows = dividends.loc[dividends["race_id"].to_numpy() == race_id]
    stake, payout, why = _settle(tickets, _paying(rows).get(race_id, {}))
    if why is not None and why.startswith("a paying"):
        raise ValueError(f"{race_id}: {why}")
    return stake, payout


# --------------------------------------------------------------------------- boards and L1


def view_at(con, race_id: str, at_utc: datetime, *, rules, selection: str = BOARD_AT_T) -> RaceView:
    """The board at t: odds_timeseries rows available at `at_utc` under `rules` (`read_asof`:
    no dividend, refund or final-capture column), `race_id`'s only, the latest snapshot per
    (pool, combination) among them."""
    if "look-ahead" in selection:
        raise ValueError(f"the board at t is not look-ahead: {selection!r}")
    rows = read_asof(con, BOARD_TABLE, at_utc, rules=rules)
    rows = rows.loc[rows["race_id"].eq(race_id)]
    order = [c for c in ("snapshot_utc", "fetched_utc") if c in rows.columns]
    rows = rows.sort_values(order, kind="stable") if order else rows
    latest = rows.drop_duplicates(["pool", "combination"], keep="last").reset_index(drop=True)
    return RaceView(race_id, latest, selection)


def _price(board: pd.DataFrame, pool: str, combination: str) -> float:
    odds = board.loc[board["pool"].eq(pool) & board["combination"].eq(combination), "odds"]
    if len(odds) > 1:
        raise ValueError(f"the board prices {pool} {combination} {len(odds)} times")
    return float(clean_odds(odds.to_numpy())[0]) if len(odds) else math.nan


def _check_express(pool: str, combination: str, threshold: float, stake: int) -> None:
    if pool not in SYNTHETIC:
        raise ValueError(f"L1 expresses {sorted(SYNTHETIC)} sets, not {pool!r}")
    horses, k = _horses(combination), math.factorial(LEGS[pool])
    if len(horses) != LEGS[pool] or horses != sorted(set(horses)):
        raise ValueError(f"{combination!r} is not a {pool} combination (ascending, distinct)")
    if not (_real(threshold) and threshold > 0):
        raise ValueError(f"threshold must be decimal odds above 0, got {threshold!r}")
    if not (isinstance(stake, int) and stake > 0 and stake % (k * _unit()) == 0):
        raise ValueError(f"stake must split into {k} whole {_unit()}-yen tickets, got {stake!r}")


def _express(pool: str, combination: str, odds: float, threshold: float, stake: int) -> list:
    if not math.isfinite(odds):
        return []
    if odds < threshold:
        return [Ticket(pool, combination, stake)]
    orders = _orders(combination, LEGS[pool])
    return [Ticket(SYNTHETIC[pool], c, stake // len(orders)) for c in orders]


def express_set(
    view: RaceView, pool: str, combination: str, *, threshold: float, stake: int
) -> list[Ticket]:
    """L1 (E11): the set in the direct pool (TRIO or QUINELLA) when its odds on the view's board
    are under `threshold`, else its ordered synthetic (six TIERCE or two FORECAST tickets, `stake`
    split evenly); no tickets when the board has no price for it (not logged yet: data pending)."""
    _check_express(pool, combination, threshold, stake)
    return _express(pool, combination, _price(view.board, pool, combination), threshold, stake)


# --------------------------------------------------------------------------- compare


def _comparison(
    race_ids: Sequence[str],
    returns: Mapping[str, tuple[float, float]],
    reasons: Mapping[str, str],
    selection: str,
    clusters: pd.Series,
    *,
    n_boot: int,
    level: float,
    seed: int,
) -> RuleComparison:
    """The paired comparison from each paired race's (rule, comparator) return."""
    paired = [r for r in race_ids if r in returns]
    index = pd.Index(paired, name="race_id", dtype=object)
    rule = pd.Series([returns[r][0] for r in paired], index=index, dtype=float)
    comparator = pd.Series([returns[r][1] for r in paired], index=index, dtype=float)
    diff = (rule - comparator).rename("diff")
    boot = {"n_boot": n_boot, "level": level, "seed": seed}
    dropped = [r for r in race_ids if r not in returns]
    return RuleComparison(
        diff=diff,
        interval=paired_bootstrap(diff, clusters, **boot) if paired else None,
        n_races=len(paired),
        dropped=dropped,
        selection=selection,
        reasons={r: reasons.get(r, "not paired") for r in dropped},
        rule_return=rule,
        comparator_return=comparator,
    )


def compare(
    race_ids: Iterable[str],
    rule: Callable[[RaceView], list[Ticket] | None],
    comparator: Callable[[RaceView], list[Ticket] | None],
    *,
    view: Callable[[str], RaceView],
    dividends: pd.DataFrame,
    clusters: pd.Series,
    n_boot: int,
    level: float,
    seed: int,
) -> RuleComparison:
    """The paired difference of `rule` against `comparator` over the races both cover and the
    dividends can settle (no rows, or a bet pool without rows, drops the race): per paired race,
    rule payout / stake minus comparator payout / stake, with `paired_bootstrap` over `clusters`.
    Views with different selection labels are a ValueError."""
    race_ids = list(race_ids)
    if len(set(race_ids)) != len(race_ids):
        raise ValueError("race_ids holds a race twice")
    paying = _paying(dividends.loc[dividends["race_id"].isin(set(race_ids))])
    selection, returns, reasons = None, {}, {}
    for race in race_ids:
        seen = view(race)
        if selection is None:
            selection = seen.selection
        elif seen.selection != selection:
            raise ValueError(f"views disagree on the selection: {selection!r}, {seen.selection!r}")
        sides = (rule(seen), comparator(seen))
        if not sides[0] or not sides[1]:
            reasons[race] = f"the {'rule' if not sides[0] else 'comparator'} does not cover it"
            continue
        settled = [_settle(tickets, paying.get(race, {})) for tickets in sides]
        why = next((w for _, _, w in settled if w), None)
        if why is None and min(s for s, _, _ in settled) <= 0:
            why = "a side stakes nothing"
        if why:
            reasons[race] = why
        else:
            returns[race] = tuple(payout / stake for stake, payout, _ in settled)
    return _comparison(
        race_ids, returns, reasons, selection or "", clusters, n_boot=n_boot, level=level, seed=seed
    )


# --------------------------------------------------------------------------- L5


def neutral_ratio(race_id: str, ordered_pool: str, unordered_pool: str) -> float:
    """The ordered pool's payout rate over the unordered pool's in `race_id` (`race_takeout` on
    the shipped boost calendar): the payout ratio of the synthetic to the direct pool if both
    priced the event alike. TakeoutUnknown before 2014-06-07 (D10)."""
    ordered, unordered = (race_takeout(race_id, p) for p in (ordered_pool, unordered_pool))
    if NEUTRAL_PAIRS.get(ordered_pool) != unordered_pool:
        raise ValueError(f"{ordered_pool} is not the ordered synthetic of {unordered_pool}")
    return ordered.payout_rate / unordered.payout_rate


# =========================================================================== the study

TICKET = 81
FAMILY_ID = "japan-family-l-exotic-pool-choice"
CONFIG_NAME = "avoidance"
BOOTSTRAP_SEED = 81  # reproducibility only; scoring.yaml signs no seed
FOLD = "development"
SCHEME = "paired_same_races:development"  # nothing is fitted: every development race once
STUDY_POOLS = ("BRACKET_QUINELLA", "QUINELLA", "FORECAST", "TRIO", "TIERCE")
SETS = ("every_set",)
TIES = ("lower_horse_no", "drop_race")
TRIFECTA_COMPARATORS = {
    "trio": "the TRIO on the same three horses, the same stake",
    "box": "the six TIERCE orders of the same three horses, the stake split evenly",
}
L5_MODES = ("report", "same_race")
GRID_KEYS = {"TRIO": "l1_threshold_trio", "QUINELLA": "l1_threshold_quinella"}
OWNER_KEYS = {
    "l1_threshold_trio": "L1 (E11) grid of X for TRIO sets, a list of distinct decimal odds above "
    "1.0 ([x] for one), each its own variant (D17): a set priced under X on the board is bet in "
    "TRIO, at X or above as its six TIERCE orders",
    "l1_threshold_quinella": "L1 grid of X for QUINELLA pairs, a list of distinct decimal odds "
    "above 1.0 ([x] for one), each its own variant: under X the QUINELLA, at X or above its two "
    "FORECAST orders",
    "sets": f"which sets L1 and L4 bet, one of {list(SETS)}: every combination of the race's "
    "starters flat (04 §3.4), and for L4 every same-bracket pair of a bracket with two starters",
    "l4_market_order_ties": f"one of {list(TIES)}: how the market-order trifecta ranks equal Win "
    "prices (the lower horse number first, or the race is not bet when a tie changes the ticket)",
    "l4_trifecta_comparators": f"a non-empty list from {list(TRIFECTA_COMPARATORS)}: the "
    "market-order trifecta's comparators (08 §5.1: 'the trio or a box')",
    "l5_mode": f"one of {list(L5_MODES)}: the calendar's rates only, or also per boosted race and "
    "pool the realised return at the paid dividends against the same at the base rate",
    "owner_signoff": "the issue where the owner signed these answers",
}
DECISIONS = {
    "D2": "the board at t is read at post - primary_min of config/decision.yaml (T-2)",
    "D3": "every JRA pool is in research scope (amended 2026-11-24, #127); exotics are tier 2",
    "D7": "meeting (venue x day) clusters, 95% one-sided (config/scoring.yaml)",
    "D10": "races from 2014-06-07 only: every comparison reads a price or the per-race takeout",
    "D14": "holdout days fully locked, outcomes and prices: no holdout row is read",
    "D18": "Family L: a paired difference against a stated comparator on the same races, "
    "one-sided; a rule only has to lose less and never claims profit",
}
RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "status", "race_class_raw"]
RUNNER_COLUMNS = ["race_id", "race_date", "horse_no", "bracket", "is_scratched", "win_odds"]
DIVIDEND_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
BOARD_COLUMNS = [
    "race_id",
    "pool",
    "combination",
    "odds",
    "snapshot_utc",
    "fetched_utc",
    "post_utc",
]


# --------------------------------------------------------------------------- inputs and spec


def _choice(raw: dict, key: str, allowed: Sequence[str]) -> str:
    if raw[key] not in allowed:
        raise ValueError(f"{key} must be one of {list(allowed)}, got {raw[key]!r}")
    return raw[key]


def _grid(raw: dict, key: str) -> list[float]:
    """An L1 threshold grid, ascending: a non-empty list of distinct decimal odds above 1.0. A
    bare number is refused (a one-member grid is [x]), so a stale config cannot shrink the
    family the owner registered."""
    grid = raw[key]
    if not (
        isinstance(grid, list)
        and grid
        and all(_real(x) and x > 1.0 for x in grid)
        and len({float(x) for x in grid}) == len(grid)
    ):
        raise ValueError(
            f"{key} must be a non-empty list of distinct decimal odds above 1.0 ([x] for one), "
            f"got {grid!r}"
        )
    return sorted(float(x) for x in grid)


def _number(x: float) -> str:
    """A threshold as written in a variant name: no exponent, no trailing '.0'."""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def owner_inputs() -> dict:
    """The owner's pre-registered inputs from config/avoidance.yaml, validated."""
    where = f"countries/{COUNTRY}/config/{CONFIG_NAME}.yaml"
    try:
        raw, problem = load_country_yaml(COUNTRY, CONFIG_NAME), "lacks"
    except FileNotFoundError:
        raw, problem = {}, "does not exist; it carries"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping of {sorted(OWNER_KEYS)}")
    missing = [key for key in OWNER_KEYS if key not in raw]
    if missing:
        needed = "; ".join(f"{key}: {OWNER_KEYS[key]}" for key in missing)
        raise MissingOwnerDecision(f"{where} {problem} the owner decision(s) {needed}")
    unknown = sorted(set(raw) - set(OWNER_KEYS))
    if unknown:
        raise ValueError(f"{where} has unknown key(s): {', '.join(unknown)}")
    grids = {key: _grid(raw, key) for key in GRID_KEYS.values()}
    comparators = raw["l4_trifecta_comparators"]
    if not (
        isinstance(comparators, list)
        and comparators
        and len(set(comparators)) == len(comparators)
        and set(comparators) <= set(TRIFECTA_COMPARATORS)
    ):
        raise ValueError(
            f"l4_trifecta_comparators must be a non-empty list from {list(TRIFECTA_COMPARATORS)}"
        )
    signoff = raw["owner_signoff"]
    if not (isinstance(signoff, int) and not isinstance(signoff, bool) and signoff > 0):
        raise ValueError(f"owner_signoff must be a positive issue number, got {signoff!r}")
    return {
        **grids,
        "sets": _choice(raw, "sets", SETS),
        "l4_market_order_ties": _choice(raw, "l4_market_order_ties", TIES),
        "l4_trifecta_comparators": [c for c in TRIFECTA_COMPARATORS if c in comparators],
        "l5_mode": _choice(raw, "l5_mode", L5_MODES),
        "owner_signoff": signoff,
    }


def bootstrap_settings() -> dict:
    """D7's bootstrap (config/scoring.yaml, checked by calibration's reader), this study's seed."""
    return {**calibration.bootstrap_settings(), "seed": BOOTSTRAP_SEED}


def _l1_name(pool: str, threshold: float, tag: str) -> str:
    return f"L1:{pool}:threshold_{_number(threshold)}:{tag}"


def variant_names(owner: dict) -> list[str]:
    """Every variant of the family, in the order the study scores them: L1 per pool, per member
    of its threshold grid, per selection (D17: each grid member counts)."""
    names = [
        _l1_name(pool, x, tag)
        for pool, key in GRID_KEYS.items()
        for x in owner[key]
        for tag in ("final_look_ahead", "board_at_t")
    ]
    names.append("L4:same_bracket_vs_quinellas:card_only")
    names += [
        f"L4:market_order_trifecta_vs_{c}:{tag}"
        for c in owner["l4_trifecta_comparators"]
        for tag in ("final_look_ahead", "board_at_t")
    ]
    names.append("L5:calendar_report")
    if owner["l5_mode"] == "same_race":
        names += [f"L5:{pool}:paid_vs_base_rate" for pool in STUDY_POOLS]
    return names


def build_spec(owner: dict, bootstrap: dict, minutes: int) -> dict:
    """The family's pre-registered spec, frozen in the registry before any race is read."""
    unit, variants = _unit(), variant_names(owner)
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "L1": "E11: a set in the direct pool when short and in the ordered synthetic when "
            "long loses less than always the direct pool, on the same sets",
            "L4": "E13: the same-bracket BRACKET_QUINELLA and the market-order trifecta return "
            "less than the equivalent ticket in another pool",
            "L5": "E16's arithmetic: the payout rate of each exotic pool on boosted races",
        },
        "rules": {
            "L1": {
                "thresholds": {pool: list(owner[key]) for pool, key in GRID_KEYS.items()},
                "rule": "a set priced under the threshold: the direct pool (TRIO, QUINELLA); at or "
                "above it: its ordered synthetic (six TIERCE, two FORECAST), stake split evenly. "
                "Each member of a pool's threshold grid is its own variant",
                "comparator": "always the direct pool, on the same sets",
                "sets": owner["sets"],
                "sets_rule": "every combination of the race's starters, the same stake per set; "
                "at t, those priced on the board",
                "stake_yen": {pool: math.factorial(LEGS[pool]) * unit for pool in SYNTHETIC},
            },
            "L4": {
                "same_bracket": {
                    "rule": "the BRACKET_QUINELLA on the same-bracket pair of each bracket holding "
                    "two or more starters, the same stake per pair",
                    "comparator": "the QUINELLAs on that bracket's pairs of starters, the stake "
                    "split evenly (the same event)",
                    "sets": owner["sets"],
                },
                "market_order_trifecta": {
                    "rule": "the TIERCE on the race's three shortest-priced starters on the Win "
                    "board, in price order",
                    "comparators": {
                        c: TRIFECTA_COMPARATORS[c] for c in owner["l4_trifecta_comparators"]
                    },
                    "ties": owner["l4_market_order_ties"],
                    "stake_yen": 6 * unit,
                },
            },
            "L5": {
                "mode": owner["l5_mode"],
                "pools": list(STUDY_POOLS),
                "rates": "takeout.race_takeout on config/takeout_boosts.yaml, per race and pool",
                "comparator": "ordinary races (the report); same_race: the same race's paying "
                "dividends at the base rate, dividend x base / paid rate floored to the pool's "
                "step and at least its minimum (approximate: breakage on the pool is not rebuilt)",
                "sets": "every combination of the race's starters, the same stake each",
            },
        },
        "difference": "per paired race, the rule's payout / stake minus the comparator's at final "
        "dividends; its mean over paired races, with D7's meeting-clustered paired bootstrap "
        "(D18, one-sided). L4 reads the do-not-bet ticket minus the equivalent ticket",
        "selections": {
            LOOK_AHEAD: "history: the final price (a winning combination's dividend / unit stake; "
            "the final Win odds), so the choice sees the result's price",
            BOARD_AT_T: f"odds_timeseries through read_asof at the race's post_utc - {minutes} min "
            "(post_utc of its latest capture), the latest snapshot per (pool, combination)",
            CARD_ONLY: "no price: the card's brackets and scratches",
            CALENDAR: "no price: the published boost calendar",
        },
        "decision_time_min": minutes,
        "board_fields": {
            "odds_timeseries": BOARD_COLUMNS,
            "runners": RUNNER_COLUMNS,
            "races": RACE_COLUMNS,
            "dividends": DIVIDEND_COLUMNS,
        },
        "starters": "runners not is_scratched; a scratched horse's combinations are never bet",
        "refunds": "refund rows concern scratched horses' combinations only, which no side bets",
        "races": "development races with status 'run' from 2014-06-07 (D10); holdout days unread",
        "variants": variants,
        "family_size": len(variants),  # D17: every grid member counts
        "bootstrap": bootstrap,
        "price_era_start": PRICE_ERA_START.isoformat(),
        "adoption": "none: no rule is adopted here (the owner's, 08 §9.11)",
        "decisions": DECISIONS,
        "owner_signoff": owner["owner_signoff"],
        "sources": [SOURCE],
    }


# --------------------------------------------------------------------------- development rows


@dataclass
class StudyData:
    """Development rows of the price era, and the races with a logged board of each pool."""

    races: pd.DataFrame  # RACE_COLUMNS
    runners: pd.DataFrame  # RUNNER_COLUMNS
    dividends: pd.DataFrame  # DIVIDEND_COLUMNS
    logged: dict[str, list[str]] = field(default_factory=dict)  # pool -> races
    view_at_t: Callable[[str], RaceView] | None = None
    board_files: list[Path] = field(default_factory=list)


def _store_file(name: str) -> Path:
    return paths.processed_dir(SOURCE) / f"{name}.parquet"


def logged_boards(race_ids: set[str]) -> tuple[dict[str, list[str]], dict[str, Path]]:
    """Per pool, the development races with a logged odds_timeseries board (race ids and pool
    codes only: no price is read here), and each race's day file. Holdout days are not opened."""
    folder = paths.processed_dir(SOURCE) / BOARD_TABLE
    logged: dict[str, set[str]] = {}
    files: dict[str, Path] = {}
    for path in sorted(folder.glob("*.parquet")) if folder.is_dir() else []:
        day = date.fromisoformat(path.stem)
        if day < PRICE_ERA_START or is_holdout_day(COUNTRY, day):
            continue
        pairs = pq.read_table(path, columns=["race_id", "pool"]).to_pandas().drop_duplicates()
        for race, pool in pairs.itertuples(index=False):
            if race in race_ids:
                logged.setdefault(pool, set()).add(race)
                files[race] = path
    return {pool: sorted(races) for pool, races in logged.items()}, files


def board_view(files: Mapping[str, Path], rules, minutes: int) -> Callable[[str], RaceView]:
    """race -> its board at t = post_utc (of its latest capture) - `minutes`, through `view_at`."""
    selection = f"{BOARD_AT_T} (T-{minutes})"

    def view(race_id: str) -> RaceView:
        rows = pq.read_table(files[race_id], filters=[("race_id", "==", race_id)])
        posts = rows.select(["snapshot_utc", "post_utc"]).to_pandas().dropna()
        if posts.empty:
            return RaceView(
                race_id, pd.DataFrame(columns=["pool", "combination", "odds"]), selection
            )
        post = posts.sort_values("snapshot_utc")["post_utc"].iloc[-1].to_pydatetime()
        con = duckdb.connect(":memory:")
        try:
            con.register(BOARD_TABLE, rows)
            return view_at(
                con, race_id, post - timedelta(minutes=minutes), rules=rules, selection=selection
            )
        finally:
            con.close()

    return view


def read_development(minutes: int) -> StudyData:
    """Column- and date-scoped store reads from 2014-06-07, holdout days dropped, then
    `split_off`'s development rows; dividends of the study's pools for those races only."""
    frames = {}
    for name, columns in (("races", RACE_COLUMNS), ("runners", RUNNER_COLUMNS)):
        table = read_store_table(
            name, columns=columns, start=PRICE_ERA_START, end=date.max, touches_price=True
        )
        frames[name], _ = split_off(table.to_pandas(date_as_object=False), "race_date", COUNTRY)
    races = frames["races"]
    run = races.loc[races["status"].eq("run"), "race_id"].tolist()
    scope = ds.field("pool").isin(list(STUDY_POOLS)) & ds.field("race_id").isin(run)
    dataset = ds.dataset(_store_file("dividends"), format="parquet")
    dividends = dataset.to_table(columns=DIVIDEND_COLUMNS, filter=scope).to_pandas()
    logged, files = logged_boards(set(run))
    view = board_view(files, load_availability(COUNTRY), minutes)
    return StudyData(races, frames["runners"], dividends, logged, view, sorted(set(files.values())))


def _digest(files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def store_builds(board_files: Sequence[Path]) -> dict[str, str]:
    """Each store file read, as '<path>@<sha256 prefix>'."""
    builds = {
        f"{SOURCE}_{name}": f"processed/{SOURCE}/{name}.parquet@{_digest([_store_file(name)])}"
        for name in ("races", "runners", "dividends")
    }
    if board_files:
        where = f"processed/{SOURCE}/{BOARD_TABLE} ({len(board_files)} days)"
        builds[f"{SOURCE}_{BOARD_TABLE}"] = f"{where}@{_digest(board_files)}"
    return builds


# --------------------------------------------------------------------------- the pure study

Sides = tuple[Callable[[RaceView], list[Ticket] | None], Callable[[RaceView], list[Ticket] | None]]
Paired = tuple[list[str], dict[str, tuple[float, float]], dict[str, str]]


class _Context:
    """The run races, their meetings, starters and dividends, and the frozen spec."""

    def __init__(self, data: StudyData, spec: dict) -> None:
        self.data, self.spec = data, spec
        self.races = data.races.loc[data.races["status"].eq("run")]
        self.race_ids = sorted(self.races["race_id"])
        known = set(self.race_ids)
        runners = data.runners.loc[data.runners["race_id"].isin(known)]
        scratched = runners["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)
        self.starters = runners.loc[~scratched]
        div = data.dividends
        self.dividends = div.loc[div["race_id"].isin(known) & div["pool"].isin(STUDY_POOLS)]
        self.clusters = self.races.set_index("race_id")["meeting_id"]
        self.n_starters = self.starters.groupby("race_id").size()
        self.at_t = f"{BOARD_AT_T} (T-{spec['decision_time_min']})"
        self._win: dict[str, tuple[list[str], np.ndarray]] | None = None
        self._rates: tuple[pd.DataFrame, dict[str, str]] | None = None

    def _final_win(self) -> dict[str, tuple[list[str], np.ndarray]]:
        if self._win is None:
            groups = self.starters.groupby("race_id")[["horse_no", "win_odds"]]
            self._win = {
                race: (
                    [str(int(h)) for h in g["horse_no"]],
                    g["win_odds"].to_numpy(dtype=float, na_value=np.nan),
                )
                for race, g in groups
            }
        return self._win

    def starters_of(self, race_id: str) -> frozenset[int]:
        horses, _ = self._final_win().get(race_id, ([], None))
        return frozenset(int(h) for h in horses)

    def final_win_view(self, race_id: str) -> RaceView:
        """The final Win board of the race's starters, from runners.win_odds: look-ahead."""
        horses, odds = self._final_win().get(race_id, ([], np.zeros(0)))
        board = pd.DataFrame({"pool": "WIN", "combination": horses, "odds": odds})
        return RaceView(race_id, board, LOOK_AHEAD)

    def rates(self) -> tuple[pd.DataFrame, dict[str, str]]:
        if self._rates is None:
            self._rates = _rates(self.races)
        return self._rates

    def boot(self) -> dict:
        boot = self.spec["bootstrap"]
        return {"n_boot": boot["n_boot"], "level": boot["level"], "seed": boot["seed"]}

    def paired(self, paired: Paired, selection: str) -> RuleComparison:
        return _comparison(*paired, selection, self.clusters, **self.boot())

    def compare(self, race_ids: list[str], sides: Sides, view) -> RuleComparison:
        rule, comparator = sides
        kwargs = {"view": view, "dividends": self.dividends, "clusters": self.clusters}
        return compare(race_ids, rule, comparator, **kwargs, **self.boot())

    def logged(self, pool: str) -> list[str]:
        return sorted(set(self.data.logged.get(pool, ())) & set(self.race_ids))


def _first(*checks: tuple[bool, str]) -> str | None:
    return next((why for failed, why in checks if failed), None)


def _pool_rows(div: pd.DataFrame, pools: Sequence[str]) -> tuple[dict, pd.DataFrame, set]:
    """The races with rows in each pool, the paying (not refund) rows, and the races with a
    paying row without a dividend."""
    div = div.loc[div["pool"].isin(pools)]
    has = {p: set(div.loc[div["pool"].eq(p), "race_id"]) for p in pools}
    won = div.loc[~div["is_refund"].astype(bool)]
    return has, won, set(won.loc[won["dividend"].isna(), "race_id"])


def _no_rows(race: str, has: Mapping[str, set], pools: Sequence[str]) -> tuple[bool, str]:
    pool = next((p for p in pools if race not in has[p]), None)
    return pool is not None, f"no {pool} dividend row"


def _l1_history(ctx: _Context, pool: str, threshold: float) -> Paired:
    """L1 on every set of the race's starters at the final price. A losing set stakes the same
    and pays nothing on both sides, so per race the difference comes from the paying sets: the
    rule's payout (the direct dividend under the threshold, else the set's ordered dividends
    over k) minus the direct dividend, over the number of sets."""
    unit, synthetic, legs = _unit(), SYNTHETIC[pool], LEGS[pool]
    k = math.factorial(legs)
    has, won, no_dividend = _pool_rows(ctx.dividends, (pool, synthetic))
    direct = won.loc[won["pool"].eq(pool), ["race_id", "combination", "dividend"]]
    ordered = won.loc[won["pool"].eq(synthetic), ["race_id", "combination", "dividend"]]
    ordered = (
        ordered.assign(combination=[_joined(sorted(_horses(c))) for c in ordered["combination"]])
        .groupby(["race_id", "combination"], as_index=False)["dividend"]
        .sum()
    )
    sets = direct.merge(
        ordered,
        on=["race_id", "combination"],
        how="outer",
        suffixes=("", "_ordered"),
        indicator=True,
    )
    disagree = set(sets.loc[sets["_merge"].ne("both"), "race_id"])
    odds = sets["dividend"] / unit  # the winning set's final price
    rule = np.where(odds < threshold, odds, sets["dividend_ordered"] / unit / k)
    sums = pd.DataFrame({"race_id": sets["race_id"], "rule": rule, "direct": odds})
    sums = sums.groupby("race_id").sum()
    returns, reasons = {}, {}
    for race in ctx.race_ids:
        n = int(ctx.n_starters.get(race, 0))
        why = _first(
            _no_rows(race, has, (pool, synthetic)),
            (n < legs, f"fewer than {legs} starters"),
            (race in no_dividend, "a paying row has no dividend"),
            (race in disagree, f"the paying {pool} and {synthetic} rows disagree"),
            (race not in sums.index, f"no paying {pool} row"),
        )
        if why:
            reasons[race] = why
            continue
        sets_bet = math.comb(n, legs)
        returns[race] = (sums.at[race, "rule"] / sets_bet, sums.at[race, "direct"] / sets_bet)
    return ctx.race_ids, returns, reasons


def _two_legs(rows: pd.DataFrame) -> pd.DataFrame:
    legs = [_horses(c) for c in rows["combination"]]
    if any(len(leg) != 2 for leg in legs):
        raise ValueError("a two-leg pool holds a combination of another length")
    a = pd.Series([leg[0] for leg in legs], index=rows.index, dtype="int64")
    return rows.assign(a=a, b=pd.Series([leg[1] for leg in legs], index=rows.index, dtype="int64"))


def _bracket_sizes(ctx: _Context) -> tuple[pd.DataFrame, set[str], pd.DataFrame]:
    """Starters per (race, bracket), the races with a starter whose bracket is missing, and the
    starters with a bracket."""
    starters = ctx.starters[["race_id", "horse_no", "bracket"]]
    missing = set(starters.loc[starters["bracket"].isna(), "race_id"])
    starters = starters.dropna(subset=["bracket"]).astype({"horse_no": "int64", "bracket": "int64"})
    sizes = starters.groupby(["race_id", "bracket"]).size().rename("size").reset_index()
    return sizes, missing, starters


def _same_bracket_history(ctx: _Context) -> Paired:
    """L4: every same-bracket pair flat against the QUINELLAs within its bracket, the stake of a
    pair split evenly over the bracket's C(k, 2) pairs of starters."""
    unit, pools = _unit(), ("BRACKET_QUINELLA", "QUINELLA")
    has, won, no_dividend = _pool_rows(ctx.dividends, pools)
    sizes, no_bracket, starters = _bracket_sizes(ctx)
    zorome = sizes.loc[sizes["size"] >= 2]
    n_zorome = zorome.groupby("race_id").size()
    bq = _two_legs(won.loc[won["pool"].eq("BRACKET_QUINELLA")])
    bq = bq.loc[bq["a"].eq(bq["b"])].rename(columns={"a": "bracket"})
    q = _two_legs(won.loc[won["pool"].eq("QUINELLA")])
    for leg in ("a", "b"):
        renamed = starters.rename(columns={"horse_no": leg, "bracket": f"bracket_{leg}"})
        q = q.merge(renamed, on=["race_id", leg], how="left")
    q = q.loc[q["bracket_a"].eq(q["bracket_b"])].rename(columns={"bracket_a": "bracket"})
    q = q.merge(zorome, on=["race_id", "bracket"], how="inner")
    keys = [set(zip(f["race_id"], f["bracket"], strict=True)) for f in (bq, q)]
    disagree = {race for race, _ in keys[0] ^ keys[1]}
    rule = (bq["dividend"] / unit).groupby(bq["race_id"]).sum()
    comparator = (
        (q["dividend"] / unit / (q["size"] * (q["size"] - 1) / 2)).groupby(q["race_id"]).sum()
    )
    returns, reasons = {}, {}
    for race in ctx.race_ids:
        pairs = int(n_zorome.get(race, 0))
        why = _first(
            _no_rows(race, has, pools),
            (race in no_bracket, "a starter has no bracket"),
            (pairs == 0, "no bracket holds two starters"),
            (race in no_dividend, "a paying row has no dividend"),
            (race in disagree, "the paying BRACKET_QUINELLA and QUINELLA rows disagree"),
        )
        if why:
            reasons[race] = why
        else:
            returns[race] = (rule.get(race, 0.0) / pairs, comparator.get(race, 0.0) / pairs)
    return ctx.race_ids, returns, reasons


def _priced_sets(view: RaceView, pool: str, starters: frozenset[int]) -> list[tuple[str, float]]:
    """The pool's sets of the race's starters priced on the board."""
    board = view.board
    rows = board.loc[board["pool"].eq(pool) & board["combination"].ne(POOL_TOTAL)]
    odds = clean_odds(rows["odds"].to_numpy())
    return [
        (c, float(o))
        for c, o in zip(rows["combination"], odds, strict=True)
        if not np.isnan(o) and set(_horses(c)) <= starters
    ]


def _l1_sides(ctx: _Context, pool: str, threshold: float, stake: int) -> Sides:
    """L1 at t: every set priced on the board at t, by express_set's rule, against the direct
    pool on the same sets; a race without a priced set is not covered (data pending)."""

    def rule(view: RaceView) -> list[Ticket] | None:
        sets = _priced_sets(view, pool, ctx.starters_of(view.race_id))
        return [t for c, o in sets for t in _express(pool, c, o, threshold, stake)] or None

    def comparator(view: RaceView) -> list[Ticket] | None:
        sets = _priced_sets(view, pool, ctx.starters_of(view.race_id))
        return [Ticket(pool, c, stake) for c, _ in sets] or None

    return rule, comparator


def _market_order(view: RaceView, starters: frozenset[int], ties: str) -> list[int] | None:
    """The three shortest Win prices among the priced starters, in price order; None under three
    or, with ties 'drop_race', when equal prices change the ticket."""
    win = view.board.loc[view.board["pool"].eq("WIN")]
    horses = np.array([int(c) if str(c).isdigit() else -1 for c in win["combination"]], dtype=int)
    odds = clean_odds(win["odds"].to_numpy())
    keep = ~np.isnan(odds) & np.isin(horses, list(starters))
    horses, odds = horses[keep], odds[keep]
    order = np.lexsort((horses, odds))  # by price, then the lower horse number
    horses, odds = horses[order], odds[order]
    if len(horses) < 3:
        return None
    if ties == "drop_race" and bool(
        np.any(odds[1 : min(4, len(odds))] == odds[: min(3, len(odds) - 1)])
    ):
        return None
    return [int(h) for h in horses[:3]]


def _market_order_sides(ctx: _Context, comparator_name: str, ties: str, stake: int) -> Sides:
    def top(view: RaceView) -> list[int] | None:
        return _market_order(view, ctx.starters_of(view.race_id), ties)

    def rule(view: RaceView) -> list[Ticket] | None:
        horses = top(view)
        return None if horses is None else [Ticket("TIERCE", _joined(horses), stake)]

    def comparator(view: RaceView) -> list[Ticket] | None:
        horses = top(view)
        if horses is None:
            return None
        trio = _joined(sorted(horses))
        if comparator_name == "trio":
            return [Ticket("TRIO", trio, stake)]
        return [Ticket("TIERCE", c, stake // 6) for c in trio_as_trifectas(trio)]

    return rule, comparator


def _rates(races: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Per race and pool, the base and paid rate (`race_takeout`, each day's calendar entries),
    and the races whose rate needs a conditions label they lack."""
    calendar, by_day, rows, unrated = load_boosts(), {}, [], {}
    columns = ["race_id", "race_date", "race_class_raw"]
    for race, day, label in races[columns].itertuples(index=False):
        day = pd.Timestamp(day).date()
        if day not in by_day:
            entries = (e for e in calendar.entries if e.date <= day <= (e.until or e.date))
            by_day[day] = BoostCalendar(tuple(entries))
        label = label if isinstance(label, str) else None
        try:
            found = [
                race_takeout(race, p, calendar=by_day[day], race_class_raw=label)
                for p in STUDY_POOLS
            ]
        except RaceConditionsNeeded:
            unrated[race] = "an age-restricted boost matches and the race has no conditions label"
            continue
        rows += [
            (
                race,
                t.pool,
                t.base_rate,
                t.payout_rate,
                t.boost or "ordinary",
                t.boost_may_be_withheld,
            )
            for t in found
        ]
    names = ["race_id", "pool", "base_rate", "payout_rate", "boost", "may_be_withheld"]
    return pd.DataFrame(rows, columns=names), unrated


def _calendar(ctx: _Context) -> tuple[list, dict]:
    """L5's report: payout rates by pool and boost, the takeout-neutral ratios and the unrated."""
    rates, unrated = ctx.rates()
    metrics, by_boost, neutral = [], [], []
    for (pool, boost), part in rates.groupby(["pool", "boost"], sort=True):
        by_boost.append(
            {"pool": pool, "boost": boost, "races": len(part)}
            | {"base rate": part["base_rate"].mean(), "paid rate": part["payout_rate"].mean()}
            | {"boost may be withheld": int(part["may_be_withheld"].sum())}
        )
    for pool, part in rates.groupby("pool", sort=True):
        boosted = part.loc[part["boost"].ne("ordinary")]
        metrics.append(_metric(FOLD, f"boosted_races[{pool}]", len(boosted), len(part)))
        if len(boosted):
            gain = (boosted["payout_rate"] - boosted["base_rate"]).mean()
            metrics.append(_metric(FOLD, f"mean_rate_gain_boosted[{pool}]", gain, len(boosted)))
    paid = rates.pivot(index="race_id", columns="pool", values="payout_rate").astype(float)
    boosts = rates.pivot(index="race_id", columns="pool", values="boost")
    for ordered, unordered in NEUTRAL_PAIRS.items():
        ratio = paid[ordered] / paid[unordered]
        kinds = boosts[ordered] + " / " + boosts[unordered]
        for kind, part in ratio.groupby(kinds, sort=True):
            neutral.append(
                {"ordered / direct": f"{ordered} / {unordered}", "boosts": kind, "races": len(part)}
                | {"neutral ratio": part.mean(), "min": part.min(), "max": part.max()}
            )
    left_out = [{"reason": why, "races": n} for why, n in Counter(unrated.values()).items()]
    return metrics, {
        "calendar_rates": by_boost,
        "calendar_neutral": neutral,
        "calendar_unrated": left_out,
    }


def _sets_bet(pool: str, starters: int, bracket_sizes: Sequence[int]) -> int:
    """Every combination of the race's starters in `pool`."""
    if pool == "BRACKET_QUINELLA":
        return math.comb(len(bracket_sizes), 2) + sum(1 for s in bracket_sizes if s >= 2)
    if pool in NEUTRAL_PAIRS:  # ordered
        return math.perm(starters, LEGS[pool])
    return math.comb(starters, LEGS[pool])


def _same_race(ctx: _Context, pool: str) -> Paired:
    """L5 same_race: on each race the pool was boosted, every combination flat at the paid
    dividends against the same paying rows at the base rate."""
    rates, _ = ctx.rates()
    boosted = rates.loc[rates["pool"].eq(pool) & rates["boost"].ne("ordinary")]
    race_ids = sorted(boosted["race_id"])
    unit, cfg = _unit(), load_source(COUNTRY, SOURCE).pools[pool]
    _, won, no_dividend = _pool_rows(
        ctx.dividends.loc[ctx.dividends["race_id"].isin(set(race_ids))], (pool,)
    )
    paid = {race: g.to_numpy(dtype=float) for race, g in won.groupby("race_id")["dividend"]}
    sizes, no_bracket, _ = _bracket_sizes(ctx)
    sizes = sizes.loc[sizes["race_id"].isin(set(race_ids))]
    by_race = {race: g.tolist() for race, g in sizes.groupby("race_id")["size"]}
    returns, reasons = {}, {}
    for race, base, rate in boosted[["race_id", "base_rate", "payout_rate"]].itertuples(
        index=False
    ):
        sets_bet = _sets_bet(pool, int(ctx.n_starters.get(race, 0)), by_race.get(race, []))
        why = _first(
            (race not in paid, f"no paying {pool} row"),
            (race in no_dividend, "a paying row has no dividend"),
            (pool == "BRACKET_QUINELLA" and race in no_bracket, "a starter has no bracket"),
            (sets_bet == 0, "no combination of starters"),
        )
        if why:
            reasons[race] = why
            continue
        steps = np.floor(paid[race] * base / rate / cfg.dividend_step + 1e-9) * cfg.dividend_step
        at_base = np.maximum(steps, cfg.min_dividend)
        returns[race] = (paid[race].sum() / unit / sets_bet, at_base.sum() / unit / sets_bet)
    return race_ids, returns, reasons


def _score(
    study: Study,
    variant: Variant,
    comparator: str,
    compute: Callable[[], RuleComparison],
    pending: str = "no paired race",
) -> None:
    """One comparison variant: its table rows always; its metrics, or why it has none."""
    row = {"rule": variant.hypothesis_id, "variant": variant.name}
    row |= {"selection": variant.components["selection"], "comparator": comparator}
    try:
        found = compute()
    except Exception as exc:  # every failure is a registry record, never dropped
        study.variants.append(replace(variant, error=f"{type(exc).__name__}: {exc}"))
        study.tables.setdefault("comparisons", []).append(row | {"status": "failed"})
        return
    row |= {"races": found.n_races, "dropped": len(found.dropped)}
    for why, n in Counter(found.reasons.values()).most_common():
        study.tables.setdefault("dropped_reasons", []).append(
            {"variant": variant.name, "reason": why, "races": n}
        )
    study.tables.setdefault("dropped_races", []).extend(
        {"variant": variant.name, "race_id": race, "reason": found.reasons[race]}
        for race in found.dropped
    )
    iv, n = found.interval, found.n_races
    if iv is None:
        study.variants.append(replace(variant, error=pending))
        study.tables.setdefault("comparisons", []).append(row | {"status": pending})
        return
    rule, other = float(found.rule_return.mean()), float(found.comparator_return.mean())
    metrics = (
        _metric(FOLD, "paired_return_difference", iv.mean, n, iv.lo, iv.hi),
        _metric(FOLD, "rule_mean_return", rule, n),
        _metric(FOLD, "comparator_mean_return", other, n),
        _metric(FOLD, "dropped_races", len(found.dropped), n + len(found.dropped)),
        _metric(FOLD, "meetings", iv.n_clusters, n),
    )
    study.variants.append(replace(variant, metrics=metrics))
    study.tables.setdefault("comparisons", []).append(
        row
        | {"meetings": iv.n_clusters, "rule return": rule, "comparator return": other}
        | {"paired difference": iv.mean, "lower": iv.lo, "upper": iv.hi, "status": "scored"}
    )


def compute_study(data: StudyData, spec: dict) -> Study:
    """Every variant of the frozen `spec` on `data`, in `variant_names` order. No I/O beyond the
    board-at-t reads of `data.view_at_t`."""
    ctx = _Context(data, spec)
    study = Study(
        counts={
            "races_read": len(data.races),
            "run_races": len(ctx.race_ids),
            "runners_read": len(data.runners),
            "starters": len(ctx.starters),
            "dividend_rows": len(ctx.dividends),
            "refund_rows": int(ctx.dividends["is_refund"].astype(bool).sum()),
        }
        | {f"races_with_a_logged_{p}_board": len(ctx.logged(p)) for p in ("WIN", *SYNTHETIC)}
    )
    l1 = spec["rules"]["L1"]
    for pool, threshold in ((p, x) for p in SYNTHETIC for x in l1["thresholds"][p]):
        stake = l1["stake_yen"][pool]
        components = {
            "rule": f"{pool} under {_number(threshold)}, else its {SYNTHETIC[pool]} orders"
        }
        components |= {"comparator": f"always {pool}", "sets": l1["sets"]}
        params = {"pool": pool, "threshold": threshold, "stake_yen": stake}
        comparator = f"always {pool}, the same sets"
        history = Variant(
            "L1",
            _l1_name(pool, threshold, "final_look_ahead"),
            SCHEME,
            {**components, "selection": LOOK_AHEAD},
            params,
        )
        _score(
            study,
            history,
            comparator,
            lambda pool=pool, x=threshold: ctx.paired(_l1_history(ctx, pool, x), LOOK_AHEAD),
        )
        at_t = Variant(
            "L1",
            _l1_name(pool, threshold, "board_at_t"),
            SCHEME,
            {**components, "selection": ctx.at_t},
            params,
        )
        sides = _l1_sides(ctx, pool, threshold, stake)
        _score(
            study,
            at_t,
            comparator,
            lambda pool=pool, sides=sides: ctx.compare(ctx.logged(pool), sides, data.view_at_t),
            pending=f"data pending: no {pool} board logged at t (#4, #21)",
        )

    l4 = spec["rules"]["L4"]
    same_bracket = Variant(
        "L4",
        "L4:same_bracket_vs_quinellas:card_only",
        SCHEME,
        {"rule": "same-bracket BRACKET_QUINELLA", "comparator": "QUINELLAs within the bracket"}
        | {"sets": l4["same_bracket"]["sets"], "selection": CARD_ONLY},
        {"stake": "the same per same-bracket pair; split evenly over the bracket's quinellas"},
    )
    _score(
        study,
        same_bracket,
        "the QUINELLAs within the bracket",
        lambda: ctx.paired(_same_bracket_history(ctx), CARD_ONLY),
    )
    market = l4["market_order_trifecta"]
    for name, definition in market["comparators"].items():
        sides = _market_order_sides(ctx, name, market["ties"], market["stake_yen"])
        components = {"rule": "market-order TIERCE", "comparator": name, "ties": market["ties"]}
        params = {"stake_yen": market["stake_yen"], "comparator": definition}
        for tag, selection, races, view, pending in (
            ("final_look_ahead", LOOK_AHEAD, ctx.race_ids, ctx.final_win_view, "no paired race"),
            (
                "board_at_t",
                ctx.at_t,
                ctx.logged("WIN"),
                data.view_at_t,
                "data pending: no WIN board",
            ),
        ):
            variant = Variant(
                "L4",
                f"L4:market_order_trifecta_vs_{name}:{tag}",
                SCHEME,
                {**components, "selection": selection},
                params,
            )
            _score(
                study,
                variant,
                definition,
                lambda sides=sides, races=races, view=view: ctx.compare(races, sides, view),
                pending,
            )

    l5 = spec["rules"]["L5"]
    calendar_report = Variant(
        "L5",
        "L5:calendar_report",
        SCHEME,
        {"rates": "takeout.race_takeout", "selection": CALENDAR},
        {"pools": l5["pools"]},
    )
    study.attempt(calendar_report, lambda: _calendar(ctx))
    if l5["mode"] == "same_race":
        for pool in STUDY_POOLS:
            variant = Variant(
                "L5",
                f"L5:{pool}:paid_vs_base_rate",
                SCHEME,
                {"rule": f"{pool} at the paid rate", "comparator": f"{pool} at the base rate"}
                | {"selection": CALENDAR},
                {"pool": pool},
            )
            _score(
                study,
                variant,
                f"{pool} at the base rate",
                lambda pool=pool: ctx.paired(_same_race(ctx, pool), CALENDAR),
            )
    return study


# --------------------------------------------------------------------------- report and runner

SECTIONS = (
    (
        "comparisons",
        "Paired comparisons: rule minus comparator on the same races",
        "Per paired race: the rule's payout / stake minus the comparator's, at final dividends."
        " 'paired difference' is its mean over the paired races; lower and upper are the"
        " (1 - level) and level quantiles of the paired bootstrap over meetings (D7, one-sided;"
        " D18). L1: a positive difference is the price rule losing less than always the direct"
        " pool. L4: the do-not-bet ticket minus the equivalent ticket, so a negative difference"
        " is the do-not-bet ticket losing more. 'final (look-ahead)' rows choose by the final"
        " price and are look-ahead; 'board at t' rows choose by odds_timeseries at t. No rule is"
        " adopted here.",
    ),
    (
        "dropped_reasons",
        "Dropped races by reason",
        "Every comparison runs on the races both sides cover; each dropped race, with its"
        " reason, is in the companion CSV.",
    ),
    ("calendar_rates", "L5: payout rates by pool and boost", ""),
    (
        "calendar_neutral",
        "L5: the takeout-neutral ratio of the ordered synthetic to the direct pool",
        "The ordered pool's payout rate over the unordered pool's: the payout ratio of the"
        " synthetic to the direct pool if both priced the event alike.",
    ),
    ("calendar_unrated", "L5: races without a rate", ""),
)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """The full report: aggregate tables only (the dropped races go to the companion CSV)."""
    lines = [
        f"# Exotic pool choice, Family L rules L1, L4 and L5 ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        f"- owner sign-off of the pre-registered inputs: #{spec['owner_signoff']}",
        "- development rows only: holdout days are never read and the holdout is not opened",
        "- no rule is adopted here: adoption is the owner's (08 §9.11)",
        "",
        "## Frozen rules",
        "",
        "```json",
        json.dumps(spec["rules"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Owner decisions used",
        "",
        *(f"- {key}: {value}" for key, value in spec["decisions"].items()),
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title, note in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", ""]
        lines += [note, ""] if note else []
        lines.append(_markdown_table(rows) if rows else "_None: see the variants below._")
    variants = [
        {"rule": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | ({"why": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def _record(variant: Variant, git_sha: str, builds: dict[str, str]) -> RunRecord:
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=variant.params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def run_study(out_dir: Path | None = None) -> dict:
    """Run the pre-registered study on Japan's development data and write its report.

    Raises `MissingOwnerDecision` before anything is read when config/avoidance.yaml lacks an
    input, `paths.DataDirInsideRepo` for an `out_dir` inside the repository, and
    `FamilyAlreadyFrozen` when the family is frozen with another spec."""
    owner = owner_inputs()
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    minutes = int(load_country_yaml(COUNTRY, "decision")["primary_min"])
    spec = build_spec(owner, bootstrap_settings(), minutes)
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any race is read or settled

    data = read_development(minutes)
    builds = store_builds(data.board_files)
    study = compute_study(data, spec)
    del data

    existing = registry.records(FAMILY_ID)
    records = [_record(v, git_sha, builds) for v in study.variants]
    appended = [registry.append(r) for r in records if r not in existing]
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    stem = out_dir / f"exotic-pool-choice-{COUNTRY}-{spec_hash[:12]}"
    report = write_report(
        stem.with_suffix(".md"), render_report(study, spec, about), sources=(SOURCE,)
    )
    dropped = pd.DataFrame(
        study.tables.get("dropped_races", []), columns=["variant", "race_id", "reason"]
    )
    csv = write_report(
        stem.with_name(f"{stem.name}-dropped.csv"), dropped.to_csv(index=False), sources=(SOURCE,)
    )
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "dropped_path": str(csv),
        "registry_path": str(registry.path),
        "counts": study.counts,
        "records_appended": len(appended),
        "records_already_present": len(records) - len(appended),
        "failed_or_pending_variants": [v.name for v in study.variants if v.error],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.avoidance")
    parser.add_argument("--run", action="store_true", required=True, help="run the study")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
