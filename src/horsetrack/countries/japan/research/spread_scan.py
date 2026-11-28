"""N8: popularity-rank spreads (box, wheel, formation) by race class at the paid dividends (#503).

A spread is a mark card filled from the Win popularity ranks: the favourite over ranks 2-4 on the
exacta, a trio box of ranks 1-5, and so on (`SPREADS`, pre-registered in
docs/specs/spread-scan-prereg.md). In each race the ranks come from the Win board at post - 2 min
(D2's primary T-2, `win_ranks`); each combination of the card is one 100-yen ticket (#139's
`betting.formations` box, wheel and formation, expanded on the rank numbers and mapped to the
horses, `expand`); the card returns the paid dividends of its hitting combinations (`settle`).
The 'lookahead' variant ranks on the final Win board instead: a ceiling, not a strategy.

Per race the cells are 'all', one level of each segment (class with G1, G2 and G3 apart, field,
favourite at T-2, surface, pool size in tertiles) and class x favourite, class x field
(`race_cells`). Per (variant, spread, cell with a bet): the flat return per yen staked (rho), the
D7 meeting bootstrap, and a normal empirical-Bayes shrink toward the spread's own return, the
shrink N1 (#430) uses for Place and Wide: a spread's return is not one combination's odds, so
#78's likelihood does not apply. Every cell is a ranked row, none is closed (owner, 2026-11-27).

`run_scan` reads the jravan store a calendar year at a time, development days from 2014-06-07
only (D10, D14), refuses unless the registry holds every family frozen with its spec, writes the
private report (one file per run, named in its records' `data_builds`, `REPORT_BUILD`) and the
candidate cells through `write_report`, and appends one record per (family, variant), failures
included; a range without a bet is a failed run and writes nothing. D21: every figure stays under
the data directory.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer
from scipy import special

from horsetrack.core import config, env, paths
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day, read_development
from horsetrack.core.eval.registry import (
    FoldMetric,
    Registry,
    RunRecord,
    _canonical,
    _record_data,
    _sha256,
    _spec_hash,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_inside, write_report
from horsetrack.countries.japan.betting import formations
from horsetrack.countries.japan.betting.formations import TicketSet
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.research import pool_scan
from horsetrack.countries.japan.takeout import RaceConditionsNeeded, race_takeout

COUNTRY, SOURCE, STUDY = "japan", "jravan", "spread-scan"
TICKET, HYPOTHESIS_ID = 503, "P503"
FAMILY_PREFIX = "japan-n8-spread-scan"
FOLD_SCHEME = "in-sample:development-days"
# the data_builds entry naming a run's private report after its '@': run provenance, not variant
# identity (`Registry.family_size` ignores what follows a build's last '@'), so a rerun at
# another commit or on a refreshed store is new runs of the same variants (#524)
REPORT_BUILD, NO_REPORT = "report", "not-written"
VARIANTS = ("t2", "lookahead")
STAKE = 100
DEFAULT_SEED, DEFAULT_N_BOOT = 20261127, 1000
ORDERED_POOLS = frozenset({"FORECAST", "TIERCE"})
POOL_ROW = pool_scan.POOL_ROW
CLASS_LEVELS = {
    "DEBUT": "newcomer",
    "MAIDEN": "maiden",
    "C1": "conditions",
    "C2": "conditions",
    "C3": "conditions",
    "OP": "open",
    "L": "open",
    "G3": "G3",
    "G2": "G2",
    "G1": "G1",
}
SEGMENT_KINDS = ("all", "class", "field", "fav", "surface", "pool_size", "class+fav", "class+field")

RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "race_class", "surface", "status"]
RUNNER_COLUMNS = ["race_id", "horse_no", "bracket", "is_scratched"]
PAID_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
BOARD_COLUMNS = ["race_id", "pool", "combination", "snapshot_utc", "post_utc", "odds", "investment"]
FINAL_COLUMNS = ["race_id", "combination", "odds"]


# --------------------------------------------------------------------------- the spreads


@dataclass(frozen=True)
class Spread:
    """A mark card in Win popularity ranks. `kind` box (legs: the boxed ranks), wheel (legs: the
    axis, then the opponents; `positions` the axis's places on an ordered pool), formation (one
    leg per finishing position) or bracket (the favourite's bracket with those of ranks 2 and 3)."""

    pool: str
    kind: str
    legs: tuple[tuple[int, ...], ...]
    positions: tuple[int, ...] | None = None

    @property
    def top_rank(self) -> int:
        return max(max(leg) for leg in self.legs)


def _r(a: int, b: int) -> tuple[int, ...]:
    return tuple(range(a, b + 1))


def _wheel(pool: str, opponents: tuple[int, ...], at: int | None = None) -> Spread:
    return Spread(pool, "wheel", ((1,), opponents), None if at is None else (at,))


SPREADS: dict[str, Spread] = {
    "exacta:1/2-3": _wheel("FORECAST", _r(2, 3), 1),
    "exacta:1/2-4": _wheel("FORECAST", _r(2, 4), 1),
    "exacta:1/2-5": _wheel("FORECAST", _r(2, 5), 1),
    "exacta:2-3/1": _wheel("FORECAST", _r(2, 3), 2),
    "exacta:2-4/1": _wheel("FORECAST", _r(2, 4), 2),
    "exacta:2-5/1": _wheel("FORECAST", _r(2, 5), 2),
    "exacta:box1-3": Spread("FORECAST", "box", (_r(1, 3),)),
    "exacta:box1-4": Spread("FORECAST", "box", (_r(1, 4),)),
    "quinella:1/2-3": _wheel("QUINELLA", _r(2, 3)),
    "quinella:1/2-4": _wheel("QUINELLA", _r(2, 4)),
    "quinella:1/2-5": _wheel("QUINELLA", _r(2, 5)),
    "quinella:box1-4": Spread("QUINELLA", "box", (_r(1, 4),)),
    "quinella:box2-5": Spread("QUINELLA", "box", (_r(2, 5),)),
    "wide:1/2-3": _wheel("QUINELLA_PLACE", _r(2, 3)),
    "wide:1/2-4": _wheel("QUINELLA_PLACE", _r(2, 4)),
    "wide:1/2-5": _wheel("QUINELLA_PLACE", _r(2, 5)),
    "wide:box1-4": Spread("QUINELLA_PLACE", "box", (_r(1, 4),)),
    "wide:box2-5": Spread("QUINELLA_PLACE", "box", (_r(2, 5),)),
    "trio:1/2-5": _wheel("TRIO", _r(2, 5)),
    "trio:1/2-6": _wheel("TRIO", _r(2, 6)),
    "trio:box1-4": Spread("TRIO", "box", (_r(1, 4),)),
    "trio:box1-5": Spread("TRIO", "box", (_r(1, 5),)),
    "trio:1-2/1-4/2-6": Spread("TRIO", "formation", (_r(1, 2), _r(1, 4), _r(2, 6))),
    "trifecta:1/2-4/2-4": _wheel("TIERCE", _r(2, 4), 1),
    "trifecta:1/2-4/2-7": Spread("TIERCE", "formation", ((1,), _r(2, 4), _r(2, 7))),
    "trifecta:2-4/1/2-7": Spread("TIERCE", "formation", (_r(2, 4), (1,), _r(2, 7))),
    "bracket:1/2-3": Spread("BRACKET_QUINELLA", "bracket", ((1,), _r(2, 3))),
}
SPREAD_IDS = tuple(SPREADS)
FAMILIES: dict[str, tuple[str, ...]] = {
    f"{FAMILY_PREFIX}-{name}": tuple(s for s in SPREADS if s.split(":")[0] == name)
    for name in ("exacta", "quinella", "wide", "trio", "trifecta", "bracket")
}
FAMILY_OF = {s: family for family, spreads in FAMILIES.items() for s in spreads}


@cache
def _rank_card(spread_id: str) -> tuple[tuple[int, ...], ...]:
    """The card's combinations on the rank numbers themselves (#139's expansion)."""
    s = SPREADS[spread_id]
    if s.kind == "box":
        return formations.box(s.pool, s.legs[0]).combinations
    if s.kind == "wheel":
        return formations.wheel(s.pool, s.legs[0], s.legs[1], positions=s.positions).combinations
    return formations.formation(s.pool, s.legs).combinations


def expand(
    spread_id: str, ranked: Sequence[int], *, brackets: Mapping[int, int] | None = None
) -> TicketSet:
    """The distinct combinations the spread's mark card buys when `ranked` is the field in Win
    popularity order (rank 1 first); one 100-yen ticket each. ValueError when the ranking is shorter
    than the spread's highest rank (no bet: the scan skips that race), names a horse twice, or, for
    the bracket spread, `brackets` (horse -> bracket) does not cover ranks 1-3."""
    spread = SPREADS[spread_id]
    horses = [int(h) for h in ranked]
    if len(set(horses)) != len(horses):
        raise ValueError(f"the ranking names a horse twice: {horses!r}")
    if len(horses) < spread.top_rank:
        raise ValueError(f"{spread_id} needs ranks 1-{spread.top_rank}, the board ranks {horses!r}")
    if spread.kind == "bracket":
        try:
            fav, *others = (int(brackets[horses[r - 1]]) for r in (1, 2, 3))  # type: ignore[index]
        except (KeyError, TypeError):
            raise ValueError(f"{spread_id} needs the brackets of ranks 1-3") from None
        return TicketSet(spread.pool, tuple(sorted({tuple(sorted((fav, b))) for b in others})))
    ordered = spread.pool in ORDERED_POOLS
    mapped = (tuple(horses[r - 1] for r in card) for card in _rank_card(spread_id))
    return TicketSet(spread.pool, tuple(sorted(c if ordered else tuple(sorted(c)) for c in mapped)))


# --------------------------------------------------------------------------- boards and pricing


def _latest_win(board: pd.DataFrame, cutoff: Any) -> pd.DataFrame:
    """Each (race, combination)'s latest WIN row at or before `cutoff` (a time, or one per row)."""
    keep = (board["pool"].to_numpy() == "WIN") & (board["snapshot_utc"] <= cutoff).to_numpy()
    rows = board.loc[keep].sort_values(["race_id", "combination", "snapshot_utc"], kind="stable")
    return rows.drop_duplicates(["race_id", "combination"], keep="last")


def _orders(rows: pd.DataFrame, scratched: set[tuple[str, int]]) -> dict[str, tuple[list, float]]:
    """race -> (horse numbers by ascending price, ties by the lower number; the first's price) from
    one row per (race, combination); the pool row, unpriced and `scratched` runners dropped."""
    rows = rows.loc[rows["combination"].to_numpy() != POOL_ROW]
    if rows.empty:  # e.g. a day file the builder wrote from the O2 feed alone: no WIN row
        return {}
    odds = pd.to_numeric(rows["odds"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    frame = pd.DataFrame(
        {
            "race_id": rows["race_id"].to_numpy(),
            "horse": rows["combination"].astype(int).to_numpy(),
            "odds": odds,
        }
    )
    keep = np.isfinite(frame["odds"].to_numpy()) & (frame["odds"].to_numpy() > 0)
    if scratched:
        keep &= np.fromiter(
            (
                (r, int(h)) not in scratched
                for r, h in zip(frame["race_id"], frame["horse"], strict=True)
            ),
            dtype=bool,
            count=len(frame),
        )
    frame = frame.loc[keep].sort_values(["race_id", "odds", "horse"], kind="stable")
    return {
        race: ([int(h) for h in part["horse"]], float(part["odds"].iloc[0]))
        for race, part in frame.groupby("race_id", sort=False)
    }


def win_ranks(board: pd.DataFrame, *, at_utc: Any, scratched: Iterable[int] = ()) -> list[int]:
    """The horse numbers in Win popularity order on the board at `at_utc` (module docstring)."""
    latest = _latest_win(board, pd.Timestamp(at_utc))
    gone = {(race, int(h)) for race in latest["race_id"].unique() for h in scratched}
    orders = _orders(latest, gone)
    return next(iter(orders.values()))[0] if orders else []


@dataclass(frozen=True)
class Settlement:
    cost: int  # yen: 100 per ticket
    ret: float  # yen returned


Paid = dict[str, tuple[dict[tuple[int, ...], float], set[tuple[int, ...]]]]


def _paid(rows: pd.DataFrame) -> Paid:
    """pool -> (combination -> summed dividend per 100 yen of its paying rows, refunded set)."""
    out: Paid = {}
    refund = rows["is_refund"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    values = rows[["pool", "combination", "dividend"]].itertuples(index=False, name=None)
    for (pool, text, dividend), refunded in zip(values, refund, strict=True):
        combo = tuple(int(x) for x in str(text).split(","))
        if pool not in ORDERED_POOLS:
            combo = tuple(sorted(combo))
        paying, refunds = out.setdefault(pool, ({}, set()))
        if refunded:
            refunds.add(combo)
            continue
        value = float(dividend) if dividend is not None else float("nan")
        if not np.isfinite(value):
            raise ValueError(f"a paying {pool} {text} row has no dividend")
        paying[combo] = paying.get(combo, 0.0) + value
    return out


def _settle(tickets: TicketSet, paid: Paid) -> Settlement:
    paying, refunds = paid.get(tickets.pool, ({}, set()))
    ordered = tickets.pool in ORDERED_POOLS
    ret = 0.0
    for combo in tickets.combinations:
        key = tuple(int(h) for h in combo)
        key = key if ordered else tuple(sorted(key))
        ret += STAKE if key in refunds else paying.get(key, 0.0) * STAKE / 100
    return Settlement(STAKE * tickets.n_tickets, ret)


def settle(tickets: TicketSet, paid: pd.DataFrame) -> Settlement:
    """Cost and return of `tickets` against one race's `dividends` rows (module docstring)."""
    return _settle(tickets, _paid(paid))


# --------------------------------------------------------------------------- segments


def class_level(race_class: Any) -> str:
    """The class level of a `races.race_class` code; ValueError for a null or unknown one."""
    if not isinstance(race_class, str) or race_class not in CLASS_LEVELS:
        raise ValueError(f"unknown race class {race_class!r} (have {sorted(CLASS_LEVELS)})")
    return CLASS_LEVELS[race_class]


def race_cells(
    *,
    race_class: str,
    field_size: int,
    fav_odds: float,
    pool_total: float,
    pool_edges: tuple[float, float],
    surface: str,
) -> frozenset[str]:
    """'all', one level of each segment and the two class crosses; the field, favourite and pool
    size bands are `pool_scan.race_cells`'s (which also validates them)."""
    level = class_level(race_class)
    marginal = pool_scan.race_cells(
        race_class=race_class,
        field_size=field_size,
        fav_odds=fav_odds,
        pool_total=pool_total,
        pool_edges=pool_edges,
        surface=surface,
        venue="",
    )
    band = dict(cell.split(":", 1) for cell in marginal if cell != "all")
    return frozenset(
        {
            "all",
            f"class:{level}",
            f"field:{band['field']}",
            f"fav:{band['fav']}",
            f"surface:{surface}",
            f"pool_size:{band['pool_size']}",
            f"class:{level}+fav:{band['fav']}",
            f"class:{level}+field:{band['field']}",
        }
    )


def pool_tertiles(totals: Iterable[float]) -> tuple[float, float]:
    """The 1/3 and 2/3 quantiles (numpy, linear) of `totals`: `race_cells`' pool-size edges."""
    values = np.asarray([float(t) for t in totals], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("pool totals must be one or more finite numbers")
    low, high = np.quantile(values, [1 / 3, 2 / 3])
    return float(low), float(high)


def _kind(label: str) -> str:
    return "+".join(part.split(":")[0] for part in label.split("+"))


def _parts(label: str) -> dict[str, str]:
    return {} if label == "all" else dict(part.split(":", 1) for part in label.split("+"))


# --------------------------------------------------------------------------- the spec


def _decision_minutes() -> int:
    return int(config.load_country_yaml(COUNTRY, "decision")["primary_min"])


def _level() -> float:
    boot = config.load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["cluster_unit"] != "meeting" or boot["sided"] != "one-sided":
        raise ValueError(f"the scan implements meeting clusters, one-sided; scoring.yaml: {boot}")
    return float(boot["level"])


def _common_spec() -> dict[str, Any]:
    return {
        "study": STUDY,
        "ticket": TICKET,
        "hypothesis_id": HYPOTHESIS_ID,
        "sources": [SOURCE],
        "hypothesis": "some pre-registered spread, bought on the T-2 Win ranks, returns more than "
        "its stake at the paid dividends in some cell (shrunk_edge above 0); the null, the "
        "starting point: no spread beats the takeout in any cell",
        "stake_yen_per_ticket": STAKE,
        "rank_rule": "Win popularity rank at post_utc - primary_min of config/decision.yaml "
        "(D2, T-2): per runner its latest odds_timeseries WIN row with snapshot_utc at or before "
        "that time; a runner without a price there, or is_scratched in runners, is dropped before "
        "ranking; ascending price, ties by the lower horse number",
        "decision_minutes": _decision_minutes(),
        "variants": {
            "t2": "ranks from the T-2 Win board: the strategy",
            "lookahead": "ranks from the final Win board (final_odds WIN rows, the same drops): "
            "a labelled ceiling, not a strategy; same races, same cells",
        },
        "bet": "per race every spread is one mark card: its distinct combinations "
        "(#139's box, wheel and formation on the ranks, mapped to the horses), 100 yen each",
        "pricing": "a ticket returns the dividend of every paying dividends row of its pool with "
        "its combination (dead heats and Wide pay several), 100 yen on an is_refund row, else 0",
        "no_bet": [
            "the board ranks fewer runners than the spread's highest rank (each spread keeps one "
            "definition; counted per spread in the report)",
            "the race has no dividends row of the spread's pool (pool not sold or not settled)",
            "bracket:1/2-3 when a rank 1-3 runner has no bracket",
        ],
        "race_skipped": [
            "status other than run",
            "no T-2 Win board, no T-2 Win pool total or no final Win board (both variants bet "
            "the same races)",
            "segments undefined (race_cells refuses)",
        ],
        "segments": {
            "class": {"newcomer": ["DEBUT"], "maiden": ["MAIDEN"], "conditions": ["C1", "C2", "C3"]}
            | {"open": ["OP", "L"], "G3": ["G3"], "G2": ["G2"], "G1": ["G1"]},
            "field": "starters (runners not is_scratched): le4, 5-10, 11-14, 15-18",
            "fav": "the T-2 favourite's Win price: le1.9, 2.0-2.9, 3.0-4.9, ge5.0, lower edge "
            "inclusive (both variants)",
            "surface": "races.surface",
            "pool_size": "the T-2 Win __pool__ investment; tertiles per calendar year of the "
            "year's scored races (numpy linear 1/3 and 2/3 quantiles): low below the first edge, "
            "mid below the second, high from it",
            "cells": "per race 'all', class, field, fav, surface, pool_size, class+fav, "
            "class+field; a row per cell with at least one bet",
        },
        "metric": {
            "mean_return": "rho = total return / total stake (100 yen x tickets)",
            "n_bets": "races the spread was bought in; n_tickets its combinations",
            "payout_rate": "stake-weighted mean of takeout.race_takeout(race, pool).payout_rate; "
            "the base rate where an age-restricted boost entry matches (the jravan class code "
            "carries no age label); descriptive, the return is the paid dividends",
            "mean_odds": "mean Win price of the variant board's favourite over the bets",
        },
        "bootstrap": {
            "unit": "meeting (races.meeting_id), D7",
            "level": _level(),
            "draws": "rng.multinomial(n meetings, uniform) per replicate, numpy default_rng(seed), "
            "shared by every (variant, spread, cell); a replicate without the cell's meetings is "
            "skipped",
            "outputs": "se_mean_return the replicates' sd of sum(return) / sum(stake); ci_lo and "
            "ci_hi their (1 - level) and level quantiles (one-sided bounds each)",
            "run_of_record": {"seed": DEFAULT_SEED, "n_boot": DEFAULT_N_BOOT},
        },
        "shrinkage": {
            "why": "a spread's return is not one combination's odds, so #78's likelihood "
            "(core.eval.segment_roi) does not apply; N1's (#430) normal empirical-Bayes shrink of "
            "the realised return, its Place and Wide method, is used instead",
            "baseline": "per (variant, spread) rho_all of the 'all' cell, not shrunk",
            "effect": "e = rho_cell / rho_all, se_e = se / rho_all, se = max(se_mean_return, "
            "sqrt(s2 / stake_cell)), s2 = sum over meetings (R_m - rho_all C_m)^2 / sum C_m of the "
            "'all' cell (a cell with one or two meetings is never taken as exact)",
            "tau2": "per (variant, spread, segment kind): max(0, mean over its cells of "
            "(e - 1)^2 - se_e^2)",
            "shrunk": "B = tau2 / (tau2 + se_e^2) (0 when both are 0); shrunk rho = rho_all (1 + "
            "B (e - 1)), posterior sd rho_all sqrt(B) se_e; 'all' keeps rho_all with sd "
            "max(se_mean_return, sqrt(s2 / its stake)); a spread whose 'all' rho is 0 keeps every "
            "cell's rho with sd 0",
            "shrunk_edge": "shrunk rho - 1",
            "p_return_above_100": "the normal posterior's P(rho > 1)",
        },
        "ranking": "per variant, every row by shrunk_edge descending, rows without one last, "
        "ties in spread then segment order; t2 rows first, then lookahead",
        "data_rules": {
            "D10": "nothing before 2014-06-07 is read or scored (an earlier start is clamped)",
            "D14": "is_holdout_day days are never read: store reads drop them in the scan, "
            "dividends are filtered by development race ids, a holdout day file is never opened; "
            "LockedHoldout is never opened",
            "reads": "one calendar year at a time, named columns only",
        },
        "family_size": {
            "spreads": len(SPREADS),
            "variants_per_family": len(VARIANTS),
            "families": len(FAMILIES),
            "cell_labels_per_spread": 1 + 7 + 4 + 4 + 3 + 3 + 7 * 4 + 7 * 4,
        },
        "verdicts": "none: every cell is a ranked row (owner reframe, 2026-11-27)",
    }


def _spread_spec(spread_id: str) -> dict[str, Any]:
    s = SPREADS[spread_id]
    return {
        "pool": s.pool,
        "kind": s.kind,
        "legs": [list(leg) for leg in s.legs],
        "axis_positions": None if s.positions is None else list(s.positions),
        "tickets": "1-2" if s.kind == "bracket" else len(_rank_card(spread_id)),
    }


def family_spec(family_id: str) -> dict[str, Any]:
    """The plain-JSON spec the family is frozen with: its spreads and the scan's rules."""
    if family_id not in FAMILIES:
        raise KeyError(f"unknown family {family_id!r} (have {sorted(FAMILIES)})")
    spreads = {s: _spread_spec(s) for s in FAMILIES[family_id]}
    return {"family_id": family_id, "spreads": spreads, **_common_spec()}


def spec_hash() -> str:
    """The sha256 of every family's spec together: the pre-registration's hash."""
    return _spec_hash({family: family_spec(family) for family in FAMILIES})


# --------------------------------------------------------------------------- reading the store


def _read_final(start: date, end: date) -> pa.Table:
    """The final Win board's rows of the development days in range, read by byte range."""
    path = paths.processed_dir(SOURCE) / "final_odds.parquet"
    with pa.OSFile(str(path)) as handle:
        where = ds.field("pool") == "WIN"
        return read_development(
            handle, COUNTRY, columns=FINAL_COLUMNS, start=start, end=end, where=where
        )


def _read_paid(race_ids: list[str]) -> pd.DataFrame:
    path = paths.processed_dir(SOURCE) / "dividends.parquet"
    keep = ds.field("race_id").isin(pa.array(race_ids, pa.string()))
    return pq.read_table(path, columns=PAID_COLUMNS, filters=keep).to_pandas()


def _read_boards(
    days: Iterable[date], scratched: set[tuple[str, int]], minutes: int
) -> tuple[dict, dict]:
    """T-2 orders and T-2 Win pool totals of the development day files of `days`."""
    folder = paths.processed_dir(SOURCE) / "odds_timeseries"
    orders: dict[str, tuple[list, float]] = {}
    totals: dict[str, float] = {}
    for day in sorted(days):
        path = folder / f"{day.isoformat()}.parquet"
        if is_holdout_day(COUNTRY, day) or not path.is_file():  # a holdout file is never opened
            continue
        board = pq.read_table(path, columns=BOARD_COLUMNS, filters=ds.field("pool") == "WIN")
        board = board.to_pandas()
        latest = _latest_win(board, board["post_utc"] - pd.Timedelta(minutes=minutes))
        pool = latest.loc[latest["combination"].to_numpy() == POOL_ROW]
        totals.update(
            (race, float(x))
            for race, x in zip(pool["race_id"], pool["investment"], strict=True)
            if x == x
        )
        orders.update(_orders(latest, scratched))
    return orders, totals


def _rate(race_id: str, pool: str) -> float:
    try:
        return race_takeout(race_id, pool).payout_rate
    except RaceConditionsNeeded:
        day = date.fromisoformat(race_id.split(":")[1])
        return 1 - config.load_source(COUNTRY, "jra").pools[pool].takeout_on(day)


@dataclass
class _Bets:
    """Per scored race its facts and, per column (variant x spread), its bet."""

    race_id: list[str]
    meeting: list[str]
    day: list[date]
    cells: list[frozenset[str]]
    fav: list[tuple[float, float]]
    bought: list[np.ndarray]
    cost: list[np.ndarray]
    ret: list[np.ndarray]
    tickets: list[np.ndarray]
    rate: list[np.ndarray]


def _scan_year(start: date, end: date, bets: _Bets, skipped: Counter) -> None:
    """Read one year's development rows and settle every spread in every race; keep the races
    holding a bet (the scored races), whose pool totals cut the year's pool-size tertiles."""
    common = {"start": start, "end": end, "touches_price": True, "source": SOURCE}
    races = read_store_table("races", columns=RACE_COLUMNS, **common).to_pandas()
    if races.empty:
        return
    skipped["race: status other than run"] += int((races["status"] != "run").sum())
    races = races.loc[races["status"].to_numpy() == "run"]
    runners = read_store_table("runners", columns=RUNNER_COLUMNS, **common).to_pandas()
    gone = runners["is_scratched"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    scratched = set(
        zip(runners["race_id"][gone], runners["horse_no"][gone].astype(int), strict=True)
    )
    starters = runners.loc[~gone].groupby("race_id").size()
    brackets: dict[str, dict[int, int]] = {}
    for race, horse, bracket in runners.loc[~gone, ["race_id", "horse_no", "bracket"]].itertuples(
        index=False, name=None
    ):
        if bracket == bracket and bracket is not None:
            brackets.setdefault(race, {})[int(horse)] = int(bracket)
    paid = _read_paid(races["race_id"].tolist())
    lookups = {race: _paid(rows) for race, rows in paid.groupby("race_id", sort=False)}
    finals = _orders(_read_final(start, end).to_pandas(), scratched)
    orders, totals = _read_boards(set(races["race_date"]), scratched, _decision_minutes())

    kept = []  # (race, its race_cells arguments but the edges, its bets): races holding a bet
    for race in races.itertuples(index=False):
        rid = race.race_id
        if rid not in orders:
            skipped["race: no T-2 Win board"] += 1
            continue
        if rid not in totals:
            skipped["race: no T-2 Win pool total"] += 1
            continue
        if rid not in finals:
            skipped["race: no final Win board"] += 1
            continue
        segment = {
            "race_class": race.race_class,
            "field_size": int(starters.get(rid, 0)),
            "fav_odds": orders[rid][1],
            "pool_total": totals[rid],
            "surface": race.surface,
        }
        try:  # every segment but the pool size, whose edges need the year's races holding a bet
            race_cells(**segment, pool_edges=(totals[rid], totals[rid]))
        except ValueError:
            skipped["race: segments undefined"] += 1
            continue
        bought, row = _race_bets(
            rid,
            (orders[rid][0], finals[rid][0]),
            lookups.get(rid, {}),
            brackets.get(rid, {}),
            skipped,
        )
        if not bought.any():
            skipped["race: no bet in any spread"] += 1
            continue
        kept.append((race, segment, bought, row))
    if not kept:
        return
    edges = pool_tertiles(segment["pool_total"] for _, segment, _, _ in kept)
    for race, segment, bought, row in kept:
        rid = race.race_id
        bets.race_id.append(rid)
        bets.meeting.append(race.meeting_id)
        bets.day.append(race.race_date)
        bets.cells.append(race_cells(**segment, pool_edges=edges))
        bets.fav.append((orders[rid][1], finals[rid][1]))
        bets.bought.append(bought)
        for key in ("cost", "ret", "tickets", "rate"):
            getattr(bets, key).append(row[key])


def _race_bets(
    rid: str,
    rankings: tuple[list, list],
    paid_race: Paid,
    horse_brackets: Mapping[int, int],
    skipped: Counter,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One race's bets per column (variant x spread). A spread is bought in both variants or in
    neither (the spec's 'same races'): both rankings reach its highest rank, the race pays its
    pool, and for the bracket spread both rankings' ranks 1-3 have a bracket."""
    n = len(VARIANTS) * len(SPREAD_IDS)
    row = {k: np.zeros(n) for k in ("cost", "ret", "tickets", "rate")}
    bought = np.zeros(n, dtype=bool)
    rates: dict[str, float] = {}
    for k, spread_id in enumerate(SPREAD_IDS):
        spread = SPREADS[spread_id]
        if min(len(ranked) for ranked in rankings) < spread.top_rank:
            skipped[f"no bet: field below the top rank: {spread_id}"] += 1
            continue
        if spread.pool not in paid_race:
            skipped[f"no bet: pool not settled: {spread_id}"] += 1
            continue
        if spread.kind == "bracket" and not all(
            set(ranked[:3]) <= set(horse_brackets) for ranked in rankings
        ):
            skipped[f"no bet: bracket unknown: {spread_id}"] += 1
            continue
        if spread.pool not in rates:
            rates[spread.pool] = _rate(rid, spread.pool)
        for v, ranked in enumerate(rankings):
            j = v * len(SPREAD_IDS) + k
            tickets = expand(spread_id, ranked, brackets=horse_brackets)
            settled = _settle(tickets, paid_race)
            bought[j] = True
            row["cost"][j], row["ret"][j] = settled.cost, settled.ret
            row["tickets"][j], row["rate"][j] = tickets.n_tickets, rates[spread.pool]
    return bought, row


# --------------------------------------------------------------------------- scoring


def _label_key(label: str) -> tuple[int, str]:
    return SEGMENT_KINDS.index(_kind(label)), label


def _normal_p(mean: float, sd: float) -> float:
    if sd > 0:
        return float(special.ndtr((mean - 1.0) / sd))
    return float(mean > 1.0)


def _score(bets: _Bets, *, seed: int, n_boot: int, level: float) -> list[dict[str, Any]]:
    """Every (variant, spread, cell with a bet) row, ranked (module docstring)."""
    if not bets.race_id:
        return []
    bought, cost, ret = np.array(bets.bought), np.array(bets.cost), np.array(bets.ret)
    tickets, rate, fav = np.array(bets.tickets), np.array(bets.rate), np.array(bets.fav)
    meeting, meetings = pd.factorize(np.array(bets.meeting, dtype=object), sort=True)
    day = pd.factorize(np.array(bets.day, dtype=object))[0]
    members: dict[str, list[int]] = {}
    for i, cells in enumerate(bets.cells):
        for label in cells:
            members.setdefault(label, []).append(i)
    labels = sorted(members, key=_label_key)

    pairs = []  # (label, column, the cell's races holding a bet)
    for label in labels:
        idx = np.array(members[label])
        for j in range(bought.shape[1]):
            races = idx[bought[idx, j]]
            if len(races):
                pairs.append((label, j, races))
    m = len(meetings)
    big_r, big_c = np.zeros((m, len(pairs))), np.zeros((m, len(pairs)))
    for p, (_, j, races) in enumerate(pairs):
        big_r[:, p] = np.bincount(meeting[races], weights=ret[races, j], minlength=m)
        big_c[:, p] = np.bincount(meeting[races], weights=cost[races, j], minlength=m)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(m, np.full(m, 1 / m), size=n_boot).astype(float)
    top, bottom = w @ big_r, w @ big_c
    del w
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(bottom > 0, top / bottom, np.nan)
    del top, bottom

    cell: dict[tuple[str, int], dict[str, Any]] = {}
    for p, (label, j, races) in enumerate(pairs):
        stake, back = float(cost[races, j].sum()), float(ret[races, j].sum())
        rho = back / stake
        good = ratio[np.isfinite(ratio[:, p]), p]
        se = float(good.std(ddof=1)) if len(good) > 1 else 0.0
        lo, hi = (
            (float(q) for q in np.quantile(good, [1 - level, level])) if len(good) else (rho,) * 2
        )
        v = j // len(SPREAD_IDS)
        cell[(label, j)] = {
            "races": races,
            "stake": stake,
            "rho": rho,
            "se": se,
            "lo": lo,
            "hi": hi,
            # the spread's between-meeting variance per yen staked, from its 'all' cell
            "spread_var": (
                float(((big_r[:, p] - rho * big_c[:, p]) ** 2).sum() / stake)
                if label == "all"
                else None
            ),
            "payout_rate": float((rate[races, j] * cost[races, j]).sum() / stake),
            "mean_odds": float(fav[races, v].mean()),
        }
    del ratio, big_r, big_c
    _shrink(cell)

    rows = []
    for (label, j), c in cell.items():
        v, k = divmod(j, len(SPREAD_IDS))
        spread_id, parts, races = SPREAD_IDS[k], _parts(label), c["races"]
        rows.append(
            {
                "study": STUDY,
                "family_id": FAMILY_OF[spread_id],
                "cell_id": f"{STUDY}:{VARIANTS[v]}:{spread_id}:{label}",
                "pool": SPREADS[spread_id].pool,
                "class_group": parts.get("class"),
                "odds_band": parts.get("fav"),
                "payout_rate": c["payout_rate"],
                "n_bets": len(races),
                "n_race_days": len(np.unique(day[races])),
                "n_meetings": len(np.unique(meeting[races])),
                "mean_return": c["rho"],
                "se_mean_return": c["se"],
                "shrunk_edge": c["shrunk"] - 1.0,
                "mean_odds": c["mean_odds"],
                "spread_id": spread_id,
                "variant": VARIANTS[v],
                "segment": label,
                "rank": 0,
                "n_tickets": int(tickets[races, j].sum()),
                "ci_lo": c["lo"],
                "ci_hi": c["hi"],
                "p_return_above_100": _normal_p(c["shrunk"], c["sd"]),
            }
        )
    out = []
    for variant in VARIANTS:
        block = [r for r in rows if r["variant"] == variant]
        block.sort(
            key=lambda r: (
                r["shrunk_edge"] is None,
                -(r["shrunk_edge"] or 0.0),
                SPREAD_IDS.index(r["spread_id"]),
                _label_key(r["segment"]),
            )
        )
        for rank, row in enumerate(block, 1):
            row["rank"] = rank
        out += block
    return out


def _shrink(cell: dict[tuple[str, int], dict[str, Any]]) -> None:
    """Per (variant, spread) the normal empirical-Bayes shrink of its cells toward its 'all' rho
    (the spec's shrinkage); sets `shrunk` and `sd` on every cell."""
    for j in {j for _, j in cell}:
        base = cell[("all", j)]
        rho_all, s2 = base["rho"], base["spread_var"]
        base["shrunk"], base["sd"] = rho_all, max(base["se"], float(np.sqrt(s2 / base["stake"])))
        others = [(label, c) for (label, jj), c in cell.items() if jj == j and label != "all"]
        if rho_all <= 0:  # nothing ever paid: every cell returned nothing
            for _, c in others:
                c["shrunk"], c["sd"] = c["rho"], 0.0
            continue
        effects: dict[str, list[tuple[dict, float, float]]] = {}
        for label, c in others:
            se = max(c["se"], float(np.sqrt(s2 / c["stake"])))
            effects.setdefault(_kind(label), []).append((c, c["rho"] / rho_all, se / rho_all))
        for members in effects.values():
            tau2 = max(0.0, float(np.mean([(e - 1) ** 2 - s**2 for _, e, s in members])))
            for c, e, s in members:
                b = tau2 / (tau2 + s**2) if tau2 + s**2 > 0 else 0.0
                c["shrunk"] = rho_all * (1 + b * (e - 1))
                c["sd"] = rho_all * float(np.sqrt(b)) * s


# --------------------------------------------------------------------------- the run


class FamilyNotFrozen(ValueError):
    """The registry does not hold a family frozen with `family_spec`: nothing was read."""


class NoBetInRange(ValueError):
    """The range holds no race with a bet: the run is recorded as failed and writes no output."""


@dataclass
class ScanResult:
    rows: list[dict[str, Any]]
    report_path: Path
    cells_path: Path
    spec_hash: str
    run_ids: dict[str, str]  # "<family_id>/<variant>" -> run id
    skipped: dict[str, int]


def _builds() -> dict[str, str]:
    """Each store file read: its path under the data directory @ a size-and-mtime fingerprint."""
    out = {}
    folder = paths.processed_dir(SOURCE)
    for name in ("races", "runners", "dividends", "final_odds", "odds_timeseries/_build.json"):
        path = folder / (name if name.endswith(".json") else f"{name}.parquet")
        try:
            stat = path.stat()
            stamp = _sha256(f"{stat.st_size}:{stat.st_mtime_ns}")[:16]
        except OSError:
            stamp = "unavailable"
        out[name.split("/")[0]] = f"processed/{SOURCE}/{path.name}@{stamp}"
    return out


def _commit() -> str:
    try:
        return _git_sha()
    except RuntimeError:
        return "unknown"


def _metrics(rows: list[dict[str, Any]], family_id: str, variant: str) -> list[FoldMetric]:
    out = []
    for r in rows:
        if r["family_id"] == family_id and r["variant"] == variant:
            sid, n = r["spread_id"], r["n_bets"]
            out.append(
                FoldMetric(
                    r["segment"], f"mean_return[{sid}]", r["mean_return"], r["ci_lo"], r["ci_hi"], n
                )  # fmt: skip
            )
            out.append(FoldMetric(r["segment"], f"shrunk_edge[{sid}]", r["shrunk_edge"], None,
                                  None, n))  # fmt: skip
    return out


def _append(registry: Registry, records: list[RunRecord]) -> dict[str, str]:
    """Append the records the registry does not already hold; every record's run id."""
    ids = {}
    for record in records:
        key = f"{record.family_id}/{record.variant}"
        if record in registry.records(record.family_id):
            ids[key] = _sha256(_canonical(_record_data(record)))
        else:
            ids[key] = registry.append(record)
    return ids


def _report(rows: list[dict], summary: dict[str, Any], skipped: Mapping[str, int]) -> str:
    """The private report: the run, the skips, each spread's 'all' row and the top rows."""
    cols = ["rank", "spread_id", "segment", "n_bets", "n_meetings", "mean_return", "ci_lo"]
    cols += ["ci_hi", "shrunk_edge", "p_return_above_100"]

    def table(selected: list[dict]) -> list[str]:
        lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
        for r in selected:
            cells = [f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
            lines.append("| " + " | ".join(cells) + " |")
        return lines

    lines = [
        "# N8 spread scan (Japan, #503): private report",
        "",
        "Built on the JRA-VAN store (licensed, D21): never copy a figure into the repository.",
        "",
        *(f"- {key}: {value}" for key, value in summary.items()),
        "",
        "## Skipped races and bets",
        "",
        *(f"- {key}: {n}" for key, n in sorted(skipped.items())),
    ]
    for variant in VARIANTS:
        block = [r for r in rows if r["variant"] == variant]
        lines += ["", f"## {variant}: each spread on all races", ""]
        lines += table([r for r in block if r["segment"] == "all"])
        lines += ["", f"## {variant}: the top 60 rows", ""]
        lines += table(block[:60])
    return "\n".join(lines) + "\n"


def run_scan(
    *,
    date_from: date,
    date_to: date,
    out_dir: Path | str,
    seed: int,
    n_boot: int,
    registry: Registry,
) -> ScanResult:
    """Score every spread on the development days from `date_from` (clamped to 2014-06-07) to
    `date_to` and record the run (module docstring)."""
    start, end = max(pd.Timestamp(date_from).date(), PRICE_ERA_START), pd.Timestamp(date_to).date()
    if end < start:
        raise ValueError(f"date_to {end} is before the scan's start {start}")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 2:
        raise ValueError(f"n_boot must be an integer of at least 2, got {n_boot!r}")
    out_dir = Path(out_dir)
    if is_inside(out_dir, env.repo_root()):
        raise LicensedOutputInRepo(f"{out_dir} is inside the repository: the scan reads jravan")
    for family_id in FAMILIES:
        held = registry.frozen_spec_hash(family_id)
        if held != _spec_hash(family_spec(family_id)):
            raise FamilyNotFrozen(f"family {family_id} is frozen with {held}, not its spec")
    params = {
        "spec_hash": spec_hash(),
        "date_from": start.isoformat(),
        "date_to": end.isoformat(),
        "seed": seed,
        "n_boot": n_boot,
    }
    base = {"git_sha": _commit(), "data_builds": _builds()}
    # one report per run: its name changes with the spec, range, seed, replicates, code and store
    report_name = f"spread-scan-japan-{_sha256(_canonical({**params, **base}))[:16]}.md"

    def records(metrics: Any, status: str, report: str) -> list[RunRecord]:
        builds = {**base["data_builds"], REPORT_BUILD: f"reports/{STUDY}@{report}"}
        return [
            RunRecord(
                hypothesis_id=HYPOTHESIS_ID,
                family_id=family_id,
                variant=variant,
                ticket=TICKET,
                git_sha=base["git_sha"],
                data_builds=builds,
                sources=(SOURCE,),
                fold_scheme=FOLD_SCHEME,
                components={},
                params={**params, "family_spec_hash": _spec_hash(family_spec(family_id))},
                metrics=tuple(metrics(family_id, variant)),
                status=status,
            )
            for family_id in FAMILIES
            for variant in VARIANTS
        ]

    skipped: Counter = Counter()
    written = NO_REPORT  # what a failed run's records name: the report once it is on disk
    try:
        bets = _Bets([], [], [], [], [], [], [], [], [], [])
        for year in range(start.year, end.year + 1):
            lo, hi = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
            _scan_year(lo, hi, bets, skipped)
        if not bets.race_id:  # nothing to score: no report, the last candidate cells stay
            raise NoBetInRange(f"no development race from {start} to {end} holds a bet")
        rows = _score(bets, seed=seed, n_boot=n_boot, level=_level())
        summary = {
            **params,
            "git_sha": base["git_sha"],
            "families": ", ".join(f"{f} {_spec_hash(family_spec(f))}" for f in FAMILIES),
            "races scored": len(bets.race_id),
            "rows": len(rows),
        }
        report_path = write_report(
            out_dir / report_name, _report(rows, summary, skipped), sources=(SOURCE,)
        )
        written = report_name
        cells_path = write_report(
            out_dir / "candidate-cells" / "spread-scan.json",
            json.dumps(rows, allow_nan=False, indent=1) + "\n",
            sources=(SOURCE,),
        )
    except Exception:
        _append(registry, records(lambda f, v: (), "failed", written))
        raise
    scored = records(lambda f, v: _metrics(rows, f, v), "scored", report_name)
    run_ids = _append(registry, scored)
    return ScanResult(rows, report_path, cells_path, params["spec_hash"], run_ids, dict(skipped))


# --------------------------------------------------------------------------- the command


class _Families:
    """The registry's records of the families named: what `write_summary` reads."""

    def __init__(self, registry: Registry, families: set[str]) -> None:
        self._registry, self._families = registry, families

    def records(self, family_id: str | None = None) -> list[RunRecord]:
        return [r for r in self._registry.records(family_id) if r.family_id in self._families]


_INDEX_ENTRY = re.compile(r"- \[([^\]]+)\]\(\1\.md\)")  # a family's line in write_summary's index


def _table_lines(text: str) -> Counter:
    """A summary file's table lines (its header and one per run and metric), as a multiset."""
    return Counter(line for line in text.splitlines() if line.startswith("|"))


def write_committed_summary(registry: Registry) -> tuple[Path, list[str]]:
    """The six families' files, by `write_summary` over their records alone, and the index with
    their lines added: every family the index lists stays and no other family's file is written,
    so a fresh local registry drops nothing (`studies/ability.py`'s way). A family file holding a
    row the local registry would not write again (runs recorded under another data directory) is
    kept as it was; the second value names the files kept."""
    directory = summary_dir(COUNTRY)
    mine = _Families(registry, set(FAMILIES))
    if not mine.records():
        return directory, []
    kept = []
    with tempfile.TemporaryDirectory() as scratch:
        write_summary(mine, scratch)  # type: ignore[arg-type]
        for path in sorted(Path(scratch).glob("*.md")):
            if path.stem == "README":
                continue
            text, target = path.read_text("utf-8"), directory / path.name
            published = _table_lines(target.read_text("utf-8")) if target.is_file() else Counter()
            if published - _table_lines(text):  # a published row this registry would drop
                kept.append(path.name)
                continue
            write_report(target, text, sources=())
        fresh = Path(scratch, "README.md").read_text("utf-8").splitlines()
    index = directory / "README.md"
    listed = index.read_text("utf-8").splitlines() if index.exists() else []
    entries = {m[1]: line for line in listed + fresh if (m := _INDEX_ENTRY.fullmatch(line))}
    head = [line for line in fresh if not _INDEX_ENTRY.fullmatch(line)]
    write_report(
        index, "\n".join([*head, *(entries[f] for f in sorted(entries))]) + "\n", sources=()
    )
    return directory, kept


spread_scan_app = typer.Typer(no_args_is_help=True, help="N8: the popularity-rank spread scan.")


@spread_scan_app.command("freeze")
def freeze_command() -> None:
    """Freeze every family's spec in the japan registry (irreversible; again is a no-op)."""
    registry = default_registry(COUNTRY)
    for family_id in FAMILIES:
        typer.echo(
            f"frozen {family_id} {registry.freeze_family(family_id, family_spec(family_id))}"
        )
    typer.echo(f"spec_hash={spec_hash()}")


@spread_scan_app.command("run")
def run_command(
    date_to: str = typer.Option(..., "--to", help="Last day to scan (YYYY-MM-DD)."),
    date_from: str = typer.Option("2014-06-07", "--from", help="First day (clamped to D10)."),
    seed: int = typer.Option(DEFAULT_SEED, help="Bootstrap seed."),
    n_boot: int = typer.Option(DEFAULT_N_BOOT, help="Bootstrap replicates."),
) -> None:
    """Scan the frozen spreads on development days and record the run; prints no figure."""
    registry = default_registry(COUNTRY)
    try:
        result = run_scan(
            date_from=date.fromisoformat(date_from),
            date_to=date.fromisoformat(date_to),
            out_dir=paths.data_dir() / "reports",
            seed=seed,
            n_boot=n_boot,
            registry=registry,
        )
    except NoBetInRange as exc:
        typer.echo(f"not scored (recorded as failed): {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"spec_hash={result.spec_hash}")
    typer.echo(f"report={result.report_path}\ncells={result.cells_path}")
    for key, run_id in result.run_ids.items():
        typer.echo(f"run {key} {run_id}")
    directory, kept = write_committed_summary(registry)
    typer.echo(f"summary={directory}")
    if kept:
        typer.echo(
            f"run recorded, summary kept: {', '.join(kept)} publish runs this data directory's "
            "registry does not hold; write the summary from the registry that holds them",
            err=True,
        )
        raise typer.Exit(1)
