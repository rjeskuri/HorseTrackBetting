"""Ticket #432 (N3): pool depth and stake capacity from the final votes (synthetic data only).

Contract pinned here for `horsetrack.countries.japan.research.pool_depth` (the coder implements
what these tests call):

- `capacity_stake(pool_total, combo_money, edge, *, share_lost=0.5) -> float`: the continuous
  stake s (no breakage) at which our own money on a single-winner combination eats `share_lost`
  of an edge e. V is the pool, W the money already on the combination, both in yen; s solves
  (1 + e)(V + s)W / (V(W + s)) = 1 + e(1 - share_lost). With k = e(1 - share_lost) that is
  s = (e - k)VW / ((1 + k)V - (1 + e)W), which at the default share_lost 0.5 (e - k = k) is the
  ticket's s = kVW / ((1 + k)V - (1 + e)W). ValueError when the edge is not a positive finite
  number, W >= V, W < 0, V or W is not finite, or the denominator is not positive.
  `share_lost` is keyword-only.
- `pool_totals(*, date_from, date_to, minutes_before=2.0) -> DataFrame` with exactly the columns
  race_id, pool, final_yen, at_t_yen: one row per race and pool the jravan store sold on a
  development day from `date_from` to `date_to` (both inclusive). final_yen is the pool's final
  total in yen from the store's `__pool__` rows (`processed/jravan/pool_votes.parquet` votes x 100,
  equal to `final_odds.parquet` investment: the tests keep the two equal, so either may be read).
  at_t_yen is, for the WIN, PLACE and QUINELLA pools, the `investment` of the race's latest
  (`snapshot_utc`) `__pool__` row with minutes_to_post >= minutes_before in the jravan time series
  (`processed/jravan/odds_timeseries/<YYYY-MM-DD>.parquet`, `logger.compact.SCHEMA`): a capture
  exactly at t counts; rows with a null minutes_to_post, after t (post-race included) and
  combination rows do not. NaN for every other pool and for a race without such a capture or
  without a day file. A holdout day (`core.eval.holdout.is_holdout_day("japan", d)`) gives no row
  and its day file is never opened.
- `depth_rows(votes, board, races, *, band_edges) -> DataFrame`: `votes` are `pool_votes` rows,
  `board` `final_odds` rows and `races` the store's `races` rows (pandas frames in the store's
  columns, many races at once); `band_edges` maps each pool to its odds-band edges, and only the
  pools it names give rows. One row per combination with a final price (non-null `odds`; the
  `__pool__` row is no combination) with at least: race_id, pool, combination, class_group
  (`pool_scan.class_group` of the race's `race_class`), band (`pool_scan.odds_band(odds,
  band_edges[pool])`, the board's `odds`, i.e. the low end of the range for PLACE and
  QUINELLA_PLACE), combo_yen (the combination's votes x 100) and pool_yen (the race's `__pool__`
  total of that pool in yen, not the sum of its combinations).
- `depth_table(rows, *, edges, quantiles) -> DataFrame`: `rows` hold at least race_id, pool,
  class_group, band, combo_yen and pool_yen (other columns are ignored). Long format: one row per
  (pool, class_group, band) cell with rows and per q in `quantiles`, with the columns pool,
  class_group, band, quantile (q), n_combinations (the cell's rows), n_races (its distinct
  races), combo_yen (the q-quantile of the cell's combo_yen), pool_yen (the q-quantile of the
  pool over the cell's distinct races, each race once) and, per e in `edges`, `capacity_{e:g}`
  (`capacity_0.05`, `capacity_0.1`, ...): the q-quantile over the cell's combinations of
  `capacity_stake(pool_yen, combo_yen, e)` at share_lost 0.5. Quantiles are numpy's default
  (linear interpolation). The capacity of the co-placer pools (PLACE, QUINELLA_PLACE), whose
  dividend is not the single-winner identity, is not pinned here (the spec registers both
  treatments as variants).
- The pre-registered spec is `src/horsetrack/countries/japan/config/pool_depth.yaml`, read with
  `core.config.load_country_yaml("japan", "pool_depth")` and frozen as loaded: `family_id`
  (non-blank), `edges` [0.05, 0.1, 0.2], `share_lost` 0.5, `quantiles` (ascending, in [0, 1]) and
  `odds_bands`, a mapping from each pool of `pool_scan.POOLS` (others allowed) to its ascending
  odds-band edges. Plain JSON, so it can be frozen.
- `run_depth(*, date_from, date_to, out_dir, registry=None) -> DepthResult`: `registry` None is
  `default_registry("japan")`. `NotPreRegistered` (a name in `pool_depth`) unless the registry has
  the shipped spec frozen under its `family_id`, before anything is read or written. A
  `date_from` before 2014-06-07 (D10) raises ValueError (`PreReformPriceError` is one). It reads
  `pool_votes`, `final_odds` and `races` from the jravan store on development days only and never
  opens `LockedHoldout`. `DepthResult` has `table` (`depth_table` of the run's `depth_rows`, with
  the spec's `odds_bands`, `edges` and `quantiles`), `report_path` (the private report, written
  under `out_dir` through `core.eval.reports.write_report(..., sources=('jravan',))`, so an
  `out_dir` inside the repository raises `LicensedOutputInRepo` and writes nothing), `family_id`
  and `spec_hash` (the frozen hash). Every run after the freeze is one RunRecord (hypothesis_id
  P432, the spec's family_id, ticket 432, sources ('jravan',)): status "scored" when it ran,
  "failed" when it raised. The same run again does not raise. It does not write the committed
  registry summary (`summary_dir("japan")`).
- `render_public_report(result) -> str`: the public text; it carries the marker `withheld (D21)`
  and no figure computed from the data: two runs on different data render the same text (hex
  run identities and timestamps aside).

Every board, vote and race here is invented; HORSETRACK_DATA_DIR points at tmp_path. The holdout
tests use a stand-in holdout definition (`stand_in_holdout`), so they do not depend on the
committed seed; elsewhere every day is before the committed holdout start.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import builtins
import collections.abc as abc
import copy
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.core import env, paths, pricing
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval import holdout as holdout_module
from horsetrack.core.eval import registry as registry_module
from horsetrack.core.eval.registry import (
    FoldMetric,
    Registry,
    RunRecord,
    _spec_hash,
    default_registry,
    summary_dir,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.jravan import odds as jravan_odds
from horsetrack.countries.japan.jravan import votes as jravan_votes
from horsetrack.countries.japan.logger import compact
from horsetrack.countries.japan.research import pool_scan
from horsetrack.countries.japan.research import pool_depth as depth

REPO = env.repo_root()
SPEC_NAME = "pool_depth"
HYPOTHESIS_ID = "P432"
TICKET = 432
SOURCES = ("jravan",)
WITHHELD = "withheld (D21)"
POOL_ROW = "__pool__"
POOLS = ("WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE", "FORECAST", "TRIO")
SINGLE_WINNER = ("WIN", "QUINELLA", "FORECAST", "TRIO")
RANGED = ("PLACE", "QUINELLA_PLACE")  # the board's odds are the low end of a range
CO_PLACER = ("PLACE", "QUINELLA_PLACE")  # stake-back-share pools
VARIANTS = ("depth", "co-placer", "exacta-own-impact")
PRICE_ERA = date(2014, 6, 7)  # D10
# payout rates (1 - takeout, config/jra.yaml since 2014-06-07), hard-coded here
RATES = {
    "WIN": 0.80,
    "PLACE": 0.80,
    "QUINELLA": 0.775,
    "QUINELLA_PLACE": 0.775,
    "FORECAST": 0.75,
    "TRIO": 0.75,
    "TIERCE": 0.725,
}
BASE_VOTES = {
    "WIN": 4000,
    "PLACE": 2500,
    "QUINELLA": 900,
    "QUINELLA_PLACE": 500,
    "FORECAST": 300,
    "TRIO": 150,
    "TIERCE": 60,
}
# `races.race_class` -> class group (the jravan vocabulary, as #429's scan groups it)
CLASS_GROUPS = {
    "DEBUT": "newcomer",
    "MAIDEN": "maiden",
    "C1": "conditions",
    "C2": "conditions",
    "C3": "conditions",
    "OP": "open",
    "L": "open",
    "G1": "graded",
    "G2": "graded",
    "G3": "graded",
}


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


# --- the identity, derived here ------------------------------------------------------------------


def value_after_stake(pool: float, combo: float, edge: float, stake: float) -> float:
    """(1 + e)(V + s)W / (V(W + s)): the edge's value once our stake is in."""
    return (1 + edge) * (pool + stake) * combo / (pool * (combo + stake))


def capacity(pool: float, combo: float, edge: float, share_lost: float = 0.5) -> float:
    """The stake solving value_after_stake = 1 + e(1 - share_lost), by hand."""
    k = edge * (1 - share_lost)
    return (edge - k) * pool * combo / ((1 + k) * pool - (1 + edge) * combo)


# --- synthetic races and the store ---------------------------------------------------------------


def combinations_of(pool: str, n: int) -> list[str]:
    """Every combination of `pool` over horses 1..n, written as the store writes it."""
    horses = range(1, n + 1)
    if pool in ("WIN", "PLACE"):
        return [str(h) for h in horses]
    if pool in ("QUINELLA", "QUINELLA_PLACE"):
        legs = itertools.combinations(horses, 2)
    elif pool == "FORECAST":
        legs = itertools.permutations(horses, 2)
    elif pool == "TRIO":
        legs = itertools.combinations(horses, 3)
    else:  # TIERCE
        legs = itertools.permutations(horses, 3)
    return [",".join(str(h) for h in c) for c in legs]


@dataclass
class Race:
    day: date
    venue: str
    race_no: int
    race_class: str
    votes: dict[str, dict[str, int]]  # pool -> combination -> votes (100-yen tickets)
    pool_votes: dict[str, int]  # pool -> the `__pool__` row's votes
    odds: dict[str, dict[str, float | None]]  # pool -> combination -> final odds (None: none)
    refunds: dict[str, int] = field(default_factory=dict)  # pool -> the pool row's refund_votes
    order: tuple[tuple[int, ...], ...] = ()  # finishing positions; a dead heat shares one
    status: str = "result"  # races.status
    votes_kubun: str = (
        "4"  # データ区分 of its pool_votes records (4 確定, 5 確定(月曜), 2 前日売最終)
    )
    board_kubun: str = "4"  # and of its final_odds records

    @property
    def race_id(self) -> str:
        return f"jra:{self.day.isoformat()}:{self.venue}:{self.race_no:02d}"

    def pool_yen(self, pool: str) -> float:
        """The gross `__pool__` total in yen (votes x 100, `final_odds` investment)."""
        return 100.0 * self.pool_votes[pool]

    def net_yen(self, pool: str) -> float:
        """The money that pays: (votes - refund_votes) x 100."""
        return 100.0 * (self.pool_votes[pool] - self.refunds.get(pool, 0))

    def unprice(self, pool: str, combination: str) -> None:
        """No final price and no votes (a zero-vote or cancelled slot)."""
        self.votes[pool][combination] = 0
        self.odds[pool][combination] = None


def make_race(
    seed: int,
    day: date,
    venue: str,
    race_no: int,
    race_class: str,
    n: int,
    *,
    scale: int = 1,
    pools: tuple[str, ...] = POOLS,
) -> Race:
    """A race whose money is drawn from `seed`; `scale` multiplies every vote (the odds, from the
    unscaled money, are unchanged). The `__pool__` row holds more than the combinations' sum."""
    rng = np.random.default_rng(seed)
    votes, pool_votes, odds = {}, {}, {}
    for pool in pools:
        combos = combinations_of(pool, n)
        raw = np.exp(rng.normal(math.log(BASE_VOTES[pool]), 0.8, len(combos)))
        v = np.maximum(1, np.round(raw)).astype(int)
        total = int(v.sum()) + int(rng.integers(20, 2000))
        assert v.max() / total < 0.8
        votes[pool] = {c: int(x) * scale for c, x in zip(combos, v, strict=True)}
        pool_votes[pool] = total * scale
        odds[pool] = {
            c: max(1.0, math.floor(10 * RATES[pool] * total / int(x)) / 10)
            for c, x in zip(combos, v, strict=True)
        }
    return Race(day, venue, race_no, race_class, votes, pool_votes, odds)


def votes_rows(race: Race) -> list[dict]:
    base = {
        "jurisdiction": "jra",
        "race_id": race.race_id,
        "race_date": race.day,
        "venue": race.venue,
        "race_no": race.race_no,
        "fav_rank": None,
        "data_kubun": race.votes_kubun,
    }
    rows = []
    for pool, combos in race.votes.items():
        for combination, v in combos.items():
            rows.append(
                {**base, "pool": pool, "combination": combination, "votes": v, "refund_votes": None}
            )
        rows.append(
            {
                **base,
                "pool": pool,
                "combination": POOL_ROW,
                "votes": race.pool_votes[pool],
                "refund_votes": race.refunds.get(pool, 0),
            }
        )
    return rows


def board_rows(race: Race) -> list[dict]:
    base = {
        "jurisdiction": "jra",
        "race_id": race.race_id,
        "race_date": race.day,
        "venue": race.venue,
        "race_no": race.race_no,
        "fav_rank": None,
        "published_utc": None,
        "data_kubun": race.board_kubun,
    }
    rows = []
    for pool, combos in race.odds.items():
        for combination, price in combos.items():
            hi = round(price * 1.5, 1) if pool in RANGED and price is not None else None
            rows.append(
                {
                    **base,
                    "pool": pool,
                    "combination": combination,
                    "odds": price,
                    "odds_hi": hi,
                    "investment": None,
                    "status_raw": None if price is not None else "0000",
                }
            )
        rows.append(
            {
                **base,
                "pool": pool,
                "combination": POOL_ROW,
                "odds": None,
                "odds_hi": None,
                "investment": race.pool_yen(pool),
                "status_raw": None,
            }
        )
    return rows


def races_schema() -> pa.Schema:
    columns = schemas.TABLES["races"].columns
    return pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in columns.items()])


def race_row(race: Race) -> dict:
    return {
        "jurisdiction": "jra",
        "race_id": race.race_id,
        "meeting_id": f"jra:{race.day.isoformat()}:{race.venue}",
        "race_date": race.day,
        "venue": race.venue,
        "race_no": race.race_no,
        "race_class": race.race_class,
        "field_size": len(race.votes["WIN"]) if "WIN" in race.votes else None,
        "surface": "turf",
        "status": race.status,
    }


def dividends_schema() -> pa.Schema:
    columns = schemas.TABLES["dividends"].columns
    return pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in columns.items()])


# JRA's stake-back-share rule for Place and Wide, derived here: of the payout rate x the net pool,
# the paying combinations' own money goes back to them and the rest is split into P equal shares,
# P the race's paying combinations of that pool; per 100 yen on paying combination i
# 100 + 100 (rV - sum W) / (P W_i), broken down to 10 yen and at least 110 (Plus 10).
PAYOUT_RATE = {"PLACE": Fraction(4, 5), "QUINELLA_PLACE": Fraction(31, 40)}  # 1 - takeout


def ranks(race: Race) -> dict[int, int]:
    """horse -> finishing rank; dead-heaters share a rank and the next rank skips."""
    out, rank = {}, 1
    for group in race.order:
        for horse in group:
            out[horse] = rank
        rank += len(group)
    return out


def paying(race: Race, pool: str) -> list[str]:
    """The race's paying combinations: Place, the horses ranked within the places paid (2 in a
    field of 5-7, 3 in one of 8 or more); Wide, the pairs of horses ranked in the first three
    (two horses dead-heating for third are not a pair)."""
    rank = ranks(race)
    if pool == "PLACE":
        places = 3 if len(race.votes["PLACE"]) >= 8 else 2
        return sorted((str(h) for h, r in rank.items() if r <= places), key=int)
    top = sorted(h for h, r in rank.items() if r <= 3)
    return [
        f"{a},{b}" for a, b in itertools.combinations(top, 2) if not (rank[a] == 3 and rank[b] == 3)
    ]


def per_100(race: Race, pool: str, combination: str) -> Fraction:
    """JRA's rule before breakage and the floor, exactly."""
    pays = paying(race, pool)
    net = 100 * (race.pool_votes[pool] - race.refunds.get(pool, 0))
    paid = 100 * sum(race.votes[pool][c] for c in pays)
    stake = 100 * race.votes[pool][combination]
    return 100 + Fraction(100) * (PAYOUT_RATE[pool] * net - paid) / (len(pays) * stake)


def jra_dividend(race: Race, pool: str, combination: str) -> int:
    return max(110, math.floor(per_100(race, pool, combination) / 10) * 10)


def dividend_rows(race: Race) -> list[dict]:
    """The race's `dividends` rows (a race without a finishing order has none): its Place and
    Wide payouts by JRA's rule, and a Win row."""
    if not race.order:
        return []
    base = {"jurisdiction": "jra", "race_id": race.race_id, "is_refund": False}
    rows = []
    if "WIN" in race.votes:
        winner = str(race.order[0][0])
        price = race.odds["WIN"][winner] or 1.0
        rows.append({**base, "pool": "WIN", "combination": winner, "dividend": 100.0 * price})
    for pool in CO_PLACER:
        if pool in race.votes:
            for c in paying(race, pool):
                rows.append(
                    {**base, "pool": pool, "combination": c,
                     "dividend": float(jra_dividend(race, pool, c))}
                )  # fmt: skip
    return rows


def store_tables(races: list[Race], *, drop=None) -> dict[str, pa.Table]:
    """The store tables of `races`; `drop(name, row)` True leaves a row out (a gap in the
    store). `dividends` holds the races with a finishing order."""
    shuffle = random.Random(432).shuffle  # nothing may rely on the file's row order
    votes = [row for race in races for row in votes_rows(race)]
    board = [row for race in races for row in board_rows(race)]
    meta = [race_row(race) for race in races]
    paid = [row for race in races for row in dividend_rows(race)]
    rows = {"pool_votes": votes, "final_odds": board, "races": meta, "dividends": paid}
    if drop is not None:
        rows = {name: [r for r in part if not drop(name, r)] for name, part in rows.items()}
    for part in rows.values():
        shuffle(part)
    return {
        "pool_votes": pa.Table.from_pylist(
            rows["pool_votes"], schema=jravan_votes.POOL_VOTES_SCHEMA
        ),
        "final_odds": pa.Table.from_pylist(
            rows["final_odds"], schema=jravan_odds.FINAL_ODDS_SCHEMA
        ),
        "races": pa.Table.from_pylist(rows["races"], schema=races_schema()),
        "dividends": pa.Table.from_pylist(rows["dividends"], schema=dividends_schema()),
    }


def write_store(data: Path, races: list[Race], *, skip: tuple[str, ...] = (), drop=None) -> None:
    """`processed/jravan/{pool_votes,final_odds,races,dividends}.parquet`, holdout days included,
    as the builders write them (only readers drop holdout days)."""
    folder = data / "processed" / "jravan"
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in store_tables(races, drop=drop).items():
        target = folder / f"{name}.parquet"
        if name in skip:
            target.unlink(missing_ok=True)
            continue
        pq.write_table(table, target)


def post_time(race: Race) -> datetime:
    return datetime.combine(race.day, time(1, 0), tzinfo=timezone.utc) + timedelta(
        minutes=30 * race.race_no
    )


def capture(
    race: Race,
    pool: str,
    minutes: float | None,
    investment: float | None,
    *,
    combination: str = POOL_ROW,
    odds: float | None = None,
) -> dict:
    """One time-series row; `minutes` None is a row without a post time (null minutes)."""
    post = post_time(race)
    snapshot = post - timedelta(minutes=7 if minutes is None else minutes)
    return {
        "race_id": race.race_id,
        "venue": race.venue,
        "race_no": race.race_no,
        "pool": pool,
        "combination": combination,
        "snapshot_utc": snapshot,
        "fetched_utc": None,
        "post_utc": None if minutes is None else post,
        "minutes_to_post": None if minutes is None else float(minutes),
        "odds": odds,
        "odds_hi": None,
        "investment": investment,
        "status_raw": None,
        "is_final_capture": False,
        "dividend": None,
        "is_refund": None,
    }


def day_file(data: Path, day: date) -> Path:
    folder = data / "processed" / "jravan" / "odds_timeseries"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{day.isoformat()}.parquet"


def write_day(data: Path, day: date, rows: list[dict]) -> Path:
    """A jravan time-series day file in `logger.compact.SCHEMA`, final captures marked."""
    latest: dict[tuple, int] = {}
    for i, row in enumerate(rows):
        key = (row["race_id"], row["pool"], row["combination"])
        if key not in latest or row["snapshot_utc"] >= rows[latest[key]]["snapshot_utc"]:
            latest[key] = i
    final = set(latest.values())
    rows = [{**row, "is_final_capture": i in final} for i, row in enumerate(rows)]
    random.Random(day.toordinal()).shuffle(rows)
    path = day_file(data, day)
    pq.write_table(pa.Table.from_pylist(rows, schema=compact.SCHEMA), path)
    return path


# --- what the module must give, derived here -----------------------------------------------------


def expected_rows(races: list[Race], band_edges: dict) -> pd.DataFrame:
    rows = []
    for race in races:
        if race.status == "abandoned":  # every ticket refunded: no money on it pays
            continue
        group = CLASS_GROUPS[race.race_class]
        for pool, combos in race.votes.items():
            if pool not in band_edges:
                continue
            edges = [float(e) for e in band_edges[pool]]
            for combination, v in combos.items():
                price = race.odds[pool][combination]
                if price is None:
                    continue
                rows.append(
                    {
                        "race_id": race.race_id,
                        "pool": pool,
                        "combination": combination,
                        "class_group": group,
                        "band": pool_scan.odds_band(price, edges),
                        "odds": price,
                        "combo_yen": 100.0 * v,
                        "pool_yen": race.net_yen(pool),
                    }
                )
    return pd.DataFrame(rows)


def expected_table(rows: pd.DataFrame, edges, quantiles) -> dict[tuple, dict[str, float]]:
    """(pool, class_group, band, quantile) -> the values `depth_table` must give."""
    out = {}
    for (pool, group, band), cell in rows.groupby(["pool", "class_group", "band"]):
        combo = cell["combo_yen"].to_numpy(dtype=float)
        pools = cell.drop_duplicates("race_id")["pool_yen"].to_numpy(dtype=float)
        pairs = list(zip(cell["pool_yen"], cell["combo_yen"], strict=True))
        caps = {e: np.array([capacity(v, w, float(e)) for v, w in pairs]) for e in edges}
        for q in quantiles:
            values = {
                "n_combinations": len(cell),
                "n_races": int(cell["race_id"].nunique()),
                "combo_yen": float(np.quantile(combo, float(q))),
                "pool_yen": float(np.quantile(pools, float(q))),
            }
            if pool in SINGLE_WINNER:
                for e in edges:
                    values[f"capacity_{float(e):g}"] = float(np.quantile(caps[e], float(q)))
            out[(pool, group, band, round(float(q), 9))] = values
    return out


def table_by_cell(table: pd.DataFrame, edges) -> dict[tuple, dict]:
    needed = [
        "pool",
        "class_group",
        "band",
        "quantile",
        "n_combinations",
        "n_races",
        "combo_yen",
        "pool_yen",
        *(f"capacity_{float(e):g}" for e in edges),
    ]
    missing = [c for c in needed if c not in table.columns]
    assert not missing, f"depth table lacks {missing}; has {list(table.columns)}"
    out: dict[tuple, dict] = {}
    for row in table.to_dict("records"):
        key = (row["pool"], row["class_group"], row["band"], round(float(row["quantile"]), 9))
        assert key not in out, f"cell {key} twice"
        out[key] = row
    return out


def assert_depth_table(table: pd.DataFrame, expected: dict, edges) -> None:
    got = table_by_cell(table, edges)
    assert set(got) == set(expected), (
        f"extra cells {sorted(set(got) - set(expected))[:5]}, "
        f"missing cells {sorted(set(expected) - set(got))[:5]}"
    )
    for key, want in expected.items():
        row = got[key]
        for column, value in want.items():
            if column.startswith("n_"):
                assert int(row[column]) == value, (key, column, row[column], value)
            else:
                assert float(row[column]) == pytest.approx(value, rel=1e-9, abs=1e-6), (
                    key,
                    column,
                    row[column],
                    value,
                )


# --- the spec, the registry and the holdout ------------------------------------------------------


def shipped_spec() -> dict:
    return load_country_yaml("japan", SPEC_NAME)


def frozen_registry(tmp_path: Path, name: str = "registry") -> tuple[Registry, dict]:
    """A registry under tmp_path with the shipped spec frozen under its family."""
    spec = shipped_spec()
    registry = Registry(tmp_path / f"{name}.jsonl", holdout_log=tmp_path / f"{name}-holdout.jsonl")
    registry.freeze_family(spec["family_id"], spec)
    return registry, spec


def summary_snapshot() -> dict[str, bytes]:
    folder = summary_dir("japan")
    if not folder.is_dir():
        return {}
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}


STAND_IN_START = date(2021, 1, 1)
STAND_IN_SEED = "t432-stand-in-seed"
DEV_DAYS = (date(2020, 12, 27), date(2021, 1, 2), date(2021, 1, 9), date(2021, 1, 17))
DEV_AFTER = date(2021, 1, 31)  # a development day after the ranges read here
HOLDOUT_DAYS = (date(2021, 1, 3), date(2021, 1, 10), date(2021, 1, 16))


def stand_in_rule(day: date) -> bool:
    """#71's rule under the stand-in definition, derived here."""
    digest = hashlib.sha256(f"{STAND_IN_SEED}:{day.isoformat()}".encode()).hexdigest()
    return day >= STAND_IN_START and int(digest, 16) % 2 == 0


@pytest.fixture
def stand_in_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """`core.eval.holdout` answers from a stand-in definition (start 2021-01-01, a test seed),
    however the module under test reaches it; `LockedHoldout.open` fails the test."""

    def definition(countries_dir, country):
        return STAND_IN_START, STAND_IN_SEED, ZoneInfo("Asia/Tokyo")

    def never(*args, **kwargs):
        raise AssertionError("LockedHoldout.open was called: N3 never opens the holdout")

    monkeypatch.setattr(holdout_module, "_load_definition", definition)
    monkeypatch.setattr(holdout_module.LockedHoldout, "open", never)
    assert not any(stand_in_rule(d) for d in (*DEV_DAYS, DEV_AFTER))
    assert all(stand_in_rule(d) for d in HOLDOUT_DAYS)
    for d in (*DEV_DAYS, DEV_AFTER, *HOLDOUT_DAYS):
        assert holdout_module.is_holdout_day("japan", d) == stand_in_rule(d), d


def record_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """From now on, every path (or SQL text) handed to the usual file and Parquet readers."""
    seen: list[str] = []

    def note(*args, **kwargs) -> None:
        for value in (*args, *kwargs.values()):
            for item in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(item, (str, os.PathLike)):
                    seen.append(os.fspath(item))

    def wrap(owner, name: str) -> None:
        original = getattr(owner, name, None)
        if original is None:
            return

        def wrapper(*args, **kwargs):
            note(*args, **kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapper)

    owners = [
        (pq, "read_table"),
        (pq, "read_metadata"),
        (pq, "read_schema"),
        (ds, "dataset"),
        (pa, "memory_map"),
        (pa, "OSFile"),
        (pd, "read_parquet"),
        (builtins, "open"),
        (io, "open"),
        (Path, "open"),
    ]
    try:
        import polars as pl

        owners += [(pl, "read_parquet"), (pl, "scan_parquet")]
    except ImportError:  # pragma: no cover
        pass
    try:
        import duckdb

        owners += [
            (duckdb, "read_parquet"),
            (duckdb, "sql"),
            (duckdb, "query"),
            (duckdb, "execute"),
        ]
    except ImportError:  # pragma: no cover
        pass
    for owner, name in owners:
        wrap(owner, name)

    original_file = pq.ParquetFile

    class RecordingParquetFile(original_file):
        def __init__(self, source, *args, **kwargs):
            note(source)
            super().__init__(source, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", RecordingParquetFile)
    return seen


def held_reads(seen: list[str]) -> list[str]:
    """The reads that name a holdout day's file (a path, or SQL text holding one)."""
    return [p for p in seen if any(f"{d.isoformat()}.parquet" in p for d in HOLDOUT_DAYS)]


# --- capacity ------------------------------------------------------------------------------------


def test_capacity_stake_halves_the_edge():
    """Pool 1,000,000, 10,000 on the combination: edge 0.10 gives 481.23 yen, 0.05 gives 246.43
    and 0.20 gives 919.12; at that stake (1 + e)(V + s)W / (V(W + s)) = 1 + e/2."""
    for edge, want in ((0.10, 481.23), (0.05, 246.43), (0.20, 919.12)):
        stake = depth.capacity_stake(1_000_000, 10_000, edge)
        assert stake == pytest.approx(want, abs=0.01), edge
        assert value_after_stake(1e6, 1e4, edge, stake) == pytest.approx(1 + edge / 2, rel=1e-12)
        same = depth.capacity_stake(1_000_000, 10_000, edge, share_lost=0.5)
        assert same == pytest.approx(stake, rel=1e-12)

    # Any pool, money and edge: the stake solves the identity and is the closed form at 0.5.
    rng = np.random.default_rng(4320)
    for _ in range(500):
        pool = float(np.exp(rng.uniform(math.log(1e5), math.log(1e10))))
        combo = pool * float(np.exp(rng.uniform(math.log(1e-6), math.log(0.6))))
        edge = float(rng.uniform(0.005, 0.5))
        stake = depth.capacity_stake(pool, combo, edge)
        assert isinstance(stake, float) and stake > 0, (pool, combo, edge, stake)
        assert value_after_stake(pool, combo, edge, stake) == pytest.approx(1 + edge / 2, rel=1e-9)
        k = edge / 2
        closed = k * pool * combo / ((1 + k) * pool - (1 + edge) * combo)
        assert stake == pytest.approx(closed, rel=1e-9)

    # More money on the combination takes more stake to halve the edge; so does a bigger edge.
    by_money = [depth.capacity_stake(1e6, w, 0.1) for w in (1e3, 1e4, 1e5, 3e5)]
    assert by_money == sorted(by_money) and len(set(by_money)) == 4
    by_edge = [depth.capacity_stake(1e6, 1e4, e) for e in (0.02, 0.05, 0.1, 0.2, 0.4)]
    assert by_edge == sorted(by_edge) and len(set(by_edge)) == 5


def test_capacity_in_a_deep_pool_is_a_share_of_the_money_on_the_combination():
    """Pool 1e9, W 10,000, edge 0.10: 476.20 yen (exactly 476.1955...), about W x 0.05 / 1.05;
    as the pool deepens the capacity falls to W x k / (1 + k), k = e / 2."""
    stake = depth.capacity_stake(1e9, 10_000, 0.10)
    assert stake == pytest.approx(476.20, abs=0.01)
    assert stake == pytest.approx(10_000 * 0.05 / 1.05, rel=1e-4)

    for combo in (1_000, 10_000, 100_000, 1_000_000):
        for edge in (0.05, 0.10, 0.20):
            k = edge / 2
            deep = depth.capacity_stake(1e12, combo, edge)
            assert deep == pytest.approx(combo * k / (1 + k), rel=1e-5), (combo, edge)
            # a shallower pool allows more: the pool's other money pays for part of our impact
            shallow_to_deep = [depth.capacity_stake(combo * m, combo, edge) for m in (4, 20, 1e3)]
            shallow_to_deep.append(deep)
            assert shallow_to_deep == sorted(shallow_to_deep, reverse=True), (combo, edge)


@pytest.mark.parametrize(
    "pool, combo, edge",
    [
        (1e6, 1e4, 0.0),  # no edge
        (1e6, 1e4, -0.05),
        (1e6, 1e4, float("nan")),
        (1e6, 1e4, float("inf")),
        (1e6, 1e6, 0.10),  # all the pool on the combination
        (1e6, 1.2e6, 0.10),  # more than the pool
        (0.0, 0.0, 0.10),
        (-1e6, 1e4, 0.10),
        (1e6, -1e4, 0.10),  # negative money
        (1e6, float("nan"), 0.10),
        (float("nan"), 1e4, 0.10),
        (1e6, 0.96e6, 0.10),  # (1 + k)V - (1 + e)W < 0
        (1e6, 0.93e6, 0.20),
    ],
)
def test_capacity_rejects_a_non_positive_edge_or_money_above_the_pool(pool, combo, edge):
    with pytest.raises(ValueError):
        depth.capacity_stake(pool, combo, edge)


def test_capacity_share_lost_is_the_part_of_the_edge_given_up():
    """share_lost generalises the half: the stake leaves 1 + e(1 - share_lost) of value (the
    ticket's closed form kVW / ((1 + k)V - (1 + e)W) holds at 0.5 only); share_lost is
    keyword-only."""
    for share in (0.25, 0.5, 0.75, 1.0):
        for pool, combo, edge in ((1e6, 1e4, 0.10), (3e7, 2.5e5, 0.05), (5e5, 1e5, 0.20)):
            stake = depth.capacity_stake(pool, combo, edge, share_lost=share)
            assert stake == pytest.approx(capacity(pool, combo, edge, share), rel=1e-9)
            left = value_after_stake(pool, combo, edge, stake)
            assert left == pytest.approx(1 + edge * (1 - share), rel=1e-9), (share, pool)
    more = [depth.capacity_stake(1e6, 1e4, 0.1, share_lost=x) for x in (0.25, 0.5, 0.75, 1.0)]
    assert more == sorted(more) and len(set(more)) == 4
    with pytest.raises(TypeError):
        depth.capacity_stake(1e6, 1e4, 0.1, 0.5)


# --- pool totals ---------------------------------------------------------------------------------


def totals_by_key(out: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    assert list(out.columns) == ["race_id", "pool", "final_yen", "at_t_yen"], list(out.columns)
    keys = list(zip(out["race_id"], out["pool"], strict=True))
    assert len(keys) == len(set(keys)), "a race and pool twice"
    finals = out["final_yen"].to_numpy(dtype=float)
    at_t = out["at_t_yen"].to_numpy(dtype=float, na_value=np.nan)
    return {k: (f, a) for k, f, a in zip(keys, finals, at_t, strict=True)}


def assert_totals(out: pd.DataFrame, races: list[Race], at_t: dict[tuple[str, str], float]):
    got = totals_by_key(out)
    want_keys = {(r.race_id, pool) for r in races for pool in r.pool_votes}
    assert set(got) == want_keys, (sorted(set(got) - want_keys), sorted(want_keys - set(got)))
    for race in races:
        for pool in race.pool_votes:
            final, at = got[(race.race_id, pool)]
            assert final == pytest.approx(race.pool_yen(pool)), (race.race_id, pool)
            want = at_t.get((race.race_id, pool))
            if want is None:
                assert math.isnan(at), (race.race_id, pool, at)
            else:
                assert at == pytest.approx(want), (race.race_id, pool, at, want)


def test_pool_totals_take_the_latest_total_at_or_before_t(data_dir):
    """Win `__pool__` captures at minutes_to_post 10, 3 and 1.5: the T-2 total is the 3-minute
    capture's. A capture exactly at t counts; post-race, undated and combination rows do not;
    pools without a time series (and races without a capture at t) have no total at t; the final
    total is the store's, whatever the time series says."""
    day = date(2019, 6, 1)
    a = make_race(11, day, "TOK", 1, "DEBUT", 6)
    b = make_race(12, day, "TOK", 2, "MAIDEN", 7)
    write_store(data_dir, [a, b])
    write_day(
        data_dir,
        day,
        [
            capture(a, "WIN", 10, 1_111_100),
            capture(a, "WIN", 3, 3_333_300),
            capture(a, "WIN", 1.5, 1_555_500),
            capture(a, "WIN", -2, 9_999_900),  # after the post: the final board
            capture(a, "WIN", 3, None, combination="1", odds=2.4),
            capture(a, "WIN", 1.5, None, combination="2", odds=5.1),
            capture(a, "PLACE", 2.5, 2_525_200),
            capture(a, "PLACE", 2.0, 2_020_200),  # exactly at t
            capture(a, "PLACE", 1.0, 1_010_100),
            capture(a, "QUINELLA", 30, 3_030_000),
            capture(a, "QUINELLA", 2.01, 2_010_000),
            capture(a, "QUINELLA", -1.0, 7_000_000),
            capture(a, "QUINELLA", None, 8_888_800),  # no post time: no minutes to post
            capture(b, "WIN", 4.0, 4_040_400),
            capture(b, "WIN", 2.5, 2_520_000),
            capture(b, "PLACE", -0.5, 5_000_000),  # only after the post
            capture(b, "PLACE", 0.5, None, combination="3", odds=1.4),
        ],
    )
    span = dict(date_from=day, date_to=day)

    at_two = {
        (a.race_id, "WIN"): 3_333_300,
        (a.race_id, "PLACE"): 2_020_200,
        (a.race_id, "QUINELLA"): 2_010_000,
        (b.race_id, "WIN"): 2_520_000,
    }
    assert_totals(depth.pool_totals(**span), [a, b], at_two)
    assert_totals(depth.pool_totals(**span, minutes_before=2.0), [a, b], at_two)

    at_one = {
        (a.race_id, "WIN"): 1_555_500,
        (a.race_id, "PLACE"): 1_010_100,
        (a.race_id, "QUINELLA"): 2_010_000,
        (b.race_id, "WIN"): 2_520_000,
    }
    assert_totals(depth.pool_totals(**span, minutes_before=1.0), [a, b], at_one)

    at_five = {(a.race_id, "WIN"): 1_111_100, (a.race_id, "QUINELLA"): 3_030_000}
    assert_totals(depth.pool_totals(**span, minutes_before=5.0), [a, b], at_five)

    assert_totals(depth.pool_totals(**span, minutes_before=45.0), [a, b], {})


def test_pool_totals_keep_development_days_only(data_dir, stand_in_holdout, monkeypatch):
    """Final and T-2 totals per race and pool, development days in date_from..date_to only: a
    holdout day's races give no row and its day file is never opened (one of them is not even a
    Parquet file)."""
    first, last = DEV_DAYS[1], DEV_DAYS[3]  # both development days, both read
    races = {}
    for i, day in enumerate((*DEV_DAYS, DEV_AFTER, *HOLDOUT_DAYS)):
        held = day in HOLDOUT_DAYS
        races[day] = make_race(
            100 + i, day, "NAK", 1, "MAIDEN" if i % 2 else "C2", 6, scale=500 if held else 1
        )
    write_store(data_dir, list(races.values()))
    at_t = {}
    for i, (day, race) in enumerate(races.items()):
        total = 1_000_000 * (i + 1) + 12_300
        rows = [capture(race, "WIN", 3, total), capture(race, "QUINELLA", 2.5, total + 4_500)]
        write_day(data_dir, day, rows)
        at_t[(race.race_id, "WIN")] = total
        at_t[(race.race_id, "QUINELLA")] = total + 4_500
    day_file(data_dir, HOLDOUT_DAYS[0]).write_bytes(b"PAR1 not a parquet file: never open it")

    seen = record_reads(monkeypatch)
    out = depth.pool_totals(date_from=first, date_to=last)

    assert not held_reads(seen), f"holdout day files were opened: {held_reads(seen)}"
    read = [races[d] for d in DEV_DAYS if first <= d <= last]
    assert len(read) == 3
    assert_totals(out, read, {k: v for k, v in at_t.items() if k[0] in {r.race_id for r in read}})


# --- depth rows and the depth table --------------------------------------------------------------

ROW_EDGES = {
    "WIN": (2.0, 5.0, 10.0),
    "PLACE": (1.5, 3.0),
    "QUINELLA": (10.0, 50.0),
    "QUINELLA_PLACE": (5.0, 20.0),
    "FORECAST": (20.0, 100.0),
    "TRIO": (50.0, 200.0),
}


def test_depth_rows_give_each_priced_combination_its_money_pool_class_and_band():
    """One row per priced combination of the pools named in band_edges: the money on it (votes x
    100), the race's pool row in yen, its class group and its band on that pool's edges; unpriced
    combinations and pools without edges (TIERCE here) give no row."""
    day = date(2019, 6, 2)
    with_tierce = (*POOLS, "TIERCE")
    races = [
        make_race(21, day, "KYO", 1, "DEBUT", 5, pools=with_tierce),
        make_race(22, day, "KYO", 2, "MAIDEN", 6, pools=with_tierce),
        make_race(23, day, "KYO", 11, "G2", 7, pools=with_tierce),
        make_race(24, day, "HSN", 3, "C1", 6, pools=with_tierce, scale=3),
    ]
    races[0].unprice("WIN", "5")
    races[1].unprice("FORECAST", "2,1")
    races[2].unprice("TRIO", "1,2,3")
    frames = {name: t.to_pandas() for name, t in store_tables(races).items()}

    out = depth.depth_rows(
        frames["pool_votes"], frames["final_odds"], frames["races"], band_edges=ROW_EDGES
    )

    want = expected_rows(races, ROW_EDGES)
    assert set(want["pool"]) == set(POOLS)
    needed = {"race_id", "pool", "combination", "class_group", "band", "combo_yen", "pool_yen"}
    assert needed <= set(out.columns), sorted(out.columns)
    got_keys = list(zip(out["race_id"], out["pool"], out["combination"], strict=True))
    assert len(got_keys) == len(set(got_keys)), "a combination twice"
    got = dict(zip(got_keys, out.to_dict("records"), strict=True))
    want_by = {(r["race_id"], r["pool"], r["combination"]): r for r in want.to_dict("records")}
    assert set(got) == set(want_by), (
        sorted(set(got) - set(want_by))[:5],
        sorted(set(want_by) - set(got))[:5],
    )
    for key, w in want_by.items():
        g = got[key]
        assert (g["class_group"], g["band"]) == (w["class_group"], w["band"]), (key, g, w)
        assert float(g["combo_yen"]) == pytest.approx(w["combo_yen"]), key
        assert float(g["pool_yen"]) == pytest.approx(w["pool_yen"]), key
    # the pool is the pool row, not the combinations' sum
    race = races[1]
    assert race.pool_yen("WIN") > 100 * sum(race.votes["WIN"].values())


def depth_cells(seed: int) -> pd.DataFrame:
    """Rows for `depth_table` across pools, class groups and bands, a race's pool constant."""
    rng = np.random.default_rng(seed)
    groups = ("newcomer", "maiden", "conditions", "open", "graded")
    rows = []
    for i in range(40):
        race_id = f"jra:2019-07-{6 + i % 2:02d}:X{i:02d}:01"
        group = groups[i % len(groups)]
        for pool in ("WIN", "PLACE", "QUINELLA", "FORECAST", "TRIO"):
            pool_yen = float(round(np.exp(rng.uniform(math.log(3e5), math.log(3e8))), -2))
            for j in range(int(rng.integers(1, 7))):
                share = float(np.exp(rng.uniform(math.log(1e-4), math.log(0.5))))
                rows.append(
                    {
                        "race_id": race_id,
                        "pool": pool,
                        "combination": str(j + 1),
                        "class_group": group,
                        "band": ("lt5", "5-20", "ge20")[int(rng.integers(3))],
                        "odds": float(rng.uniform(1.1, 300.0)),
                        "combo_yen": max(100.0, round(pool_yen * share, -2)),
                        "pool_yen": pool_yen,
                    }
                )
    # cells of a single combination: every quantile is its own value
    for pool, group, combo_yen, pool_yen in (
        ("WIN", "newcomer", 2_300.0, 4_100_000.0),
        ("FORECAST", "graded", 87_600.0, 9_870_000.0),
        ("PLACE", "maiden", 12_500.0, 1_250_000.0),
    ):
        rows.append(
            {
                "race_id": f"jra:2019-07-13:S{len(rows):04d}:01",
                "pool": pool,
                "combination": "1",
                "class_group": group,
                "band": "ge500",
                "odds": 612.3,
                "combo_yen": combo_yen,
                "pool_yen": pool_yen,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_depth_table_groups_by_pool_class_and_band():
    """Per pool x class group x odds band and quantile: the cell's combinations and races, the
    quantile of the money on a combination, of the pool (each race once) and of the capacity at
    each edge; other edges and quantiles give other columns and rows."""
    rows = depth_cells(4321)
    assert rows.groupby(["pool", "class_group", "band"]).size().min() == 1  # single-row cells too
    edges, quantiles = (0.05, 0.10, 0.20), (0.1, 0.5, 0.9)

    table = depth.depth_table(rows, edges=edges, quantiles=quantiles)

    assert_depth_table(table, expected_table(rows, edges, quantiles), edges)
    assert {"capacity_0.05", "capacity_0.1", "capacity_0.2"} <= set(table.columns)

    other = depth.depth_table(rows, edges=(0.15,), quantiles=(0.25, 0.75))
    assert_depth_table(other, expected_table(rows, (0.15,), (0.25, 0.75)), (0.15,))
    assert "capacity_0.15" in other.columns
    assert not {"capacity_0.05", "capacity_0.1", "capacity_0.2"} & set(other.columns)


# --- the spec and the run ------------------------------------------------------------------------


def test_the_shipped_spec_registers_the_cells_edges_and_share_lost(tmp_path):
    """The pre-registered spec: the edges 5, 10 and 20%, half the edge lost, the quantiles and an
    odds band per pool; plain JSON, so the registry can freeze it."""
    spec = shipped_spec()
    assert isinstance(spec.get("family_id"), str) and spec["family_id"].strip()
    assert [float(e) for e in spec["edges"]] == [0.05, 0.1, 0.2]
    assert float(spec["share_lost"]) == 0.5
    quantiles = [float(q) for q in spec["quantiles"]]
    assert quantiles and quantiles == sorted(set(quantiles))
    assert all(0 <= q <= 1 for q in quantiles)
    assert set(POOLS) <= set(spec["odds_bands"]), sorted(spec["odds_bands"])
    for pool, edges in spec["odds_bands"].items():
        edges = [float(e) for e in edges]
        assert edges and all(a < b for a, b in itertools.pairwise(edges)), pool
        assert edges[0] > 0, pool
    json.dumps(spec, allow_nan=False)
    registry = Registry(tmp_path / "r.jsonl", holdout_log=tmp_path / "h.jsonl")
    assert registry.freeze_family(spec["family_id"], spec) == _spec_hash(spec)


RUN_CLASSES = ("DEBUT", "MAIDEN", "C1", "DEBUT", "MAIDEN", "C3", "OP", "MAIDEN", "G1")


def run_races(days: tuple[date, ...], seeds, *, scale: int = 1, venue: str = "TOK") -> list[Race]:
    """A race per seed on the days in turn; a seed's class, field and odds do not depend on
    `scale` or the day, so the same seeds fill the same cells."""
    races = []
    for i, seed in enumerate(seeds):
        races.append(
            make_race(
                seed,
                days[i % len(days)],
                venue,
                i // len(days) + 1,
                RUN_CLASSES[seed % len(RUN_CLASSES)],
                6 + seed % 2,
                scale=scale,
            )
        )
    return races


JUNE = (date(2019, 6, 1), date(2019, 6, 2), date(2019, 6, 8))


def run_expected(races: list[Race], spec: dict) -> dict:
    rows = expected_rows(races, spec["odds_bands"])
    return expected_table(rows, spec["edges"], spec["quantiles"])


def test_run_refuses_an_unfrozen_family(tmp_path, data_dir):
    """No run before the freeze: nothing frozen, the family frozen with another spec, or the spec
    frozen under another family all refuse with NotPreRegistered, and nothing is written."""
    spec = shipped_spec()
    write_store(data_dir, run_races(JUNE, range(9)))
    out_dir = data_dir / "reports"
    span = dict(date_from=date(2019, 6, 1), date_to=date(2019, 6, 30), out_dir=out_dir)

    empty = Registry(tmp_path / "empty.jsonl", holdout_log=tmp_path / "h.jsonl")
    with pytest.raises(depth.NotPreRegistered):
        depth.run_depth(**span, registry=empty)
    with pytest.raises(depth.NotPreRegistered):
        depth.run_depth(**span)  # the default registry, nothing frozen in it

    changed = copy.deepcopy(dict(spec))
    changed["share_lost"] = 0.25
    other = Registry(tmp_path / "other.jsonl", holdout_log=tmp_path / "h.jsonl")
    other.freeze_family(spec["family_id"], changed)
    with pytest.raises(depth.NotPreRegistered):
        depth.run_depth(**span, registry=other)

    elsewhere = Registry(tmp_path / "elsewhere.jsonl", holdout_log=tmp_path / "h.jsonl")
    elsewhere.freeze_family(f"{spec['family_id']}-not-this-one", spec)
    with pytest.raises(depth.NotPreRegistered):
        depth.run_depth(**span, registry=elsewhere)

    assert not out_dir.exists() or not any(p.is_file() for p in out_dir.rglob("*"))
    for registry in (empty, other, elsewhere):
        assert not [r for r in registry.records() if r.status == "scored"]


def test_run_gives_the_depth_table_and_records_the_run(tmp_path, data_dir):
    """DoD: per pool x class group x odds band, the quantiles of the money on a combination and of
    the pool and the capacity stake at edges 5, 10 and 20%, with the newcomer and maiden rows in
    every pool; the private report under out_dir; one scored RunRecord; the committed summary
    untouched. Races outside the dates are not read."""
    registry, spec = frozen_registry(tmp_path)
    inside = run_races(JUNE, range(9))
    outside = [
        make_race(31, date(2019, 5, 26), "TOK", 1, "G3", 6),
        make_race(32, date(2019, 7, 6), "TOK", 1, "L", 6),
    ]
    write_store(data_dir, inside + outside)
    out_dir = data_dir / "reports"
    before = summary_snapshot()
    span = dict(date_from=JUNE[0], date_to=date(2019, 6, 30), out_dir=out_dir)

    result = depth.run_depth(**span, registry=registry)

    assert_depth_table(result.table, run_expected(inside, spec), spec["edges"])
    for pool in POOLS:
        groups = set(result.table.loc[result.table["pool"] == pool, "class_group"])
        assert {"newcomer", "maiden"} <= groups, (pool, groups)
    assert result.family_id == spec["family_id"]
    assert result.spec_hash == _spec_hash(spec) == registry.frozen_spec_hash(spec["family_id"])

    report = Path(result.report_path)
    assert report.is_file() and report.resolve().is_relative_to(out_dir.resolve())
    text = report.read_text(encoding="utf-8")
    assert all(pool in text for pool in POOLS)
    assert "newcomer" in text and "maiden" in text

    records = registry.records(spec["family_id"])
    assert [r.status for r in records] == ["scored"]
    record = records[0]
    assert (record.hypothesis_id, record.ticket, tuple(record.sources)) == (
        HYPOTHESIS_ID,
        TICKET,
        SOURCES,
    )
    registry.verify()

    depth.run_depth(**span, registry=registry)  # the same run again does not raise
    registry.verify()
    assert summary_snapshot() == before


def test_run_records_in_the_default_registry_when_none_is_given(data_dir):
    spec = shipped_spec()
    default_registry("japan").freeze_family(spec["family_id"], spec)
    write_store(data_dir, run_races(JUNE, range(9)))

    depth.run_depth(date_from=JUNE[0], date_to=JUNE[-1], out_dir=data_dir / "reports")

    registry = default_registry("japan")
    assert registry.path.resolve().is_relative_to(data_dir)
    statuses = [r.status for r in registry.records(spec["family_id"])]
    assert statuses == ["scored"], statuses


def test_a_run_that_fails_after_the_freeze_is_a_registry_record(tmp_path, data_dir):
    registry, spec = frozen_registry(tmp_path)
    write_store(data_dir, run_races(JUNE, range(9)), skip=("pool_votes",))

    with pytest.raises(Exception):  # noqa: B017 - any failure, as long as it is recorded
        depth.run_depth(
            date_from=JUNE[0], date_to=JUNE[-1], out_dir=data_dir / "reports", registry=registry
        )

    records = registry.records(spec["family_id"])
    assert [r.status for r in records] == ["failed"], [r.status for r in records]
    record = records[0]
    assert (record.hypothesis_id, record.ticket, tuple(record.sources)) == (
        HYPOTHESIS_ID,
        TICKET,
        SOURCES,
    )


def test_the_private_report_is_refused_inside_the_repository(tmp_path, data_dir):
    """The report is built on JRA-VAN data, so it goes through write_report, which refuses the
    repository: nothing is written there and the run is not scored."""
    registry, spec = frozen_registry(tmp_path)
    write_store(data_dir, run_races(JUNE, range(9)))
    inside = REPO / f".pool-depth-test-{uuid.uuid4().hex[:12]}"
    try:
        with pytest.raises(LicensedOutputInRepo):
            depth.run_depth(date_from=JUNE[0], date_to=JUNE[-1], out_dir=inside, registry=registry)
        assert not inside.exists() or not any(p.is_file() for p in inside.rglob("*"))
    finally:
        shutil.rmtree(inside, ignore_errors=True)
    assert not [r for r in registry.records(spec["family_id"]) if r.status == "scored"]


def test_run_refuses_a_start_before_the_price_era(tmp_path, data_dir):
    """D10: no price row before 2014-06-07. A start before it raises ValueError and writes no
    report; a start on it reads that day."""
    registry, spec = frozen_registry(tmp_path)
    early = make_race(41, date(2014, 6, 1), "TOK", 1, "MAIDEN", 6)
    first = make_race(42, date(2014, 6, 7), "TOK", 1, "DEBUT", 7)
    write_store(data_dir, [early, first])
    out_dir = data_dir / "reports"

    for start in (date(2014, 6, 6), date(2014, 6, 1), date(2010, 1, 1)):
        with pytest.raises(ValueError):
            depth.run_depth(
                date_from=start, date_to=date(2014, 6, 30), out_dir=out_dir, registry=registry
            )
    assert not out_dir.exists() or not any(p.is_file() for p in out_dir.rglob("*"))
    assert not [r for r in registry.records(spec["family_id"]) if r.status == "scored"]

    result = depth.run_depth(
        date_from=date(2014, 6, 7), date_to=date(2014, 6, 30), out_dir=out_dir, registry=registry
    )
    assert_depth_table(result.table, run_expected([first], spec), spec["edges"])


def test_run_reads_no_holdout_day(tmp_path, data_dir, stand_in_holdout, monkeypatch):
    """Holdout days' rows sit in the store tables (huge money, classes no development race has)
    and their time-series day files sit beside the development ones (one not even Parquet): the
    run's table is the development races' alone, no holdout day file is opened and
    LockedHoldout is never opened."""
    registry, spec = frozen_registry(tmp_path)
    development = run_races(DEV_DAYS[1:], range(9), venue="KOK")
    holdout = [
        make_race(51, HOLDOUT_DAYS[0], "KOK", 1, "G2", 6, scale=1000),
        make_race(52, HOLDOUT_DAYS[1], "KOK", 1, "L", 7, scale=1000),
        make_race(53, HOLDOUT_DAYS[2], "KOK", 1, "MAIDEN", 6, scale=1000),
    ]
    write_store(data_dir, development + holdout)
    boards: dict[date, list[dict]] = {}
    for race in development + holdout:
        boards.setdefault(race.day, []).append(
            capture(race, "WIN", 3, race.pool_yen("WIN") - 50_000)
        )
    for day, rows in boards.items():
        write_day(data_dir, day, rows)
    day_file(data_dir, HOLDOUT_DAYS[0]).write_bytes(b"PAR1 not a parquet file: never open it")

    seen = record_reads(monkeypatch)
    result = depth.run_depth(
        date_from=STAND_IN_START,
        date_to=DEV_DAYS[-1],
        out_dir=data_dir / "reports",
        registry=registry,
    )

    assert not held_reads(seen), f"holdout day files were opened: {held_reads(seen)}"
    # the holdout races (G2, L, MAIDEN at 1000x the money) would change these cells' counts
    assert_depth_table(result.table, run_expected(development, spec), spec["edges"])


TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?")
HEX = re.compile(r"\b[0-9a-f]{12,}\b")


def without_identities(text: str) -> str:
    """The text with run identities (hex hashes) and timestamps blanked."""
    return HEX.sub("<hex>", TIMESTAMP.sub("<time>", text))


def renderings(value: float) -> set[str]:
    """How a figure from the table would print: to the yen when it is whole, else to 2 places.
    Small and round numbers are skipped: they may be the spec's worked arithmetic."""
    if abs(value) < 1000 or value % 1000 == 0:
        return set()
    if value == int(value):
        return {f"{value:.0f}", f"{value:,.0f}"} if abs(value) >= 10_000 else set()
    return {f"{value:.2f}", f"{value:,.2f}"}


def test_public_report_carries_no_figures(tmp_path, data_dir):
    """D21: the public report holds the identity and worked arithmetic, never a figure from the
    data. Two runs over different data (other days, other money, more races, the same cells)
    render the same text, and no figure of either run's table appears in it."""
    out_dir = data_dir / "reports"
    span = dict(date_from=date(2019, 6, 1), date_to=date(2019, 6, 30), out_dir=out_dir)

    registry_a, spec = frozen_registry(tmp_path, "a")
    write_store(data_dir, run_races(JUNE, range(9)))
    result_a = depth.run_depth(**span, registry=registry_a)

    registry_b, _ = frozen_registry(tmp_path, "b")
    later = (date(2019, 6, 9), date(2019, 6, 15), date(2019, 6, 16), date(2019, 6, 22))
    write_store(
        data_dir,
        run_races(later, range(9), scale=3) + run_races(later, (0, 4, 8), scale=7, venue="HSN"),
    )
    result_b = depth.run_depth(**span, registry=registry_b)

    text_a = depth.render_public_report(result_a)
    text_b = depth.render_public_report(result_b)
    assert isinstance(text_a, str) and text_a.strip()
    assert WITHHELD in text_a
    assert without_identities(text_a) == without_identities(text_b), (
        "the public report changes with the data: it carries a figure computed from it"
    )

    figure_columns = ["combo_yen", "pool_yen", *(f"capacity_{float(e):g}" for e in spec["edges"])]
    for result in (result_a, result_b):
        values = result.table[figure_columns].to_numpy(dtype=float).ravel()
        for value in values[np.isfinite(values)]:
            for text in renderings(float(value)):
                assert text not in text_a, f"the figure {text} is in the public report"


# --- review round 1: the co-placer rule, the readers, the run identities and the command --------
#
# Contract added here (interfaces the tests call, beyond the ones above):
# - `coplacer_capacities(pool, combo, paid_money, places, rate, edge, share)` (numpy arrays).
# - `run_coplacer(...)` and `run_exacta_impact(...)` take run_depth's arguments; their results
#   carry `run_id` and `report_path`; `CoplacerResult` carries `status` and `rule_match` (the
#   pre-registered rule's share of the paid dividends reproduced: one number, or a mapping per
#   pool); `ImpactResult.bins` has `n_combinations` and `n_races` per pool-size bin.
# - A run that fails (raises, or fails its gate) is one registry record with status "failed".
# - `horsetrack japan pool-depth run [--no-public]` resolves the committed summary through
#   `summary_dir` (patched here, in this module and in core.eval.registry).

JUNE_DAYS = tuple(date(2019, 6, d) for d in (1, 2, 8, 9, 15, 16))
JUNE_SPAN = dict(date_from=date(2019, 6, 1), date_to=date(2019, 6, 30))


def set_votes(race: Race, pool: str, new: dict[str, int]) -> None:
    """New votes on some combinations of `pool`: the pool row keeps its money beyond the
    combinations' sum and the final odds follow the money."""
    extra = race.pool_votes[pool] - sum(race.votes[pool].values())
    race.votes[pool].update(new)
    total = sum(race.votes[pool].values()) + extra
    race.pool_votes[pool] = total
    race.odds[pool] = {
        c: max(1.0, math.floor(10 * RATES[pool] * total / v) / 10)
        for c, v in race.votes[pool].items()
    }


def finished(race: Race, seed: int, *, dead_heat: int | None = None) -> Race:
    """`race` with a finishing order drawn from `seed`; with `dead_heat` k, the k-th and the next
    horse of the order share the k-th position."""
    n = len(race.votes["PLACE"])
    groups = [(int(h),) for h in np.random.default_rng(seed).permutation(np.arange(1, n + 1))]
    if dead_heat is not None:
        groups[dead_heat - 1 : dead_heat + 1] = [groups[dead_heat - 1] + groups[dead_heat]]
    race.order = tuple(groups)
    return race


def coplacer_races(
    n_races: int, *, seed: int = 0, pools: tuple[str, ...] = CO_PLACER, venue: str = "TOK"
) -> list[Race]:
    """Races of 5 to 10 runners over June 2019 with the pools `pools` and a finishing order;
    every third one has refunded votes in its pools."""
    races = []
    for i in range(n_races):
        s = seed + i
        race = make_race(
            1000 + s,
            JUNE_DAYS[i % len(JUNE_DAYS)],
            venue,
            1 + i // len(JUNE_DAYS),
            RUN_CLASSES[s % len(RUN_CLASSES)],
            5 + s % 6,
            pools=pools,
        )
        if s % 3 == 0:
            race.refunds = {p: max(1, race.pool_votes[p] // 40) for p in pools}
        races.append(finished(race, s))
    return races


def jra_rule_races() -> list[Race]:
    """Eighteen Place and Wide races (fields 5 to 10), with dead heats for third in a field of
    nine and for second in one of six, and two races whose paying horses hold so much of the
    Place pool that the rule pays the 110-yen floor."""
    races = coplacer_races(18)
    by_field = {n: [r for r in races if len(r.votes["PLACE"]) == n] for n in range(5, 11)}
    finished(by_field[9][0], 7, dead_heat=3)
    finished(by_field[6][0], 8, dead_heat=2)
    for race, heavy in zip(
        by_field[8][:2],
        ({"1": 60_000, "2": 50_000, "3": 40_000}, {"1": 18_000, "2": 16_000, "3": 15_000}),
        strict=True,
    ):
        race.order = tuple((h,) for h in range(1, 9))
        set_votes(race, "PLACE", heavy)
    return races


def statuses(registry: Registry, variant: str) -> list[str]:
    family = shipped_spec()["family_id"]
    return [r.status for r in registry.records(family) if r.variant == variant]


def run_quietly(run, **kwargs):
    """The run's result, or None when it raised (a run that fails is recorded, then raised)."""
    try:
        return run(**kwargs)
    except Exception:  # noqa: BLE001 - the registry says what happened
        return None


def match_shares(value) -> list[float]:
    """The rule's match share(s): one number, or one per pool."""
    values = list(value.values()) if isinstance(value, abc.Mapping) else [value]
    return [float(v) for v in values]


def bump_dividends(data: Path, chosen) -> int:
    """Pay 10 yen more on the store's `dividends` rows `chosen(frame)` selects; how many."""
    path = data / "processed" / "jravan" / "dividends.parquet"
    paid = pq.read_table(path).to_pandas()
    rows = chosen(paid)
    paid.loc[rows, "dividend"] += 10.0
    table = pa.Table.from_pandas(paid, schema=dividends_schema(), preserve_index=False)
    pq.write_table(table, path)
    return int(np.asarray(rows).sum()) if np.asarray(rows).dtype == bool else len(rows)


def test_the_coplacer_run_reproduces_dividends_paid_by_jra_rule(tmp_path, data_dir):
    """Place and Wide dividends paid by JRA's rule (the net pool, P = 2 places in a field of 5-7
    and 3 in one of 8 or more, dead heats adding paying combinations, breakage to 10 yen, the
    110-yen floor): the co-placer run reproduces every one of them and is scored. The same store
    with one paid dividend in eight 10 yen off, in each pool, is recorded failed."""
    registry, spec = frozen_registry(tmp_path)
    races = jra_rule_races()
    place_p = {len(paying(r, "PLACE")) for r in races}
    wide_p = {len(paying(r, "QUINELLA_PLACE")) for r in races}
    assert {2, 3, 4} <= place_p and {3, 5} <= wide_p, (place_p, wide_p)
    raw = [per_100(r, p, c) for r in races for p in CO_PLACER for c in paying(r, p)]
    assert any(x < 100 for x in raw), "no payout where the paying money exceeds rV"
    assert any(100 <= x < 110 for x in raw), "no payout the floor lifts after breakage"
    assert any(x >= 110 and x % 10 for x in raw), "no payout breakage cuts"
    assert any(r.refunds for r in races)
    write_store(data_dir, races)
    out_dir = data_dir / "reports"

    result = depth.run_coplacer(**JUNE_SPAN, out_dir=out_dir, registry=registry)

    assert result.status == "scored"
    assert match_shares(result.rule_match) and all(
        v == 1.0 for v in match_shares(result.rule_match)
    ), result.rule_match
    assert statuses(registry, "co-placer") == ["scored"]

    bumped = bump_dividends(
        data_dir, lambda paid: [i for p in CO_PLACER for i in paid.index[paid["pool"] == p][::8]]
    )
    assert bumped >= 8
    run_quietly(depth.run_coplacer, **JUNE_SPAN, out_dir=out_dir, registry=registry)
    assert statuses(registry, "co-placer") == ["scored", "failed"]


def stake_back_value(pool, combo, paid, places, rate, stake):
    """d(s), JRA's rule per yen once our stake s is on combination i: s goes into the pool, into
    W_i and into the paying money."""
    return 1 + (rate * (pool + stake) - (paid + stake)) / (places * (combo + stake))


def test_coplacer_capacity_solves_the_stake_back_share_identity():
    """With V the pool, W_i the money on a paying combination, sum W the money on the race's P
    paying combinations and r the payout rate, the capacity s solves
    (1 + e) d(s) / d(0) = 1 + e(1 - share_lost); inf only where no stake gets there; NaN where
    rV <= sum W; at P = 1 (sum W = W_i) it is the single-winner capacity_stake."""
    rng = np.random.default_rng(43200)
    m = 300
    pool = np.exp(rng.uniform(math.log(1e5), math.log(1e9), m))
    paid = pool * rng.uniform(0.02, 0.7, m)
    places = rng.integers(2, 6, m).astype(float)
    combo = paid * rng.uniform(0.05, 0.95, m)
    rate = np.where(rng.random(m) < 0.5, 0.8, 0.775)
    for edge in (0.05, 0.1, 0.2):
        for share in (0.25, 0.5, 1.0):
            k = edge * (1 - share)
            s = np.asarray(
                depth.coplacer_capacities(pool, combo, paid, places, rate, edge, share), float
            )
            assert s.shape == (m,)
            finite = np.isfinite(s)
            assert finite.mean() > 0.5, (edge, share)
            d0 = stake_back_value(pool, combo, paid, places, rate, 0.0)
            at = np.where(finite, s, 0.0)
            left = (1 + edge) * stake_back_value(pool, combo, paid, places, rate, at) / d0
            assert np.all(s[finite] > 0)
            np.testing.assert_allclose(left[finite], 1 + k, rtol=1e-9)
            far = (1 + edge) * stake_back_value(pool, combo, paid, places, rate, 1e15) / d0
            assert np.all(far[~finite] > 1 + k - 1e-9), (edge, share)

    one = np.ones(m)
    for edge in (0.05, 0.1, 0.2):
        s = depth.coplacer_capacities(pool, combo, combo, one, rate, edge, 0.5)
        want = [depth.capacity_stake(v, w, edge) for v, w in zip(pool, combo, strict=True)]
        np.testing.assert_allclose(np.asarray(s, float), want, rtol=1e-9)

    losing = depth.coplacer_capacities(pool, combo, rate * pool * 1.01, places, rate, 0.1, 0.5)
    assert np.isnan(np.asarray(losing, float)).all()


def coverage_races() -> tuple[list[Race], list[Race]]:
    """Forty Place and Wide races and forty Place-only races (Wide not sold), every payout by
    JRA's rule."""
    return coplacer_races(40, seed=0), coplacer_races(40, seed=40, pools=("PLACE",), venue="HSN")


@pytest.mark.parametrize(
    "gap, k, status",
    [
        ("no Wide odds in the store", 40, "failed"),
        ("the Wide votes and odds of 4 races missing", 4, "failed"),
        ("Wide paid 10 yen more in 4 races", 4, "failed"),
        ("the Wide votes and odds of 1 race missing", 1, "scored"),
    ],
)
def test_coplacer_payouts_are_checked_per_pool_and_every_payout_counts(
    tmp_path, data_dir, gap, k, status
):
    """Every eligible payout counts: a paid combination with no votes or odds row is one the run
    does not reproduce, and the share is checked per pool. A store with no Wide odds fails
    rather than passing on Place alone; with 4 of 40 Wide races unmatched or mispaid (Wide 90%,
    both pools together over 95%) the run fails; with 1 of 40 (97.5%) it is scored."""
    registry, _ = frozen_registry(tmp_path)
    wide, place_only = coverage_races()
    missing = {r.race_id for r in wide[:k]}
    n_wide = sum(len(paying(r, "QUINELLA_PLACE")) for r in wide)
    n_place = sum(len(paying(r, "PLACE")) for r in wide + place_only)
    off = sum(len(paying(r, "QUINELLA_PLACE")) for r in wide[:k])
    if k == 4:  # the pools together would pass; Wide alone does not
        assert (n_wide - off) / n_wide < 0.95 <= (n_wide + n_place - off) / (n_wide + n_place)
    if k == 1:
        assert (n_wide - off) / n_wide >= 0.95

    def drop(name: str, row: dict) -> bool:
        if gap.startswith("no Wide odds"):
            return name == "final_odds" and row["pool"] == "QUINELLA_PLACE"
        if gap.startswith("the Wide votes"):
            return (
                name in ("pool_votes", "final_odds")
                and row["pool"] == "QUINELLA_PLACE"
                and row["race_id"] in missing
            )
        return False

    write_store(data_dir, wide + place_only, drop=drop)
    if gap.startswith("Wide paid"):
        bump_dividends(
            data_dir,
            lambda paid: (
                paid["race_id"].isin(missing).to_numpy()
                & (paid["pool"] == "QUINELLA_PLACE").to_numpy()
            ),
        )

    run_quietly(depth.run_coplacer, **JUNE_SPAN, out_dir=data_dir / "reports", registry=registry)

    assert statuses(registry, "co-placer") == [status], gap


def n_priced(races: list[Race], pool: str) -> int:
    return sum(1 for r in races for price in r.odds[pool].values() if price is not None)


@pytest.mark.parametrize("table", ["pool_votes", "final_odds"])
def test_day_before_records_are_not_final(tmp_path, data_dir, table):
    """A race whose `table` records are the day before's only (データ区分 2, 前日売最終) has no
    final money: no final_yen in pool_totals (for pool_votes; its T-2 capture still gives its
    row, final_yen NaN and at_t_yen the capture's), no depth row and no exacta row. Confirmed
    records (4 確定, and 5 確定(月曜)) count."""
    registry, spec = frozen_registry(tmp_path)
    races = run_races(JUNE, range(9))
    races[1].votes_kubun = races[1].board_kubun = "5"
    held = races[4]
    if table == "pool_votes":
        held.votes_kubun = "2"
    else:
        held.board_kubun = "2"
    kept = [r for r in races if r is not held]
    write_store(data_dir, races)
    at_t = {(r.race_id, "WIN"): r.pool_yen("WIN") - 50_000 - 700 * i for i, r in enumerate(races)}
    for day in JUNE:
        write_day(data_dir, day, [capture(r, "WIN", 3, at_t[(r.race_id, "WIN")])
                                  for r in races if r.day == day])  # fmt: skip
    out_dir = data_dir / "reports"

    if table == "pool_votes":
        totals = depth.pool_totals(**JUNE_SPAN)
        assert totals.loc[totals["race_id"] == held.race_id, "final_yen"].isna().all()
        got = totals_by_key(totals.loc[totals["race_id"] != held.race_id])
        assert set(got) == {(r.race_id, p) for r in kept for p in r.pool_votes}
        for r in kept:
            for p in r.pool_votes:
                assert got[(r.race_id, p)][0] == pytest.approx(r.pool_yen(p)), (r.race_id, p)
                want, at = at_t.get((r.race_id, p)), got[(r.race_id, p)][1]
                assert math.isnan(at) if want is None else at == pytest.approx(want), (r.race_id, p)
        # the day-before record drops the final total, not the race's capture at t
        mine = totals_by_key(totals.loc[totals["race_id"] == held.race_id])
        assert (held.race_id, "WIN") in mine, "the T-2 capture was dropped with the final total"
        final, at = mine[(held.race_id, "WIN")]
        assert math.isnan(final), final
        assert at == pytest.approx(at_t[(held.race_id, "WIN")]), at
        assert all(math.isnan(a) for (_, p), (_, a) in mine.items() if p != "WIN"), mine

    result = depth.run_depth(**JUNE_SPAN, out_dir=out_dir, registry=registry)
    assert_depth_table(result.table, run_expected(kept, spec), spec["edges"])

    impact = depth.run_exacta_impact(**JUNE_SPAN, out_dir=out_dir, registry=registry)
    assert int(impact.bins["n_combinations"].sum()) == n_priced(kept, "FORECAST")
    assert int(impact.bins["n_races"].sum()) == len(kept)


def test_exacta_impact_without_a_priced_exacta_is_recorded_failed(tmp_path, data_dir):
    """With the votes but no exacta final odds every pool-size bin is empty: the run is recorded
    failed, not scored. With the odds, the bins hold every priced exacta and it is scored."""
    registry, _ = frozen_registry(tmp_path)
    races = run_races(JUNE, range(9))
    out_dir = data_dir / "reports"
    write_store(data_dir, races)
    impact = depth.run_exacta_impact(**JUNE_SPAN, out_dir=out_dir, registry=registry)
    assert int(impact.bins["n_combinations"].sum()) == n_priced(races, "FORECAST")
    assert statuses(registry, "exacta-own-impact") == ["scored"]

    write_store(
        data_dir, races, drop=lambda name, row: name == "final_odds" and row["pool"] == "FORECAST"
    )
    run_quietly(depth.run_exacta_impact, **JUNE_SPAN, out_dir=out_dir, registry=registry)

    assert statuses(registry, "exacta-own-impact") == ["scored", "failed"]


def test_run_depth_takes_the_net_pool_and_drops_abandoned_races(tmp_path, data_dir):
    """V is the money that pays, (votes - refund_votes) x 100, where a pool has refunded votes
    (the store's pool votes count them); an abandoned race gives no row."""
    registry, spec = frozen_registry(tmp_path)
    races = run_races(JUNE, range(9))
    for race in races[::2]:
        race.refunds = {p: max(1, race.pool_votes[p] // 20) for p in race.pool_votes}
    races[3].status = "abandoned"
    write_store(data_dir, races)

    result = depth.run_depth(**JUNE_SPAN, out_dir=data_dir / "reports", registry=registry)

    assert_depth_table(result.table, run_expected(races, spec), spec["edges"])
    assert races[3].race_id not in set(expected_rows(races, spec["odds_bands"])["race_id"])


# The runs' inputs beside ineligible rows: development races on 2014-06-07/08 and on DEV_DAYS,
# holdout races on HOLDOUT_DAYS (stand-in) and a race on 2014-06-01, before the price era.
EARLY_DAY = date(2014, 6, 1)
ERA_DAYS = (date(2014, 6, 7), date(2014, 6, 8))
ELIGIBLE_SPAN = dict(date_from=PRICE_ERA, date_to=DEV_DAYS[-1])
RUNNERS = {
    "depth": "run_depth",
    "co-placer": "run_coplacer",
    "exacta-own-impact": "run_exacta_impact",
}


def runner(variant: str):
    return getattr(depth, RUNNERS[variant])


def mixed_store(data: Path, *, held: int = 1, early: int = 1, dev: int = 1) -> list[Race]:
    """The store and the time-series day files of every day, holdout and pre-era days included;
    `held`, `early` and `dev` change the money and finishing order of the holdout races, of the
    pre-era race and of one development race. Returns the development races."""
    days = (*ERA_DAYS, *DEV_DAYS)
    development = [
        finished(
            make_race(60 + i, days[i % len(days)], "KOK", 1 + i // len(days),
                      RUN_CLASSES[i % len(RUN_CLASSES)], 6 + i % 3, scale=dev if i == 0 else 1),
            i + 100 * (dev - 1) * (i == 0),
        )
        for i in range(8)
    ]  # fmt: skip
    holdout = [
        finished(make_race(70 + i, d, "KOK", 1, ("G2", "L", "MAIDEN")[i], 7, scale=1000 * held),
                 90 + i + 10 * held)
        for i, d in enumerate(HOLDOUT_DAYS)
    ]  # fmt: skip
    pre_era = finished(make_race(80, EARLY_DAY, "KOK", 1, "G3", 8, scale=early), 95 + early)
    races = [*development, *holdout, pre_era]
    write_store(data, races)
    bump = {id(r): held for r in holdout} | {id(pre_era): early}
    for day in sorted({r.day for r in races}):
        rows = [
            capture(r, "WIN", 3, r.pool_yen("WIN") - 50_000 * bump.get(id(r), 1))
            for r in races
            if r.day == day
        ]
        write_day(data, day, rows)
    return development


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_run_identity_is_derived_from_the_eligible_inputs(
    tmp_path, data_dir, stand_in_holdout, variant
):
    """A run's identity (run id and data_builds) comes from what it may read: a store whose
    holdout days' rows and day files changed, or whose race before 2014-06-07 changed, gives the
    same run id and data_builds; a change on a development day gives other data_builds."""

    def identity(name: str, **change) -> tuple[str, dict]:
        registry, spec = frozen_registry(tmp_path, name)
        mixed_store(data_dir, **change)
        run = runner(variant)(**ELIGIBLE_SPAN, out_dir=data_dir / "reports" / name,
                              registry=registry)  # fmt: skip
        [record] = [r for r in registry.records(spec["family_id"]) if r.variant == variant]
        assert record.status == "scored", (name, record.status)
        return run.run_id, record.data_builds

    base = identity("base")
    assert identity("holdout-refreshed", held=3) == base, "holdout rows changed the identity"
    assert identity("pre-era-refreshed", early=3) == base, "pre-2014-06-07 rows changed it"
    assert identity("development-refreshed", dev=2)[1] != base[1]


def row_days(frame) -> list[date]:
    """The race days of a table, batch or frame: its race_date, else the day in its race_id."""
    names = list(frame.columns) if isinstance(frame, pd.DataFrame) else frame.schema.names
    column = "race_date" if "race_date" in names else "race_id" if "race_id" in names else None
    if column is None:
        return []
    if isinstance(frame, pd.DataFrame):
        values = list(pd.unique(frame[column].dropna()))
    else:
        values = [v for v in pc.unique(frame.column(column)).to_pylist() if v is not None]
    days = []
    for value in values:
        if column == "race_date":
            days.append(pd.Timestamp(value).date())
            continue
        try:
            days.append(date.fromisoformat(str(value).split(":")[1]))
        except (IndexError, ValueError):
            continue
    return days


def bare(value):
    """`value` with every Watched object unwrapped, inside lists and tuples too."""
    if isinstance(value, Watched):
        return value._inner
    if isinstance(value, (list, tuple)):
        return type(value)(bare(v) for v in value)
    return value


def bare_kwargs(kwargs: dict) -> dict:
    return {k: bare(v) for k, v in kwargs.items()}


class RowSpy:
    """Checks every table, batch or frame a Parquet reader hands to Python: each row must be of a
    development day on or after 2014-06-07, by its race_date or, without one, the day in its
    race_id. `bad` lists the reads that broke it, `rows` counts the rows checked."""

    LAZY = (ds.Dataset, ds.Scanner, ds.Fragment, pa.RecordBatchReader)  # the real classes

    def __init__(self) -> None:
        self.bad: list[tuple[str, list[str]]] = []
        self.rows = 0

    def wrap(self, value, where: str):
        frame = value.record_batch if isinstance(value, ds.TaggedRecordBatch) else value
        if isinstance(frame, (pa.Table, pa.RecordBatch, pd.DataFrame)):
            self.rows += len(frame)
            wrong = sorted(
                {
                    d.isoformat()
                    for d in row_days(frame)
                    if d < PRICE_ERA or holdout_module.is_holdout_day("japan", d)
                }
            )
            if wrong:
                self.bad.append((where, wrong))
            return value
        if isinstance(value, self.LAZY):
            return Watched(value, self, where)
        if isinstance(value, list):
            return [self.wrap(v, where) for v in value]
        if isinstance(value, abc.Iterator):
            return (self.wrap(v, where) for v in value)
        return value


class Watched:
    """A dataset, scanner, fragment or batch reader whose every table or batch is checked."""

    def __init__(self, inner, spy: RowSpy, where: str) -> None:
        self.__dict__.update(_inner=inner, _spy=spy, _where=where)

    def __getattr__(self, name: str):
        value = getattr(self._inner, name)
        if not callable(value) or isinstance(value, type):
            return value

        def call(*args, **kwargs):
            out = value(*bare(args), **bare_kwargs(kwargs))
            return self._spy.wrap(out, f"{self._where}.{name}")

        return call

    def __iter__(self):
        return (self._spy.wrap(v, self._where) for v in self._inner)

    def __next__(self):
        return self._spy.wrap(next(self._inner), self._where)

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)


def install_row_spy(monkeypatch: pytest.MonkeyPatch) -> RowSpy:
    """From now on every row a Parquet reader hands to Python goes through a RowSpy, whatever
    route reads it: pq.read_table (and so core.eval.holdout.read_development and
    pd.read_parquet), pq.ParquetFile's reads, ds.dataset, the ds.Scanner constructors, and the
    fragments of ds.ParquetFileFormat().make_fragment and the ds.FileSystemDataset built on them
    (their to_table, to_batches, scanner, head, take and readers)."""
    spy = RowSpy()
    real_read_table, real_dataset = pq.read_table, ds.dataset
    real_scanner, real_format, real_fsd = ds.Scanner, ds.ParquetFileFormat, ds.FileSystemDataset

    def read_table(*args, **kwargs):
        return spy.wrap(real_read_table(*bare(args), **bare_kwargs(kwargs)), "pq.read_table")

    def dataset(*args, **kwargs):
        return spy.wrap(real_dataset(*bare(args), **bare_kwargs(kwargs)), "ds.dataset")

    monkeypatch.setattr(pq, "read_table", read_table)
    monkeypatch.setattr(ds, "dataset", dataset)
    for name in ("iter_batches", "read", "read_row_group", "read_row_groups"):
        original = getattr(pq.ParquetFile, name)

        def method(self, *args, _original=original, _name=name, **kwargs):
            return spy.wrap(_original(self, *args, **kwargs), f"ParquetFile.{_name}")

        monkeypatch.setattr(pq.ParquetFile, name, method)

    class Scanner:
        @staticmethod
        def from_dataset(*args, **kwargs):
            out = real_scanner.from_dataset(*bare(args), **bare_kwargs(kwargs))
            return spy.wrap(out, "Scanner.from_dataset")

        @staticmethod
        def from_fragment(*args, **kwargs):
            out = real_scanner.from_fragment(*bare(args), **bare_kwargs(kwargs))
            return spy.wrap(out, "Scanner.from_fragment")

        @staticmethod
        def from_batches(*args, **kwargs):
            out = real_scanner.from_batches(*bare(args), **bare_kwargs(kwargs))
            return spy.wrap(out, "Scanner.from_batches")

    class ParquetFileFormat(real_format):
        def make_fragment(self, *args, **kwargs):
            out = super().make_fragment(*bare(args), **bare_kwargs(kwargs))
            return spy.wrap(out, "make_fragment")

    class FileSystemDataset(real_fsd):
        def __init__(self, fragments, *args, **kwargs):
            super().__init__(bare(list(fragments)), *bare(args), **bare_kwargs(kwargs))

        def to_table(self, *args, **kwargs):
            return spy.wrap(super().to_table(*args, **kwargs), "FileSystemDataset.to_table")

        def to_batches(self, *args, **kwargs):
            return spy.wrap(super().to_batches(*args, **kwargs), "FileSystemDataset.to_batches")

        def scanner(self, *args, **kwargs):
            return spy.wrap(super().scanner(*args, **kwargs), "FileSystemDataset.scanner")

        def head(self, *args, **kwargs):
            return spy.wrap(super().head(*args, **kwargs), "FileSystemDataset.head")

        def take(self, *args, **kwargs):
            return spy.wrap(super().take(*args, **kwargs), "FileSystemDataset.take")

    monkeypatch.setattr(ds, "Scanner", Scanner)
    monkeypatch.setattr(ds, "ParquetFileFormat", ParquetFileFormat)
    monkeypatch.setattr(ds, "FileSystemDataset", FileSystemDataset)
    return spy


def test_no_holdout_or_pre_era_row_reaches_python(
    tmp_path, data_dir, stand_in_holdout, monkeypatch
):
    """The store's row groups hold holdout days and a day before 2014-06-07 beside the
    development days. The development predicate runs inside the scan: no table, batch or frame
    any reader hands to Python, in the three runs or pool_totals, holds a holdout day or a day
    before 2014-06-07; and the depth table is the development races' own."""
    registry, spec = frozen_registry(tmp_path)
    development = mixed_store(data_dir)
    out_dir = data_dir / "reports"
    spy = install_row_spy(monkeypatch)

    result = depth.run_depth(**ELIGIBLE_SPAN, out_dir=out_dir, registry=registry)
    depth.run_coplacer(**ELIGIBLE_SPAN, out_dir=out_dir, registry=registry)
    depth.run_exacta_impact(**ELIGIBLE_SPAN, out_dir=out_dir, registry=registry)
    depth.pool_totals(**ELIGIBLE_SPAN)

    assert spy.rows > 0, "no read went through the watched readers"
    assert not spy.bad, f"rows of ineligible days reached Python: {spy.bad[:5]}"
    assert_depth_table(result.table, run_expected(development, spec), spec["edges"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_rerun_on_refreshed_data_keeps_the_first_report(tmp_path, data_dir, variant):
    """The private report is the run's own: a run after the store changed writes another file
    and leaves the first run's report as it was."""
    registry, _ = frozen_registry(tmp_path)
    span = dict(**JUNE_SPAN, out_dir=data_dir / "reports", registry=registry)
    write_store(data_dir, [finished(r, i) for i, r in enumerate(run_races(JUNE, range(9)))])
    first = runner(variant)(**span)
    first_path = Path(first.report_path)
    first_bytes = first_path.read_bytes()
    refreshed = run_races(JUNE, range(9), scale=3)
    write_store(data_dir, [finished(r, 20 + i) for i, r in enumerate(refreshed)])

    second = runner(variant)(**span)

    assert second.run_id != first.run_id
    assert Path(second.report_path) != first_path
    assert Path(second.report_path).is_file()
    assert first_path.read_bytes() == first_bytes, "the first run's report was overwritten"


def write_plain_day(data: Path, day: date, rows: list[dict]) -> Path:
    """A day file without compression or dictionaries: a new value of the same type changes its
    bytes, not its size."""
    path = day_file(data, day)
    table = pa.Table.from_pylist(rows, schema=compact.SCHEMA)
    pq.write_table(table, path, compression="NONE", use_dictionary=False)
    return path


def test_a_same_size_correction_of_a_day_file_changes_the_run_identity(tmp_path, data_dir):
    """A corrected development-day time-series file of the same size (another total at t) is
    another input: the run's identity changes."""
    registry, _ = frozen_registry(tmp_path)
    races = run_races(JUNE, range(9))
    write_store(data_dir, races)
    day = JUNE[0]
    on_day = [r for r in races if r.day == day]

    def rows(correction: int) -> list[dict]:
        return [
            capture(r, "WIN", 3, r.pool_yen("WIN") - 50_000 + (correction if i == 0 else 0))
            for i, r in enumerate(on_day)
        ]

    path = write_plain_day(data_dir, day, rows(0))
    size, before = path.stat().st_size, path.read_bytes()
    span = dict(**JUNE_SPAN, out_dir=data_dir / "reports", registry=registry)
    first = depth.run_depth(**span)
    write_plain_day(data_dir, day, rows(1_000))
    assert path.stat().st_size == size and path.read_bytes() != before

    second = depth.run_depth(**span)

    assert second.run_id != first.run_id


# --- the command ---------------------------------------------------------------------------------


def invoke_run(*args: str):
    from horsetrack.cli import app

    return CliRunner().invoke(app, ["japan", "pool-depth", "run", "--no-public", *args])


def output_of(result) -> str:
    text = result.output or ""
    try:
        text += result.stderr or ""
    except ValueError:  # stderr mixed into output
        pass
    return text


@pytest.fixture
def summary_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> abc.Iterator[Path]:
    """The committed registry summary, at tmp_path, however the command resolves it (through
    `summary_dir`); the repository's own summary must come out unchanged."""
    real = REPO / "docs" / "experiments" / "registry" / "japan"
    committed = {p.name: p.read_bytes() for p in sorted(real.iterdir()) if p.is_file()}
    folder = tmp_path / "summary"
    folder.mkdir()
    monkeypatch.setattr(depth, "summary_dir", lambda country: folder, raising=False)
    monkeypatch.setattr(registry_module, "summary_dir", lambda country: folder)
    yield folder
    after = {p.name: p.read_bytes() for p in sorted(real.iterdir()) if p.is_file()}
    assert after == committed, "the command wrote the repository's committed summary"


def test_the_command_refuses_an_unfrozen_spec_before_touching_the_store(data_dir, summary_folder):
    """Unfrozen spec and no races table: the command refuses as not pre-registered (an exit, not
    a missing-file crash) and writes no record, report or summary."""
    write_store(data_dir, run_races(JUNE, range(9)), skip=("races",))

    result = invoke_run()

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert re.search(r"freez|frozen|pre-?regist", output_of(result), re.I), output_of(result)
    assert default_registry("japan").records() == []
    reports = paths.reports_dir()
    assert not reports.exists() or not any(p.is_file() for p in reports.rglob("*"))
    assert not any(summary_folder.iterdir())


def test_the_command_records_a_failed_run_when_the_races_table_is_missing(data_dir, summary_folder):
    """Frozen spec, no races table: the failure is a registry record (status failed, nothing
    scored) and the command exits non-zero without a crash."""
    spec = shipped_spec()
    default_registry("japan").freeze_family(spec["family_id"], spec)
    write_store(data_dir, run_races(JUNE, range(9)), skip=("races",))

    result = invoke_run()

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    found = [r.status for r in default_registry("japan").records(spec["family_id"])]
    assert found and set(found) == {"failed"}, found


def test_the_command_updates_only_its_family_in_the_committed_summary(data_dir, summary_folder):
    """The local registry holds the pool-depth runs and one run of another family; the committed
    summary indexes families this registry does not hold. The command adds its own family's file
    and index entry and keeps every other family's entry and file as they were."""
    spec = shipped_spec()
    family = spec["family_id"]
    registry = default_registry("japan")
    registry.freeze_family(family, spec)
    registry.append(
        RunRecord(
            hypothesis_id="P1", family_id="japan-other-a", variant="v", ticket=1,
            git_sha="0" * 40, data_builds={}, sources=("jra",), fold_scheme="development-days",
            components={}, params={}, metrics=(FoldMetric("all", "m", 0.5, None, None, 3),),
            status="scored",
        )
    )  # fmt: skip
    others = {
        name: f"# {name}: the committed summary\n" for name in ("japan-other-a", "japan-other-b")
    }
    for name, text in others.items():
        (summary_folder / f"{name}.md").write_text(text, encoding="utf-8")
    index = "# Experiment registry index\n\n" + "".join(f"- [{n}]({n}.md)\n" for n in others)
    (summary_folder / "README.md").write_text(index, encoding="utf-8")
    write_store(data_dir, [finished(r, i) for i, r in enumerate(run_races(JUNE, range(9)))])

    result = invoke_run()

    assert result.exit_code == 0, output_of(result)
    for name, text in others.items():
        assert (summary_folder / f"{name}.md").read_text(encoding="utf-8") == text, name
    linked = set(
        re.findall(r"\[([^\]]+)\]\(\1\.md\)", (summary_folder / "README.md").read_text("utf-8"))
    )
    assert {*others, family} <= linked, linked
    own = summary_folder / f"{family}.md"
    assert own.is_file() and family in own.read_text(encoding="utf-8")


# --- review round 2: the co-placer variant re-registered on the core rule (family v2) -----------
#
# Contract added here (interfaces the tests call, beyond the ones above):
# - `core_coplacer_capacities(pool, combo, paid_money, places, edge, share)` (numpy arrays): the
#   capacity of paying combinations under the rule of `core.pricing.stake_back_share_dividend`,
#   per yen d = r(1 + B / (P W_i)), B = V - sum W the losers' money, our stake s going into V, into
#   sum W and into W_i. NaN where W_i <= 0, B < 0 or a value is missing; inf where no finite
#   stake eats that much of the edge.
# - `run_coplacer(..., spec_name="pool_depth_coplacer_v2")`: the co-placer variant under
#   config/pool_depth_coplacer_v2.yaml (family japan-pool-depth-coplacer-v2), refused unless that
#   spec is frozen under that family; `coplacer_rows(..., rule="core")` takes that rule.
# - `horsetrack japan pool-depth freeze|run --spec pool_depth_coplacer_v2`.

V2 = "pool_depth_coplacer_v2"
V2_FAMILY = "japan-pool-depth-coplacer-v2"
# the frozen spec's sha256, as docs/specs/pool-depth-prereg.md ("Co-placer v2") registers it
V2_HASH = "82a471b219bfee2e65f53c5c12e2f66514757169d82184b6be45b35b2454e89a"
# the only keys v2 changes from v1, and their v2 values
V2_CHANGES = {
    ("family_id",): V2_FAMILY,
    ("place_wide", "variants"): ["co-placer"],
    ("place_wide", "co_placer", "rule"): "core-stake-back-share",
}


def v2_spec() -> dict:
    return copy.deepcopy(load_country_yaml("japan", V2))


def both_frozen(tmp_path: Path, name: str = "registry") -> Registry:
    """A registry under tmp_path with v1's and v2's shipped specs frozen under their families."""
    registry = Registry(tmp_path / f"{name}.jsonl", holdout_log=tmp_path / f"{name}-holdout.jsonl")
    for spec in (copy.deepcopy(shipped_spec()), v2_spec()):
        registry.freeze_family(spec["family_id"], spec)
    return registry


# The core rule for Place and Wide, derived here: the payout rate is taken on the whole net pool,
# the paying combinations' own money included; per 100 yen on paying combination i
# 100 r (1 + B / (P W_i)), B = V - sum W, broken down to 10 yen and at least 110.


def core_per_100(race: Race, pool: str, combination: str) -> Fraction:
    """The core rule before breakage and the floor, exactly."""
    pays = paying(race, pool)
    net = 100 * (race.pool_votes[pool] - race.refunds.get(pool, 0))
    paid = 100 * sum(race.votes[pool][c] for c in pays)
    stake = 100 * race.votes[pool][combination]
    return 100 * PAYOUT_RATE[pool] * (1 + Fraction(net - paid, len(pays) * stake))


def core_dividend(race: Race, pool: str, combination: str) -> float:
    """The dividend `core.pricing.stake_back_share_dividend` pays: shares of the net pool."""
    pays = paying(race, pool)
    net = race.pool_votes[pool] - race.refunds.get(pool, 0)
    votes = race.votes[pool]
    return pricing.stake_back_share_dividend(
        votes[combination] / net,
        [votes[c] / net for c in pays if c != combination],
        rate=float(PAYOUT_RATE[pool]),
        unit=100,
        dividend_step=10,
        min_dividend=110,
    )


def pay_by_core_rule(data: Path, races: list[Race]) -> None:
    """The store's Place and Wide `dividends` rows paid by the core rule instead (Win as it was)."""
    path = data / "processed" / "jravan" / "dividends.parquet"
    paid = pq.read_table(path).to_pandas()
    by_id = {r.race_id: r for r in races}
    co = paid["pool"].isin(CO_PLACER).to_numpy()
    keys = paid.loc[co, ["race_id", "pool", "combination"]].itertuples(index=False)
    paid.loc[co, "dividend"] = [core_dividend(by_id[r], p, c) for r, p, c in keys]
    pq.write_table(
        pa.Table.from_pandas(paid, schema=dividends_schema(), preserve_index=False), path
    )


def payouts(races: list[Race], pool: str) -> list[tuple[Race, str]]:
    return [(r, c) for r in races for c in paying(r, pool)]


def assert_rule_races(races: list[Race]) -> None:
    """The races exercise every part of both rules, and the two rules pay differently: in each
    pool, more than one payout in twenty differs between them."""
    place_p = {len(paying(r, "PLACE")) for r in races}
    wide_p = {len(paying(r, "QUINELLA_PLACE")) for r in races}
    assert {2, 3, 4} <= place_p and {3, 5} <= wide_p, (place_p, wide_p)
    assert any(r.refunds for r in races)
    raw = [core_per_100(r, p, c) for p in CO_PLACER for r, c in payouts(races, p)]
    assert any(x < 110 for x in raw), "no core payout the floor lifts"
    assert any(x >= 110 and x % 10 for x in raw), "no core payout breakage cuts"
    for pool in CO_PLACER:
        for race, c in payouts(races, pool):
            want = max(110, math.floor(core_per_100(race, pool, c) / 10) * 10)
            assert core_dividend(race, pool, c) == want, (race.race_id, pool, c)
        same = [
            jra_dividend(r, pool, c) == core_dividend(r, pool, c) for r, c in payouts(races, pool)
        ]
        assert np.mean(same) < 0.95, (pool, np.mean(same))


def core_value(pool, combo, paid, places, rate, stake):
    """d(s), the core rule per yen once our stake s is on paying combination i: s goes into the
    pool V, into the paying money sum W and into W_i."""
    return rate * (1 + ((pool + stake) - (paid + stake)) / (places * (combo + stake)))


def core_capacity(pool, combo, paid, places, edge, share, *, rate=0.8) -> float:
    """The stake solving (1 + e) d(s) / d(0) = 1 + e(1 - share), by bisection on core_value; inf
    when d(s), falling to r as s grows, never gets there."""
    k = edge * (1 - share)
    d0 = core_value(pool, combo, paid, places, rate, 0.0)

    def gap(s: float) -> float:
        return (1 + edge) * core_value(pool, combo, paid, places, rate, s) / d0 - (1 + k)

    if (1 + edge) * rate / d0 >= 1 + k:
        return math.inf
    lo, hi = 0.0, float(combo)
    while gap(hi) > 0:
        hi *= 2
    for _ in range(400):
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            break
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def linear_quantile(values, q: float) -> float:
    """numpy's linear interpolation, with +inf values sorting last: inf wherever the interpolation
    reaches one."""
    x = sorted(float(v) for v in values)
    h = (len(x) - 1) * q
    lo, hi = math.floor(h), math.ceil(h)
    if lo == hi:
        return x[lo]
    if math.isinf(x[hi]):
        return math.inf
    return x[lo] + (x[hi] - x[lo]) * (h - lo)


def expected_core_cells(races: list[Race], spec: dict) -> dict[tuple, dict[str, float]]:
    """(pool, class_group, band, quantile) -> what the v2 co-placer run's table must give: over
    the paying combinations, the money on them, the pool (each race once), the core-rule capacity
    at each edge and the share of capacities reaching each flat stake."""
    edges, stakes, share = spec["edges"], spec["flat_stakes_yen"], float(spec["share_lost"])
    rows = []
    for pool in CO_PLACER:
        bands = [float(e) for e in spec["odds_bands"][pool]]
        for race in races:
            pays = paying(race, pool)
            paid = 100.0 * sum(race.votes[pool][c] for c in pays)
            for c in pays:
                w = 100.0 * race.votes[pool][c]
                caps = {e: core_capacity(race.net_yen(pool), w, paid, len(pays), e, share)
                        for e in edges}  # fmt: skip
                rows.append(
                    {"key": (pool, CLASS_GROUPS[race.race_class],
                             pool_scan.odds_band(race.odds[pool][c], bands)),
                     "race": race.race_id, "combo": w, "pool": race.net_yen(pool), "caps": caps}
                )  # fmt: skip
    cells: dict[tuple, list[dict]] = {}
    for row in rows:
        cells.setdefault(row["key"], []).append(row)
    out = {}
    for key, cell in cells.items():
        once = {r["race"]: r["pool"] for r in cell}
        for q in spec["quantiles"]:
            q = float(q)
            values = {
                "n_combinations": len(cell),
                "n_races": len(once),
                "combo_yen": float(np.quantile([r["combo"] for r in cell], q)),
                "pool_yen": float(np.quantile(list(once.values()), q)),
            }
            for e in edges:
                caps = [r["caps"][e] for r in cell]
                values[f"capacity_{float(e):g}"] = linear_quantile(caps, q)
                for s in stakes:
                    values[f"fits_{float(e):g}_{float(s):g}"] = float(np.mean(np.array(caps) >= s))
            out[(*key, round(q, 9))] = values
    return out


def test_core_coplacer_capacity_solves_the_core_rule_identity():
    """Under the core rule d(s) = r(1 + B / (P(W_i + s))) (B = V - sum W stays as our stake goes
    into V and sum W alike), the capacity s solves (1 + e) d(s) / d(0) = 1 + e(1 - share_lost)
    whatever the payout rate r; it is inf exactly where d(s), falling to r, never gets there; at
    P = 1 (sum W = W_i) it is the single-winner capacity_stake."""
    rng = np.random.default_rng(43202)
    m = 400
    pool = np.exp(rng.uniform(math.log(1e5), math.log(1e9), m))
    paid = pool * rng.uniform(0.02, 0.98, m)
    places = rng.integers(2, 6, m).astype(float)
    combo = paid * rng.uniform(0.03, 0.97, m)
    unbounded = 0
    for edge in (0.05, 0.1, 0.2):
        for share in (0.25, 0.5, 1.0):
            k = edge * (1 - share)
            s = np.asarray(depth.core_coplacer_capacities(pool, combo, paid, places, edge, share),
                           float)  # fmt: skip
            assert s.shape == (m,)
            assert not np.isnan(s).any(), (edge, share)
            finite = np.isfinite(s)
            assert finite.mean() > 0.5, (edge, share)
            assert np.all(s[finite] > 0)
            unbounded += int((~finite).sum())
            at = np.where(finite, s, 0.0)
            for rate in (0.8, 0.775):
                d0 = core_value(pool, combo, paid, places, rate, 0.0)
                left = (1 + edge) * core_value(pool, combo, paid, places, rate, at) / d0
                np.testing.assert_allclose(left[finite], 1 + k, rtol=1e-9)
                limit = (1 + edge) * rate / d0  # (1 + e) d(s) / d(0) as s grows without bound
                clear = np.abs(limit - (1 + k)) > 1e-9
                np.testing.assert_array_equal(finite[clear], (limit < 1 + k)[clear])
            want = [core_capacity(*row, edge, share) for row in
                    zip(pool[:20], combo[:20], paid[:20], places[:20], strict=True)]  # fmt: skip
            np.testing.assert_allclose(s[:20], want, rtol=1e-9)
    assert unbounded > 0, "no unbounded capacity in the sample"

    def single_winner(v: float, w: float, edge: float, share: float) -> float:
        try:
            return depth.capacity_stake(v, w, edge, share_lost=share)
        except ValueError:  # no stake eats that much of the edge
            return math.inf

    one = np.ones(m)
    single = pool * rng.uniform(0.01, 0.99, m)
    for edge in (0.05, 0.1, 0.2):
        for share in (0.25, 0.5, 1.0):
            s = depth.core_coplacer_capacities(pool, single, single, one, edge, share)
            want = [single_winner(v, w, edge, share) for v, w in zip(pool, single, strict=True)]
            assert np.isfinite(want).any() and np.isinf(want).any(), (edge, share)
            np.testing.assert_allclose(np.asarray(s, float), want, rtol=1e-9)


def test_core_coplacer_capacity_without_money_or_losers():
    """NaN where the combination holds no money, the paying money exceeds the pool (B < 0) or a
    value is missing; inf where every yen of the pool is on the paying combinations (B = 0: the
    rule pays r whatever the stake). The rows are independent of one another."""
    nan = math.nan
    cases = [  # pool, combo, paid, places, what
        (1e6, 2e4, 6e4, 3, "finite"),
        (1e6, 0.0, 6e4, 3, "nan"),
        (1e6, -1e3, 6e4, 3, "nan"),
        (1e6, 2e4, 1.2e6, 3, "nan"),
        (nan, 2e4, 6e4, 3, "nan"),
        (1e6, nan, 6e4, 3, "nan"),
        (1e6, 2e4, nan, 3, "nan"),
        (1e6, 2e4, 6e4, nan, "nan"),
        (1e6, 4e5, 1e6, 3, "inf"),
        (3e5, 1e5, 2e5, 2, "finite"),
    ]
    columns = [np.array(c, dtype=float) for c in list(zip(*cases, strict=True))[:4]]
    for edge, share in ((0.05, 0.5), (0.1, 0.25), (0.2, 1.0)):
        s = np.asarray(depth.core_coplacer_capacities(*columns, edge, share), float)
        for (v, w, b, p, what), got in zip(cases, s, strict=True):
            if what == "nan":
                assert math.isnan(got), (v, w, b, p, got)
            elif what == "inf":
                assert got == math.inf, (v, w, b, p, got)
            else:
                assert got == pytest.approx(core_capacity(v, w, b, p, edge, share), rel=1e-9)


@pytest.mark.parametrize("paid_by", ["core", "jra"])
def test_the_registered_rule_decides_the_coplacer_status(tmp_path, data_dir, paid_by):
    """Place and Wide dividends paid by `core.pricing.stake_back_share_dividend` (the net pool,
    P = 2 places in a field of 5-7 and 3 in one of 8 or more, dead heats, breakage to 10 yen, the
    110-yen floor): the v2 co-placer run reproduces every one, is scored, and its table is the
    core-rule capacity; v1's run on the same store is recorded failed. Paid by v1's rule, the
    converse. One record per run, in each spec's own family."""
    registry = both_frozen(tmp_path)
    v1_family = shipped_spec()["family_id"]
    races = jra_rule_races()
    assert_rule_races(races)
    write_store(data_dir, races)
    if paid_by == "core":
        pay_by_core_rule(data_dir, races)
    out_dir = data_dir / "reports"

    v2 = run_quietly(depth.run_coplacer, **JUNE_SPAN, out_dir=out_dir, registry=registry,
                     spec_name=V2)  # fmt: skip
    v1 = run_quietly(depth.run_coplacer, **JUNE_SPAN, out_dir=out_dir, registry=registry)

    reproduces, rejects = (v2, v1) if paid_by == "core" else (v1, v2)
    assert reproduces is not None and reproduces.status == "scored"
    assert set(reproduces.rule_match) == set(CO_PLACER), reproduces.rule_match
    assert all(float(v) == 1.0 for v in reproduces.rule_match.values()), reproduces.rule_match
    if rejects is not None:
        assert rejects.status == "failed"
        assert min(float(v) for v in rejects.rule_match.values()) < 0.95, rejects.rule_match
    v2_status, v1_status = ("scored", "failed") if paid_by == "core" else ("failed", "scored")
    for family, status, spec in (
        (V2_FAMILY, v2_status, v2_spec()),
        (v1_family, v1_status, shipped_spec()),
    ):
        records = registry.records(family)
        assert [(r.variant, r.status) for r in records] == [("co-placer", status)], family
        record = records[0]
        assert (record.hypothesis_id, record.ticket, tuple(record.sources)) == (
            HYPOTHESIS_ID,
            TICKET,
            SOURCES,
        )
        assert record.params["spec_hash"] == _spec_hash(spec)
    assert len(registry.records()) == 2
    registry.verify()
    if paid_by == "core":
        assert_depth_table(v2.table, expected_core_cells(races, v2_spec()), v2_spec()["edges"])


def test_coplacer_v2_refuses_until_its_own_spec_is_frozen(tmp_path, data_dir):
    """No v2 run before v2's freeze: nothing frozen, only v1 frozen, v2's family frozen with
    another spec, or v2's spec frozen under another family all refuse with NotPreRegistered and
    write nothing. Freezing v2 itself lets the same call run. An unknown spec is refused."""
    races = jra_rule_races()
    write_store(data_dir, races)
    pay_by_core_rule(data_dir, races)
    out_dir = data_dir / "reports"
    span = dict(**JUNE_SPAN, out_dir=out_dir)

    def registry(name: str, *freezes: tuple[str, dict]) -> Registry:
        r = Registry(tmp_path / f"{name}.jsonl", holdout_log=tmp_path / "h.jsonl")
        for family, spec in freezes:
            r.freeze_family(family, spec)
        return r

    v1 = copy.deepcopy(shipped_spec())
    changed = v2_spec()
    changed["place_wide"]["co_placer"]["min_rule_match"] = 0.5
    refused = [
        registry("empty"),
        registry("v1-only", (v1["family_id"], v1)),
        registry("v2-changed", (V2_FAMILY, changed)),
        registry("v2-elsewhere", (f"{V2_FAMILY}-not-this-one", v2_spec()), (v1["family_id"], v1)),
    ]
    for r in refused:
        with pytest.raises(depth.NotPreRegistered):
            depth.run_coplacer(**span, registry=r, spec_name=V2)
    with pytest.raises(depth.NotPreRegistered):
        depth.run_coplacer(**span, spec_name=V2)  # the default registry, nothing frozen in it
    with pytest.raises(ValueError):
        depth.run_coplacer(**span, registry=refused[1], spec_name="pool_depth_coplacer_v9")

    assert not out_dir.exists() or not any(p.is_file() for p in out_dir.rglob("*"))
    for r in [*refused, default_registry("japan")]:
        assert r.records() == []

    unlocked = refused[1]
    unlocked.freeze_family(V2_FAMILY, v2_spec())
    result = depth.run_coplacer(**span, registry=unlocked, spec_name=V2)
    assert result.status == "scored"
    assert [r.family_id for r in unlocked.records()] == [V2_FAMILY]


def test_coplacer_rows_refuse_bad_inputs_under_the_core_rule():
    """`coplacer_rows(..., rule="core")` runs; an unknown rule, a share lost outside (0, 1] or a
    missing or non-positive edge is refused."""
    frames = store_tables(jra_rule_races()[:6])
    votes, board, meta = (frames[n].to_pandas() for n in ("pool_votes", "final_odds", "races"))
    bands = {p: v2_spec()["odds_bands"][p] for p in CO_PLACER}
    rows = depth.depth_rows(votes, board, meta, band_edges=bands)
    paid = frames["dividends"].to_pandas()
    kwargs = dict(rates={p: float(PAYOUT_RATE[p]) for p in CO_PLACER}, edges=[0.1],
                  share_lost=0.5, rule="core")  # fmt: skip
    assert len(depth.coplacer_rows(rows, paid, **kwargs)) == int(paid["pool"].isin(CO_PLACER).sum())
    for bad in (dict(rule="nope"), dict(share_lost=0.0), dict(share_lost=1.5),
                dict(edges=[0.0]), dict(edges=[-0.1]), dict(edges=[])):  # fmt: skip
        with pytest.raises(ValueError):
            depth.coplacer_rows(rows, paid, **{**kwargs, **bad})


def test_the_command_freezes_and_runs_coplacer_v2_alone(data_dir, summary_folder):
    """`pool-depth run --spec pool_depth_coplacer_v2` is refused before `pool-depth freeze --spec
    pool_depth_coplacer_v2`, which freezes v2 alone. Then it runs the co-placer variant alone
    under v2 (one scored record in v2's family, one private report) and of the committed summary
    adds v2's file and index entry, every other family's file and entry, v1's included, kept."""
    v1, v2 = shipped_spec(), v2_spec()
    races = jra_rule_races()
    write_store(data_dir, races)
    pay_by_core_rule(data_dir, races)
    others = {name: f"# {name}: the committed summary\n"
              for name in ("japan-other-a", v1["family_id"])}  # fmt: skip
    for name, text in others.items():
        (summary_folder / f"{name}.md").write_text(text, encoding="utf-8")
    index = "# Experiment registry index\n\n" + "".join(f"- [{n}]({n}.md)\n" for n in others)
    (summary_folder / "README.md").write_text(index, encoding="utf-8")
    registry = default_registry("japan")

    early = invoke_run("--spec", V2)
    assert early.exit_code != 0
    assert isinstance(early.exception, SystemExit), repr(early.exception)
    assert registry.records() == []

    from horsetrack.cli import app

    frozen = CliRunner().invoke(app, ["japan", "pool-depth", "freeze", "--spec", V2])
    assert frozen.exit_code == 0, output_of(frozen)
    assert registry.frozen_spec_hash(V2_FAMILY) == _spec_hash(v2)
    assert registry.frozen_spec_hash(v1["family_id"]) is None

    result = invoke_run("--spec", V2)

    assert result.exit_code == 0, output_of(result)
    records = registry.records()
    assert [(r.family_id, r.variant, r.status) for r in records] == [
        (V2_FAMILY, "co-placer", "scored")
    ]
    reports = [p for p in paths.reports_dir().rglob("*") if p.is_file()]
    assert len(reports) == 1, reports
    for name, text in others.items():
        assert (summary_folder / f"{name}.md").read_text(encoding="utf-8") == text, name
    linked = set(
        re.findall(r"\[([^\]]+)\]\(\1\.md\)", (summary_folder / "README.md").read_text("utf-8"))
    )
    assert linked == {*others, V2_FAMILY}, linked
    own = summary_folder / f"{V2_FAMILY}.md"
    assert own.is_file() and V2_FAMILY in own.read_text(encoding="utf-8")


def leaves(value, path: tuple = ()) -> dict[tuple, object]:
    """Every leaf of a nested mapping by its key path (lists are leaves)."""
    if isinstance(value, abc.Mapping):
        out = {}
        for key, inner in value.items():
            out |= leaves(inner, (*path, key))
        return out
    return {path: value}


def test_the_shipped_v2_spec_is_v1_with_the_core_rule_alone(tmp_path):
    """config/pool_depth_coplacer_v2.yaml is config/pool_depth.yaml with three keys changed: its
    family, the co-placer variant alone, and the core rule. It loads, is plain JSON and freezes
    to the hash the pre-registration records."""
    v1, v2 = leaves(shipped_spec()), leaves(v2_spec())
    differ = {p for p in v1.keys() | v2.keys() if v1.get(p, KeyError) != v2.get(p, KeyError)}
    assert differ == set(V2_CHANGES), sorted(differ)
    assert {p: v2[p] for p in V2_CHANGES} == V2_CHANGES
    assert v1[("place_wide", "co_placer", "rule")] == "jra-stake-back-share"

    family, spec = depth.load_spec(V2)
    assert family == V2_FAMILY and spec == v2_spec()
    json.dumps(spec, allow_nan=False)
    registry = Registry(tmp_path / "r.jsonl", holdout_log=tmp_path / "h.jsonl")
    assert registry.freeze_family(V2_FAMILY, spec) == _spec_hash(spec) == V2_HASH


# --- review fix round 2: the public run history, the gates and the private report names ---------
#
# Contract added here (interfaces the tests call, beyond the ones above):
# - `horsetrack japan pool-depth run` (public) writes `<env.repo_root()>/docs/experiments/
#   pool-depth-japan.md` (the tests point `env.repo_root` at a stand-in root). The report already
#   there is merged by run id before rendering: a run the registry holds is listed from the
#   registry, a run only the existing report lists keeps its line (variant, status, family), and
#   each run id is listed once.
# - The #489 follow-up says the ticket "can close" only when a scored co-placer run of family
#   japan-pool-depth-coplacer-v2 is listed; otherwise the report says the Place and Wide capacity
#   is "unvalidated".
# - A depth run with no eligible cell (no priced combination in any pool) is recorded failed, so it
#   never becomes the public report's latest scored depth run.
# - The co-placer rule check takes sum W from the confirmed votes of the race's paying
#   combinations: a paid combination without its final price is one miss, and only one.
# - `pool_totals`: a race and pool with a capture at t but no confirmed final record keeps its row,
#   final_yen NaN and at_t_yen the capture's.
# - The public report names a private report only when the file is in the report folder:
#   `<stem>-<first 16 hex of the run id>.md`, or a first-invocation run's
#   `<stem>-<first 12 hex of the spec hash>.md`; otherwise it names none.
# - `date_to=None` (the command's span) resolves to the store's last development day, the last
#   race day `is_holdout_day("japan", d)` does not lock, and params.date_to records that day.

V1_FAMILY = "japan-pool-depth-v1"
PUBLIC = Path("docs") / "experiments" / "pool-depth-japan.md"
REPORT_NAME = re.compile(
    r"(?:pool-depth-japan|pool-depth-coplacer-japan|exacta-impact-history-japan)-[0-9a-f]{12,16}\.md"
)
RUN_ID_IN_LINE = re.compile(r"registry run `([0-9a-f]{64})`")


@pytest.fixture
def public_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> abc.Iterator[Path]:
    """A stand-in repository root, however the command resolves it (`env.repo_root`): the public
    report goes to its docs/experiments/pool-depth-japan.md; the repository's own report must come
    out unchanged."""
    import horsetrack.cli  # noqa: F401 - before the patch: some modules read the root on import

    real = REPO / PUBLIC
    before = real.read_bytes() if real.is_file() else None
    root = (tmp_path / "repo").resolve()
    (root / PUBLIC).parent.mkdir(parents=True)
    monkeypatch.setattr(env, "repo_root", lambda: root)
    yield root
    after = real.read_bytes() if real.is_file() else None
    assert after == before, "the command wrote the repository's public report"


def invoke_public(*args: str):
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "pool-depth", "run", *args])
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        repr(result.exception),
        output_of(result),
    )
    return result


def run_lines(text: str) -> dict[str, tuple[str, str, bool]]:
    """The public report's run lines: run id -> (variant, status, whether the line names the v2
    family); a run id listed twice fails."""
    out: dict[str, tuple[str, str, bool]] = {}
    for line in text.splitlines():
        ids = RUN_ID_IN_LINE.findall(line)
        if not ids:
            continue
        assert len(ids) == 1, line
        variant = re.match(r"- `([^`]+)`", line)
        status = re.search(r": (scored|failed|rejected|adopted)\b", line)
        assert variant and status, line
        assert ids[0] not in out, f"run {ids[0]} listed twice"
        out[ids[0]] = (variant.group(1), status.group(1), f"`{V2_FAMILY}`" in line)
    return out


def marks_489_done(text: str) -> bool:
    """Whether the #489 follow-up says the co-placer re-registration is done (the ticket can
    close) rather than the Place and Wide capacity unvalidated; exactly one of them holds."""
    lines = [line for line in text.splitlines() if "#489" in line]
    assert lines, "the public report has no #489 follow-up"
    done = any("can close" in line for line in lines)
    assert done != ("unvalidated" in text), lines
    return done


def header(text: str) -> str:
    """The public report before its first section: where it names the latest depth run's
    private report."""
    return text.split("\n## ", 1)[0]


def registry_runs(registry: Registry) -> list[tuple[str, str, str, str]]:
    """(run id, family, variant, status) of every run in append order, from the registry file."""
    out = []
    for line in registry.path.read_bytes().splitlines():
        entry = json.loads(line)
        if entry["kind"] == "run":
            record = entry["record"]
            out.append((entry["run_id"], record["family_id"], record["variant"], record["status"]))
    return out


def finished_races() -> list[Race]:
    return [finished(r, i) for i, r in enumerate(run_races(JUNE, range(9)))]


def invented_history() -> str:
    """A public report of invented runs, as an earlier invocation wrote it: v1's three variants
    (one failed depth run), a private report named on some lines, and a scored v2 co-placer run."""
    v1, a1, b2, c3 = _spec_hash(shipped_spec()), "a1" * 20, "b2" * 20, "c3" * 20
    ids = [hashlib.sha256(f"t432-invented-run-{i}".encode()).hexdigest() for i in range(5)]
    named = {i: f"pool-depth-coplacer-japan-{ids[i][:16]}.md" for i in (1, 3)}
    runs = [
        depth.VariantRun("depth", "scored", ids[0], None, V1_FAMILY, v1, a1),
        depth.VariantRun("co-placer", "failed", ids[1], named[1], V1_FAMILY, v1, a1),
        depth.VariantRun("exacta-own-impact", "scored", ids[2], None, V1_FAMILY, v1, a1),
        depth.VariantRun("co-placer", "scored", ids[3], named[3], V2_FAMILY, V2_HASH, b2),
        depth.VariantRun("depth", "failed", ids[4], None, V1_FAMILY, v1, c3),
    ]
    result = depth.DepthResult(pd.DataFrame(), Path(f"pool-depth-japan-{ids[0][:16]}.md"),
                               V1_FAMILY, v1, ids[0], PRICE_ERA, date(2019, 6, 8))  # fmt: skip
    return depth.render_public_report(result, variants=runs)


@pytest.mark.parametrize("history", ["committed", "invented"])
def test_a_public_rerun_from_a_fresh_registry_keeps_the_published_runs(
    data_dir, summary_folder, public_repo, history
):
    """The public report already lists runs (the repository's committed report, or invented
    ones) this machine's registry does not hold, a scored v2 co-placer run among them. The
    default command, its registry holding new v1 runs only (v1's co-placer failed), keeps every
    published run with its variant and status, keeps #489 done, adds the new runs, and a second
    rerun lists each run once."""
    text = (REPO / PUBLIC).read_text("utf-8") if history == "committed" else invented_history()
    published = run_lines(text)
    assert len(published) >= 5 and any(v2 and s == "scored" for _, s, v2 in published.values())
    assert marks_489_done(text)
    doc = public_repo / PUBLIC
    doc.write_text(text, encoding="utf-8")
    spec = shipped_spec()
    registry = default_registry("japan")
    registry.freeze_family(spec["family_id"], spec)
    races = finished_races()
    write_store(data_dir, races)
    pay_by_core_rule(data_dir, races)  # v1's rule reproduces no payout: its co-placer run fails

    for _ in range(2):
        invoke_public()

        runs = registry_runs(registry)
        assert statuses(registry, "co-placer") == ["failed"]
        assert {family for _, family, _, _ in runs} == {V1_FAMILY} and len(runs) == 3
        listed = run_lines(doc.read_text("utf-8"))
        lost = sorted(set(published) - set(listed))
        assert not lost, f"published runs dropped from the public report: {lost}"
        for run_id, (variant, status, v2) in published.items():
            assert listed[run_id] == (variant, status, v2), run_id
        new = {run_id: (variant, status) for run_id, _, variant, status in runs}
        assert {k: listed.get(k, (None, None, False))[:2] for k in new} == new
        assert set(listed) == set(published) | set(new)
        assert marks_489_done(doc.read_text("utf-8")), "the v2 validation was dropped"


V2_ID = hashlib.sha256(b"t432-v2-coplacer").hexdigest()
V1_ID = hashlib.sha256(b"t432-v1-coplacer").hexdigest()


CLOSES_489 = [  # the runs listed, and whether #489 is done
    ((), False),
    ((("co-placer", "scored", V1_ID, V1_FAMILY),), False),
    ((("co-placer", "scored", V1_ID, None),), False),
    ((("co-placer", "failed", V2_ID, V2_FAMILY),), False),
    ((("co-placer", "scored", V1_ID, V1_FAMILY), ("co-placer", "failed", V2_ID, V2_FAMILY)), False),
    ((("depth", "scored", V1_ID, V2_FAMILY),), False),
    ((("co-placer", "scored", V2_ID, V2_FAMILY),), True),
    ((("co-placer", "failed", V1_ID, V1_FAMILY), ("co-placer", "scored", V2_ID, V2_FAMILY),
      ("depth", "scored", "d" * 64, V1_FAMILY)), True),
]  # fmt: skip


def test_only_a_scored_v2_coplacer_run_closes_489():
    """#489 is the co-placer re-registration: a scored co-placer run of family
    japan-pool-depth-coplacer-v2 marks it done; a scored v1 co-placer run (one that cleared v1's
    gate on a limited export), a failed v2 run or no co-placer run does not."""
    v1_hash = _spec_hash(shipped_spec())
    report = Path("pool-depth-japan-0123456789abcdef.md")
    result = depth.DepthResult(pd.DataFrame(), report, V1_FAMILY, v1_hash, "e" * 64, PRICE_ERA,
                               date(2019, 6, 30))  # fmt: skip
    got = []
    for runs, _ in CLOSES_489:
        variants = [
            depth.VariantRun(variant, status, run_id, None, family,
                             V2_HASH if family == V2_FAMILY else v1_hash, "f" * 40)
            for variant, status, run_id, family in runs
        ]  # fmt: skip
        got.append(marks_489_done(depth.render_public_report(result, variants=variants)))

    wrong = [runs for (runs, done), seen in zip(CLOSES_489, got, strict=True) if seen is not done]
    assert not wrong, wrong


def test_a_depth_run_without_a_priced_combination_is_recorded_failed(tmp_path, data_dir):
    """Races and confirmed votes, but only the day before's odds: no combination has a final
    price, so the depth table has no cell and the run is recorded failed, not scored. With one
    race's final odds the run is scored on that race's cells."""
    registry, spec = frozen_registry(tmp_path)
    races = run_races(JUNE, range(9))
    for race in races:
        race.board_kubun = "2"
    write_store(data_dir, races)
    span = dict(**JUNE_SPAN, out_dir=data_dir / "reports", registry=registry)

    run_quietly(depth.run_depth, **span)

    assert statuses(registry, "depth") == ["failed"]
    races[0].board_kubun = "4"
    write_store(data_dir, races)
    result = depth.run_depth(**span)
    assert statuses(registry, "depth") == ["failed", "scored"]
    assert_depth_table(result.table, run_expected([races[0]], spec), spec["edges"])


def test_a_depth_run_without_cells_is_not_the_public_reports_latest(
    data_dir, summary_folder, public_repo
):
    """A scored depth run, then a run on a store with only the day before's odds: the second is
    recorded failed, and the public report still names the scored run's private report."""
    spec = shipped_spec()
    registry = default_registry("japan")
    registry.freeze_family(spec["family_id"], spec)
    races = finished_races()
    write_store(data_dir, races)
    first = invoke_public()
    [scored] = re.findall(r"depth: scored run_id=([0-9a-f]{64})", output_of(first))
    for race in races:
        race.board_kubun = "2"
    write_store(data_dir, races)

    invoke_public()

    assert statuses(registry, "depth") == ["scored", "failed"]
    top = header((public_repo / PUBLIC).read_text("utf-8"))
    assert f"pool-depth-japan-{scored[:16]}.md" in top, top
    ids = [run_id for run_id, _, variant, _ in registry_runs(registry) if variant == "depth"]
    assert [run_id[:16] in top for run_id in ids] == [True, False], top


@pytest.mark.parametrize("k, status", [(3, "scored"), (8, "failed")])
def test_a_wide_payout_without_its_final_price_is_one_miss(tmp_path, data_dir, k, status):
    """Fifty Place and Wide races, three Wide payouts each, every payout by JRA's rule; in k of
    them one Wide winner has its confirmed votes but no final price. sum W comes from the votes,
    so the race's other payouts are still checked: each priceless payout is exactly one miss, Wide
    reproduces (150 - k) / 150 and Place all. k = 3 (98%) is scored, k = 8 (94.7%) failed."""
    registry, _ = frozen_registry(tmp_path)
    races = coplacer_races(50)
    assert [len(paying(r, "QUINELLA_PLACE")) for r in races] == [3] * 50
    for i, race in enumerate(races[1::6][:k]):
        winner = paying(race, "QUINELLA_PLACE")[i % 3]
        race.odds["QUINELLA_PLACE"][winner] = None  # no final price; its votes stay
    write_store(data_dir, races)

    result = run_quietly(depth.run_coplacer, **JUNE_SPAN, out_dir=data_dir / "reports",
                         registry=registry)  # fmt: skip

    assert statuses(registry, "co-placer") == [status]
    assert result is not None and result.status == status
    assert float(result.rule_match["QUINELLA_PLACE"]) == pytest.approx((150 - k) / 150)
    assert float(result.rule_match["PLACE"]) == 1.0


def test_the_public_report_names_only_private_reports_that_exist(
    data_dir, summary_folder, public_repo
):
    """The v2 command publishes the study's latest depth run, a v1 run. Its private report under
    the run id's name is named; renamed to the first invocation's name (the spec hash's), that
    file is named instead; deleted, no depth report is named. No private report the folder does
    not hold is ever named."""
    v1 = shipped_spec()
    registry = default_registry("japan")
    for spec in (v1, v2_spec()):
        registry.freeze_family(spec["family_id"], spec)
    races = finished_races()
    write_store(data_dir, races)
    pay_by_core_rule(data_dir, races)
    folder = paths.reports_dir()
    run = depth.run_depth(date_from=JUNE[0], date_to=JUNE[-1], out_dir=folder, registry=registry)
    by_id = folder / f"pool-depth-japan-{run.run_id[:16]}.md"
    legacy = folder / f"pool-depth-japan-{_spec_hash(v1)[:12]}.md"
    assert Path(run.report_path) == by_id and not legacy.exists()

    def publish() -> str:
        invoke_public("--spec", V2)
        text = (public_repo / PUBLIC).read_text("utf-8")
        missing = sorted(n for n in REPORT_NAME.findall(text) if not (folder / n).is_file())
        assert not missing, f"the public report names private reports never written: {missing}"
        return text

    assert by_id.name in header(publish())

    by_id.rename(legacy)
    text = publish()
    assert legacy.name in header(text), header(text)
    assert by_id.name not in text

    legacy.unlink()
    text = publish()
    named = [n for n in REPORT_NAME.findall(text) if n.startswith("pool-depth-japan-")]
    assert not named, named


def trailing_store(
    data: Path, trailing: tuple[date, ...], *, held: int = 1, after: bool = False
) -> None:
    """Development races on DEV_DAYS (and on DEV_AFTER with `after`), holdout races on
    HOLDOUT_DAYS and on the `trailing` holdout days after the last development day, with their
    time-series day files; `held` changes the holdout races' money and finishing order."""
    development = [
        finished(make_race(160 + i, DEV_DAYS[i % len(DEV_DAYS)], "KOK", 1 + i // len(DEV_DAYS),
                           RUN_CLASSES[i % len(RUN_CLASSES)], 6 + i % 3), i)
        for i in range(8)
    ]  # fmt: skip
    if after:
        development.append(finished(make_race(169, DEV_AFTER, "KOK", 1, "C2", 7), 9))
    holdout = [
        finished(make_race(170 + i, d, "KOK", 1, ("G2", "L", "MAIDEN", "OP")[i % 4], 7,
                           scale=1000 * held), 190 + i + 10 * held)
        for i, d in enumerate((*HOLDOUT_DAYS, *trailing))
    ]  # fmt: skip
    races = [*development, *holdout]
    write_store(data, races)
    bump = {id(r): held for r in holdout}
    for day in sorted({r.day for r in races}):
        write_day(data, day, [capture(r, "WIN", 3, r.pool_yen("WIN") - 50_000 * bump.get(id(r), 1))
                              for r in races if r.day == day])  # fmt: skip


def test_the_open_span_ends_on_the_stores_last_development_day(
    tmp_path, data_dir, stand_in_holdout
):
    """date_to None, as the command runs: the store's last race days are holdout days, after its
    last development day. Each variant's params.date_to is the last development day, and
    appending trailing holdout days or changing their rows and day files leaves params.date_to
    and the run id as they were; a development day after them is the new last day."""
    later = [DEV_DAYS[-1] + timedelta(days=k) for k in range(1, (DEV_AFTER - DEV_DAYS[-1]).days)]
    trailing = tuple(d for d in later if stand_in_rule(d))[:2]
    assert len(trailing) == 2 and all(holdout_module.is_holdout_day("japan", d) for d in trailing)

    def identities(name: str, **store) -> dict[str, tuple[str, str]]:
        registry, _ = frozen_registry(tmp_path, name)
        trailing_store(data_dir, **store)
        for variant in VARIANTS:
            run_quietly(runner(variant), date_from=DEV_DAYS[0], date_to=None,
                        out_dir=data_dir / "reports" / name, registry=registry)  # fmt: skip
        out = {}
        for line in registry.path.read_bytes().splitlines():
            entry = json.loads(line)
            if entry["kind"] == "run":
                record = entry["record"]
                assert record["status"] == "scored", (name, record["variant"], record["status"])
                out[record["variant"]] = (entry["run_id"], record["params"]["date_to"])
        assert set(out) == set(VARIANTS), (name, sorted(out))
        return out

    base = identities("base", trailing=trailing[:1])
    assert {v: day for v, (_, day) in base.items()} == dict.fromkeys(
        VARIANTS, DEV_DAYS[-1].isoformat()
    ), base
    assert identities("appended", trailing=trailing) == base, "a trailing holdout day changed it"
    assert identities("refreshed", trailing=trailing, held=3) == base, "holdout rows changed it"

    after = identities("development-after", trailing=trailing, after=True)
    for variant in VARIANTS:
        assert after[variant][1] == DEV_AFTER.isoformat(), (variant, after[variant])
        assert after[variant][0] != base[variant][0], variant
