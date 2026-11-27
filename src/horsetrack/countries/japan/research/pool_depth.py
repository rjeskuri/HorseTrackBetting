"""Pool depth and stake capacity from the final votes: N3 (ticket #432).

The spec is config/pool_depth.yaml (docs/specs/pool-depth-prereg.md), frozen in the japan registry
before any run. Per pool x class group x odds band it measures the money on a combination, the
pool, and the capacity stake: the stake s at which our own money on a single-winner combination
eats `share_lost` of an edge e. With V the pool and W the money already on the combination, our
stake moves the dividend from rate x V / W to rate x (V + s) / (W + s), so a bet worth 1 + e is
worth (1 + e)(V + s)W / (V(W + s)) once it is in; `capacity_stake` solves that for
1 + e(1 - share_lost): s = (e - k)VW / ((1 + k)V - (1 + e)W) with k = e(1 - share_lost). The
ticket's closed form kVW / ((1 + k)V - (1 + e)W) is the same at share_lost 0.5 only (e - k = k).
No breakage and no floor: the stake is continuous. As the pool deepens it falls to
W(e - k) / (1 + k) (W k / (1 + k) at one half); a shallower pool allows more.

V and W are the money that pays: W is the combination's final votes x 100 and V the pool's
`__pool__` votes net of its refunded votes, (votes - refund_votes) x 100 (`pool_votes` counts
refunded votes in the total; JRA pays (sales - refunds) x rate). `pool_totals`, a sales reader for
#349 and the low-turnover segment, reports the gross total instead (votes x 100, which is
`final_odds` investment), beside the time-series investment at t, which is gross too.

Place and Wide pay by the stake-back-share rule with co-placers, so the identity does not hold for
them. The spec registers two treatments: `depth-only` (`depth_table` gives no capacity for them)
and `co-placer` (`run_coplacer`): for each paying combination, given the race's realised
co-placers (the `dividends` rows), the pre-registered rule (`jra-stake-back-share` in the spec,
taken as JRA's) pays per yen d = 1 + (rV - sum W) / (P W_i), the
sum over the P paying combinations; own stake s on combination i adds s to V, to W_i and to the
sum, and the capacity solves (1 + e) d(s) / d(0) = 1 + k. With c = (1 + k) / (1 + e) and
A = rV - sum W that is s = (1 - c)(A + P W_i) / ((c d(0) - 1) P + 1 - r); at P = 1 it is the
single-winner identity again. The run checks the rule against the paid dividends, per pool, every
eligible paid combination counting (one without its votes or final price is one not reproduced),
and is recorded failed when either pool's share is below the spec's; the rule of
`core.pricing.stake_back_share_dividend`, r(1 + (V - sum W) / (P W_i)), is checked beside it as a
diagnostic only. `run_exacta_impact` is #350's measure: the
own impact of 100 yen on one exacta (FORECAST) combination by pool size
(`research.exacta_impact.own_impact`, its POOL_BIN_EDGES), from the final votes.

The readers scan `processed/jravan/{pool_votes,final_odds,races}.parquet` one row group (a year)
at a time with only the columns named, the predicate of `core.eval.holdout.
file_development_filter` (the span, holdout days dropped) and the reader's own pushed into the
scan, so no row of a holdout day or from before 2014-06-07 (D10) reaches Python; `pool_votes` and
`final_odds` keep the confirmed records only (データ区分 4 確定 and 5 確定(月曜); a race whose
kept record is the day before's close, 2, has no final money). A time-series day file of a
holdout day is skipped by its name before it is opened, and `LockedHoldout` is never opened. Every
run after the freeze is one RunRecord (failures included) in the registry, identified by what it
read: a digest of the rows each table gave the run and of the bytes of each development day file
read, so a refresh of holdout or pre-2014-06-07 rows leaves it unchanged. Every report built on
the store goes through `core.eval.reports.write_report` (sources ('jravan',)), so it never lands
in the repository, under a name carrying the run's id. `render_public_report` writes the public
text, which holds no figure from the data.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import math
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer

from horsetrack.core import env, paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.holdout import _git_sha, file_development_filter, is_holdout_day
from horsetrack.core.eval.registry import (
    FamilyAlreadyFrozen,
    FoldMetric,
    RecordRefused,
    Registry,
    RunRecord,
    _canonical,
    _sha256,
    _spec_hash,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_inside, write_report
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, PreReformPriceError
from horsetrack.countries.japan.research import exacta_impact, pool_scan

COUNTRY = "japan"
SOURCE = "jravan"
SOURCES = (SOURCE,)
SPEC_NAME = "pool_depth"
COPLACER_V2 = "pool_depth_coplacer_v2"  # the co-placer variant re-registered on the core rule
TICKET = 432
HYPOTHESIS_ID = "P432"
FOLD_SCHEME = "development-days"
WITHHELD = "withheld (D21)"
POOL_ROW = "__pool__"
KEYS = ["race_id", "pool", "combination"]
CELL = ("pool", "class_group", "band")
AT_T_POOLS = ("WIN", "PLACE", "QUINELLA")  # the pools of the jravan time series (0B41/0B42)
CO_PLACER = ("PLACE", "QUINELLA_PLACE")  # stake-back-share pools: no single-winner capacity
TERTILES = ("low", "mid", "high")
VARIANT_DEPTH, VARIANT_COPLACER, VARIANT_EXACTA = "depth", "co-placer", "exacta-own-impact"
# the spec files `pool-depth run --spec` takes, and the variants each one registers
SPECS = {
    SPEC_NAME: (VARIANT_DEPTH, VARIANT_COPLACER, VARIANT_EXACTA),
    COPLACER_V2: (VARIANT_COPLACER,),
}
# place_wide.co_placer.rule -> the rule's code: `jra` 1 + (rV - sum W) / (P W_i) per yen (frozen
# in japan-pool-depth-v1), `core` r(1 + (V - sum W) / (P W_i)) (core.pricing, japan-pool-depth-
# coplacer-v2)
RULES = {"jra-stake-back-share": "jra", "core-stake-back-share": "core"}
UNIT, STEP, FLOOR = 100, 10, 110  # yen per ticket, breakage, the Plus-10 minimum
CONFIRMED = ("4", "5")  # データ区分 確定 and 確定(月曜): the final records (2 is the day before's)
BATCH_ROWS = 65_536
RUN_SLOT = "<run-id>"  # replaced in a private report by the run's registry id
_EPS = 1e-9


class NotPreRegistered(RuntimeError):
    """The registry has no freeze of the shipped spec under its family: no run before the freeze."""


# --------------------------------------------------------------------------- the identities


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, not {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, not {value!r}")
    return number


def _check_share(share_lost: Any) -> float:
    share = _finite("share_lost", share_lost)
    if not 0 < share <= 1:
        raise ValueError(f"share_lost must be in (0, 1], not {share_lost!r}")
    return share


def capacity_stake(
    pool_total: float, combo_money: float, edge: float, *, share_lost: float = 0.5
) -> float:
    """The continuous stake (no breakage) at which our own money on a single-winner combination
    eats `share_lost` of the edge: (1 + e)(V + s)W / (V(W + s)) = 1 + e(1 - share_lost), so
    s = (e - k)VW / ((1 + k)V - (1 + e)W), k = e(1 - share_lost). ValueError when the edge is not
    a positive finite number, W < 0, W >= V, V or W is not finite, or the denominator is not
    positive (the stake never eats that much of the edge)."""
    pool = _finite("pool_total", pool_total)
    combo = _finite("combo_money", combo_money)
    e = _finite("edge", edge)
    share = _check_share(share_lost)
    if not e > 0:
        raise ValueError(f"edge must be positive, not {edge!r}")
    if combo < 0 or combo >= pool:
        raise ValueError(f"combo_money must be in [0, pool_total), not {combo_money!r}")
    k = e * (1 - share)
    denominator = (1 + k) * pool - (1 + e) * combo
    if not denominator > 0:
        raise ValueError(f"no stake eats {share:g} of the edge {e:g} here (W / V too high)")
    return float((e - k) * pool * combo / denominator)


def _capacities(pool: np.ndarray, combo: np.ndarray, edge: float, share: float) -> np.ndarray:
    """`capacity_stake` of every (V, W) at once: NaN where W < 0 or W >= V, inf where the
    denominator is not positive (the edge is never eaten that far)."""
    k = edge * (1 - share)
    denominator = (1 + k) * pool - (1 + edge) * combo
    with np.errstate(divide="ignore", invalid="ignore"):
        stake = (edge - k) * pool * combo / denominator
    stake = np.where(denominator > 0, stake, np.inf)
    return np.where((combo >= 0) & (combo < pool), stake, np.nan)


def coplacer_capacities(
    pool: np.ndarray,
    combo: np.ndarray,
    paid_money: np.ndarray,
    places: np.ndarray,
    rate: np.ndarray,
    edge: float,
    share: float,
) -> np.ndarray:
    """The co-placer capacity of paying combinations (module docstring): V `pool`, W_i `combo`,
    sum W `paid_money` over the race's P `places` paying combinations, payout rate `rate`. NaN
    where the rule pays no more than the stake back from the losers (A = rV - sum W <= 0) or a
    value is missing; inf where the edge is never eaten that far."""
    k = edge * (1 - share)
    c = (1 + k) / (1 + edge)
    a = rate * pool - paid_money
    with np.errstate(divide="ignore", invalid="ignore"):
        d0 = 1 + a / (places * combo)
        denominator = (c * d0 - 1) * places + 1 - rate
        stake = (1 - c) * (a + places * combo) / denominator
    stake = np.where(denominator > 0, stake, np.inf)
    return np.where((a > 0) & (combo > 0) & np.isfinite(d0), stake, np.nan)


def core_coplacer_capacities(
    pool: np.ndarray,
    combo: np.ndarray,
    paid_money: np.ndarray,
    places: np.ndarray,
    edge: float,
    share: float,
) -> np.ndarray:
    """The co-placer capacity by the rule of `core.pricing.stake_back_share_dividend` (family
    japan-pool-depth-coplacer-v2): per yen d = r(1 + B / (P W_i)), B = V - sum W the losers'
    money. Own stake s on combination i adds s to V and to sum W, so B stays and
    d(s) = r(1 + B / (P(W_i + s))); (1 + e) d(s) / d(0) = 1 + k gives, with c = (1 + k) / (1 + e),
    s = (1 - c)(B + P W_i) / (P(c - 1) + c B / W_i), whatever r is; at P = 1 it is
    `capacity_stake`. NaN where W_i <= 0, B < 0 or a value is missing; inf where the edge is
    never eaten that far."""
    k = edge * (1 - share)
    c = (1 + k) / (1 + edge)
    lost = pool - paid_money
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = places * (c - 1) + c * lost / combo
        stake = (1 - c) * (lost + places * combo) / denominator
    stake = np.where(denominator > 0, stake, np.inf)
    ok = (combo > 0) & (lost >= 0) & np.isfinite(lost) & np.isfinite(places)
    return np.where(ok, stake, np.nan)


def _rule_dividends(pool, combo, paid_money, places, rate, *, rule: str) -> np.ndarray:
    """The dividend per 100 yen of paying combinations at no own stake, broken to 10 yen and at
    least 110: the pre-registered rule (`jra`, 1 + (rV - sum W) / (P W_i) per yen) or the rule of
    `core.pricing.stake_back_share_dividend` (`core`, r(1 + (V - sum W) / (P W_i)))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if rule == "jra":
            per_yen = 1 + (rate * pool - paid_money) / (places * combo)
        else:
            per_yen = rate * (1 + (pool - paid_money) / (places * combo))
        raw = np.floor(UNIT * per_yen / STEP + _EPS) * STEP
    return np.where(np.isfinite(raw), np.maximum(FLOOR, raw), np.nan)


def _quantiles(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """numpy's linear quantiles of the non-NaN values; +inf values (unbounded capacity) sort last
    and give inf wherever the interpolation reaches one. NaN for no values."""
    x = np.sort(np.asarray(values, dtype=float)[~np.isnan(values)])
    qs = np.asarray(quantiles, dtype=float)
    if x.size == 0:
        return np.full(len(qs), np.nan)
    if np.isfinite(x[-1]):
        return np.quantile(x, qs)
    h = (x.size - 1) * qs
    lo, hi = np.floor(h).astype(int), np.ceil(h).astype(int)
    a, b = x[lo], x[hi]
    with np.errstate(invalid="ignore"):
        inner = np.where(np.isinf(b), np.inf, a + (b - a) * (h - lo))
    return np.where(hi == lo, a, inner)


# --------------------------------------------------------------------------- reading the store


def _check_span(date_from: date, date_to: date) -> None:
    for value in (date_from, date_to):
        if not isinstance(value, date) or isinstance(value, datetime):
            raise TypeError(f"date_from and date_to must be dates, got {value!r}")
    if date_from < PRICE_ERA_START:
        raise PreReformPriceError(
            f"no price row before {PRICE_ERA_START.isoformat()} (D10), got {date_from.isoformat()}"
        )
    if date_to < date_from:
        raise ValueError(f"date_to {date_to} is before date_from {date_from}")


def _years(date_from: date, date_to: date) -> Iterator[tuple[date, date]]:
    for year in range(date_from.year, date_to.year + 1):
        yield max(date_from, date(year, 1, 1)), min(date_to, date(year, 12, 31))


def _months(start: date, end: date) -> Iterator[tuple[date, date]]:
    first = date(start.year, start.month, 1)
    while first <= end:
        after = date(first.year + first.month // 12, first.month % 12 + 1, 1)
        yield max(start, first), min(end, after - timedelta(days=1))
        first = after


def _store_path(name: str) -> Path:
    return paths.processed_dir(SOURCE) / f"{name}.parquet"


def _row_groups(meta: pq.FileMetaData, start: date, end: date) -> list[int]:
    """The row groups whose race_date statistics may hold a day in start..end (or record none)."""
    keep = []
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        stats = next(
            (
                group.column(c).statistics
                for c in range(group.num_columns)
                if group.column(c).path_in_schema == "race_date"
            ),
            None,
        )
        if stats is None or not stats.has_min_max or (stats.max >= start and stats.min <= end):
            keep.append(g)
    return keep


def _scan(
    name: str,
    columns: Sequence[str],
    start: date,
    end: date,
    *,
    where: ds.Expression | None = None,
) -> pa.Table:
    """The development rows of the jravan store table `name` with start <= race_date <= end:
    exactly `columns`, narrowed by `where` (which may name columns not read). The predicate
    (`file_development_filter`: the span, holdout days dropped, and `where`) runs inside the
    scan, one row group at a time, so no other row reaches Python and no more than one row
    group's columns are decoded at once; strings are decoded as dictionaries there, and only the
    rows kept as plain strings."""
    _check_span(start, end)
    columns = list(columns)
    with pa.OSFile(str(_store_path(name))) as handle:
        keep = file_development_filter(handle, COUNTRY, start=start, end=end, where=where)
        whole = ds.ParquetFileFormat().make_fragment(handle)
        schema = whole.physical_schema
        missing = [c for c in columns if c not in schema.names]
        if missing:
            raise KeyError(f"{name} has no column(s): {', '.join(missing)}")
        strings = {f.name for f in schema if pa.types.is_string(f.type)}
        options = ds.ParquetReadOptions(dictionary_columns=strings)
        out = pa.schema([schema.field(c) for c in columns])
        batches = []
        for group in _row_groups(whole.metadata, start, end):
            piece = ds.ParquetFileFormat(read_options=options).make_fragment(
                handle, row_groups=[group]
            )
            scanner = ds.Scanner.from_fragment(
                piece, columns=columns, filter=keep, batch_size=BATCH_ROWS
            )
            batches.extend(b.cast(out) for b in scanner.to_batches() if b.num_rows)
    table = pa.Table.from_batches(batches, schema=out)
    del batches
    pa.default_memory_pool().release_unused()  # the decoded row group, back to the system
    return table


def _confirmed(where: ds.Expression | None = None) -> ds.Expression:
    """The confirmed records of `pool_votes` or `final_odds` (CONFIRMED), and `where`."""
    keep = ds.field("data_kubun").isin(list(CONFIRMED))
    return keep if where is None else keep & where


def _between(table: pa.Table, start: date, end: date) -> pa.Table:
    day = ds.field("race_date")
    span = (day >= pa.scalar(start, pa.date32())) & (day <= pa.scalar(end, pa.date32()))
    return table.filter(span).drop_columns(["race_date"])


def _numbers(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _net_yen(pool_rows: pd.DataFrame) -> np.ndarray:
    """The money that pays on `__pool__` rows: (votes - refund_votes) x 100, a null refund 0."""
    refunds = np.nan_to_num(_numbers(pool_rows["refund_votes"]), nan=0.0)
    return UNIT * (_numbers(pool_rows["votes"]) - refunds)


def _days(date_from: date, date_to: date) -> Iterator[date]:
    for k in range((date_to - date_from).days + 1):
        yield date_from + timedelta(days=k)


def _day_files(date_from: date, date_to: date) -> list[tuple[date, Path]]:
    """The jravan time-series day files of the development days in the range; a holdout day's
    file is dropped by its name, before anything opens it."""
    folder = paths.processed_dir(SOURCE) / "odds_timeseries"
    out = []
    for day in _days(date_from, date_to):
        if is_holdout_day(COUNTRY, day):
            continue
        path = folder / f"{day.isoformat()}.parquet"
        if path.is_file():
            out.append((day, path))
    return out


class _Inputs:
    """What a run read, for its identity: per store table a sha256 over the frames the run took
    from it (their columns and pandas row hashes, in read order), and per development day file
    read its bytes. Holdout and pre-2014-06-07 rows never reach a frame, so they never count."""

    def __init__(self) -> None:
        self._digests: dict[str, Any] = {}

    def add(self, name: str, frame: pd.DataFrame) -> None:
        digest = self._digests.setdefault(name, hashlib.sha256())
        digest.update(("\x1f".join(map(str, frame.columns)) + "\n").encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())

    def add_files(self, name: str, files: Sequence[tuple[date, Path]]) -> None:
        digest = self._digests.setdefault(name, hashlib.sha256())
        for day, path in files:
            digest.update(f"{day.isoformat()}:{_sha16(path)}\n".encode())

    def builds(self) -> dict[str, str]:
        """data_builds: each table's path, then the digest of what the run read of it."""
        out = {}
        for name, digest in sorted(self._digests.items()):
            where = name if name == "odds_timeseries" else f"{name}.parquet"
            out[name] = f"processed/{SOURCE}/{where}@{digest.hexdigest()[:16]}"
        return out


def pool_totals(*, date_from: date, date_to: date, minutes_before: float = 2.0) -> pd.DataFrame:
    """Per race and pool the store sold on a development day in date_from..date_to: race_id,
    pool, final_yen (the confirmed `__pool__` row's votes x 100, gross sales: `final_odds`
    investment; a race whose kept record is the day before's has no row) and at_t_yen (WIN,
    PLACE and QUINELLA: the investment of the latest time-series `__pool__` capture with
    minutes_to_post >= minutes_before, a capture at t included; NaN otherwise)."""
    _check_span(date_from, date_to)
    minutes = _finite("minutes_before", minutes_before)
    finals = []
    for start, end in _years(date_from, date_to):
        table = _scan(
            "pool_votes",
            ["race_id", "pool", "votes"],
            start,
            end,
            where=_confirmed(ds.field("combination") == POOL_ROW),
        )
        finals.append(table.to_pandas())
    final = pd.concat(finals, ignore_index=True)
    final = pd.DataFrame(
        {
            "race_id": final["race_id"],
            "pool": final["pool"],
            "final_yen": UNIT * _numbers(final["votes"]),
        }
    )
    captures = []
    at_t = (
        (ds.field("combination") == POOL_ROW)
        & ds.field("pool").isin(list(AT_T_POOLS))
        & (ds.field("minutes_to_post") >= minutes)
        & ds.field("investment").is_valid()
    )
    for _, path in _day_files(date_from, date_to):
        rows = pq.read_table(
            path, columns=["race_id", "pool", "snapshot_utc", "investment"], filters=at_t
        ).to_pandas()
        latest = rows.sort_values("snapshot_utc", kind="stable").drop_duplicates(
            ["race_id", "pool"], keep="last"
        )
        captures.append(latest[["race_id", "pool", "investment"]])
    if captures:
        board = pd.concat(captures, ignore_index=True).rename(columns={"investment": "at_t_yen"})
    else:
        board = pd.DataFrame({"race_id": [], "pool": [], "at_t_yen": []})
    board = board.astype({"race_id": object, "pool": object, "at_t_yen": float})
    out = final.merge(board, on=["race_id", "pool"], how="left")
    out["at_t_yen"] = out["at_t_yen"].astype(float)
    out = out.sort_values(["race_id", "pool"], kind="stable").reset_index(drop=True)
    return out[["race_id", "pool", "final_yen", "at_t_yen"]]


# --------------------------------------------------------------------------- depth rows and table


def _columns(frame: pd.DataFrame, name: str, needed: Sequence[str]) -> None:
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks column(s): {', '.join(missing)}")


def _race_groups(races: pd.DataFrame, exclude_status: Sequence[str]) -> pd.DataFrame:
    """race_id and class_group of the races, without those whose status is excluded."""
    _columns(races, "races", ["race_id", "race_class"])
    if "status" in races.columns and exclude_status:
        races = races.loc[~races["status"].isin(list(exclude_status)).to_numpy(dtype=bool)]
    groups = {c: pool_scan.class_group(c) for c in races["race_class"].unique()}
    return pd.DataFrame(
        {"race_id": races["race_id"].to_numpy(), "class_group": races["race_class"].map(groups)}
    )


def depth_rows(
    votes: pd.DataFrame,
    board: pd.DataFrame,
    races: pd.DataFrame,
    *,
    band_edges: Mapping[str, Sequence[float]],
    exclude_status: Sequence[str] = ("abandoned",),
) -> pd.DataFrame:
    """One row per combination with a final price of the pools `band_edges` names: race_id, pool,
    combination, class_group, band (its final odds on the pool's edges), odds, combo_yen (its
    votes x 100) and pool_yen (the race's `__pool__` money that pays, in yen). A combination
    without votes, or a race without its pool row or race row, gives no row; so does a race whose
    status is in `exclude_status`."""
    _columns(votes, "votes", [*KEYS, "votes", "refund_votes"])
    _columns(board, "board", [*KEYS, "odds"])
    pools = list(band_edges)
    v = votes.loc[votes["pool"].isin(pools).to_numpy(dtype=bool)]
    is_pool = v["combination"].to_numpy() == POOL_ROW
    totals = v.loc[is_pool, ["race_id", "pool"]].assign(pool_yen=_net_yen(v.loc[is_pool]))
    if totals.duplicated(["race_id", "pool"]).any():
        raise ValueError("votes has a pool row twice in a race")
    slots = v.loc[~is_pool, KEYS].assign(combo_yen=UNIT * _numbers(v.loc[~is_pool, "votes"]))
    b = board.loc[
        board["pool"].isin(pools).to_numpy(dtype=bool)
        & (board["combination"].to_numpy() != POOL_ROW)
    ]
    odds = _numbers(b["odds"])
    priced = b.loc[~np.isnan(odds), KEYS].assign(odds=odds[~np.isnan(odds)])
    if priced.duplicated(KEYS).any() or slots.duplicated(KEYS).any():
        raise ValueError("a combination is on the board or in the votes twice in a race")
    rows = priced.merge(slots, on=KEYS).merge(totals, on=["race_id", "pool"])
    rows = rows.loc[rows["combo_yen"].notna().to_numpy() & rows["pool_yen"].notna().to_numpy()]
    rows = rows.merge(_race_groups(races, exclude_status), on="race_id")
    band = np.empty(len(rows), dtype=object)
    pool_of = rows["pool"].to_numpy()
    for pool in pools:
        mine = pool_of == pool
        if mine.any():
            band[mine] = pool_scan._bands(rows["odds"].to_numpy()[mine], band_edges[pool])
    rows = rows.assign(band=band)
    columns = [*KEYS, "class_group", "band", "odds", "combo_yen", "pool_yen"]
    return rows[columns].reset_index(drop=True)


def _edges(edges: Sequence[float]) -> list[float]:
    out = [_finite("edge", e) for e in edges]
    if not out or any(e <= 0 for e in out):
        raise ValueError(f"edges must be positive numbers, not {list(edges)}")
    return out


def _quantile_list(quantiles: Sequence[float]) -> list[float]:
    out = [_finite("quantile", q) for q in quantiles]
    if not out or any(not 0 <= q <= 1 for q in out) or out != sorted(set(out)):
        raise ValueError(f"quantiles must be ascending values in [0, 1], not {list(quantiles)}")
    return out


def capacity_column(edge: float) -> str:
    return f"capacity_{float(edge):g}"


def fits_column(edge: float, stake: float) -> str:
    return f"fits_{float(edge):g}_{float(stake):g}"


def _table(
    rows: pd.DataFrame,
    *,
    by: Sequence[str],
    quantiles: Sequence[float],
    capacities: Callable[[str, np.ndarray], dict[float, np.ndarray] | None],
    edges: Sequence[float],
    stakes: Sequence[float],
) -> pd.DataFrame:
    """The long cell table: per cell of `by` and quantile, the cell's combinations and races, the
    quantiles of combo_yen and of pool_yen (each race once) and of `capacities(pool, index)` per
    edge (NaN columns where it returns None), and the share of capacities at least each stake."""
    by = list(by)
    caps = [capacity_column(e) for e in edges]
    fits = [fits_column(e, s) for e in edges for s in stakes]
    columns = [*by, "quantile", "n_combinations", "n_races", "combo_yen", "pool_yen", *caps, *fits]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    combo = rows["combo_yen"].to_numpy(dtype=float)
    pool_yen = rows["pool_yen"].to_numpy(dtype=float)
    race = rows["race_id"].to_numpy()
    out = []
    nq = len(quantiles)
    for key, index in rows.groupby(by, sort=True, observed=True).indices.items():
        key = key if isinstance(key, tuple) else (key,)
        cell = dict(zip(by, key, strict=True))
        _, first = np.unique(race[index], return_index=True)
        values = {
            "combo_yen": _quantiles(combo[index], quantiles),
            "pool_yen": _quantiles(pool_yen[index][first], quantiles),
        }
        found = capacities(str(cell["pool"]), index)
        for e, name in zip(edges, caps, strict=True):
            values[name] = np.full(nq, np.nan) if found is None else _quantiles(found[e], quantiles)
            for s in stakes:
                share = np.nan
                if found is not None and (~np.isnan(found[e])).any():
                    share = float(np.mean(found[e][~np.isnan(found[e])] >= s))
                values[fits_column(e, s)] = np.full(nq, share)
        for i, q in enumerate(quantiles):
            out.append(
                {
                    **cell,
                    "quantile": q,
                    "n_combinations": len(index),
                    "n_races": len(first),
                    **{name: float(array[i]) for name, array in values.items()},
                }
            )
    return pd.DataFrame(out, columns=columns)


def depth_table(
    rows: pd.DataFrame,
    *,
    edges: Sequence[float],
    quantiles: Sequence[float],
    share_lost: float = 0.5,
    by: Sequence[str] = CELL,
    stakes: Sequence[float] = (),
) -> pd.DataFrame:
    """Long format, one row per cell of `by` (default pool x class group x odds band) and q of
    `quantiles`: pool, class_group, band, quantile, n_combinations, n_races, combo_yen and
    pool_yen (the q-quantiles of the money on a combination and of the pool, each race once) and
    `capacity_{e:g}` per edge (the q-quantile of `capacity_stake` over the cell's combinations;
    NaN for Place and Wide, whose dividend is not the single-winner identity). With `stakes`, the
    `fits_{e:g}_{stake:g}` columns give the share of the cell's combinations whose capacity is at
    least that stake."""
    by = list(by)
    if "pool" not in by:
        raise ValueError(f"by must include 'pool', not {by}")
    _columns(rows, "rows", [*by, "race_id", "combo_yen", "pool_yen"])
    edges, quantiles = _edges(edges), _quantile_list(quantiles)
    share = _check_share(share_lost)
    stakes = [_finite("stake", s) for s in stakes]
    combo = rows["combo_yen"].to_numpy(dtype=float)
    pool_yen = rows["pool_yen"].to_numpy(dtype=float)

    def capacities(pool: str, index: np.ndarray) -> dict[float, np.ndarray] | None:
        if pool in CO_PLACER:
            return None
        return {e: _capacities(pool_yen[index], combo[index], e, share) for e in edges}

    return _table(
        rows, by=by, quantiles=quantiles, capacities=capacities, edges=edges, stakes=stakes
    )


def coplacer_rows(
    rows: pd.DataFrame, paid: pd.DataFrame, *, rates: Mapping[str, float], edges: Sequence[float],
    share_lost: float, rule: str = "jra",
) -> pd.DataFrame:  # fmt: skip
    """Every paid Place and Wide combination of `paid` (`dividends` rows, refunds dropped), each
    with its `rows` row (`depth_rows` of those pools) where it has one, given the race's realised
    co-placers: the depth columns (NaN where the combination has no votes or final price), places
    (P, the race's paying combinations), paid_yen (their money), dividend (as paid), jra_dividend
    and core_dividend (the two rules at no own stake; NaN where a money row is missing) and
    `capacity_{e:g}` per edge by `rule` (`jra`, `coplacer_capacities`, the rule frozen in
    japan-pool-depth-v1; `core`, `core_coplacer_capacities`)."""
    _columns(paid, "paid", [*KEYS, "dividend", "is_refund"])
    refund = paid["is_refund"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    paid = paid.loc[paid["pool"].isin(list(CO_PLACER)).to_numpy(dtype=bool) & ~refund]
    paid = paid[[*KEYS, "dividend"]].astype({c: object for c in KEYS})
    if rows.empty:
        rows = pd.DataFrame({c: pd.Series(dtype=object) for c in [*KEYS, "class_group", "band"]})
        rows = rows.assign(**{c: pd.Series(dtype=float) for c in ("odds", "combo_yen", "pool_yen")})
    out = paid.merge(rows, on=KEYS, how="left")
    race_pool = ["race_id", "pool"]
    group = out.groupby(race_pool, sort=False)
    places = group["combination"].transform("size").to_numpy(dtype=float)
    missing = out["combo_yen"].isna().groupby([out["race_id"], out["pool"]]).transform("any")
    paid_yen = group["combo_yen"].transform("sum").where(~missing).to_numpy(dtype=float)
    pool_yen = group["pool_yen"].transform("max").to_numpy(dtype=float)
    combo = out["combo_yen"].to_numpy(dtype=float)
    rate = out["pool"].map(dict(rates)).to_numpy(dtype=float)
    columns = {
        "places": places,
        "paid_yen": paid_yen,
        "jra_dividend": _rule_dividends(pool_yen, combo, paid_yen, places, rate, rule="jra"),
        "core_dividend": _rule_dividends(pool_yen, combo, paid_yen, places, rate, rule="core"),
    }
    share = _check_share(share_lost)
    if rule not in RULES.values():
        raise ValueError(f"rule must be one of {sorted(RULES.values())}, not {rule!r}")
    for e in _edges(edges):
        columns[capacity_column(e)] = (
            coplacer_capacities(pool_yen, combo, paid_yen, places, rate, e, share)
            if rule == "jra"
            else core_coplacer_capacities(pool_yen, combo, paid_yen, places, e, share)
        )
    out = out.assign(pool_yen=pool_yen, dividend=_numbers(out["dividend"]), **columns)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- the spec and records


def load_spec(name: str = SPEC_NAME) -> tuple[str, dict[str, Any]]:
    """The shipped spec `config/<name>.yaml` (a copy) and its family_id, checked; ValueError
    when malformed or not one of SPECS."""
    if name not in SPECS:
        raise ValueError(f"spec must be one of {sorted(SPECS)}, not {name!r}")
    spec = copy.deepcopy(dict(load_country_yaml(COUNTRY, name)))
    family = spec.get("family_id")
    if not isinstance(family, str) or not family.strip():
        raise ValueError(f"config/{name}.yaml names no family_id")
    if spec["place_wide"]["co_placer"]["rule"] not in RULES:
        raise ValueError(f"place_wide.co_placer.rule must be one of {sorted(RULES)}")
    _edges(spec["edges"])
    _check_share(spec["share_lost"])
    _quantile_list(spec["quantiles"])
    bands = spec["odds_bands"]
    if not set(pool_scan.POOLS) <= set(bands):
        raise ValueError(f"odds_bands must name every pool of {pool_scan.POOLS}")
    for edges in bands.values():
        pool_scan._band_labels(edges)
    if spec["class_groups"] != pool_scan.CLASS_GROUPS:
        raise ValueError("class_groups must be research/pool_scan.CLASS_GROUPS")
    impact = spec["exacta_impact"]
    if tuple(impact["pool_bin_edges"]) != exacta_impact.POOL_BIN_EDGES:
        raise ValueError("exacta_impact.pool_bin_edges must be exacta_impact.POOL_BIN_EDGES")
    if date.fromisoformat(spec["date_from"]) < PRICE_ERA_START:
        raise ValueError("date_from must not be before 2014-06-07 (D10)")
    return family, spec


@dataclass(frozen=True)
class _Run:
    registry: Registry
    family_id: str
    spec: dict[str, Any]
    spec_hash: str


def _preflight(
    registry: Registry | None,
    date_from: date,
    date_to: date | None,
    out_dir: Path | str,
    spec_name: str = SPEC_NAME,
) -> _Run:
    """NotPreRegistered before anything is read or written; then the span (D10; date_to None is
    the store's last race day, looked up in the recorded attempt) and out_dir."""
    family, spec = load_spec(spec_name)
    registry = default_registry(COUNTRY) if registry is None else registry
    spec_hash = _spec_hash(spec)
    frozen = registry.frozen_spec_hash(family)
    if frozen != spec_hash:
        raise NotPreRegistered(
            f"family {family!r} has no freeze of this spec (sha256 {spec_hash}, frozen: {frozen}); "
            f"run `horsetrack japan pool-depth freeze --spec {spec_name}` first"
        )
    _check_span(date_from, date_from if date_to is None else date_to)
    if is_inside(Path(out_dir), env.repo_root()):
        raise LicensedOutputInRepo(f"{out_dir} is inside the repository: the report is licensed")
    return _Run(registry, family, spec, spec_hash)


def _sha16(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _commit() -> str:
    try:
        return _git_sha()
    except RuntimeError:
        return "unknown"


def _record(run: _Run, variant: str, params: dict, builds: dict, metrics, status) -> RunRecord:
    return RunRecord(
        hypothesis_id=HYPOTHESIS_ID,
        family_id=run.family_id,
        variant=variant,
        ticket=TICKET,
        git_sha=_commit(),
        data_builds=builds,
        sources=SOURCES,
        fold_scheme=FOLD_SCHEME,
        components={},
        params={"spec_hash": run.spec_hash, **params},
        metrics=tuple(metrics),
        status=status,
    )


def _run_id(record: RunRecord) -> str:
    """The id the registry gives the record (the sha256 of its canonical form)."""
    return _sha256(_canonical(asdict(record)))


def _append(registry: Registry, record: RunRecord) -> str:
    """The run's id; the record is appended unless the registry already holds this exact run."""
    if record not in registry.records(record.family_id):
        return registry.append(record)
    return _run_id(record)


Body = Callable[[_Inputs, date], tuple[list[FoldMetric], str, str, Any]]


def _recorded(
    run: _Run,
    variant: str,
    span: tuple[date, date | None],
    out_dir: Path | str,
    stem: str,
    body: Body,
) -> tuple[str, str, Path, Any]:
    """Run `body(inputs, date_to)` (metrics, status, report text, payload) on the span, date_to
    None being the store's last race day (looked up here, inside the attempt), and record it:
    the record's data_builds are what the run read (`_Inputs`), its report is written to
    `<stem>-<first 16 hex of the run id>.md` under out_dir. Any failure after the freeze is
    recorded as failed and raised again."""
    date_from, date_to = span
    inputs = _Inputs()
    params = {"date_from": date_from.isoformat(), "date_to": None}
    try:
        date_to = _last_store_day() if date_to is None else date_to
        params["date_to"] = date_to.isoformat()
        _check_span(date_from, date_to)
        metrics, status, text, payload = body(inputs, date_to)
        record = _record(run, variant, params, inputs.builds(), metrics, status)
        run_id = _run_id(record)
        path = Path(out_dir) / f"{stem}-{run_id[:16]}.md"
        report = write_report(path, text.replace(RUN_SLOT, run_id), sources=SOURCES)
    except Exception:
        _append(run.registry, _record(run, variant, params, inputs.builds(), (), "failed"))
        raise
    return _append(run.registry, record), status, report, payload


# --------------------------------------------------------------------------- the passes


def _read_races(date_from: date, date_to: date, inputs: _Inputs) -> pd.DataFrame:
    """race_id, race_date, race_class and status of the development races in the span."""
    columns = ["race_id", "race_date", "race_class", "status"]
    races = _scan("races", columns, date_from, date_to).to_pandas()
    inputs.add("races", races)
    return races


def _chunks(
    date_from: date, date_to: date, pools: Sequence[str], inputs: _Inputs
) -> Iterator[tuple[date, date, pd.DataFrame, pd.DataFrame]]:
    """(first day, last day, votes, board) a month at a time: the confirmed `pool_votes` rows of
    `pools` (slots and pool rows) and their confirmed `final_odds` rows (priced slots and pool
    rows), read a year at a time; each frame counts toward the run's inputs."""
    in_pools = ds.field("pool").isin(list(pools))
    priced = (ds.field("combination") == POOL_ROW) | ds.field("odds").is_valid()
    for start, end in _years(date_from, date_to):
        votes = _scan(
            "pool_votes",
            ["race_date", *KEYS, "votes", "refund_votes"],
            start,
            end,
            where=_confirmed(in_pools),
        )
        board = _scan(
            "final_odds",
            ["race_date", *KEYS, "odds", "investment"],
            start,
            end,
            where=_confirmed(in_pools & priced),
        )
        for a, b in _months(start, end):
            month_votes = _between(votes, a, b).to_pandas()
            month_board = _between(board, a, b).to_pandas()
            inputs.add("pool_votes", month_votes)
            inputs.add("final_odds", month_board)
            yield a, b, month_votes, month_board
        del votes, board
        pa.default_memory_pool().release_unused()


def _pool_check(votes: pd.DataFrame, board: pd.DataFrame) -> dict[str, int]:
    """The `__pool__` rows of both tables compared: gross and net votes x 100 against the board's
    investment."""
    v = votes.loc[votes["combination"].to_numpy() == POOL_ROW]
    b = board.loc[board["combination"].to_numpy() == POOL_ROW, ["race_id", "pool", "investment"]]
    m = v.merge(b, on=["race_id", "pool"], how="outer", indicator=True)
    both = m["_merge"].to_numpy() == "both"
    investment = _numbers(m["investment"])
    gross = UNIT * _numbers(m["votes"])
    refunded = np.nan_to_num(_numbers(m["refund_votes"]), nan=0.0) > 0
    return {
        "pool_rows_in_both": int(both.sum()),
        "pool_rows_in_one_table_only": int((~both).sum()),
        "pool_rows_with_refunds": int((both & refunded).sum()),
        "gross_equals_investment": int((both & (gross == investment)).sum()),
        "net_equals_investment": int((both & (_net_yen(m) == investment)).sum()),
        "refunded_and_gross_equals_investment": int(
            (both & refunded & (gross == investment)).sum()
        ),
    }


class _Depth:
    """The depth rows of a run, compactly: per pool the race index, band code and votes of each
    priced combination, with the pool's money per race."""

    def __init__(self, races: pd.DataFrame, band_edges: Mapping[str, Sequence[float]]):
        self.races = pd.Index(races["race_id"])
        self.groups = list(dict.fromkeys(pool_scan.CLASS_GROUPS.values()))
        frame = _race_groups(races, ())
        self.group_code = np.full(len(self.races), -1, dtype=np.int8)
        codes = pd.Categorical(frame["class_group"], categories=self.groups).codes
        self.group_code[self.races.get_indexer(frame["race_id"])] = codes
        self.labels = {p: pool_scan._band_labels(e) for p, e in band_edges.items()}
        self.pool_yen = {p: np.full(len(self.races), np.nan) for p in band_edges}
        self.parts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            p: [] for p in band_edges
        }

    def add(self, rows: pd.DataFrame) -> None:
        for pool, part in rows.groupby("pool", sort=False):
            race = self.races.get_indexer(part["race_id"]).astype(np.int32)
            band = pd.Categorical(part["band"], categories=self.labels[pool]).codes
            votes = np.rint(part["combo_yen"].to_numpy(dtype=float) / UNIT)
            if (race < 0).any() or (band < 0).any() or votes.max() >= 2**31:
                raise ValueError(f"a {pool} row has an unknown race, band or too many votes")
            self.parts[pool].append((race, band.astype(np.int8), votes.astype(np.int32)))
            self.pool_yen[pool][race] = part["pool_yen"].to_numpy(dtype=float)

    def frame(self, pool: str, *, tertiles: bool = False) -> pd.DataFrame:
        parts = self.parts[pool]
        race = np.concatenate([p[0] for p in parts]) if parts else np.empty(0, np.int32)
        band = np.concatenate([p[1] for p in parts]) if parts else np.empty(0, np.int8)
        votes = np.concatenate([p[2] for p in parts]) if parts else np.empty(0, np.int32)
        pool_yen = self.pool_yen[pool][race]
        frame = {
            "race_id": race,
            "pool": pd.Categorical.from_codes(np.zeros(len(race), np.int8), [pool]),
            "class_group": pd.Categorical.from_codes(self.group_code[race], self.groups),
            "band": pd.Categorical.from_codes(band, self.labels[pool]),
            "combo_yen": UNIT * votes.astype(float),
            "pool_yen": pool_yen,
        }
        if tertiles:
            sizes = self.pool_yen[pool][np.unique(race)]
            cuts = np.quantile(sizes, [1 / 3, 2 / 3]) if sizes.size else np.array([0.0, 0.0])
            code = np.searchsorted(cuts, pool_yen, side="right").astype(np.int8)
            frame["pool_size"] = pd.Categorical.from_codes(code, list(TERTILES))
        return pd.DataFrame(frame)


def _counts(rows: pd.DataFrame, by: Sequence[str] = ("pool",)) -> pd.DataFrame:
    return (
        rows.groupby(list(by), observed=True, sort=True)
        .agg(n_combinations=("race_id", "size"), n_races=("race_id", "nunique"))
        .reset_index()
    )


# --------------------------------------------------------------------------- the depth run


@dataclass(frozen=True)
class DepthResult:
    table: pd.DataFrame
    report_path: Path
    family_id: str
    spec_hash: str
    run_id: str
    date_from: date
    date_to: date
    tertiles: pd.DataFrame | None = None
    totals: pd.DataFrame | None = None


def run_depth(
    *,
    date_from: date,
    date_to: date | None,
    out_dir: Path | str,
    registry: Registry | None = None,
) -> DepthResult:
    """The depth variant on development days from date_from to date_to (None: the store's last
    race day): `depth_table` of the run's `depth_rows` with the spec's odds bands, edges and
    quantiles (Place and Wide depth only), the pool-size tertile cells, the flat-stake shares and
    the pool totals at t, written to the run's private report under `out_dir` and recorded as one
    RunRecord ("failed" when it raised). NotPreRegistered unless the registry has the shipped
    spec frozen."""
    run = _preflight(registry, date_from, date_to, out_dir)
    spec = run.spec

    def body(inputs: _Inputs, date_to: date):
        races = _read_races(date_from, date_to, inputs)
        depth = _Depth(races.loc[~races["status"].isin(spec["exclude_status"])], spec["odds_bands"])
        check: dict[str, int] = {}
        for _, _, votes, board in _chunks(date_from, date_to, list(spec["odds_bands"]), inputs):
            for key, n in _pool_check(votes, board).items():
                check[key] = check.get(key, 0) + n
            depth.add(depth_rows(votes, board, races, band_edges=spec["odds_bands"],
                                 exclude_status=spec["exclude_status"]))  # fmt: skip
        tables, tertiles, counts = [], [], []
        for pool in spec["odds_bands"]:
            frame = depth.frame(pool, tertiles=True)
            counts.append(_counts(frame, ("pool", "class_group")))
            options = dict(edges=spec["edges"], quantiles=spec["quantiles"],
                           share_lost=spec["share_lost"])  # fmt: skip
            tables.append(depth_table(frame, **options, stakes=spec["flat_stakes_yen"]))
            tertiles.append(depth_table(frame, **options, by=("pool", "pool_size", "band")))
            del frame
        table = pd.concat(tables, ignore_index=True)
        tertile_table = pd.concat(tertiles, ignore_index=True)
        totals = _totals_by_class(date_from, date_to, races, spec, inputs)
        result = DepthResult(table, Path(), run.family_id, run.spec_hash, "", date_from, date_to,
                             tertile_table, totals)  # fmt: skip
        text = _depth_report(result, spec, check, pd.concat(counts, ignore_index=True))
        return _depth_metrics(table, spec), "scored", text, result

    run_id, _, report, result = _recorded(
        run, VARIANT_DEPTH, (date_from, date_to), out_dir, "pool-depth-japan", body
    )
    return replace(result, report_path=report, run_id=run_id)


def _totals_by_class(
    date_from: date,
    date_to: date,
    races: pd.DataFrame,
    spec: Mapping[str, Any],
    inputs: _Inputs,
) -> pd.DataFrame:
    """Per pool at t (WIN, PLACE, QUINELLA) x class group: races, races with a total at t, and the
    quantiles of the final total, the total at t and their ratio (#349's history). The totals
    count toward the run's pool_votes input and the day files read toward its odds_timeseries."""
    inputs.add_files("odds_timeseries", _day_files(date_from, date_to))
    totals = pool_totals(
        date_from=date_from, date_to=date_to, minutes_before=spec["minutes_before"]
    )
    inputs.add("pool_votes", totals)
    totals = totals.loc[totals["pool"].isin(list(AT_T_POOLS)).to_numpy()]
    totals = totals.merge(_race_groups(races, spec["exclude_status"]), on="race_id")
    rows = []
    for (pool, group), cell in totals.groupby(["pool", "class_group"], sort=True):
        at_t = cell["at_t_yen"].to_numpy(dtype=float)
        final = cell["final_yen"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = at_t / final
        for q, f, a, r in zip(
            spec["quantiles"],
            _quantiles(final, spec["quantiles"]),
            _quantiles(at_t, spec["quantiles"]),
            _quantiles(np.where(np.isfinite(ratio), ratio, np.nan), spec["quantiles"]),
            strict=True,
        ):
            rows.append({"pool": pool, "class_group": group, "quantile": q,
                         "n_races": len(cell), "n_at_t": int(np.isfinite(at_t).sum()),
                         "final_yen": f, "at_t_yen": a, "at_t_share": r})  # fmt: skip
    return pd.DataFrame(rows)


def _median_row(table: pd.DataFrame, quantiles: Sequence[float]) -> pd.DataFrame:
    middle = min(quantiles, key=lambda q: abs(float(q) - 0.5))
    return table.loc[np.isclose(table["quantile"].to_numpy(dtype=float), float(middle))]


def _depth_metrics(table: pd.DataFrame, spec: Mapping[str, Any]) -> list[FoldMetric]:
    """Per cell, the median capacity at the middle edge (Place and Wide: the median money on a
    combination), n its combinations."""
    edge = spec["edges"][len(spec["edges"]) // 2]
    out = []
    for row in _median_row(table, spec["quantiles"]).to_dict("records"):
        co = row["pool"] in CO_PLACER
        metric = "median_combo_yen" if co else f"median_{capacity_column(edge)}"
        value = row["combo_yen"] if co else row[capacity_column(edge)]
        if math.isfinite(value):
            fold = f"{row['pool']}|{row['class_group']}|{row['band']}"
            out.append(
                FoldMetric(fold, metric, float(value), None, None, int(row["n_combinations"]))
            )
    return out


# --------------------------------------------------------------------------- the co-placer run


@dataclass(frozen=True)
class CoplacerResult:
    """Per pool (PLACE, QUINELLA_PLACE): `rule_match`, the share of the eligible paid
    combinations the spec's rule (`rule`, a RULES code) reproduces, the one the gate reads;
    `jra_match` and `core_match`, the shares of both rules; `coverage`, the share with their
    votes and final price (NaN for a pool with no eligible payout)."""

    table: pd.DataFrame
    status: str
    rule: str
    rule_match: Mapping[str, float]
    jra_match: Mapping[str, float]
    core_match: Mapping[str, float]
    coverage: Mapping[str, float]
    n_paid: Mapping[str, int]
    report_path: Path
    run_id: str


def _paid(races: pd.DataFrame, inputs: _Inputs) -> pd.DataFrame:
    """The Place and Wide `dividends` rows of `races` (the table has no day column: the scan keeps
    the race ids of the development races read)."""
    keep = ds.field("race_id").isin(races["race_id"].tolist()) & ds.field("pool").isin(
        list(CO_PLACER)
    )
    columns = [*KEYS, "dividend", "is_refund"]
    paid = pq.read_table(_store_path("dividends"), columns=columns, filters=keep).to_pandas()
    inputs.add("dividends", paid)
    return paid


def _rates(races: pd.DataFrame) -> dict[str, float]:
    """1 - each Place/Wide pool's base takeout (config/jra.yaml) on the first race day read; the
    run refuses a span over which the base rate changes."""
    days = pd.to_datetime(races["race_date"]).dt.date
    pools = load_source(COUNTRY, "jra").pools
    rates = {}
    for pool in CO_PLACER:
        taken = {pools[pool].takeout_on(d) for d in (days.min(), days.max())}
        if len(taken) != 1 or None in taken:
            raise ValueError(f"the {pool} takeout is not one known rate over the span")
        rates[pool] = 1 - taken.pop()
    return rates


def _shares(rows: pd.DataFrame, hits: np.ndarray) -> dict[str, float]:
    """Per co-placer pool, the share of its rows where `hits` holds (NaN for no row)."""
    pool = rows["pool"].to_numpy()
    return {p: float(hits[pool == p].mean()) if (pool == p).any() else math.nan for p in CO_PLACER}


def run_coplacer(
    *,
    date_from: date,
    date_to: date | None,
    out_dir: Path | str,
    registry: Registry | None = None,
    spec_name: str = SPEC_NAME,
) -> CoplacerResult:
    """The co-placer variant of Place and Wide under the spec `spec_name` (default the frozen
    japan-pool-depth-v1; COPLACER_V2, its re-registration on the core rule): the capacity of each
    paying combination given the race's realised co-placers by the spec's
    `place_wide.co_placer.rule` (module docstring), per pool x class group x odds band, with the
    rule checked against every eligible paid dividend (refunds and the races of an excluded
    status aside), per pool: a paid combination without its votes or final price is one the rule
    does not reproduce. Recorded "failed" when either pool's share is below the spec's
    `place_wide.co_placer.min_rule_match` (or the pool has no payout), or when it raised."""
    run = _preflight(registry, date_from, date_to, out_dir, spec_name)
    spec = run.spec
    bands = {p: spec["odds_bands"][p] for p in CO_PLACER}
    gate = float(spec["place_wide"]["co_placer"]["min_rule_match"])
    rule = RULES[spec["place_wide"]["co_placer"]["rule"]]

    def body(inputs: _Inputs, date_to: date):
        races = _read_races(date_from, date_to, inputs)
        eligible = races.loc[~races["status"].isin(spec["exclude_status"]).to_numpy(dtype=bool)]
        paid, rates = _paid(eligible, inputs), _rates(races)
        day = pd.to_datetime(paid["race_id"].map(eligible.set_index("race_id")["race_date"]))
        parts = []
        for a, b, votes, board in _chunks(date_from, date_to, CO_PLACER, inputs):
            rows = depth_rows(votes, board, races, band_edges=bands,
                              exclude_status=spec["exclude_status"])  # fmt: skip
            month = ((day >= pd.Timestamp(a)) & (day <= pd.Timestamp(b))).to_numpy(dtype=bool)
            parts.append(coplacer_rows(rows, paid.loc[month], rates=rates, edges=spec["edges"],
                                       share_lost=spec["share_lost"], rule=rule))  # fmt: skip
        rows = pd.concat(parts, ignore_index=True)
        dividend = rows["dividend"].to_numpy(dtype=float)
        priced = rows["class_group"].notna().to_numpy(dtype=bool)
        jra = _shares(rows, rows["jra_dividend"].to_numpy(dtype=float) == dividend)
        core = _shares(rows, rows["core_dividend"].to_numpy(dtype=float) == dividend)
        registered = jra if rule == "jra" else core
        coverage = _shares(rows, priced)
        n_paid = {p: int((rows["pool"].to_numpy() == p).sum()) for p in CO_PLACER}
        matched = rows.loc[priced].reset_index(drop=True)
        caps = {e: matched[capacity_column(e)].to_numpy(dtype=float) for e in spec["edges"]}
        table = _table(
            matched,
            by=CELL,
            quantiles=spec["quantiles"],
            capacities=lambda pool, index: {e: c[index] for e, c in caps.items()},
            edges=spec["edges"],
            stakes=spec["flat_stakes_yen"],
        )
        status = "scored" if all(registered[p] >= gate for p in CO_PLACER) else "failed"
        result = CoplacerResult(
            table, status, rule, registered, jra, core, coverage, n_paid, Path(), ""
        )
        text = _coplacer_report(result, matched, spec, run, date_from, date_to, gate)
        metrics = []
        for pool in CO_PLACER:
            if n_paid[pool]:
                metrics += [
                    FoldMetric(pool, "jra_rule_match", jra[pool], None, None, n_paid[pool]),
                    FoldMetric(pool, "core_rule_match", core[pool], None, None, n_paid[pool]),
                    FoldMetric(pool, "paid_with_votes_and_price", coverage[pool], None, None,
                               n_paid[pool]),
                ]  # fmt: skip
        metrics += _depth_metrics(table.assign(pool="co-placer:" + table["pool"]), spec)
        return metrics, status, text, result

    run_id, _, report, result = _recorded(
        run, VARIANT_COPLACER, (date_from, date_to), out_dir, "pool-depth-coplacer-japan", body
    )
    return replace(result, report_path=report, run_id=run_id)


# --------------------------------------------------------------------------- #350's measure


@dataclass(frozen=True)
class ImpactResult:
    bins: pd.DataFrame
    report_path: Path
    run_id: str
    status: str = "scored"


def run_exacta_impact(
    *,
    date_from: date,
    date_to: date | None,
    out_dir: Path | str,
    registry: Registry | None = None,
) -> ImpactResult:
    """#350's measure on history: |own_impact| of the spec's stake on each priced FORECAST
    combination, from its final votes (odds (1 - t) V / W, so own_impact sees W), by the final
    pool's size bin (exacta_impact.POOL_BIN_EDGES); one RunRecord, "failed" when every bin is
    empty (no priced exacta with its votes) or when it raised."""
    run = _preflight(registry, date_from, date_to, out_dir)
    spec = run.spec
    impact = spec["exacta_impact"]
    pool, stake, takeout = impact["pool"], float(impact["stake"]), float(impact["takeout"])
    edges = tuple(impact["pool_bin_edges"])

    def body(inputs: _Inputs, date_to: date):
        races = _read_races(date_from, date_to, inputs)
        values: list[list[np.ndarray]] = [[] for _ in edges]
        race_sets: list[set[str]] = [set() for _ in edges]
        for _, _, votes, board in _chunks(date_from, date_to, (pool,), inputs):
            rows = depth_rows(votes, board, races, band_edges={pool: spec["odds_bands"][pool]},
                              exclude_status=spec["exclude_status"])  # fmt: skip
            rows = rows.loc[(rows["combo_yen"].to_numpy() > 0) & (rows["pool_yen"].to_numpy() > 0)]
            bins = np.searchsorted(np.asarray(edges, float), rows["pool_yen"].to_numpy(), "right")
            for b, part in rows.groupby(np.maximum(bins - 1, 0), sort=False):
                v, w = part["pool_yen"].to_numpy(float), part["combo_yen"].to_numpy(float)
                odds = (1 - takeout) * v / w
                values[b].append(np.abs([
                    exacta_impact.own_impact(p, o, stake=stake, takeout=takeout)
                    for p, o in zip(v, odds, strict=True)
                ]))  # fmt: skip
                race_sets[b] |= set(part["race_id"])
        rows = []
        for i, lo in enumerate(edges):
            x = np.concatenate(values[i]) if values[i] else np.empty(0)
            q = _quantiles(x, spec["quantiles"])
            rows.append({"lo": lo, "hi": edges[i + 1] if i + 1 < len(edges) else None,
                         "n_races": len(race_sets[i]), "n_combinations": int(x.size),
                         "median_abs_impact": float(np.median(x)) if x.size else np.nan,
                         "max_abs_impact": float(x.max()) if x.size else np.nan,
                         **{f"q{qq:g}": float(v) for qq, v in zip(spec["quantiles"], q,
                                                                     strict=True)}})  # fmt: skip
        bins = pd.DataFrame(rows)
        status = "scored" if int(bins["n_combinations"].sum()) else "failed"
        text = _impact_report(bins, spec, run, date_from, date_to, status)
        metrics = [
            FoldMetric(f"pool {r['lo']}..{'inf' if r['hi'] is None else r['hi']}",
                       "median_abs_impact", r["median_abs_impact"], None, r["max_abs_impact"],
                       r["n_races"])
            for r in rows if r["n_combinations"]
        ]  # fmt: skip
        return metrics, status, text, ImpactResult(bins, Path(), "", status)

    run_id, _, report, result = _recorded(
        run, VARIANT_EXACTA, (date_from, date_to), out_dir, "exacta-impact-history-japan", body
    )
    return replace(result, report_path=report, run_id=run_id)


# --------------------------------------------------------------------------- the reports


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return "-"
        if math.isinf(value):
            return "inf"
        return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.4g}"
    return str(value)


def _markdown(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty:
        return ["(no rows)", ""]
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + " --- |" * len(frame.columns)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return [*lines, ""]


def _about(family: str, spec_hash: str, variant: str, date_from: date, date_to: date) -> list[str]:
    return [
        "Built on JRA-VAN (licensed): every figure here stays out of the repository (D21).",
        "",
        f"- family `{family}`, spec sha256 `{spec_hash}`, variant `{variant}`",
        f"- registry run `{RUN_SLOT}`",
        f"- race days {date_from} to {date_to}, development days only (holdout days dropped "
        "by the scan; no day before 2014-06-07); confirmed vote and odds records only",
        f"- git `{_commit()}`; written {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]


def _depth_report(
    result: DepthResult, spec: Mapping[str, Any], check: Mapping[str, int], counts: pd.DataFrame
) -> str:
    table = result.table
    caps = [capacity_column(e) for e in spec["edges"]]
    fits = [fits_column(e, s) for e in spec["edges"] for s in spec["flat_stakes_yen"]]
    base = ["pool", "class_group", "band", "quantile", "n_combinations", "n_races"]
    lines = [
        "# Pool depth and stake capacity, Japan (N3, #432): private report",
        "",
        *_about(
            result.family_id,
            result.spec_hash,
            "depth (Place and Wide: depth-only)",
            result.date_from,
            result.date_to,
        ),
        "## The pool's definition",
        "",
        "V is the `__pool__` row's (votes - refund_votes) x 100, W the combination's votes x 100. "
        "The `__pool__` rows of `pool_votes` and `final_odds` compared:",
        "",
        *_markdown(pd.DataFrame([check])),
        "## Combinations and races per pool",
        "",
        *_markdown(counts),
        "## Money, pool and capacity per pool x class group x odds band",
        "",
        "combo_yen and pool_yen in yen; capacity in yen at each edge, share lost "
        f"{spec['share_lost']}; inf: the edge is never eaten that far. Place and Wide: depth only.",
        "",
    ]
    for pool in spec["odds_bands"]:
        part = table.loc[table["pool"] == pool, [*base, "combo_yen", "pool_yen", *caps]]
        lines += [f"### {pool}", "", *_markdown(part)]
    lines += [
        "## The flat stake beside the capacity",
        "",
        "Per cell, the share of combinations whose capacity at edge e is at least the stake "
        f"(`fits_<e>_<stake>`, stakes {spec['flat_stakes_yen']} yen).",
        "",
    ]
    once = table.drop_duplicates(["pool", "class_group", "band"])
    for pool in spec["odds_bands"]:
        if pool in CO_PLACER:
            continue
        part = once.loc[once["pool"] == pool, ["pool", "class_group", "band", *fits]]
        lines += [f"### {pool}", "", *_markdown(part)]
    lines += [
        "## Pool-size tertiles x odds band",
        "",
        "Per pool, tertiles of the final pool over the run's races (low, mid, high).",
        "",
    ]
    tertiles = result.tertiles
    for pool in spec["odds_bands"]:
        part = tertiles.loc[
            tertiles["pool"] == pool,
            ["pool", "pool_size", "band", "quantile", "n_combinations", "n_races", "combo_yen",
             "pool_yen", *caps],
        ]  # fmt: skip
        lines += [f"### {pool}", "", *_markdown(part)]
    lines += [
        "## Pool totals at t and final (#349)",
        "",
        f"t = {spec['minutes_before']} minutes before the post; gross totals (votes x 100 and the "
        "time series' investment); at_t_share = at t / final.",
        "",
        *_markdown(result.totals),
    ]
    return "\n".join(lines)


def _coplacer_report(result, rows, spec, run, date_from, date_to, gate) -> str:
    caps = [capacity_column(e) for e in spec["edges"]]
    fits = [fits_column(e, s) for e in spec["edges"] for s in spec["flat_stakes_yen"]]
    base = ["pool", "class_group", "band", "quantile", "n_combinations", "n_races"]
    no_cap = int(np.isnan(rows[caps[0]].to_numpy(dtype=float)).sum()) if len(rows) else 0
    check = pd.DataFrame(
        {
            "pool": list(CO_PLACER),
            "paid_combinations": [result.n_paid[p] for p in CO_PLACER],
            "with_votes_and_price": [result.coverage[p] for p in CO_PLACER],
            "jra_rule_match": [result.jra_match[p] for p in CO_PLACER],
            "core_rule_match": [result.core_match[p] for p in CO_PLACER],
        }
    )
    registered = {
        "jra": "1 + (rV - sum W) / (P W_i) (`jra-stake-back-share`)",
        "core": "r(1 + (V - sum W) / (P W_i)) (`core-stake-back-share`, "
        "core.pricing.stake_back_share_dividend)",
    }[result.rule]
    lines = [
        "# Place and Wide capacity with the realised co-placers (N3, #432): private report",
        "",
        *_about(run.family_id, run.spec_hash, VARIANT_COPLACER, date_from, date_to),
        "## The rule against the paid dividends",
        "",
        f"The registered rule, per yen: {registered}, broken to 10 yen, at least 110, V net of "
        "refunds, P the race's paying combinations, r the base payout rate (boosted days are not "
        "adjusted). jra_rule_match and core_rule_match: the two rules' shares. Every eligible "
        "paid combination counts; one without its votes or final price is not reproduced. Gate: "
        f"each pool's share by the registered rule >= {gate}.",
        "",
        *_markdown(check),
        f"- matched paying combinations without a capacity (no positive stake or W): {no_cap:,}",
        f"- status: **{result.status}**",
        "",
        "## Capacity of the paying combinations per pool x class group x odds band",
        "",
    ]
    for pool in CO_PLACER:
        part = result.table.loc[result.table["pool"] == pool, [*base, "combo_yen", "pool_yen",
                                                               *caps]]  # fmt: skip
        lines += [f"### {pool}", "", *_markdown(part)]
        once = result.table.drop_duplicates(["pool", "class_group", "band"])
        lines += _markdown(once.loc[once["pool"] == pool, ["pool", "class_group", "band", *fits]])
    return "\n".join(lines)


def _impact_report(bins: pd.DataFrame, spec, run, date_from, date_to, status) -> str:
    impact = spec["exacta_impact"]
    return "\n".join(
        [
            "# The 100-yen exacta own impact on history (#350 in N3, #432): private report",
            "",
            *_about(run.family_id, run.spec_hash, VARIANT_EXACTA, date_from, date_to),
            f"|own_impact| of {impact['stake']} yen on one {impact['pool']} combination at its "
            f"final votes, takeout {impact['takeout']}, by the final pool (net of refunds) in "
            "exacta_impact.POOL_BIN_EDGES. Status: "
            f"**{status}** (failed when every bin is empty).",
            "",
            *_markdown(bins),
        ]
    )


@dataclass(frozen=True)
class VariantRun:
    """A run listed in the public report: its variant, status and registry id, and the file name
    of its private report when the listing knows it."""

    variant: str
    status: str
    run_id: str
    report: str | None = None
    family_id: str | None = None


def render_public_report(result: DepthResult, *, variants: Sequence[VariantRun] = ()) -> str:
    """The public experiment report (docs/experiment-report.md headings): what ran, the identity
    and worked arithmetic on invented numbers, every figure from the data withheld (D21).
    `variants` lists the runs to show (every run of the family, failures included); without it,
    the depth run of `result` alone."""
    runs = list(variants) or [VariantRun(VARIANT_DEPTH, "scored", result.run_id)]
    listed = [
        f"- `{v.variant}`"
        + ("" if v.family_id in (None, result.family_id) else f" (family `{v.family_id}`)")
        + f": {v.status}, registry run `{v.run_id}`"
        + ("" if v.report is None else f", private report `{v.report}`")
        + f"; every figure {WITHHELD}."
        for v in runs
    ]
    report = Path(result.report_path).name
    validated = any(v.variant == VARIANT_COPLACER and v.status == "scored" for v in runs)
    coplacer_follow_up = (
        []
        if validated
        else [
            "- #489: the Place and Wide co-placer capacity re-registered and run on a rule that "
            "reproduces the paid dividends. Until then the Place and Wide capacity is "
            "unvalidated, newcomer and maiden cells included: this study gives their depth only."
        ]
    )
    return "\n".join(
        [
            "# Pool depth and stake capacity, Japan (N3, #432)",
            "",
            f"Family `{result.family_id}`, spec sha256 `{result.spec_hash}` (frozen before the "
            "run; docs/specs/pool-depth-prereg.md). Race days from "
            f"{result.date_from.isoformat()} to the run's last day (a span of the store: "
            f"{WITHHELD}), development days only, confirmed vote and odds records only. The "
            "JRA-VAN store is licensed: every figure computed from it is "
            f"{WITHHELD}, in the private report `{report}` under `$HORSETRACK_DATA_DIR/reports/` "
            "(each run's report carries the first 16 hex digits of its registry run id).",
            "",
            "## Hypothesis",
            "",
            "Null: in every pool x class group x odds band, newcomer and maiden races included, "
            "the stake at which our own money eats half of a 5, 10 or 20% edge is below the flat "
            "stake, so betting big is not possible. A cell whose capacity quantiles sit above "
            "the flat stake shows it wrong for that cell. Nothing here tests whether an edge "
            "exists: the capacity is a property of the pool, given an edge.",
            "",
            "## What ran",
            "",
            "From the jravan store (`pool_votes`, `final_odds`, `races`, and `dividends` for the "
            "co-placer variant), streamed a year at a time: per priced combination, the money on "
            "it (W, its final votes x 100), the pool (V, the `__pool__` votes net of refunds x "
            "100), its class group and its final-odds band. Per cell, the quantiles of W, of V "
            "(each race once) and of the capacity stake at each edge, the share of combinations "
            "whose capacity reaches each flat stake, the same by pool-size tertile and odds "
            "band, and the Win, Place and Quinella pool totals two minutes before the post "
            f"against the final. Figures: {WITHHELD}.",
            "",
            "## The identity and worked arithmetic",
            "",
            "A single-winner pool pays rate x V / W per yen. Our stake s goes into both, so a "
            "bet worth 1 + e is worth (1 + e)(V + s)W / (V(W + s)) once it is in. The capacity "
            "stake solves that for 1 + e(1 - share_lost): s = (e - k)VW / ((1 + k)V - (1 + e)W), "
            "k = e(1 - share_lost); at share_lost 1/2, s = kVW / ((1 + k)V - (1 + e)W) with "
            "k = e/2. No breakage and no floor.",
            "",
            "- Invented pool of 1,000,000 yen with 10,000 on the combination: the capacity is "
            "246.43 yen at e = 0.05, 481.23 at e = 0.10 and 919.12 at e = 0.20.",
            "- The same 10,000 yen in a pool of 1,000,000,000 yen: 476.20 at e = 0.10, close to "
            "the deep-pool limit W k / (1 + k) = 10,000 x 0.05 / 1.05. A shallower pool allows "
            "more, because the rest of the pool pays for part of our impact.",
            "- The dividend itself (`models.decision.own_stake_dividend`): floor(100 x rate x "
            "(V + s) / (W + s)) to 10 yen, at least 110. Pool 1,000,000, W 10,000, rate 0.725: "
            "7,250 per 100 yen with no stake, 6,590 with a stake of 1,000.",
            "- Place and Wide (co-placer variant): the pre-registered rule pays "
            "1 + (rV - sum W) / (P W_i) per yen, the sum over the P paying combinations. With "
            "A = rV - sum W and c = (1 + k) / (1 + e) the capacity is s = (1 - c)(A + P W_i) / "
            "((c d0 - 1) P + 1 - r), d0 the dividend with no stake; at P = 1 it is the "
            "single-winner identity. The run checks the rule against every eligible paid "
            "dividend, Place and Wide apart, a paid combination without its votes or final price "
            "counting as not reproduced, and is recorded failed when either pool's share is "
            f"below the spec's (95%); the shares themselves are {WITHHELD}.",
            "- The exacta (#350): 100 yen on one combination moves its dividend by "
            "(V + s) / V x W / (W + s) - 1, with W the combination's final money.",
            "",
            "## Variants tried",
            "",
            *listed,
            "",
            "## Alternatives rejected and why",
            "",
            "- TIERCE and BRACKET_QUINELLA: not in the cells. The trifecta has several times "
            "the trio's combinations, too many for exact quantiles in memory on this machine; "
            "the bracket quinella needs the horse-to-bracket map.",
            "- Quantile sketches: rejected for exact quantiles over the six pools.",
            "- Capacity with breakage and the 110-yen floor: rejected; breakage makes the stake "
            "a step function of the price, and the continuous stake is the stated quantity.",
            "- The gross pool for V: rejected; refunded votes do not pay.",
            "- The rule of `core.pricing.stake_back_share_dividend`, r(1 + (V - sum W) / "
            "(P W_i)), the stake-back-share rule as `config/jra.yaml` states it: not the "
            "pre-registered co-placer rule. Its agreement with the paid dividends is a "
            f"diagnostic in the private report ({WITHHELD}).",
            "",
            "## What this rules out",
            "",
            f"Which cells the flat stake fits, and so where betting big is possible: {WITHHELD}. "
            "By the identity alone, whatever the pool, a stake below W k / (1 + k), k = e / 2, "
            "never eats half of the edge: about 2.4% of the money already on the combination at "
            "e = 0.05, 4.8% at 0.10 and 9.1% at 0.20.",
            "",
            "## Follow-ups filed",
            "",
            "Filed as proposed tickets:",
            "",
            *coplacer_follow_up,
            "- #490: the trifecta (TIERCE) and bracket-quinella depth, with a streaming quantile.",
            "- #491: the capacity beside each leaderboard row.",
            "",
            "Publishable figures need the logged public days (#130).",
            "",
        ]
    )


# --------------------------------------------------------------------------- the command

pool_depth_app = typer.Typer(no_args_is_help=True, help="Pool depth and stake capacity (#432).")


def _last_store_day() -> date:
    """The last race day the jravan races table holds, from its footer statistics (no row)."""
    meta = pq.ParquetFile(_store_path("races")).metadata
    days = [
        meta.row_group(g).column(c).statistics.max
        for g in range(meta.num_row_groups)
        for c in range(meta.num_columns)
        if meta.row_group(g).column(c).path_in_schema == "race_date"
    ]
    return max(days)


SPEC_HELP = f"the spec file under config/ ({', '.join(SPECS)})"


@pool_depth_app.command("freeze")
def freeze_command(
    spec_name: Annotated[str, typer.Option("--spec", help=SPEC_HELP)] = SPEC_NAME,
) -> None:
    """Freeze config/<spec>.yaml in the japan registry (irreversible; again is a no-op)."""
    try:
        family, spec = load_spec(spec_name)
        spec_hash = default_registry(COUNTRY).freeze_family(family, spec)
    except (ValueError, FamilyAlreadyFrozen) as exc:
        typer.echo(f"not frozen: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"frozen family={family} spec_hash={spec_hash}")


@pool_depth_app.command("run")
def run_command(
    out: Annotated[Path | None, typer.Option(help="report folder, outside the repo")] = None,
    public: Annotated[bool, typer.Option(help="write docs/experiments/pool-depth-japan.md")] = True,
    spec_name: Annotated[str, typer.Option("--spec", help=SPEC_HELP)] = SPEC_NAME,
) -> None:
    """Run the variants the spec registers (SPECS: pool_depth the three, pool_depth_coplacer_v2
    the co-placer one) on every development day from the spec's date_from to the store's last
    race day, record each run, rewrite the public report from every run of the study's families
    and update their files and index entries in the committed summary. The freeze is checked
    before the store is touched; the last race day is looked up inside each recorded attempt.
    Prints run identities and file names only; exits 1 when a run failed."""
    try:
        _, spec = load_spec(spec_name)
    except ValueError as exc:
        typer.echo(f"not run: {exc}", err=True)
        raise typer.Exit(1) from None
    registry = default_registry(COUNTRY)
    folder = paths.reports_dir() if out is None else paths.outside_repo(out, what="out")
    date_from = date.fromisoformat(spec["date_from"])
    try:
        _preflight(registry, date_from, None, folder, spec_name)
    except (NotPreRegistered, LicensedOutputInRepo, ValueError) as exc:
        typer.echo(f"not run: {exc}", err=True)
        raise typer.Exit(1) from None
    runners = {
        VARIANT_DEPTH: run_depth,
        VARIANT_COPLACER: functools.partial(run_coplacer, spec_name=spec_name),
        VARIANT_EXACTA: run_exacta_impact,
    }
    reports, failed = {}, False
    for variant in SPECS[spec_name]:
        try:
            result = runners[variant](
                date_from=date_from, date_to=None, out_dir=folder, registry=registry
            )
        except (NotPreRegistered, RecordRefused) as exc:
            typer.echo(f"{variant}: not run ({exc})", err=True)
            raise typer.Exit(1) from None
        except Exception as exc:  # recorded as failed by the run itself
            typer.echo(f"{variant}: failed ({type(exc).__name__}), recorded as failed", err=True)
            failed = True
            continue
        status = getattr(result, "status", "scored")
        failed = failed or status == "failed"
        reports[result.run_id] = result.report_path.name
        typer.echo(f"{variant}: {status} run_id={result.run_id} report={result.report_path.name}")
    families = [load_spec(name)[0] for name in SPECS]
    depth = _latest_depth(registry, reports)
    if public and depth is not None:
        doc = env.repo_root() / "docs" / "experiments" / "pool-depth-japan.md"
        runs = family_runs(registry, families, reports)
        write_report(doc, render_public_report(depth, variants=runs), sources=())
        typer.echo(f"wrote {doc.relative_to(env.repo_root())}")
    update_summary(registry, families, summary_dir(COUNTRY))
    typer.echo(f"updated the committed summary of {', '.join(families)}")
    if failed:
        raise typer.Exit(1)


def _latest_depth(registry: Registry, reports: Mapping[str, str]) -> DepthResult | None:
    """The study's latest scored depth run, as the public report names it (no table), or None:
    its private report is the one `reports` names, else `pool-depth-japan-<id16>.md`."""
    family, _ = load_spec(SPEC_NAME)
    scored = [
        r for r in registry.records(family) if r.variant == VARIANT_DEPTH and r.status == "scored"
    ]
    if not scored:
        return None
    record = scored[-1]
    run_id = _run_id(record)
    report = Path(reports.get(run_id, f"pool-depth-japan-{run_id[:16]}.md"))
    days = [date.fromisoformat(record.params[k]) for k in ("date_from", "date_to")]
    return DepthResult(pd.DataFrame(), report, family, record.params["spec_hash"], run_id, *days)


def family_runs(
    registry: Registry, families: Sequence[str], reports: Mapping[str, str] | None = None
) -> list[VariantRun]:
    """Every run of `families` in the registry, in append order, failures included, with the
    private report name of those `reports` (run id -> file name) knows."""
    reports = {} if reports is None else reports
    out = []
    for record in registry.records():
        if record.family_id in families:
            run_id = _run_id(record)
            out.append(VariantRun(record.variant, record.status, run_id, reports.get(run_id),
                                  record.family_id))  # fmt: skip
    return out


_INDEX_ENTRY = re.compile(r"^- \[([^\]]+)\]\(\1\.md\)$", re.MULTILINE)


class _Records:
    """Records for `write_summary` to render (it reads `records()` only)."""

    def __init__(self, records: Sequence[RunRecord]) -> None:
        self._records = list(records)

    def records(self, family_id: str | None = None) -> list[RunRecord]:
        return [r for r in self._records if family_id is None or r.family_id == family_id]


def update_summary(registry: Registry, families: Sequence[str], directory: Path | str) -> None:
    """The committed summary in `directory` with `families`' files and index entries written by
    `write_summary` from their records in `registry`, and every other family's file and index
    entry kept: the index is `write_summary`'s rendering of the families it already lists plus
    these (stand-in records name the others; their files are not written)."""
    directory = Path(directory)
    index = directory / "README.md"
    listed = _INDEX_ENTRY.findall(index.read_text(encoding="utf-8")) if index.is_file() else []
    own = [r for r in registry.records() if r.family_id in families]
    names = [*dict.fromkeys(r.family_id for r in own), "README"]
    stand_ins = [
        RunRecord("P0", name, "-", 1, "-", {}, (), "-", {}, {}, (), "failed")
        for name in dict.fromkeys(listed)
        if name not in families
    ]
    with tempfile.TemporaryDirectory() as tmp:
        write_summary(_Records([*own, *stand_ins]), tmp)
        for name in names:
            text = (Path(tmp) / f"{name}.md").read_text(encoding="utf-8")
            write_report(directory / f"{name}.md", text, sources=())
