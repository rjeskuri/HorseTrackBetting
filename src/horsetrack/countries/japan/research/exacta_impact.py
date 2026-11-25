"""`horsetrack japan exacta-impact`: the ¥100 exacta own-impact check (ticket #283).

#131 assumed that ¥100 per race has negligible own price impact on the exacta pool. This measures
it from the payout identity on the logged exacta pool totals (#130): the dividend per ¥100 is
100 (1 - t) P / W, P the pool's money and W the money on the combination. Adding a stake s to each
of n combinations (the winner among them) gives 100 (1 - t) (P + n s) / (W + s), so the relative
own impact is (P + n s) / P x W / (W + s) - 1, with W = (1 - t) P / O for decimal odds O.

Source: `$HORSETRACK_DATA_DIR/odds_snapshots/jra/<day>/*.jsonl.gz`, the JRA public odds log, rows of
the FORECAST pool (the exacta, tier 2). Development days only: a holdout day is never opened
(`is_holdout_day`, D14). Per race the latest pool total and the latest price of each combination
are used; the impact of `stake` on each combination alone is grouped by pool size. The command
records the run in the japan registry and prints aggregates only; the same run again (same logs,
same commit) adds nothing.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from datetime import date
from typing import Any

import typer

from horsetrack.core import paths
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import FoldMetric, RecordRefused, RunRecord, default_registry
from horsetrack.countries.japan import odds_log

TAKEOUT = 0.25  # the exacta's takeout
STAKE = 100
POOL_BIN_EDGES = (0, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000)
POOL = "FORECAST"  # the log's pool name for the exacta
SOURCE = "jra"
COUNTRY = "japan"
TICKET = 283
HYPOTHESIS_ID = "P283"
FAMILY_ID = "exacta-own-impact"
VARIANT = "payout-identity"
METRIC = "median_abs_impact"
FOLD_SCHEME = "development-days"


def _check_takeout(takeout: float) -> None:
    if not (isinstance(takeout, (int, float)) and math.isfinite(takeout) and 0 <= takeout < 1):
        raise ValueError(f"takeout must be in [0, 1), got {takeout!r}")


def _positive_finite(name: str, value: float) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")


def stake_on_combination(pool: float, odds: float, takeout: float = TAKEOUT) -> float:
    """The money W on a combination at decimal odds `odds` in a pool of `pool`:
    (1 - takeout) x pool / odds."""
    _positive_finite("pool", pool)
    _positive_finite("odds", odds)
    _check_takeout(takeout)
    return (1 - takeout) * pool / odds


def own_impact(
    pool: float,
    odds: float,
    stake: float = STAKE,
    n_combinations: int = 1,
    takeout: float = TAKEOUT,
) -> float:
    """The relative change in a combination's dividend when `stake` is added to each of
    `n_combinations` combinations (it among them): (P + n s) / P x W / (W + s) - 1. Negative when
    the dividend falls."""
    if n_combinations < 1:
        raise ValueError(f"n_combinations must be at least 1, got {n_combinations!r}")
    w = stake_on_combination(pool, odds, takeout)
    impact = (pool + n_combinations * stake) / pool * w / (w + stake) - 1
    if not math.isfinite(impact):
        raise ValueError(f"the impact is not finite for pool={pool!r} odds={odds!r}")
    return impact


def _bin_of(pool: float, edges: tuple[float, ...]) -> int:
    index = 0
    for i, edge in enumerate(edges):
        if pool >= edge:
            index = i
    return index


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def _day_folders() -> list[tuple[date, list[Any]]]:
    root = paths.data_dir() / "odds_snapshots" / SOURCE
    if not root.is_dir():
        return []
    days = []
    for folder in sorted(root.iterdir()):
        try:
            day = date.fromisoformat(folder.name)
        except ValueError:
            continue
        logs = sorted(folder.glob("*.jsonl.gz")) if folder.is_dir() else []
        if logs:
            days.append((day, logs))
    return days


def _merge_latest(best: dict[tuple, dict[str, Any]], newer: dict[tuple, dict[str, Any]]) -> None:
    """Fold the latest rows of a later member (`newer`) into `best`: the later snapshot_utc wins,
    and at the same instant the later row in the log wins, as in `odds_log.latest_*` (whose rows
    all have a readable snapshot_utc)."""
    for key, row in newer.items():
        current = best.get(key)
        if current is None or odds_log._instant(row["snapshot_utc"]) >= odds_log._instant(
            current["snapshot_utc"]
        ):
            best[key] = row


def _races_of(log_path: Any) -> tuple[dict[str, float | None], dict[str, list[float]], bool]:
    """(latest FORECAST pool by race, latest valid odds by race, whether any FORECAST row).

    The log is streamed one decompressed member at a time (docs/specs/odds-log.md: a tier-2 day
    runs to millions of rows). Only a member's FORECAST rows are kept, and across members only the
    latest row per key."""
    latest_total: dict[tuple, dict[str, Any]] = {}
    latest_price: dict[tuple, dict[str, Any]] = {}
    seen = False
    for member in odds_log.iter_members(log_path):
        rows = [r for r in member if r.get("pool") == POOL]
        if not rows:
            continue
        seen = True
        _merge_latest(latest_total, odds_log.latest_totals(rows))
        _merge_latest(latest_price, odds_log.latest_prices(rows))
    totals: dict[str, float | None] = {}
    for (race_id, _), row in latest_total.items():
        totals[race_id] = _number(row.get("investment"))
    odds: dict[str, list[float]] = {}
    for (race_id, _, _), row in latest_price.items():
        odds.setdefault(race_id, [])
        value = _number(row.get("odds"))
        if value is not None:
            odds[race_id].append(value)
    return totals, odds, seen


def summarize_impact(
    *,
    stake: float = STAKE,
    takeout: float = TAKEOUT,
    bin_edges: tuple[float, ...] = POOL_BIN_EDGES,
) -> dict[str, Any]:
    """The own impact of `stake` on each logged exacta combination, by pool size, development days
    only. See the module docstring; the keys are days_read, days_with_forecast,
    days_holdout_excluded, races, races_skipped_no_pool and bins."""
    edges = tuple(bin_edges)
    impacts: list[list[float]] = [[] for _ in edges]
    races_in = [0] * len(edges)
    days_read = days_with_forecast = days_holdout = races = skipped = 0
    for day, logs in _day_folders():
        if is_holdout_day(COUNTRY, day):
            days_holdout += 1
            continue
        days_read += 1
        any_forecast = False
        for log_path in logs:
            totals, odds, seen = _races_of(log_path)
            any_forecast = any_forecast or seen
            for race_id, race_odds in odds.items():
                pool = totals.get(race_id)
                if pool is None:
                    skipped += 1
                    continue
                if not race_odds:
                    continue
                races += 1
                index = _bin_of(pool, edges)
                races_in[index] += 1
                impacts[index] += [
                    abs(own_impact(pool, o, stake=stake, takeout=takeout)) for o in race_odds
                ]
        days_with_forecast += any_forecast
    bins = []
    for i, lo in enumerate(edges):
        values = impacts[i]
        bins.append(
            {
                "lo": lo,
                "hi": edges[i + 1] if i + 1 < len(edges) else None,
                "n_races": races_in[i],
                "n_combinations": len(values),
                "median_abs_impact": statistics.median(values) if values else None,
                "max_abs_impact": max(values) if values else None,
            }
        )
    return {
        "days_read": days_read,
        "days_with_forecast": days_with_forecast,
        "days_holdout_excluded": days_holdout,
        "races": races,
        "races_skipped_no_pool": skipped,
        "bins": bins,
    }


def _log_build() -> str:
    """The logs' identity for the registry: what they hold, then a digest of their names and
    sizes after the `@` (more logged days are a refresh of the same table)."""
    digest = hashlib.sha256()
    for day, logs in _day_folders():
        if is_holdout_day(COUNTRY, day):
            continue
        for log_path in logs:
            digest.update(f"{day.isoformat()}/{log_path.name}:{log_path.stat().st_size}\n".encode())
    return f"odds_snapshots/{SOURCE}/{POOL}@{digest.hexdigest()}"


def _commit() -> str:
    try:
        return _git_sha()
    except RuntimeError:
        return "unknown"


def _metrics(result: dict[str, Any]) -> tuple[FoldMetric, ...]:
    out = []
    for b in result["bins"]:
        if not b["n_races"]:
            continue
        hi = "inf" if b["hi"] is None else b["hi"]
        out.append(
            FoldMetric(
                fold=f"pool {b['lo']}..{hi}",
                metric=METRIC,
                value=b["median_abs_impact"],
                lo=None,
                hi=b["max_abs_impact"],
                n=b["n_races"],
            )
        )
    return tuple(out)


def exacta_impact_command() -> None:
    """Measure the ¥100 exacta own impact on logged development days and record the run."""
    result = summarize_impact()
    typer.echo(
        f"days_read={result['days_read']} days_with_forecast={result['days_with_forecast']} "
        f"days_holdout_excluded={result['days_holdout_excluded']} races={result['races']} "
        f"races_skipped_no_pool={result['races_skipped_no_pool']}"
    )
    for b in result["bins"]:
        typer.echo(" ".join(f"{k}={v}" for k, v in b.items()))
    record = RunRecord(
        hypothesis_id=HYPOTHESIS_ID,
        family_id=FAMILY_ID,
        variant=VARIANT,
        ticket=TICKET,
        git_sha=_commit(),
        data_builds={"odds_log": _log_build()},
        sources=(SOURCE,),
        fold_scheme=FOLD_SCHEME,
        components={},
        params={"stake": STAKE, "takeout": TAKEOUT, "bin_edges": list(POOL_BIN_EDGES)},
        metrics=_metrics(result),
        status="scored",
    )
    registry = default_registry(COUNTRY)
    if record in registry.records(FAMILY_ID):
        typer.echo("this run is already recorded (same logs and commit); nothing added")
        return
    try:
        run_id = registry.append(record)
    except RecordRefused as refused:
        typer.echo(f"not recorded: {refused}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"recorded run_id={run_id}")
