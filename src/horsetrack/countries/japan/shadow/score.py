"""Score shadow records against the final dividends (ticket #564).

``score_shadow(records, dividends)`` puts one notional flat 100 yen on every scored record and,
per (pool, tau_min), sets the return the model claimed at the decision-time odds beside the return
the dividends paid. A record is scored when it is version 2 with a ``p_model``, status ``ok`` with
a ``board_odds``, labelled ``development``, and its race and pool have dividend rows; a combination
with a refund row is left out (its stake comes back). Records come in as ``ShadowLog.read()``
returns them, which refuses a holdout day, and a holdout-labelled record is skipped here too:
nothing in this module reads a file or opens the holdout. The rules are in
docs/specs/shadow-log.md ("Scoring").
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from horsetrack.countries.japan.bets.log import _in_store_order
from horsetrack.countries.japan.bets.settle import _paying_rows
from horsetrack.countries.japan.shadow.log import DEVELOPMENT, RECORD_VERSION, record_version


@dataclass(frozen=True)
class PoolScore:
    pool: str
    tau_min: int
    n: int  # records scored
    hits: int  # of them, those whose combination paid
    expected_per_100: float  # 100 x mean(p_model x board_odds)
    realised_per_100: float  # dividends paid (yen per 100 yen) summed over the records / n


def _scored(record: dict[str, Any]) -> bool:
    return (
        record_version(record) == RECORD_VERSION
        and record.get("p_model") is not None
        and record.get("status") == "ok"
        and record.get("board_odds") is not None
        and record.get("day_status") == DEVELOPMENT
    )


def score_shadow(
    records: Iterable[dict[str, Any]], dividends: pd.DataFrame
) -> dict[tuple[str, int], PoolScore]:
    """(pool, tau_min) -> PoolScore of the scored records; a (pool, tau) with none has no entry.
    ``dividends`` has ``bets.settle.DIVIDEND_COLUMNS`` (race_id, pool, combination, dividend,
    is_refund), the dividend in yen per 100 yen. Ordered pools match in finishing order, the
    others in any order (both sides are read in the store's order)."""
    scored = [r for r in records if _scored(r)]
    paying = _paying_rows(dividends, {(r["race_id"], r["pool"]) for r in scored})
    sums: dict[tuple[str, int], list[float]] = {}  # [n, hits, sum of p x odds, dividends paid]
    for r in scored:
        rows = paying.get((r["race_id"], r["pool"]))
        if rows is None:
            continue  # unsettled: no dividend row of its race and pool yet
        paid = rows.get(_in_store_order(r["pool"], r["combination"]))
        if paid is not None and paid[1]:
            continue  # refunded: not measured
        if paid is not None and paid[0] is None:
            raise ValueError(f"{r['race_id']} {r['pool']} {r['combination']}: no dividend")
        dividend = 0.0 if paid is None else paid[0]
        acc = sums.setdefault((r["pool"], r["tau_min"]), [0, 0, 0.0, 0.0])
        acc[0] += 1
        acc[1] += dividend > 0
        acc[2] += float(r["p_model"]) * float(r["board_odds"])
        acc[3] += dividend
    return {
        (pool, tau): PoolScore(pool, tau, int(n), int(hits), 100 * value / n, paid / n)
        for (pool, tau), (n, hits, value, paid) in sums.items()
    }
