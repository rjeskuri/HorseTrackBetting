"""The top of the private leaderboard beside the day's cap (#455).

``shortlist`` takes the top ranked, measured options of ``core.eval.leaderboard.rank``;
``shortlist_text`` prints them one block each, with the day's staked total, the cap left, the flat
stake and what the owner has played on each option. The private leaderboard holds withheld
figures (D21): the text is for the owner's terminal and is never written to a file.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from horsetrack.core.eval.leaderboard import LeaderboardRow, RankedRow, rank, shrink
from horsetrack.countries.japan.bets.log import CapCheck, Limits

LOOK_AHEAD = "look_ahead"


def shortlist(rows: Sequence[LeaderboardRow], *, top: int) -> list[RankedRow]:
    """The ``top`` best ranked rows of ``rank(shrink(rows))``, in rank order: the shrinkage runs
    over every row, as the private leaderboard's does, before the list is cut. Unmeasured and
    descriptive rows are unranked and never on it."""
    return [r for r in rank(shrink(rows)) if r.rank is not None][: max(top, 0)]


def _yen(amount: float) -> str:
    return f"¥{amount:,.0f}"


def _figure(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.1f}"


def _played(option_id: str, played: pd.DataFrame | None) -> str:
    if played is None or played.empty:
        return "none"
    mine = played.loc[played["option_id"] == option_id]
    if mine.empty:
        return "none"
    got = mine.iloc[0]
    line = (
        f"{int(got['n_bets'])} bets, {_yen(got['staked_yen'])} staked, "
        f"return per ¥100 {_figure(got['return_per_100'])}"
    )
    extra = [
        f"{int(got[column])} {name}"
        for column, name in (("n_refunded", "refunded"), ("n_pending", "pending"))
        if int(got[column])
    ]
    return f"{line} ({', '.join(extra)})" if extra else line


def _block(item: RankedRow, played: pd.DataFrame | None) -> str:
    row = item.row
    tag = "  [look-ahead]" if row.selection == LOOK_AHEAD else ""
    head = f"{item.rank:>3}. {row.option_id}  {row.pool or '-'}  {row.segment}  {row.rule}{tag}"
    # as the private leaderboard: each bound is one-sided at the level, not a two-sided interval
    if row.ci_lo is not None and row.ci_hi is not None:
        ci, sided = f"CI [{row.ci_lo:.1f}, {row.ci_hi:.1f}]", "each bound one-sided"
    elif row.ci_lo is not None:
        ci, sided = f"CI ≥ {row.ci_lo:.1f}", "one-sided"
    elif row.ci_hi is not None:
        ci, sided = f"CI ≤ {row.ci_hi:.1f}", "one-sided"
    else:
        ci, sided = "CI n/a", None
    if sided is not None and row.ci_level is not None:
        ci = f"{ci} ({sided} {row.ci_level * 100:g}%)"
    figures = (
        f"selection {row.selection.replace('_', '-')}; "
        f"shrunk return per ¥100 {_figure(row.shrunk_per_100)}; {ci}; "
        f"variants searched {row.variants_searched:,}"
    )
    return f"{head}\n     {figures}\n     played: {_played(row.option_id, played)}"


def shortlist_text(
    ranked: list[RankedRow],
    *,
    day: date,
    check: CapCheck,
    limits: Limits,
    played: pd.DataFrame | None,
) -> str:
    """The head (staked total, cap left, flat stake) and one block per option, in rank order."""
    remaining = check.remaining_yen
    if remaining >= 0:
        left = f"remaining {_yen(remaining)}"
    else:
        left = f"remaining ¥0, {_yen(-remaining)} past the cap"
    flat = "not set" if limits.flat_stake_yen is None else _yen(limits.flat_stake_yen)
    lines = [
        f"{day.isoformat()}: staked {_yen(check.staked_yen)} of the {_yen(limits.day_cap_yen)} "
        f"day cap, {left}",
        f"flat stake: {flat}",
        "",
    ]
    if not ranked:
        lines.append("no measured, ranked options on the leaderboard")
    lines.extend(_block(item, played) for item in ranked)
    return "\n".join(lines)
