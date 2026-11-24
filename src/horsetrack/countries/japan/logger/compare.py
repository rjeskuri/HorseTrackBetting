"""Compare a day's odds snapshot log with HKTrack's, for the logger cutover (ticket #28).

`compare_logs(ours, theirs)` takes rows as `odds_log.read_rows` returns them and reports: the races
and pools in each log and the differences, the snapshot count per race and pool in each, and the
odds agreement on their price rows. It reports and does not judge: the pass threshold is the
owner's call.

HKTrack's known bugs are normalised first, on their rows only: a `source_update_utc` more than 1 h
after its `snapshot_utc` was dated by the race day and is moved back one day. On both sides a
`status_raw` other than "SCR" (time text, "", 最終オッズ) counts as null.

Each of their price rows is matched to the nearest of our rows with the same race, pool and
combination, captured within 20 s, whose source update agrees with their normalised one (null with
null; otherwise within 20 s, which is equality for JRA's minute-precision printed time).

`render` writes markdown that carries the first 20 mismatched rows with their prices: it belongs
under $HORSETRACK_DATA_DIR, not in git.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import typer

from horsetrack.core import paths
from horsetrack.countries.japan.odds_log import POOL_TOTAL, _instant, read_rows

MATCH_WINDOW = timedelta(seconds=20)
DAY_OFF_THRESHOLD = timedelta(hours=1)
MAX_MISMATCHES = 20
SCRATCHED = "SCR"

Row = Mapping[str, Any]
PoolKey = tuple[str, str]


@dataclass(frozen=True)
class Mismatch:
    """One of their price rows matched to one of ours, with differing odds, odds_hi or status."""

    race_id: str
    pool: str
    combination: str
    ours: dict[str, Any]
    theirs: dict[str, Any]


@dataclass(frozen=True)
class Comparison:
    races_ours: frozenset[str]
    races_theirs: frozenset[str]
    races_only_ours: frozenset[str]
    races_only_theirs: frozenset[str]
    pools_ours: frozenset[PoolKey]
    pools_theirs: frozenset[PoolKey]
    pools_only_ours: frozenset[PoolKey]
    pools_only_theirs: frozenset[PoolKey]
    snapshots_ours: dict[PoolKey, int]
    snapshots_theirs: dict[PoolKey, int]
    compared: int
    matched: int
    equal: int
    mismatches: tuple[Mismatch, ...]


def _status(row: Row) -> str | None:
    return SCRATCHED if row.get("status_raw") == SCRATCHED else None


def _their_source_update(row: Row) -> datetime | None:
    """Their source_update_utc, moved back a day when it lands more than 1 h after the capture."""
    source = _instant(row.get("source_update_utc"))
    snapshot = _instant(row.get("snapshot_utc"))
    if source is not None and snapshot is not None and source - snapshot > DAY_OFF_THRESHOLD:
        source -= timedelta(days=1)
    return source


def _same_source_update(ours: datetime | None, theirs: datetime | None) -> bool:
    """Null matches null. JRA prints the update to the minute, so on real logs the window is plain
    equality; it only absorbs sub-minute differences in how a source time was derived."""
    if ours is None or theirs is None:
        return ours is theirs
    return abs(ours - theirs) <= MATCH_WINDOW


def _snapshot_counts(rows: Iterable[Row]) -> dict[PoolKey, int]:
    seen: dict[PoolKey, set] = defaultdict(set)
    for row in rows:
        raw = row.get("snapshot_utc")
        seen[(row.get("race_id"), row.get("pool"))].add(_instant(raw) or raw)
    return {key: len(instants) for key, instants in seen.items()}


def _equal(ours: Row, theirs: Row) -> bool:
    return (
        ours.get("odds") == theirs.get("odds")
        and ours.get("odds_hi") == theirs.get("odds_hi")
        and _status(ours) == _status(theirs)
    )


def compare_logs(ours: list[dict], theirs: list[dict]) -> Comparison:
    """Compare our day's log with HKTrack's; neither list nor its rows is changed."""
    races_ours = frozenset(row.get("race_id") for row in ours)
    races_theirs = frozenset(row.get("race_id") for row in theirs)
    pools_ours = frozenset((row.get("race_id"), row.get("pool")) for row in ours)
    pools_theirs = frozenset((row.get("race_id"), row.get("pool")) for row in theirs)

    # Our price rows by (race, pool, combination): (snapshot, position, source update), sorted by
    # time; positions are unique, so sorting never compares source updates.
    index: dict[tuple, list[tuple[datetime, int, datetime | None]]] = defaultdict(list)
    for position, row in enumerate(ours):
        if row.get("combination") == POOL_TOTAL:
            continue
        snapshot = _instant(row.get("snapshot_utc"))
        if snapshot is None:
            continue
        key = (row.get("race_id"), row.get("pool"), row.get("combination"))
        index[key].append((snapshot, position, _instant(row.get("source_update_utc"))))
    for captures in index.values():
        captures.sort()

    compared = matched = equal = 0
    mismatches: list[Mismatch] = []
    for row in theirs:
        if row.get("combination") == POOL_TOTAL:
            continue
        compared += 1
        snapshot = _instant(row.get("snapshot_utc"))
        if snapshot is None:
            continue
        key = (row.get("race_id"), row.get("pool"), row.get("combination"))
        captures = index.get(key)
        if not captures:
            continue
        source = _their_source_update(row)
        at = bisect.bisect_left(captures, (snapshot - MATCH_WINDOW,))
        best: tuple[timedelta, int] | None = None
        while at < len(captures) and captures[at][0] <= snapshot + MATCH_WINDOW:
            when, position, our_source = captures[at]
            gap = abs(when - snapshot)
            if _same_source_update(our_source, source) and (best is None or gap < best[0]):
                best = (gap, position)
            at += 1
        if best is None:
            continue
        matched += 1
        mine = ours[best[1]]
        if _equal(mine, row):
            equal += 1
        elif len(mismatches) < MAX_MISMATCHES:
            mismatches.append(Mismatch(*key, ours=dict(mine), theirs=dict(row)))

    return Comparison(
        races_ours=races_ours,
        races_theirs=races_theirs,
        races_only_ours=races_ours - races_theirs,
        races_only_theirs=races_theirs - races_ours,
        pools_ours=pools_ours,
        pools_theirs=pools_theirs,
        pools_only_ours=pools_ours - pools_theirs,
        pools_only_theirs=pools_theirs - pools_ours,
        snapshots_ours=_snapshot_counts(ours),
        snapshots_theirs=_snapshot_counts(theirs),
        compared=compared,
        matched=matched,
        equal=equal,
        mismatches=tuple(mismatches),
    )


# ---------------------------------------------------------------- the report


def _sort_key(value: Any) -> tuple:
    items = value if isinstance(value, tuple) else (value,)
    return tuple("" if item is None else str(item) for item in items)


def _rate(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def _cell(value: Any) -> str:
    return "" if value is None else str(value).replace("|", "\\|")


def _id_list(values: Iterable[Any]) -> str:
    return ", ".join(f"`{_cell(v)}`" for v in sorted(values, key=_sort_key)) or "none"


def _pool_list(values: Iterable[PoolKey]) -> list[str]:
    return [f"- `{_cell(r)}` {_cell(p)}" for r, p in sorted(values, key=_sort_key)] or ["- none"]


def render(comparison: Comparison) -> str:
    """The comparison as markdown. It carries prices (the mismatch rows): keep it under
    $HORSETRACK_DATA_DIR, not in git."""
    c = comparison
    lines = [
        "# Odds log comparison: ours against HKTrack's",
        "",
        "## Races",
        "",
        f"- ours: {len(c.races_ours)}, theirs: {len(c.races_theirs)}",
        f"- only in ours: {_id_list(c.races_only_ours)}",
        f"- only in theirs: {_id_list(c.races_only_theirs)}",
        "",
        "## Pools",
        "",
        f"- ours: {len(c.pools_ours)}, theirs: {len(c.pools_theirs)}",
        "",
        "Only in ours:",
        "",
        *_pool_list(c.pools_only_ours),
        "",
        "Only in theirs:",
        "",
        *_pool_list(c.pools_only_theirs),
        "",
        "## Snapshots per race and pool",
        "",
        "| race | pool | ours | theirs |",
        "| --- | --- | ---: | ---: |",
    ]
    for key in sorted(set(c.snapshots_ours) | set(c.snapshots_theirs), key=_sort_key):
        race_id, pool = key
        ours, theirs = c.snapshots_ours.get(key, 0), c.snapshots_theirs.get(key, 0)
        lines.append(f"| {_cell(race_id)} | {_cell(pool)} | {ours} | {theirs} |")
    lines += [
        "",
        "## Odds agreement",
        "",
        f"Their price rows matched to ours by race, pool, combination and source update, captured"
        f" within {MATCH_WINDOW.seconds} s; equal means odds, odds_hi and status agree.",
        "",
        "| | rows | rate |",
        "| --- | ---: | ---: |",
        f"| compared (their price rows) | {c.compared} | |",
        f"| matched | {c.matched} | {_rate(c.matched, c.compared)} of compared |",
        f"| unmatched | {c.compared - c.matched} | {_rate(c.compared - c.matched, c.compared)}"
        " of compared |",
        f"| equal | {c.equal} | {_rate(c.equal, c.matched)} of matched |",
        f"| unequal | {c.matched - c.equal} | {_rate(c.matched - c.equal, c.matched)} of matched |",
        "",
        f"## Mismatches (up to the first {MAX_MISMATCHES}, their log order; carries prices)",
        "",
    ]
    if not c.mismatches:
        lines.append("none")
    else:
        lines += [
            "| race | pool | combination | snapshot (theirs) | snapshot (ours) | odds (theirs) |"
            " odds (ours) | odds_hi (theirs) | odds_hi (ours) | status (theirs) | status (ours) |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for m in c.mismatches:
            cells = [
                m.race_id,
                m.pool,
                m.combination,
                m.theirs.get("snapshot_utc"),
                m.ours.get("snapshot_utc"),
                m.theirs.get("odds"),
                m.ours.get("odds"),
                m.theirs.get("odds_hi"),
                m.ours.get("odds_hi"),
                _status(m.theirs),
                _status(m.ours),
            ]
            lines.append("| " + " | ".join(_cell(v) for v in cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the command


def compare_logs_command(
    race_date: Annotated[str, typer.Option("--date", help="Race day, YYYY-MM-DD.")],
    venue: Annotated[str, typer.Option("--venue", help="Venue code, e.g. TOK.")],
    theirs: Annotated[
        Path,
        typer.Option(
            "--theirs", exists=True, dir_okay=False, help="HKTrack's log for the day (read only)."
        ),
    ],
    ours: Annotated[
        Path | None,
        typer.Option(
            "--ours", dir_okay=False, help="Our log; defaults to the day's log in the data dir."
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            dir_okay=False,
            help="Write the markdown report here instead of stdout; it carries prices, so keep it"
            " under $HORSETRACK_DATA_DIR, not in git.",
        ),
    ] = None,
) -> None:
    """Compare a day's odds snapshot log with HKTrack's and print a markdown report.

    The report carries the first 20 mismatched rows with their prices: keep it under
    $HORSETRACK_DATA_DIR, not in git. It reports differences and exits 0 whatever they are.
    """
    try:
        date.fromisoformat(race_date)
    except ValueError as exc:
        raise typer.BadParameter(
            f"not a YYYY-MM-DD date: {race_date!r}", param_hint="--date"
        ) from exc
    ours = ours if ours is not None else paths.odds_snapshot_path("jra", race_date, venue)
    if not ours.is_file():
        raise typer.BadParameter(f"no log at {ours}", param_hint="--ours")
    if out is not None and out.resolve() == theirs.resolve():
        raise typer.BadParameter("--out would overwrite their log", param_hint="--out")

    text = render(compare_logs(read_rows(ours), read_rows(theirs)))
    if out is None:
        typer.echo(text, nl=False)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
