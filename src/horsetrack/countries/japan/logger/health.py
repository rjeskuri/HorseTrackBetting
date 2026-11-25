"""The log-day health metric and health lines (ticket #25).

The gap metric is the #34 criterion: no gap over 2 minutes inside T-20. For a race, the window is
[post - within_min, post] (both ends closed), and its gap is the longest stretch inside it without
a capture: from the window start to the first capture, between captures, and from the last capture
to the post. A race with no capture inside its window has the whole window as its gap.

A health line: `<venue> snapshots=N rows=N failures=N wrong_meeting=N max_gap_t20_s=N
stopped=<reason>`, with `<date> ` in front when the date is given; max_gap_t20_s is in whole
seconds, `-` when the venue has no post time; with tier 2 on (ticket #130) the line ends
` tier2_skipped=N`, the venue's tier-2 grid points not captured; with the source given (ticket
#38) it ends ` source=<key>` (`jra`, `nar`). A venue that logged 0 rows, stopped other than
'ended', or whose max_gap_t20_s is over 120 also gets
`ALERT [<date> ]<venue> [source=<key> ]<reasons>`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from typing import Any

from horsetrack.countries.japan.logger.schedule import DaySummary, VenueSummary
from horsetrack.countries.japan.odds_log import _instant

MAX_GAP_S = 120  # the #34 criterion: no gap over 2 minutes inside T-20
WITHIN_MIN = 20


def gaps_inside(
    rows: Iterable[Mapping[str, Any]],
    post_times: Mapping[str, datetime],
    within_min: int = WITHIN_MIN,
) -> dict[str, float]:
    """race_id -> the longest gap in seconds inside [post - within_min, post], for every race of
    `post_times`. Rows of a race without a post time, or without a readable snapshot_utc, do not
    count; neither do captures outside the race's window."""
    window = timedelta(minutes=within_min)
    seen: dict[str, set[datetime]] = {race_id: set() for race_id in post_times}
    for row in rows:
        race_id = row.get("race_id")
        if race_id not in seen:
            continue
        at = _instant(row.get("snapshot_utc"))
        if at is not None and post_times[race_id] - window <= at <= post_times[race_id]:
            seen[race_id].add(at)
    out = {}
    for race_id, times in seen.items():
        post = post_times[race_id]
        edges = [post - window, *sorted(times), post]
        out[race_id] = max((b - a).total_seconds() for a, b in zip(edges, edges[1:], strict=False))
    return out


def _venue_of(race_id: str) -> str | None:
    """The venue of a race_id jra:<YYYY-MM-DD>:<VENUE>:<NN>; None when it is not of that shape."""
    parts = str(race_id).split(":")
    return parts[2] if len(parts) == 4 else None


def _alert_reasons(v: VenueSummary, gap_s: int | None) -> list[str]:
    reasons = []
    if v.rows == 0:
        reasons.append("logged 0 rows")
    if v.stopped != "ended":
        reasons.append(f"stopped early: {v.stopped or 'no stop reason'}")
    if gap_s is not None and gap_s > MAX_GAP_S:
        reasons.append(f"max_gap_t20_s={gap_s} over {MAX_GAP_S}")
    return reasons


def health_lines(
    summary: DaySummary,
    rows_by_venue: Mapping[str, Iterable[Mapping[str, Any]]],
    post_times: Mapping[str, datetime],
    *,
    race_date: date | None = None,
    source: str | None = None,
) -> list[str]:
    """One health line per venue of `summary`, each followed by its ALERT line when it has one.
    `post_times` is race_id -> aware UTC post, for any venue: each venue's races are picked out by
    the venue in their race_id. With `source`, every line names it."""
    head = f"{race_date.isoformat()} " if race_date is not None else ""
    tail = f" source={source}" if source is not None else ""
    lines = []
    for venue, v in summary.venues.items():
        posts = {rid: at for rid, at in post_times.items() if _venue_of(rid) == venue}
        gaps = gaps_inside(rows_by_venue.get(venue, ()), posts)
        gap_s = round(max(gaps.values())) if gaps else None
        lines.append(
            f"{head}{venue} snapshots={v.snapshots} rows={v.rows} failures={v.failures}"
            f" wrong_meeting={v.wrong_meeting} max_gap_t20_s={'-' if gap_s is None else gap_s}"
            f" stopped={v.stopped or '-'}"
            + (f" tier2_skipped={v.tier2_skipped}" if v.tier2_skipped is not None else "")
            + tail
        )
        reasons = _alert_reasons(v, gap_s)
        if reasons:
            lines.append(f"ALERT {head}{venue}{tail} {'; '.join(reasons)}")
    return lines
