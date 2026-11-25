"""The log-day health metric and health lines (ticket #25).

The gap metric is the #34 criterion: no gap over 2 minutes inside T-20.

Contract pinned here (`horsetrack.countries.japan.logger.health`):
- `gaps_inside(rows, post_times, within_min=20) -> dict[str, float]`:
  `rows` are odds-log rows (as captured, `snapshot_utc` an aware datetime, or as
  `odds_log.read_rows` returns them, `snapshot_utc` ISO text); every row of one capture shares its
  `snapshot_utc`. `post_times` maps race_id -> aware UTC post time. For each race_id, the longest
  gap in seconds between that race's captures inside [post - within_min, post] (both ends closed),
  the first gap counted from the window start. Captures outside the window, and rows of another
  race_id, do not count.
- `health_lines(summary, rows_by_venue, post_times) -> list[str]`: `summary` is the schedule's
  `DaySummary`, `rows_by_venue` {venue: rows}, `post_times` race_id -> aware UTC post (as above).
  One line per venue holding `<venue> snapshots=N rows=N failures=N wrong_meeting=N
  max_gap_t20_s=N stopped=<reason>` (the counts from the summary; max_gap_t20_s the venue's
  largest race gap from gaps_inside), and a line starting `ALERT` naming the venue when it logged
  0 rows, stopped early ('wrong_meeting', 'circuit_open') or max_gap_t20_s > 120.

Not pinned here: the date at the head of each line (the CLI test pins it in health.log), the
alert's reason text, the gap of a race with no capture inside its window, and whether the stretch
from the last capture to the post counts (every case here has a capture at the post itself).
All rows are synthetic.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from horsetrack.countries.japan.logger.schedule import DaySummary, VenueSummary

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a
# red run reports, and the order lints the same before and after it exists.
from horsetrack.countries.japan.logger.health import gaps_inside, health_lines

UTC = timezone.utc
DAY = "2026-11-28"
WINDOW_S = 20 * 60


def rid(venue: str, number: int) -> str:
    return f"jra:{DAY}:{venue}:{number:02d}"


def post_at(hh: int, mm: int) -> datetime:
    """A post time given in UTC on the race date."""
    return datetime(2026, 11, 28, hh, mm, tzinfo=UTC)


def capture_rows(race_id: str, at: datetime, as_logged: bool) -> list[dict]:
    """The rows one capture of a race writes: a few prices and a pool total, one snapshot_utc."""
    stamp = at.isoformat() if as_logged else at
    base = {"snapshot_utc": stamp, "race_id": race_id, "fetched_utc": stamp}
    return [
        {**base, "pool": "WIN", "combination": "1", "odds": 3.4},
        {**base, "pool": "WIN", "combination": "2", "odds": 5.1},
        {**base, "pool": "QUINELLA", "combination": "1,2", "odds": 12.0},
        {**base, "pool": "WIN", "combination": "__pool__", "investment": 1000},
    ]


def rows_at(race_id: str, post: datetime, offsets_s, as_logged: bool) -> list[dict]:
    """Rows of captures at window start + each offset (window = 20 min before the post)."""
    start = post - timedelta(seconds=WINDOW_S)
    out = []
    for off in offsets_s:
        out += capture_rows(race_id, start + timedelta(seconds=off), as_logged)
    return out


def every(first: int, step: int, last: int = WINDOW_S) -> list[int]:
    """first, first + step, ... below `last`, then `last` (the post) itself."""
    return [*range(first, last, step), last]


FORMS = pytest.mark.parametrize("as_logged", [True, False], ids=["as-logged", "as-captured"])


@FORMS
def test_gap_inside_t20_counts_from_the_window_start(as_logged):
    # (race, post, capture offsets from the window start, expected longest gap)
    cases = [
        (rid("TOK", 1), post_at(0, 55), [90, *every(110, 20)], 90.0),  # first capture 90 s in
        (rid("TOK", 2), post_at(1, 25), every(0, 20), 20.0),  # a capture at the window start
        (rid("KYO", 1), post_at(1, 0), every(300, 100), 300.0),  # the leading gap is the longest
        (rid("KYO", 2), post_at(1, 35), [150, 250, 400, 600, 800, 1000, 1100, 1200], 200.0),
        (rid("FKS", 7), post_at(5, 10), [45, 60, 75, 1200], 1125.0),  # a hole late in the window
    ]
    rows, posts = [], {}
    for race_id, post, offsets, _ in cases:
        rows += rows_at(race_id, post, offsets, as_logged)
        posts[race_id] = post
    gaps = gaps_inside(rows, posts)
    for race_id, _, _, expected in cases:
        assert gaps[race_id] == pytest.approx(expected, abs=1e-6), race_id

    # within_min moves the window start, and the leading gap with it.
    post = post_at(2, 0)
    offsets = [0, 700, 900, 1200]  # post - 20 min, post - 500 s, post - 300 s, post
    rows = rows_at(rid("HSN", 4), post, offsets, as_logged)
    assert gaps_inside(rows, {rid("HSN", 4): post}, within_min=20)[rid("HSN", 4)] == pytest.approx(
        700.0
    )
    assert gaps_inside(rows, {rid("HSN", 4): post}, within_min=10)[rid("HSN", 4)] == pytest.approx(
        300.0  # window from post - 600 s: 100 s to the first capture, then 200 and 300
    )


@FORMS
def test_gap_ignores_captures_outside_t20(as_logged):
    x_post, y_post = post_at(1, 0), post_at(1, 30)
    x, y = rid("NAK", 3), rid("NAK", 4)
    x_start = x_post - timedelta(seconds=WINDOW_S)
    rows = []
    # Race X: two captures before its window, the first inside 240 s after the window start,
    # then every 60 s to the post, then two after the post with a long hole between them.
    for at in (x_post - timedelta(minutes=30), x_post - timedelta(seconds=1300)):
        rows += capture_rows(x, at, as_logged)
    rows += rows_at(x, x_post, every(240, 60), as_logged)
    for at in (x_post + timedelta(seconds=30), x_post + timedelta(seconds=600)):
        rows += capture_rows(x, at, as_logged)
    # Race Y: captures inside X's window (outside its own), which must not fill X's leading gap,
    # then its own captures every 100 s through its window.
    for off in (10, 60, 120, 180):
        rows += capture_rows(y, x_start + timedelta(seconds=off), as_logged)
    rows += rows_at(y, y_post, every(0, 100), as_logged)

    gaps = gaps_inside(rows, {x: x_post, y: y_post})
    assert gaps[x] == pytest.approx(240.0)  # not 340 (from the capture before), nor 570 (after)
    assert gaps[y] == pytest.approx(100.0)  # its captures in X's window do not count


LINE = re.compile(
    r"(?P<venue>\S+) snapshots=(?P<snapshots>\d+) rows=(?P<rows>\d+) failures=(?P<failures>\d+)"
    r" wrong_meeting=(?P<wrong_meeting>\d+) max_gap_t20_s=(?P<gap>\S+) stopped=(?P<stopped>\S+)$"
)


def parse(lines: list[str]) -> tuple[dict[str, list[dict]], list[str]]:
    """({venue: [health line fields, ...]}, [ALERT lines])."""
    health: dict[str, list[dict]] = {}
    alerts = []
    for line in lines:
        assert "\n" not in line
        if line.startswith("ALERT"):
            alerts.append(line)
            continue
        m = LINE.search(line)
        assert m, f"not a health line: {line!r}"
        health.setdefault(m["venue"], []).append(m.groupdict())
    return health, alerts


def alerted(alerts: list[str]) -> set[str]:
    venues = {"TOK", "KYO", "NAK", "HSN", "SAP", "FKS"}
    return {word for line in alerts for word in line.split() if word in venues}


def day(venues: dict[str, tuple[list[int], dict]], posts: dict[str, datetime]):
    """A summary, rows and post times for venues: {venue: (gap offsets of race 1, summary)}."""
    summary = DaySummary()
    rows_by_venue = {}
    for venue, (offsets, counts) in venues.items():
        summary.venues[venue] = VenueSummary(**counts)
        race_id = rid(venue, 1)
        rows_by_venue[venue] = rows_at(race_id, posts[race_id], offsets, as_logged=True)
    return summary, rows_by_venue


def test_a_gap_over_120_s_is_an_alert():
    posts = {rid(v, 1): post_at(1, 5 * i) for i, v in enumerate(["TOK", "KYO", "NAK", "HSN"])}
    venues = {
        # venue: (capture offsets in race 1's window, summary counts), expected gap
        "TOK": ([121, *every(141, 20)], dict(snapshots=40, rows=900, failures=1, stopped="ended")),
        "KYO": (
            [120, *every(140, 20)],
            dict(snapshots=55, rows=1210, wrong_meeting=2, stopped="ended"),
        ),
        "NAK": (every(0, 20), dict(snapshots=61, rows=3050, failures=0, stopped="ended")),
        "HSN": ([400, *every(420, 20)], dict(snapshots=7, rows=77, failures=3, stopped="ended")),
    }
    expected_gap = {"TOK": 121.0, "KYO": 120.0, "NAK": 20.0, "HSN": 400.0}
    summary, rows_by_venue = day(venues, posts)

    health, alerts = parse(health_lines(summary, rows_by_venue, posts))

    assert set(health) == set(venues)
    for venue, (_, counts) in venues.items():
        assert len(health[venue]) == 1, venue
        line = health[venue][0]
        want = VenueSummary(**counts)
        got = (int(line["snapshots"]), int(line["rows"]), int(line["failures"]))
        assert got == (want.snapshots, want.rows, want.failures), venue
        assert int(line["wrong_meeting"]) == want.wrong_meeting, venue
        assert float(line["gap"]) == pytest.approx(expected_gap[venue], abs=1.0), venue
    # Over 120 s alerts; exactly 120 s does not.
    assert alerted(alerts) == {"TOK", "HSN"}


@pytest.mark.parametrize("reason", ["wrong_meeting", "circuit_open"])
def test_a_venue_stopped_early_or_logging_nothing_is_an_alert(reason):
    posts = {rid(v, 1): post_at(1, 5 * i) for i, v in enumerate(["SAP", "FKS", "KYO"])}
    venues = {
        "SAP": (every(0, 20), dict(snapshots=30, rows=600, stopped=reason)),  # gaps fine
        "FKS": (every(0, 20), dict(snapshots=61, rows=1220, stopped="ended")),  # clean
        "KYO": ([], dict(snapshots=0, rows=0, failures=12, stopped="ended")),  # logged nothing
    }
    summary, rows_by_venue = day(venues, posts)

    health, alerts = parse(health_lines(summary, rows_by_venue, posts))

    assert set(health) == set(venues)
    assert health["SAP"][0]["stopped"] == reason
    assert int(health["KYO"][0]["rows"]) == 0
    assert alerted(alerts) == {"SAP", "KYO"}
