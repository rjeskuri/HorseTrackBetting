"""The measured publication time in place of #138's proxy (ticket #225), and the offset summary
the ledger's publication-proxy row is measured with.

Contract pinned here (`horsetrack.countries.japan.handicap.track_bias_report`):
- `RACE_COLUMNS` includes `result_published_utc`, so the report reads the store's logged times.
- `published_times(races) -> dict[str, datetime]`: race_id -> the aware instant passed to
  `track_bias_features(published=...)`. A race with a logged `result_published_utc` gets that
  instant, earlier or later than the proxy; one without gets the proxy, its local post time
  (`post_time_local` "HH:MM", Asia/Tokyo, on its `race_date`) + PUBLISH_LAG. A race with neither
  is left out. A frame without the column (a store built before #225) gets the proxy throughout.
  Ruling: a logged time stands without a post time (it is measured, not derived).
- `publication_lag_summary(races) -> dict`: aggregates only, over the races having both a post
  time and a logged time, lag = result_published_utc - local post time, in minutes:
  `races` (count), `min_minutes`, `median_minutes`, `p90_minutes`, `max_minutes` and
  `share_within_30_minutes` (the share with lag <= 30 minutes, the proxy's claim). With no such
  race: `races` 0 and every other value None. No race id, date or per-race value is returned.

The bracket (review of PR #261: a run's first poll stamps every race already shown, so a restart
or a late start must not count as a measured lag):
- `RACE_COLUMNS` also includes `result_unpublished_utc`, the store's lower bound (the last logger
  poll that did not show the result; null for a late sighting or a store row without one).
- `publication_lag_summary(races, max_bracket_s=None)` counts a race only when its bracket,
  result_published_utc - result_unpublished_utc, is no wider than `max_bracket_s` seconds
  (inclusive). The default (None) is 2 x logger.yaml's `results.every_s` (Ruling: polls ride on
  the logging rounds, so a healthy gap is every_s plus up to one round; twice every_s keeps those
  and drops a bracket that spans a failed poll). Of the races having a post time and a logged
  time, the summary adds `excluded_late` (no lower bound: a late sighting, and every race of a
  frame without the column) and `excluded_wide` (bracket wider than the limit), both ints, 0 when
  none; `races` and the statistics are over the counted races only.

Data: small synthetic polars frames shaped like the store's races rows.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.countries.japan.handicap import track_bias_report as report

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
D1, D2 = date(2025, 6, 14), date(2025, 6, 15)
SCHEMA = {
    "race_id": pl.String,
    "race_date": pl.Date,
    "post_time_local": pl.String,
    "result_published_utc": pl.Datetime("us", "UTC"),
    "result_unpublished_utc": pl.Datetime("us", "UTC"),
}
SUMMARY_KEYS = {
    "races", "min_minutes", "median_minutes", "p90_minutes", "max_minutes",
    "share_within_30_minutes",
}  # fmt: skip
COUNTS = {"races", "excluded_late", "excluded_wide"}
NARROW = timedelta(seconds=45)  # a bracket every limit tested here counts


def local(day: date, hhmm: str) -> datetime:
    hh, mm = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=JST)


def rid(day: date, venue: str, n: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{n:02d}"


def frame(rows: list[tuple]) -> pl.DataFrame:
    """Rows (race_id, race_date, post_time_local, published[, unpublished]); no bound: null."""

    def utc(t: datetime | None) -> datetime | None:
        return None if t is None else t.astimezone(UTC)

    full = [(*r, None) if len(r) == 4 else tuple(r) for r in rows]
    return pl.DataFrame([(r, d, p, utc(t), utc(b)) for r, d, p, t, b in full], schema=SCHEMA,
                        orient="row")  # fmt: skip


def test_the_report_reads_the_logged_column():
    assert "result_published_utc" in report.RACE_COLUMNS


def test_logged_times_replace_the_proxy_where_present():
    proxy = timedelta(minutes=30)
    rows = [
        (rid(D1, "TOK", 1), D1, "10:00", local(D1, "10:12") + timedelta(seconds=30)),  # earlier
        (rid(D1, "TOK", 2), D1, "10:30", None),  # proxy
        (rid(D1, "TOK", 3), D1, None, local(D1, "11:15")),  # logged, no post
        (rid(D1, "TOK", 4), D1, None, None),  # neither: left out
        (rid(D1, "KYO", 5), D1, "11:30", local(D1, "12:20")),  # later than the proxy
        (rid(D2, "KYO", 6), D2, "15:45", None),  # proxy, on its own date
    ]
    got = report.published_times(frame(rows))
    expected = {
        rid(D1, "TOK", 1): local(D1, "10:12") + timedelta(seconds=30),
        rid(D1, "TOK", 2): local(D1, "10:30") + proxy,
        rid(D1, "TOK", 3): local(D1, "11:15"),
        rid(D1, "KYO", 5): local(D1, "12:20"),
        rid(D2, "KYO", 6): local(D2, "15:45") + proxy,
    }
    assert got == expected
    assert all(t.utcoffset() is not None for t in got.values()), "tz-aware for track_bias"


def test_a_store_without_the_column_gets_the_proxy():
    rows = [(rid(D1, "TOK", 1), D1, "09:55"), (rid(D2, "HSN", 2), D2, "13:05")]
    races = pl.DataFrame(
        rows, schema={k: SCHEMA[k] for k in ("race_id", "race_date", "post_time_local")},
        orient="row",
    )  # fmt: skip
    got = report.published_times(races)
    lag = report.PUBLISH_LAG
    assert got == {rid(D1, "TOK", 1): local(D1, "09:55") + lag,
                   rid(D2, "HSN", 2): local(D2, "13:05") + lag}  # fmt: skip


LAGS_A = [7.0, 12.5, 9.0, 31.0, 15.0, 22.0, 40.25]
LAGS_B = [30.0, 18.5, 33.0, 11.0, 26.75]


@pytest.mark.parametrize("lags", [LAGS_A, LAGS_B], ids=["seven", "five-with-30"])
def test_the_lag_summary_is_aggregates_of_logged_minus_post(lags):
    rows = []
    for i, lag in enumerate(lags):
        day, hhmm = (D1, f"{10 + i}:05") if i % 2 else (D2, f"{9 + i}:40")
        seen = local(day, hhmm) + timedelta(minutes=lag)
        rows.append((rid(day, "NAK", i + 1), day, hhmm, seen, seen - NARROW))
    rows += [
        (rid(D1, "NAK", 20), D1, "16:10", None),  # no logged time: not counted
        (rid(D1, "NAK", 21), D1, None, local(D1, "12:00")),  # no post: not counted
    ]
    got = report.publication_lag_summary(frame(rows))
    assert SUMMARY_KEYS <= set(got)
    assert got["races"] == len(lags)
    assert got["min_minutes"] == pytest.approx(min(lags))
    assert got["max_minutes"] == pytest.approx(max(lags))
    assert got["median_minutes"] == pytest.approx(statistics.median(lags))  # odd n: unambiguous
    ordered = sorted(lags)
    assert ordered[-2] - 1e-9 <= got["p90_minutes"] <= ordered[-1] + 1e-9
    within = sum(lag <= 30.0 for lag in lags) / len(lags)
    assert got["share_within_30_minutes"] == pytest.approx(within)
    for key, value in got.items():
        assert value is None or isinstance(value, int | float), f"{key} is not an aggregate"


def test_the_lag_summary_of_nothing_logged_is_empty():
    rows = [(rid(D1, "TOK", 1), D1, "10:00", None)]
    got = report.publication_lag_summary(frame(rows))
    assert got["races"] == 0
    assert all(got[k] is None for k in SUMMARY_KEYS - {"races"})


# ---------------------------------------------------------------- the bracket


def test_the_report_reads_the_lower_bound_column():
    assert "result_unpublished_utc" in report.RACE_COLUMNS


# (lag in minutes, bracket in seconds or None for a late sighting)
BRACKETED = [
    (6.0, 10), (9.5, 59), (11.0, 60), (14.25, 61), (8.0, 119), (21.0, 120), (33.0, 121),
    (12.0, 300), (45.0, 301), (52.0, None), (70.5, None), (2.0, None),
]  # fmt: skip


def bracketed_rows(cases) -> list[tuple]:
    rows = []
    for i, (lag, bracket) in enumerate(cases):
        day, hhmm = (D1, f"{10 + i % 6}:{5 * i:02d}") if i % 2 else (D2, f"{9 + i % 7}:{4 * i:02d}")
        seen = local(day, hhmm) + timedelta(minutes=lag)
        bound = None if bracket is None else seen - timedelta(seconds=bracket)
        rows.append((rid(day, "HAN", i + 1), day, hhmm, seen, bound))
    return rows


def expected_summary(cases, limit_s: float) -> dict:
    """From first principles: the counted lags' aggregates and the two exclusion counts."""
    counted = sorted(lag for lag, b in cases if b is not None and b <= limit_s)
    late = sum(b is None for _, b in cases)
    wide = sum(b is not None and b > limit_s for _, b in cases)
    return {"counted": counted, "excluded_late": late, "excluded_wide": wide}


def check(got: dict, want: dict) -> None:
    counted = want["counted"]
    assert SUMMARY_KEYS | COUNTS <= set(got)
    assert got["races"] == len(counted)
    assert got["excluded_late"] == want["excluded_late"]
    assert got["excluded_wide"] == want["excluded_wide"]
    assert got["min_minutes"] == pytest.approx(counted[0])
    assert got["max_minutes"] == pytest.approx(counted[-1])
    assert got["median_minutes"] == pytest.approx(statistics.median(counted))
    assert counted[-2] - 1e-9 <= got["p90_minutes"] <= counted[-1] + 1e-9
    assert got["share_within_30_minutes"] == pytest.approx(
        sum(lag <= 30.0 for lag in counted) / len(counted)
    )
    for key, value in got.items():
        assert value is None or isinstance(value, int | float), f"{key} is not an aggregate"


@pytest.mark.parametrize("limit_s", [60, 120, 300])
def test_the_lag_summary_counts_only_brackets_within_the_limit(limit_s):
    rows = bracketed_rows(BRACKETED) + [
        (rid(D1, "HAN", 40), D1, None, local(D1, "12:00"), local(D1, "11:59")),  # no post
        (rid(D1, "HAN", 41), D1, "16:10", None, None),  # nothing logged
    ]
    got = report.publication_lag_summary(frame(rows), max_bracket_s=limit_s)
    check(got, expected_summary(BRACKETED, limit_s))


def test_the_default_limit_is_twice_the_loggers_poll_interval():
    every_s = float(load_country_yaml("japan", "logger")["results"]["every_s"])
    cases = [
        (7.0, every_s / 2), (12.0, every_s + 1), (16.0, 2 * every_s), (28.0, 2 * every_s + 1),
        (40.0, 4 * every_s), (19.0, None),
    ]  # fmt: skip
    got = report.publication_lag_summary(frame(bracketed_rows(cases)))
    check(got, expected_summary(cases, 2 * every_s))
    assert got["races"] == 3 and got["excluded_wide"] == 2 and got["excluded_late"] == 1


def test_a_restart_full_of_late_sightings_leaves_the_measured_lags_alone():
    """A run's first poll after a gap stamps old results late: they must not move the summary."""
    measured = [(8.0, 60), (11.5, 60), (14.0, 60), (9.0, 80), (17.0, 60)]
    restart = [(95.0, None), (71.0, None), (48.0, None)]
    alone = report.publication_lag_summary(frame(bracketed_rows(measured)), max_bracket_s=120)
    mixed = report.publication_lag_summary(
        frame(bracketed_rows(measured + restart)), max_bracket_s=120
    )
    for key in SUMMARY_KEYS:
        assert mixed[key] == pytest.approx(alone[key]), key
    assert mixed["excluded_late"] == len(restart) and alone["excluded_late"] == 0
    assert mixed["share_within_30_minutes"] == pytest.approx(1.0)


def test_a_frame_without_the_bound_counts_every_logged_race_as_late():
    rows = [(rid(D1, "TOK", n), D1, f"1{n}:00", local(D1, f"1{n}:00") + timedelta(minutes=9 + n))
            for n in range(4)]  # fmt: skip
    rows.append((rid(D1, "TOK", 9), D1, "15:00", None))
    races = frame(rows).drop("result_unpublished_utc")
    got = report.publication_lag_summary(races, max_bracket_s=600)
    assert got["races"] == 0 and got["excluded_late"] == 4 and got["excluded_wide"] == 0
    assert all(got[k] is None for k in SUMMARY_KEYS - {"races"})


def test_the_summary_of_nothing_logged_excludes_nothing():
    got = report.publication_lag_summary(frame([(rid(D1, "TOK", 1), D1, "10:00", None)]))
    assert (got["races"], got["excluded_late"], got["excluded_wide"]) == (0, 0, 0)
