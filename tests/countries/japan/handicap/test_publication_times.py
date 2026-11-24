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

Data: small synthetic polars frames shaped like the store's races rows.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from horsetrack.countries.japan.handicap import track_bias_report as report

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
D1, D2 = date(2025, 6, 14), date(2025, 6, 15)
SCHEMA = {
    "race_id": pl.String,
    "race_date": pl.Date,
    "post_time_local": pl.String,
    "result_published_utc": pl.Datetime("us", "UTC"),
}
SUMMARY_KEYS = {
    "races", "min_minutes", "median_minutes", "p90_minutes", "max_minutes",
    "share_within_30_minutes",
}  # fmt: skip


def local(day: date, hhmm: str) -> datetime:
    hh, mm = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=JST)


def rid(day: date, venue: str, n: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{n:02d}"


def frame(rows: list[tuple[str, date, str | None, datetime | None]]) -> pl.DataFrame:
    utc = [(r, d, p, None if t is None else t.astimezone(UTC)) for r, d, p, t in rows]
    return pl.DataFrame(utc, schema=SCHEMA, orient="row")


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
        rows.append((rid(day, "NAK", i + 1), day, hhmm, local(day, hhmm) + timedelta(minutes=lag)))
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
