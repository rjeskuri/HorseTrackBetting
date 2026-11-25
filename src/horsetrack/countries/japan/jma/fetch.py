"""Fetching JMA hourly downloads through `horsetrack.core.ingest` (ticket #89), built to be wired
later: no CLI and no race-day command calls it, and no request has been sent with it yet.

The 過去の気象データ・ダウンロード page (https://www.data.jma.go.jp/risk/obsdl/) posts a form to
`show/table` and gets one CSV back (Shift_JIS). `month_task` builds that post for one venue's
station: hourly 気温, 降水量, 日照時間 and 風向・風速 over one calendar month, with the quality
columns and dates as literals (24時 as 0時 of the next day). The form's names are read from the
page's script (top.bundle.js, 2026-11-25); JMA documents none of them, so they hold until the
first wired fetch shows otherwise. A response without the CSV header counts as blocked.

The Fetcher (cache-first, rate-limited per host, circuit breaker; the http block of
`config/weather.yaml`) keeps each month under `raw/jma/amedas_hourly/<year>/<station>_<YYYY-MM>`.
JMA serves data up to the day before, so a month that has not ended in JST is refetched
(`mutable`); an ended month is never requested twice.
"""

from __future__ import annotations

import calendar
import json
from datetime import date, datetime
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest.fetcher import Fetcher, FetchError
from horsetrack.core.ingest.types import STATUS_BLOCKED, STATUS_FETCHED, FetchTask, RawPage
from horsetrack.countries.japan.jma import parse

SOURCE = "jma"
PAGE_TYPE = "amedas_hourly"
TABLE_URL = "https://www.data.jma.go.jp/risk/obsdl/show/table"
HOURLY = "9"  # aggrgPeriod 時別値
# 気温, 降水量 (前1時間), 日照時間 (前1時間), 風向・風速
ELEMENTS = (("201", ""), ("101", ""), ("401", ""), ("301", ""))
ENCODING = "cp932"  # Shift_JIS as JMA sends it


def make_fetcher(**kwargs: Any) -> Fetcher:
    """The ingest Fetcher with weather.yaml's per-host rate and circuit breaker."""
    return Fetcher(SOURCE, http=load_country_yaml(parse.COUNTRY, parse.CONFIG)["http"], **kwargs)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))  # as JSON.stringify


def form(venue: str, year: int, month: int) -> dict[str, str]:
    """The download form for one venue's station over one calendar month (JST). The station is
    named kind + block_no (`s47412`, `a1133`), the form's own pattern; JMA's mapping is unseen."""
    entry = parse.station(venue)
    last = calendar.monthrange(year, month)[1]
    return {
        "stationNumList": _json([f"{entry['kind']}{entry['station_id']}"]),
        "aggrgPeriod": HOURLY,
        "elementNumList": _json([list(e) for e in ELEMENTS]),
        "interAnnualType": "1",  # one continuous period
        "ymdList": _json([str(year), str(year), str(month), str(month), "1", str(last)]),
        "option": _json({"list": ["obs"], "nyear": 0}),
        "downloadFlag": "true",
        "rmkFlag": "1",  # 品質情報, 現象なし情報, 均質番号
        "disconnectFlag": "1",  # every value, across homogeneity breaks
        "youbiFlag": "0",
        "fukenFlag": "0",
        "kijiFlag": "0",
        "csvFlag": "1",  # numbers, not symbols
        "jikantaiFlag": "0",
        "jikantaiList": _json([1, 24]),
        "ymdLiteral": "1",
    }


def month_task(venue: str, year: int, month: int, *, today: date | None = None) -> FetchTask:
    """The fetch of one venue's month; refetched (`mutable`) until the month has ended in JST."""
    today = today or datetime.now(parse.JST).date()
    first = date(year, month, 1)
    if first > today:
        raise ValueError(f"{year}-{month:02d} has not started: JMA has no data for it yet")
    last = date(year, month, calendar.monthrange(year, month)[1])
    key = f"{parse.station_for_venue(venue)}_{year:04d}-{month:02d}"
    body = urlencode(form(venue, year, month))
    return FetchTask(SOURCE, PAGE_TYPE, key, TABLE_URL, year, method="POST", body=body,
                     mutable=last >= today)  # fmt: skip


def classify(body: bytes) -> str:
    """A download is a CSV with the 年月日時 header row; an error or block page is not."""
    try:
        text = body.decode(ENCODING)
    except UnicodeDecodeError:
        return STATUS_BLOCKED
    lines = text.splitlines()
    return STATUS_FETCHED if any(line.startswith("年月日時,") for line in lines) else STATUS_BLOCKED


def fetch_month(fetcher: Fetcher, venue: str, year: int, month: int) -> RawPage:
    """One venue's month, cache-first. A body that is not a download is moved aside (quarantined)
    and raised as a FetchError, so the next attempt goes back to JMA."""
    task = month_task(venue, year, month)
    page = fetcher.fetch(task)
    if classify(page.body) != STATUS_FETCHED:
        fetcher.quarantine(task)
        raise FetchError(f"not a JMA download for {PAGE_TYPE}/{task.key}")
    return page


def parse_page(page: RawPage, venue: str) -> pd.DataFrame:
    """The page's hourly rows (`parse.parse_amedas_csv`) plus `ingested_utc`, the fetch time.
    No `available_utc`: `parse.with_publication_time` adds it with the owner's lag."""
    if page.fetched_utc is None:
        raise ValueError(f"{page.page_type}/{page.key} has no fetch time")
    station_id = parse.station_for_venue(venue)
    frame = parse.parse_amedas_csv(page.body.decode(ENCODING), station_id=station_id)
    frame["ingested_utc"] = pd.Timestamp(page.fetched_utc).tz_convert("UTC").as_unit("us")
    return frame
