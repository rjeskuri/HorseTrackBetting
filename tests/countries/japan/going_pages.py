"""Synthetic JRA going pages (馬場情報) and a raw-cache writer for the store-build tests (#253).

`going_page` builds the structure of tests/fixtures/synthetic/jra_going_rail_a.html from values:
the date line, the 発表 line, the going rows, an optional rail block, the cushion and the
moisture table. `add_raw` writes one raw file plus one manifest row under $HORSETRACK_DATA_DIR the
way core.ingest does (gzip file under raw_page_path, a fetch_queue row), for any source.
Every value is invented.
"""

from __future__ import annotations

import gzip
import hashlib
import sqlite3
from datetime import date, datetime

from horsetrack.core import paths

_WEEKDAY = "月火水木金土日"

MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_queue (
    jurisdiction TEXT NOT NULL, page_type TEXT NOT NULL, key TEXT NOT NULL, url TEXT NOT NULL,
    year INTEGER NOT NULL, priority INTEGER NOT NULL DEFAULT 0, method TEXT NOT NULL DEFAULT 'GET',
    body TEXT, mutable INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending',
    http_status INTEGER, final_url TEXT, sha256 TEXT, bytes INTEGER, fetched_utc TEXT,
    attempts INTEGER NOT NULL DEFAULT 0, error TEXT, parser_version TEXT,
    PRIMARY KEY (page_type, key)
)
"""


def _day(d: date) -> str:
    return f"{d.month}月{d.day}日（{_WEEKDAY[d.weekday()]}曜）"


def going_page(race_date: date, meeting: str, published_jst: datetime, rail: str | None) -> bytes:
    """A cp932 going page of `meeting` ('5回京都6日'), published at the naive JST `published_jst`;
    `rail` is the inner text of the rail value ('Aコース', '－'), None leaves the block out."""
    block = (
        ""
        if rail is None
        else f'<div class="rail"><h3>使用コース（芝）</h3><p class="value">{rail}</p></div>'
    )
    p = published_jst
    html = (
        '<!-- synthetic: JRA going page -->\n<!doctype html><html lang="ja">'
        '<head><meta charset="Shift_JIS"><title>馬場情報</title></head><body><div id="baba">'
        f'<h2 class="date_line">{race_date.year}年{_day(race_date)} {meeting}</h2>'
        f'<p class="publish">{_day(p.date())}{p.hour}時{p.minute:02d}分 発表</p>'
        '<table class="going"><tr><th>芝</th><td>良</td></tr>'
        "<tr><th>ダート</th><td>良</td></tr></table>"
        f"{block}"
        '<div class="cushion"><h3>クッション値（芝）</h3><p class="value">9.0</p></div>'
        '<table class="moisture"><caption>含水率</caption>'
        "<tr><th></th><th>ゴール前</th><th>4コーナー</th></tr>"
        "<tr><th>芝</th><td>10.0%</td><td>9.0%</td></tr>"
        "<tr><th>ダート</th><td>5.0%</td><td>4.0%</td></tr></table>"
        "</div></body></html>"
    )
    return html.encode("cp932")


def add_raw(
    source: str, page_type: str, key: str, year: int, body: bytes, fetched_utc: str
) -> None:
    """One gzip raw file and one `fetched` manifest row for (page_type, key)."""
    path = paths.raw_page_path(source, page_type, year, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body, mtime=0))
    url = f"https://synthetic.example.test/{page_type}/{key}"
    manifest = paths.manifest_path(source)
    conn = sqlite3.connect(manifest)
    with conn:
        conn.execute(MANIFEST_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO fetch_queue (jurisdiction, page_type, key, url, year, status,"
            " final_url, sha256, bytes, fetched_utc, http_status, attempts)"
            " VALUES (?, ?, ?, ?, ?, 'fetched', ?, ?, ?, ?, 200, 1)",
            (source, page_type, key, url, year, url, hashlib.sha256(body).hexdigest(),
             len(body), fetched_utc),
        )  # fmt: skip
    conn.close()
