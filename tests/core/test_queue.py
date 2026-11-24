"""FetchQueue: the SQLite manifest that queues and records every fetch. All synthetic.

Ported from HKTrack tests/unit/ingest/test_ingest_queue.py (all 19 tests, one per behavior, same
names), plus `test_opens_a_manifest_written_by_hktrack` (ticket #43).

Seams the implementation must honour (pinned here):

* `horsetrack.core.ingest.queue` exposes `FetchQueue(path)` and `STATUSES`; tasks are
  `horsetrack.core.ingest.types.FetchTask` (field `source`, not HKTrack's `jurisdiction`).
* The on-disk schema is HKTrack's, unchanged: the table `fetch_queue` keeps the column
  `jurisdiction`, which holds `FetchTask.source`. A manifest HKTrack wrote opens without migration.
* `get` and `iter_status` return row dicts keyed by column name; `next_pending` returns FetchTasks.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime

import pytest

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a red
# run reports, and the order lints the same before and after it exists.
from horsetrack.core.ingest.queue import STATUSES, FetchQueue

# isort: split
from fakes import SOURCE, make_task

from horsetrack.core.ingest.types import FetchTask

# In HKTrack's on-disk order.
COLUMNS = [
    "jurisdiction", "page_type", "key", "url", "year", "priority", "method", "body", "mutable",
    "status", "http_status", "final_url", "sha256", "bytes", "fetched_utc", "attempts", "error",
    "parser_version",
]  # fmt: skip
EXPECTED_COLUMNS = set(COLUMNS)

# Copied literally from HKTrack src/hktrack/ingest/queue.py (`_SCHEMA`), so this test does not need
# HKTrack on disk.
HKTRACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_queue (
    jurisdiction   TEXT NOT NULL,
    page_type      TEXT NOT NULL,
    key            TEXT NOT NULL,
    url            TEXT NOT NULL,
    year           INTEGER NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 0,
    method         TEXT NOT NULL DEFAULT 'GET',
    body           TEXT,
    mutable        INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'pending',
    http_status    INTEGER,
    final_url      TEXT,
    sha256         TEXT,
    bytes          INTEGER,
    fetched_utc    TEXT,
    attempts       INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    parser_version TEXT,
    PRIMARY KEY (page_type, key)
);
CREATE INDEX IF NOT EXISTS fetch_queue_next ON fetch_queue (status, priority DESC, key, page_type);
"""


@pytest.fixture
def queue(tmp_path):
    q = FetchQueue(tmp_path / "raw" / "manifest.sqlite")
    yield q
    q.close()


def test_creates_parent_dirs_wal_mode_and_spec_columns(tmp_path):
    path = tmp_path / "deep" / "er" / "manifest.sqlite"
    q = FetchQueue(path)
    q.close()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    columns = {row[1] for row in conn.execute("PRAGMA table_info(fetch_queue)")}
    conn.close()
    assert columns == EXPECTED_COLUMNS


def test_statuses_match_the_spec():
    assert set(STATUSES) == {"pending", "fetched", "empty", "abandoned", "failed"}


def test_enqueue_is_idempotent(queue):
    tasks = [make_task("results", f"k{i}") for i in range(5)]
    assert queue.enqueue(tasks) == 5
    assert queue.enqueue(tasks) == 0
    assert queue.enqueue([*tasks, make_task("results", "k9")]) == 1
    assert queue.counts() == {("results", "pending"): 6}


def test_enqueue_accepts_a_generator_and_duplicates_inside_one_batch(queue):
    assert queue.enqueue(make_task("horse", k) for k in ["A", "B", "A"]) == 2


def test_same_key_in_two_page_types_are_two_rows(queue):
    assert (
        queue.enqueue(
            [make_task("results", "2026-09-16_R1"), make_task("sectional", "2026-09-16_R1")]
        )
        == 2
    )


def test_reenqueue_never_resets_an_immutable_task(queue):
    task = make_task("results", "k1", priority=5)
    queue.enqueue([task])
    queue.mark(task, "fetched", http_status=200, sha256="abc", n_bytes=10)
    changed = dataclasses.replace(task, url="https://site.test/other", priority=99)
    assert queue.enqueue([changed]) == 0
    row = queue.get("results", "k1")
    assert (row["status"], row["url"], row["priority"]) == ("fetched", task.url, 5)
    assert queue.next_pending(10) == []


def test_mutable_task_is_reset_to_pending_and_updated(queue):
    task = make_task(
        "racecard", "2026-09-23_HV", mutable=True, priority=1, method="POST", body='{"v":1}'
    )
    queue.enqueue([task])
    queue.mark(task, "fetched", http_status=200)
    newer = dataclasses.replace(
        task, url="https://site.test/racecard/v2", priority=7, body='{"v":2}'
    )
    assert queue.enqueue([newer]) == 0
    row = queue.get("racecard", "2026-09-23_HV")
    assert (row["status"], row["url"], row["priority"], row["body"]) == (
        "pending", "https://site.test/racecard/v2", 7, '{"v":2}',
    )  # fmt: skip
    assert queue.next_pending(1) == [newer]


def test_next_pending_orders_by_priority_desc_then_key(queue):
    queue.enqueue(
        [
            make_task("results", "b", priority=1),
            make_task("results", "a", priority=1),
            make_task("horse", "z", priority=9),
            make_task("results", "c", priority=5),
        ]
    )
    assert [t.key for t in queue.next_pending(10)] == ["z", "c", "a", "b"]
    assert [t.key for t in queue.next_pending(2)] == ["z", "c"]
    assert [t.key for t in queue.next_pending()] == ["z"]


def test_next_pending_filters_page_types_and_skips_done(queue):
    tasks = [make_task("results", "r1", priority=3), make_task("horse", "h1", priority=9)]
    queue.enqueue(tasks)
    assert [t.key for t in queue.next_pending(10, page_types=["results"])] == ["r1"]
    assert queue.next_pending(10, page_types=["sectional"]) == []
    queue.mark(tasks[1], "empty")
    assert [t.key for t in queue.next_pending(10)] == ["r1"]


def test_next_pending_rebuilds_every_task_field(queue):
    task = make_task(
        "odds", "snap", year=2031, priority=-4, method="POST", body='{"query":"x"}', mutable=True
    )
    plain = make_task("results", "plain")
    queue.enqueue([task, plain])
    assert queue.next_pending(10) == [plain, task]


def test_mark_records_outcome_and_counts_attempts(queue):
    task = make_task("results", "k1")
    queue.enqueue([task])
    queue.mark(task, "failed", http_status=500, error="HTTP 500")
    row = queue.get("results", "k1")
    assert (row["status"], row["attempts"], row["http_status"], row["error"]) == (
        "failed",
        1,
        500,
        "HTTP 500",
    )

    queue.mark(
        task, "fetched", http_status=200, final_url=task.url + "?x=1", sha256="f" * 64, n_bytes=1234
    )
    row = queue.get("results", "k1")
    assert row["status"] == "fetched"
    assert row["attempts"] == 2
    assert row["error"] is None
    assert (row["http_status"], row["final_url"], row["sha256"], row["bytes"]) == (
        200, task.url + "?x=1", "f" * 64, 1234,
    )  # fmt: skip
    fetched = datetime.fromisoformat(row["fetched_utc"])
    assert fetched.utcoffset().total_seconds() == 0


def test_mark_rejects_unknown_status(queue):
    task = make_task("results", "k1")
    queue.enqueue([task])
    with pytest.raises(ValueError):
        queue.mark(task, "blocked")
    assert queue.get("results", "k1")["status"] == "pending"


def test_mark_of_an_unqueued_task_creates_the_row(queue):
    task = make_task("racecard", "adhoc", mutable=True)
    queue.mark(task, "fetched", http_status=200)
    row = queue.get("racecard", "adhoc")
    assert (row["status"], row["attempts"], row["url"], row["mutable"]) == (
        "fetched",
        1,
        task.url,
        1,
    )


def test_counts_and_get(queue):
    tasks = [make_task("results", f"k{i}") for i in range(4)] + [make_task("horse", "H1")]
    queue.enqueue(tasks)
    queue.mark(tasks[0], "fetched")
    queue.mark(tasks[1], "empty")
    queue.mark(tasks[2], "abandoned")
    assert queue.counts() == {
        ("results", "fetched"): 1,
        ("results", "empty"): 1,
        ("results", "abandoned"): 1,
        ("results", "pending"): 1,
        ("horse", "pending"): 1,
    }
    assert queue.get("results", "nope") is None
    # HKTrack's column name is kept on disk; it holds the task's source.
    assert queue.get("horse", "H1")["jurisdiction"] == SOURCE


def test_state_survives_reopen(tmp_path):
    path = tmp_path / "manifest.sqlite"
    q = FetchQueue(path)
    tasks = [make_task("results", "a", priority=2), make_task("results", "b", priority=1)]
    q.enqueue(tasks)
    q.mark(tasks[0], "fetched", http_status=200, sha256="abc", n_bytes=3)
    q.close()

    again = FetchQueue(path)
    assert again.counts() == {("results", "fetched"): 1, ("results", "pending"): 1}
    assert again.next_pending(10) == [tasks[1]]
    assert again.enqueue(tasks) == 0
    assert again.get("results", "a")["sha256"] == "abc"
    again.close()


def test_mark_is_committed_immediately(tmp_path):
    """kill -9 safety: another connection sees the mark without the writer closing."""
    path = tmp_path / "manifest.sqlite"
    q = FetchQueue(path)
    task = make_task("results", "a")
    q.enqueue([task])
    q.mark(task, "fetched")
    other = sqlite3.connect(path)
    assert other.execute("SELECT status FROM fetch_queue").fetchone()[0] == "fetched"
    other.close()
    q.close()


def test_iter_status_yields_rows_and_tolerates_enqueue_while_iterating(queue):
    tasks = [make_task("results", f"k{i:04d}") for i in range(1200)]
    queue.enqueue(tasks)
    for t in tasks[:1100]:
        queue.mark(t, "fetched")
    seen = []
    for row in queue.iter_status("results"):
        seen.append(row["key"])
        queue.enqueue([make_task("horse", f"H{row['key']}")])
    assert seen == [t.key for t in tasks[:1100]]
    assert [r["key"] for r in queue.iter_status("results", "pending")] == [
        t.key for t in tasks[1100:]
    ]
    assert queue.counts()[("horse", "pending")] == 1100


def test_reset_failed(queue):
    tasks = [make_task("results", "r1"), make_task("results", "r2"), make_task("horse", "h1")]
    queue.enqueue(tasks)
    for t in tasks:
        queue.mark(t, "failed", error="HTTP 500")
    assert queue.reset_failed(page_types=["horse"]) == 1
    assert queue.get("horse", "h1")["status"] == "pending"
    assert queue.get("results", "r1")["status"] == "failed"
    assert queue.reset_failed() == 2
    assert queue.reset_failed() == 0
    assert {t.key for t in queue.next_pending(10)} == {"r1", "r2", "h1"}


def test_set_error_and_parser_version(queue):
    task = make_task("results", "r1")
    queue.enqueue([task])
    queue.mark(task, "fetched", error="parse: boom")
    queue.set_error("results", "r1", None)
    queue.set_parser_version("results", "r1", "3")
    row = queue.get("results", "r1")
    assert (row["status"], row["error"], row["parser_version"], row["attempts"]) == (
        "fetched",
        None,
        "3",
        1,
    )


# --- ticket #43: a manifest HKTrack wrote opens unchanged ---------------------------------------

# (jurisdiction, page_type, key, url, year, priority, method, body, mutable, status, http_status,
#  final_url, sha256, bytes, fetched_utc, attempts, error, parser_version), as HKTrack writes them.
# mutable = 2 is HKTrack's "owed one final fetch" marker: still handed out as mutable.
HKTRACK_ROWS = [
    ("hkjc", "results", "2025-01-01_ST_R1", "https://site.test/results/1", 2025, 3, "GET", None,
     0, "fetched", 200, "https://site.test/results/1?x", "a" * 64, 4321,
     "2025-01-02T03:04:05+00:00", 1, None, "2"),
    ("hkjc", "results", "2025-01-01_ST_R2", "https://site.test/results/2", 2025, 3, "GET", None,
     0, "failed", 500, None, None, None, "2025-01-02T03:04:06+00:00", 4, "HTTP 500", None),
    ("hkjc", "horse", "H123", "https://site.test/horse/H123", 2024, 7, "GET", None, 0,
     "pending", None, None, None, None, None, 0, None, None),
    ("hkjc", "racecard", "2025-01-08_HV", "https://site.test/racecard", 2025, 9, "POST",
     '{"d":"x"}', 2, "pending", 200, None, "b" * 64, 99, "2025-01-07T00:00:00+00:00", 1, None,
     None),
    ("hkjc", "fixture", "2025-01", "https://site.test/fixture/2025-01", 2025, 0, "GET", None, 0,
     "empty", 200, None, "c" * 64, 12, "2025-01-01T00:00:00+00:00", 1, "parse: no table", "1"),
]  # fmt: skip


def _write_hktrack_manifest(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(HKTRACK_SCHEMA)
    with conn:
        conn.executemany(f"INSERT INTO fetch_queue VALUES ({', '.join('?' * 18)})", HKTRACK_ROWS)
    conn.close()


def test_opens_a_manifest_written_by_hktrack(tmp_path):
    path = tmp_path / "manifest.sqlite"
    _write_hktrack_manifest(path)

    q = FetchQueue(path)
    try:
        assert q.counts() == {
            ("results", "fetched"): 1,
            ("results", "failed"): 1,
            ("horse", "pending"): 1,
            ("racecard", "pending"): 1,
            ("fixture", "empty"): 1,
        }
        assert q.page_types() == ["fixture", "horse", "racecard", "results"]
        assert q.count_parse_errors() == 1

        # Every stored row reads back column for column.
        for stored in HKTRACK_ROWS:
            row = q.get(stored[1], stored[2])
            assert row is not None
            assert {c: row[c] for c in COLUMNS} == dict(zip(COLUMNS, stored, strict=True))

        # Pending rows come back as FetchTasks (jurisdiction -> source), priority first.
        assert q.next_pending(10) == [
            FetchTask(
                source="hkjc", page_type="racecard", key="2025-01-08_HV", url="https://site.test/racecard",
                year=2025, priority=9, method="POST", body='{"d":"x"}', mutable=True,
            ),
            FetchTask(
                source="hkjc", page_type="horse", key="H123", url="https://site.test/horse/H123",
                year=2024, priority=7,
            ),
        ]  # fmt: skip
        assert [r["key"] for r in q.iter_status("results")] == ["2025-01-01_ST_R1"]
        assert [r["key"] for r in q.iter_status("results", "failed")] == ["2025-01-01_ST_R2"]
        assert q.reset_failed(max_attempts=4) == 0  # attempted 4 times already: stays failed

        # The old manifest keeps taking writes.
        assert q.enqueue([make_task("results", "2025-01-08_HV_R1")]) == 1
        q.mark(make_task("horse", "H123", url="https://site.test/horse/H123", year=2024), "fetched")
        assert q.get("horse", "H123")["attempts"] == 1
        assert q.get("results", "2025-01-08_HV_R1")["jurisdiction"] == SOURCE
    finally:
        q.close()

    # The schema on disk is still HKTrack's: no column added, renamed or dropped.
    conn = sqlite3.connect(path)
    assert {row[1] for row in conn.execute("PRAGMA table_info(fetch_queue)")} == EXPECTED_COLUMNS
    assert conn.execute("SELECT COUNT(*) FROM fetch_queue").fetchone()[0] == len(HKTRACK_ROWS) + 1
    conn.close()
