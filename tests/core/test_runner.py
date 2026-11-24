"""run_backfill end to end against a fake adapter and a fake multi-host site. No network, no real
sleeping.

Ported from HKTrack tests/unit/ingest/test_ingest_runner.py (all 28 tests, same names) and the
runner cases of test_ingest_adversarial.py and test_ingest_review2.py, plus the stop rules ticket
#44 names: a CircuitOpen from any host stops the run, and the quiet-hours callback.

Seams the implementation must honour (pinned here):

* `horsetrack.core.ingest.runner` exposes `RunSummary`, `plan_backfill`, `run_backfill` and
  `expand_followups`. Their first argument is the source (`FetchTask.source`).
* `plan_backfill(source, start, end, *, adapter=None, queue=None) -> int`: new rows.
* `run_backfill(source, *, max_requests=None, max_minutes=None, page_types=None, adapter=None,
  fetcher=None, queue=None, log_every=200, http=None, clock=time.monotonic, quiet=None)`
  returns a `RunSummary`. Without `queue` it opens `paths.manifest_path(source)`.
* `http` is the mapping the Fetcher takes. The runner's own limits are
  `http["circuit_breaker"]["max_consecutive_failures"]` (default 5) and
  `["max_consecutive_marker_misses"]` (default 10); the defaults also apply when `http` is omitted.
* `quiet()` is asked before every task; a truthy `why` stops the run with
  `stopped_reason == f"quiet hours: {why}"`, and nothing more is requested.
* Stop reasons as in HKTrack: "queue empty", "max_requests", "max_minutes", "interrupted",
  "circuit open: ...", "already running: ...", "quiet hours: ...".
* The run log is one `scrape-*.log` file under `$HORSETRACK_DATA_DIR/logs/` (any subfolder),
  named by `RunSummary.log_path`; nothing goes to stdout or stderr.
* `expand_followups(source, page_types=None, adapter=None, queue=None, fetcher=None) -> int`.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import os
import signal
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from horsetrack.core.paths import manifest_path, raw_page_path

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a red
# run reports, and the order lints the same before and after it exists.
from horsetrack.core.ingest.runner import RunSummary, expand_followups, plan_backfill, run_backfill

# isort: split
import httpx
from fakes import BASE, HTTP, SOURCE, FakeAdapter, FakeClock, FakeSite, make_task

from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.core.ingest.queue import FetchQueue

START, END = date(2026, 9, 1), date(2026, 9, 30)
A, B, C = "https://a.test", "https://b.test", "https://c.test"
UA_VAR = "HORSETRACK_USER_AGENT"


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch) -> Path:
    """Raw pages, manifest and logs under tmp_path; no real `.env` or environment can leak in."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv(UA_VAR, raising=False)
    return tmp_path / "data"


@pytest.fixture
def http() -> dict:
    return copy.deepcopy(HTTP)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def site() -> FakeSite:
    return FakeSite()


@pytest.fixture
def small_site(site: FakeSite) -> FakeSite:
    """One fixture month -> two meetings -> three horses (one shared between meetings)."""
    site.add(f"{BASE}/fixture/2026-09", "MARKER\nmeeting:2026-09-16_R1\nmeeting:2026-09-20_R1\n")
    site.add(f"{BASE}/results/2026-09-16_R1", "MARKER\nhorse:H1\nhorse:H2\n")
    site.add(f"{BASE}/results/2026-09-20_R1", "MARKER\nhorse:H2\nhorse:H3\n")
    for horse in ("H1", "H2", "H3"):
        site.add(f"{BASE}/horse/{horse}", f"MARKER profile {horse}")
    return site


@pytest.fixture
def queue(data):
    q = FetchQueue(manifest_path(SOURCE))
    yield q
    q.close()


@pytest.fixture
def make_fetcher(site, http, clock):
    made = []

    def _make(**overrides):
        kw = {"http": http, "transport": site.transport, "sleep": clock.sleep, "clock": clock}
        kw.update(overrides)
        fetcher = Fetcher(SOURCE, **kw)
        made.append(fetcher)
        return fetcher

    yield _make
    for f in made:
        f.close()


@pytest.fixture
def fetcher(make_fetcher):
    return make_fetcher()


@pytest.fixture
def adapter() -> FakeAdapter:
    return FakeAdapter()


def _run(adapter, fetcher, queue, **kw) -> RunSummary:
    return run_backfill(SOURCE, adapter=adapter, fetcher=fetcher, queue=queue, **kw)


def _statuses(queue) -> dict[str, str]:
    out = {}
    for (page_type, status), _n in queue.counts().items():
        for row in queue.iter_status(page_type, status):
            out[f"{page_type}/{row['key']}"] = status
    return out


def _nonzero(counts: dict) -> dict:
    return {k: n for k, n in counts.items() if n}


def _logs(data: Path) -> list[Path]:
    return sorted((data / "logs").rglob("scrape-*.log"))


def _warm_cache(site, fetcher, tasks, body: str = "MARKER cached earlier") -> None:
    """Raw files that exist while their manifest rows are still pending (a rebuilt or restored
    manifest)."""
    for t in tasks:
        site.add(t.url, body)
        fetcher.fetch(t)
        site.pages.pop(t.url)


# ---------------------------------------------------------------- the drain loop and its summary

_BODIES = {
    "fetched": "MARKER page",
    "empty": "MARKER No information.",
    "abandoned": "MARKER ABANDONED",
}


def _round_robin(mix: dict[str, int]) -> list[str]:
    left = dict(mix)
    order: list[str] = []
    while any(left.values()):
        for kind in left:
            if left[kind]:
                order.append(kind)
                left[kind] -= 1
    return order


@pytest.mark.parametrize(
    "mix",
    [
        {"fetched": 1, "empty": 0, "abandoned": 0, "failed": 0, "cached": 0},
        {"fetched": 2, "empty": 1, "abandoned": 1, "failed": 1, "cached": 0},
        {"fetched": 0, "empty": 3, "abandoned": 2, "failed": 0, "cached": 2},
        {"fetched": 4, "empty": 0, "abandoned": 1, "failed": 2, "cached": 3},
    ],
    ids=["one-page", "one-of-each", "no-plain-pages", "mixed"],
)
def test_each_outcome_is_recorded_on_the_queue_and_in_the_summary(
    site, adapter, fetcher, queue, mix
):
    kinds = _round_robin(mix)
    tasks = [make_task("results", f"k{i:02d}") for i in range(len(kinds))]
    bodies: dict[str, bytes] = {}
    for kind, t in zip(kinds, tasks, strict=True):
        if kind == "cached":
            bodies[t.key] = f"MARKER cached {t.key}".encode()
            _warm_cache(site, fetcher, [t], bodies[t.key].decode())
        elif kind == "failed":
            site.add(t.url, "gone", status=404)
        else:
            bodies[t.key] = f"{_BODIES[kind]} {t.key}".encode()
            site.add(t.url, bodies[t.key])
    queue.enqueue(tasks)
    warmed = site.total_hits

    summary = _run(adapter, fetcher, queue)

    network = len(kinds) - mix["cached"]
    assert (summary.requested, summary.from_cache, summary.processed) == (
        network, mix["cached"], len(kinds),
    )  # fmt: skip
    assert (summary.fetched, summary.empty, summary.abandoned, summary.failed) == (
        mix["fetched"] + mix["cached"], mix["empty"], mix["abandoned"], mix["failed"],
    )  # fmt: skip
    assert summary.enqueued_followups == 0
    assert summary.stopped_reason == "queue empty" and not summary.circuit_opened
    assert site.total_hits - warmed == network  # one request per network page, none per cache hit
    for kind, t in zip(kinds, tasks, strict=True):
        row = queue.get("results", t.key)
        assert (row["status"], row["attempts"]) == ("fetched" if kind == "cached" else kind, 1)
        if kind == "failed":
            assert row["http_status"] == 404 and "404" in row["error"]
            continue
        body = bodies[t.key]
        assert row["sha256"] == hashlib.sha256(body).hexdigest() and row["bytes"] == len(body)
        assert row["error"] is None
        if kind != "cached":
            assert (row["http_status"], row["final_url"]) == (200, t.url)


# ---------------------------------------------------------------- stop rules: any host, quiet hours


@pytest.mark.parametrize(("blocked_at", "status"), [(0, 429), (4, 403), (7, 429), (8, 403)])
def test_a_circuit_open_from_any_host_stops_the_run(
    site, adapter, fetcher, queue, blocked_at, status
):
    """The breaker is per host since #43, but the run still stops on the first CircuitOpen,
    whichever host raised it: the hosts that are fine are not worked on alone."""
    bases = [A, B, C]
    tasks = [make_task("results", f"k{i}", base=bases[i % 3]) for i in range(9)]
    for i, t in enumerate(tasks):
        if i == blocked_at:
            site.add(t.url, "denied", status=status)
        else:
            site.add(t.url, f"MARKER {t.key}")
    queue.enqueue(tasks)
    blocked = tasks[blocked_at]
    host = httpx.URL(blocked.url).host

    summary = _run(adapter, fetcher, queue)

    assert summary.circuit_opened
    assert str(status) in summary.stopped_reason and host in summary.stopped_reason
    assert site.total_hits == blocked_at + 1
    assert (summary.requested, summary.fetched, summary.failed) == (blocked_at + 1, blocked_at, 1)
    assert queue.counts() == _nonzero(
        {
            ("results", "fetched"): blocked_at,
            ("results", "failed"): 1,
            ("results", "pending"): 8 - blocked_at,
        }
    )
    row = queue.get("results", blocked.key)
    assert (row["status"], row["http_status"]) == ("failed", status)


def test_a_failure_streak_on_one_host_stops_the_run_while_other_hosts_succeed(
    site, adapter, make_fetcher, queue, http
):
    """a.test's successes reset the run's own failure streak every other task, so only b.test's
    breaker in the fetcher can stop this run: its CircuitOpen must."""
    http["max_retries"] = 0
    http["circuit_breaker"]["max_consecutive_failures"] = 3
    fetcher = make_fetcher()
    tasks = [make_task("results", f"k{i:02d}", base=A if i % 2 == 0 else B) for i in range(12)]
    for t in tasks:
        if t.url.startswith(B):
            site.add(t.url, "oops", status=500)
        else:
            site.add(t.url, f"MARKER {t.key}")
    queue.enqueue(tasks)

    summary = _run(adapter, fetcher, queue, http=http)

    assert summary.circuit_opened and "b.test" in summary.stopped_reason
    assert (site.hits_on("a.test"), site.hits_on("b.test")) == (3, 3)
    assert (summary.fetched, summary.failed) == (3, 3)
    assert queue.counts() == {
        ("results", "fetched"): 3,
        ("results", "failed"): 3,
        ("results", "pending"): 6,
    }
    assert queue.get("results", "k05")["http_status"] == 500

    # b.test stays shut for this fetcher: the next run stops at its first b.test task, unrequested
    again = _run(adapter, fetcher, queue, http=http)
    assert again.circuit_opened and "b.test" in again.stopped_reason
    assert (site.hits_on("a.test"), site.hits_on("b.test")) == (4, 3)
    assert (again.fetched, again.failed) == (1, 1)
    assert queue.get("results", "k07")["status"] == "failed"
    assert queue.counts()[("results", "pending")] == 4


@pytest.mark.parametrize(
    ("quiet_after", "why"),
    [(0, "racing until 17:00"), (1, "window opened"), (4, "meeting in progress")],
)
def test_quiet_hours_stop_the_run_before_the_next_task(
    site, adapter, fetcher, queue, quiet_after, why
):
    tasks = [make_task("results", f"k{i}") for i in range(7)]
    for t in tasks:
        site.add(t.url, f"MARKER {t.key}")
    queue.enqueue(tasks)

    def quiet():
        return why if site.total_hits >= quiet_after else None

    summary = _run(adapter, fetcher, queue, quiet=quiet)

    assert summary.stopped_reason == f"quiet hours: {why}"
    assert not summary.circuit_opened
    assert site.total_hits == quiet_after  # the page in flight completed, nothing more was asked
    assert (summary.requested, summary.fetched, summary.failed) == (quiet_after, quiet_after, 0)
    assert queue.counts() == _nonzero(
        {("results", "fetched"): quiet_after, ("results", "pending"): 7 - quiet_after}
    )


def test_a_quiet_callback_that_never_fires_lets_the_run_finish(site, adapter, fetcher, queue):
    tasks = [make_task("results", f"k{i}") for i in range(5)]
    for t in tasks:
        site.add(t.url, f"MARKER {t.key}")
    queue.enqueue(tasks)
    asked_after: list[int] = []

    def quiet():
        asked_after.append(site.total_hits)
        return None

    summary = _run(adapter, fetcher, queue, quiet=quiet)

    assert summary.stopped_reason == "queue empty" and summary.fetched == 5
    assert set(range(5)) <= set(asked_after)  # asked before each of the five requests


def test_without_a_queue_plan_and_run_use_the_source_manifest(small_site, adapter, fetcher):
    assert plan_backfill(SOURCE, START, END, adapter=adapter) == 1

    summary = run_backfill(SOURCE, adapter=adapter, fetcher=fetcher)

    assert summary.stopped_reason == "queue empty" and summary.fetched == 6
    with FetchQueue(manifest_path(SOURCE)) as q:
        assert q.counts() == {
            ("fixture", "fetched"): 1,
            ("results", "fetched"): 2,
            ("horse", "fetched"): 3,
        }


# ---------------------------------------------------------------- HKTrack test_ingest_runner.py


@pytest.mark.parametrize("months", [["2026-09"], ["2026-07", "2026-08", "2026-09"]])
def test_plan_enqueues_fixture_tasks_once(queue, months):
    adapter = FakeAdapter(months=months)
    assert plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue) == len(months)
    assert plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue) == 0
    assert queue.counts() == {("fixture", "pending"): len(months)}
    assert adapter.planned == [(START, END), (START, END)]


def test_fixture_to_meetings_to_followups(small_site, adapter, fetcher, queue):
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    summary = _run(adapter, fetcher, queue)

    assert queue.counts() == {
        ("fixture", "fetched"): 1,
        ("results", "fetched"): 2,
        ("horse", "fetched"): 3,
    }
    assert summary.requested == 6
    assert summary.fetched == 6
    assert (summary.from_cache, summary.empty, summary.abandoned, summary.failed) == (0, 0, 0, 0)
    assert summary.enqueued_followups == 5  # 2 results + 3 distinct horses (H2 appears twice)
    assert summary.stopped_reason == "queue empty"
    assert all(n == 1 for n in small_site.hits.values()) and small_site.total_hits == 6
    assert raw_page_path(SOURCE, "horse", 2026, "H2").exists()
    # fixture pages go through meeting_tasks, never through parse()
    assert sorted(adapter.parsed) == [
        "horse/H1", "horse/H2", "horse/H3", "results/2026-09-16_R1", "results/2026-09-20_R1",
    ]  # fmt: skip

    row = queue.get("results", "2026-09-16_R1")
    assert (row["http_status"], row["attempts"], row["error"]) == (200, 1, None)
    assert row["bytes"] == len(b"MARKER\nhorse:H1\nhorse:H2\n") and len(row["sha256"]) == 64


def test_higher_priority_followups_run_before_older_work(small_site, adapter, fetcher, queue):
    order: list[str] = []
    small_site.on_request = lambda request: order.append(request.url.path)
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    _run(adapter, fetcher, queue)
    assert order == [
        "/fixture/2026-09", "/results/2026-09-16_R1", "/results/2026-09-20_R1",
        "/horse/H1", "/horse/H2", "/horse/H3",
    ]  # fmt: skip


def test_empty_pages_are_not_followed_and_abandoned_pages_are_left_to_the_adapter(
    site, adapter, fetcher, queue
):
    """An abandoned page is parsed and the ADAPTER decides what it leads to (the fake one follows
    every horse line); an empty page is not parsed at all."""
    site.add(f"{BASE}/fixture/2026-09", "MARKER\nmeeting:none\nmeeting:washout\nmeeting:ok\n")
    site.add(f"{BASE}/results/none", "<html>MARKER No information.</html>\nhorse:H8\n")
    site.add(f"{BASE}/results/washout", "MARKER ABANDONED\nhorse:H9\n")
    site.add(f"{BASE}/results/ok", "MARKER\nhorse:H1\n")
    site.add(f"{BASE}/horse/H1", "MARKER")
    site.add(f"{BASE}/horse/H9", "MARKER")
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)

    summary = _run(adapter, fetcher, queue)

    assert _statuses(queue) == {
        "fixture/2026-09": "fetched",
        "results/none": "empty",
        "results/washout": "abandoned",
        "results/ok": "fetched",
        "horse/H1": "fetched",
        "horse/H9": "fetched",
    }
    assert (summary.fetched, summary.empty, summary.abandoned, summary.failed) == (4, 1, 1, 0)
    assert "results/none" not in adapter.parsed and queue.get("horse", "H8") is None
    assert "results/washout" in adapter.parsed
    assert fetcher.read_cached("results", 2026, "none") is not None
    assert fetcher.read_cached("results", 2026, "washout") is not None

    # cache-first: a second run has nothing to do and asks the site for nothing
    hits = site.total_hits
    again = _run(adapter, fetcher, queue)
    assert (again.requested, again.stopped_reason, site.total_hits) == (0, "queue empty", hits)


@pytest.mark.parametrize(
    ("limit", "n_tasks"), [(1, 4), (3, 6), (None, 12)], ids=["limit-1", "limit-3", "default-10"]
)
def test_blocked_streak_opens_the_circuit(site, adapter, fetcher, queue, http, limit, n_tasks):
    if limit is not None:
        http["circuit_breaker"]["max_consecutive_marker_misses"] = limit
    expected = 10 if limit is None else limit
    tasks = [make_task("results", f"k{i:02d}") for i in range(n_tasks)]
    for t in tasks:
        site.add(t.url, "<html>Please verify you are human</html>")
    queue.enqueue(tasks)

    summary = _run(adapter, fetcher, queue, http=http)

    assert summary.stopped_reason.startswith("circuit open")
    assert summary.circuit_opened
    assert "marker" in summary.stopped_reason
    assert site.total_hits == expected
    assert queue.counts() == {
        ("results", "failed"): expected,
        ("results", "pending"): n_tasks - expected,
    }
    assert summary.failed == expected and summary.fetched == 0
    row = queue.get("results", "k00")
    assert row["error"].startswith("blocked") and row["http_status"] == 200
    # a blocked body must not poison the cache, otherwise a retry would never reach the site again
    assert fetcher.read_cached("results", 2026, "k00") is None


def test_isolated_blocked_pages_do_not_open_the_circuit(site, adapter, fetcher, queue, http):
    """Also HKTrack review2's test_a_real_success_from_the_network_still_resets_the_streaks: the
    same alternating pages, the same claim."""
    http["circuit_breaker"]["max_consecutive_marker_misses"] = 2
    tasks = [make_task("results", f"k{i}") for i in range(6)]
    for i, t in enumerate(tasks):
        site.add(t.url, "captcha" if i % 2 == 0 else "MARKER")
    queue.enqueue(tasks)

    summary = _run(adapter, fetcher, queue, http=http)

    assert summary.stopped_reason == "queue empty" and not summary.circuit_opened
    assert queue.counts() == {("results", "failed"): 3, ("results", "fetched"): 3}

    # retry-failed puts them back and they are requested again (nothing was cached)
    for t in tasks:
        site.add(t.url, "MARKER")
    assert queue.reset_failed() == 3
    _run(adapter, fetcher, queue, http=http)
    assert queue.counts() == {("results", "fetched"): 6}
    assert site.total_hits == 9


def test_parse_exception_is_tolerated_and_the_raw_page_kept(site, adapter, fetcher, queue):
    broken, fine = make_task("results", "broken", priority=9), make_task("results", "fine")
    site.add(broken.url, "MARKER BROKEN\nhorse:H1\n")
    site.add(fine.url, "MARKER\nhorse:H2\n")
    site.add(f"{BASE}/horse/H2", "MARKER")
    queue.enqueue([broken, fine])

    summary = _run(adapter, fetcher, queue)

    row = queue.get("results", "broken")
    assert row["status"] == "fetched"
    assert row["error"] == "parse: unexpected table layout"
    assert fetcher.read_cached("results", 2026, "broken").body == b"MARKER BROKEN\nhorse:H1\n"
    assert summary.stopped_reason == "queue empty"
    assert (summary.fetched, summary.failed) == (3, 0)
    assert queue.get("horse", "H2")["status"] == "fetched"


def test_broken_followup_derivation_is_tolerated_too(small_site, fetcher, queue):
    class BadFollowups(FakeAdapter):
        def meeting_tasks(self, fixture_pages):
            raise KeyError("fixture layout changed")

    adapter = BadFollowups()
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    summary = _run(adapter, fetcher, queue)
    row = queue.get("fixture", "2026-09")
    assert row["status"] == "fetched" and row["error"].startswith("parse: ")
    assert summary.stopped_reason == "queue empty"


@pytest.mark.parametrize(("budget", "cached"), [(1, 0), (3, 2), (7, 0)])
def test_max_requests_budget_is_honoured(site, adapter, fetcher, queue, budget, cached):
    """max_requests counts tasks that went to the network; cache hits are free."""
    tasks = [make_task("results", f"k{i:02d}") for i in range(10)]
    _warm_cache(site, fetcher, tasks[:cached], "MARKER")
    for t in tasks[cached:]:
        site.add(t.url, "MARKER")
    queue.enqueue(tasks)
    warmed = site.total_hits

    summary = _run(adapter, fetcher, queue, max_requests=budget)

    assert site.total_hits - warmed == budget
    assert (summary.requested, summary.fetched, summary.stopped_reason) == (
        budget, budget + cached, "max_requests",
    )  # fmt: skip
    assert summary.from_cache == cached
    assert queue.counts() == {
        ("results", "fetched"): budget + cached,
        ("results", "pending"): 10 - budget - cached,
    }


def test_max_minutes_budget_is_honoured(site, adapter, fetcher, queue, clock):
    tasks = [make_task("results", f"k{i:02d}") for i in range(10)]
    for t in tasks:
        site.add(t.url, "MARKER")
    queue.enqueue(tasks)
    site.on_request = lambda request: clock.advance(25.0)  # every page takes 25 s

    summary = _run(adapter, fetcher, queue, max_minutes=1.0, clock=clock)

    assert summary.stopped_reason == "max_minutes"
    assert site.total_hits == 3
    assert summary.elapsed_s >= 60.0


def test_resume_after_interruption_rerequests_nothing(
    small_site, adapter, fetcher, queue, make_fetcher
):
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    first = _run(adapter, fetcher, queue, max_requests=4)
    assert first.requested == 4 and small_site.total_hits == 4

    # a new process: fresh queue connection and fetcher over the same files
    queue2 = FetchQueue(queue.path)
    fetcher2 = make_fetcher()
    second = _run(adapter, fetcher2, queue2)
    queue2.close()

    assert second.requested == 2 and second.from_cache == 0
    assert small_site.total_hits == 6 and all(n == 1 for n in small_site.hits.values())
    assert queue.counts() == {
        ("fixture", "fetched"): 1,
        ("results", "fetched"): 2,
        ("horse", "fetched"): 3,
    }


def test_page_written_but_not_marked_is_served_from_cache(small_site, adapter, fetcher, queue):
    """The kill -9 window: raw file on disk, manifest still says pending."""
    task = make_task("results", "2026-09-16_R1")
    fetcher.fetch(task)
    assert small_site.total_hits == 1
    queue.enqueue([task])

    summary = _run(adapter, fetcher, queue)

    assert small_site.hits[task.url] == 1
    assert (summary.from_cache, summary.requested, summary.fetched) == (1, 2, 3)
    assert queue.get("results", task.key)["status"] == "fetched"
    assert queue.get("horse", "H1")["status"] == "fetched"  # follow-ups were still derived


def test_page_types_filter(small_site, adapter, fetcher, queue):
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    summary = _run(adapter, fetcher, queue, page_types=["fixture", "results"])
    assert summary.stopped_reason == "queue empty"
    assert queue.counts() == {
        ("fixture", "fetched"): 1,
        ("results", "fetched"): 2,
        ("horse", "pending"): 3,
    }


def test_failing_page_is_marked_failed_and_the_run_continues(site, adapter, fetcher, queue):
    bad, good = make_task("results", "bad", priority=5), make_task("results", "good")
    site.add(bad.url, "server error", status=500)
    site.add(good.url, "MARKER")
    queue.enqueue([bad, good])

    summary = _run(adapter, fetcher, queue)

    row = queue.get("results", "bad")
    assert (row["status"], row["http_status"], row["attempts"]) == ("failed", 500, 1)
    assert "500" in row["error"]
    assert site.hits[bad.url] == 5
    assert queue.get("results", "good")["status"] == "fetched"
    assert (summary.failed, summary.fetched, summary.stopped_reason) == (1, 1, "queue empty")


@pytest.mark.parametrize("status", [403, 429])
def test_stop_status_stops_the_run(site, adapter, fetcher, queue, status):
    tasks = [make_task("results", f"k{i}") for i in range(4)]
    site.add(tasks[0].url, "MARKER")
    site.add(tasks[1].url, "denied", status=status)
    site.add(tasks[2].url, "MARKER")
    site.add(tasks[3].url, "MARKER")
    queue.enqueue(tasks)

    summary = _run(adapter, fetcher, queue)

    assert summary.circuit_opened and str(status) in summary.stopped_reason
    assert queue.get("results", "k1")["http_status"] == status
    assert site.total_hits == 2
    assert queue.counts() == {
        ("results", "fetched"): 1,
        ("results", "failed"): 1,
        ("results", "pending"): 2,
    }


def test_consecutive_failures_stop_the_run(site, adapter, make_fetcher, queue, http):
    http["circuit_breaker"]["max_consecutive_failures"] = 3
    tasks = [make_task("results", f"k{i}") for i in range(6)]
    for t in tasks:
        site.add(t.url, "gone", status=404)
    queue.enqueue(tasks)
    fetcher2 = make_fetcher(sleep=lambda s: None)

    summary = _run(adapter, fetcher2, queue, http=http)

    assert summary.circuit_opened and "3 consecutive" in summary.stopped_reason
    assert site.total_hits == 3
    assert queue.counts() == {("results", "failed"): 3, ("results", "pending"): 3}


def test_unexpected_errors_are_contained_and_bounded(site, adapter, queue, http):
    """A local fault (disk full, bad key) must neither crash the run nor burn through the whole
    queue."""

    class BrokenDisk:
        def __init__(self):
            self.calls = 0

        def fetch(self, task, *, force=False):
            self.calls += 1
            raise OSError("No space left on device")

    http["circuit_breaker"]["max_consecutive_failures"] = 4
    queue.enqueue([make_task("results", f"k{i}") for i in range(10)])
    broken = BrokenDisk()

    summary = _run(adapter, broken, queue, http=http)

    assert broken.calls == 4
    assert summary.circuit_opened and summary.failed == 4
    assert "No space left" in queue.get("results", "k0")["error"]
    assert queue.counts()[("results", "pending")] == 6


def test_sigterm_finishes_the_current_page_then_stops(site, adapter, fetcher, queue):
    tasks = [make_task("results", f"k{i}") for i in range(5)]
    for t in tasks:
        site.add(t.url, "MARKER")
    queue.enqueue(tasks)

    def terminate_during_second(request):
        if site.total_hits == 2:
            os.kill(os.getpid(), signal.SIGTERM)

    site.on_request = terminate_during_second
    before = signal.getsignal(signal.SIGTERM)

    summary = _run(adapter, fetcher, queue)

    assert summary.stopped_reason == "interrupted"
    assert site.total_hits == 2
    assert queue.counts() == {("results", "fetched"): 2, ("results", "pending"): 3}
    assert signal.getsignal(signal.SIGTERM) is before  # handlers are restored


def test_keyboard_interrupt_returns_a_summary_and_loses_nothing(site, adapter, fetcher, queue):
    tasks = [make_task("results", f"k{i}") for i in range(4)]
    for t in tasks:
        site.add(t.url, "MARKER")
    queue.enqueue(tasks)

    def ctrl_c(request):
        if site.total_hits == 2:
            raise KeyboardInterrupt

    site.on_request = ctrl_c
    summary = _run(adapter, fetcher, queue)

    assert summary.stopped_reason == "interrupted"
    assert summary.fetched == 1
    assert queue.counts() == {("results", "fetched"): 1, ("results", "pending"): 3}

    site.on_request = None
    assert _run(adapter, fetcher, queue).fetched == 3


def test_ignored_signals_stay_ignored(site, adapter, fetcher, queue):
    """nohup and background launchers set SIG_IGN; the run must not undo that protection."""
    task = make_task("results", "k1")
    site.add(task.url, "MARKER")
    queue.enqueue([task])
    seen: list[object] = []
    site.on_request = lambda request: seen.append(signal.getsignal(signal.SIGINT))
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _run(adapter, fetcher, queue)
        assert seen == [signal.SIG_IGN]
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


def test_mutable_followup_is_requested_at_most_once_per_run(small_site, fetcher, queue):
    """Re-enqueueing a mutable task resets it to pending; pages sharing one must not multiply
    requests."""

    class MutableHorses(FakeAdapter):
        def followup_tasks(self, parsed):
            return [
                dataclasses.replace(t, mutable=True, priority=99)
                for t in super().followup_tasks(parsed)
            ]

    adapter = MutableHorses()
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    summary = _run(adapter, fetcher, queue)

    assert small_site.hits[f"{BASE}/horse/H2"] == 1  # H2 runs in both meetings
    assert small_site.total_hits == 6 and summary.stopped_reason == "queue empty"
    assert queue.counts()[("horse", "fetched")] == 3


def test_second_run_on_the_same_manifest_is_refused(site, adapter, fetcher, queue, make_fetcher):
    """Single worker: two overlapping runs would double the request rate against the site."""
    tasks = [make_task("results", f"k{i}") for i in range(3)]
    for t in tasks:
        site.add(t.url, "MARKER")
    queue.enqueue(tasks)
    nested: list[RunSummary | None] = []

    def start_a_second_run(request):
        if not nested:
            nested.append(None)
            q2 = FetchQueue(queue.path)
            f2 = make_fetcher()
            nested[0] = _run(adapter, f2, q2)
            q2.close()

    site.on_request = start_a_second_run
    outer = _run(adapter, fetcher, queue)

    assert nested[0].stopped_reason.startswith("already running")
    assert nested[0].requested == 0
    assert outer.fetched == 3 and site.total_hits == 3
    # the lock is released afterwards
    assert _run(adapter, fetcher, queue).stopped_reason == "queue empty"


def test_crash_leaves_a_log_line_and_cleans_up(site, adapter, fetcher, queue, monkeypatch, data):
    task = make_task("results", "k1")
    site.add(task.url, "MARKER")
    queue.enqueue([task])

    def broken_mark(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    before = signal.getsignal(signal.SIGTERM)
    with monkeypatch.context() as patch:
        patch.setattr(queue, "mark", broken_mark)
        with pytest.raises(sqlite3.OperationalError):
            _run(adapter, fetcher, queue)

    assert signal.getsignal(signal.SIGTERM) is before
    (log,) = _logs(data)
    text = log.read_text()
    assert "crashed" in text and "OperationalError" in text and "disk I/O error" in text
    # lock released, page kept: the next run finishes the task from the cache without a request
    again = _run(adapter, fetcher, queue)
    assert (again.from_cache, again.fetched, site.total_hits) == (1, 1, 1)


def test_progress_goes_to_a_log_file_not_stdout(small_site, adapter, fetcher, queue, capsys, data):
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    summary = _run(adapter, fetcher, queue, log_every=2)

    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    logs = _logs(data)
    assert len(logs) == 1
    assert Path(summary.log_path).resolve() == logs[0].resolve()
    lines = logs[0].read_text().splitlines()
    progress = [ln for ln in lines if " PROGRESS " in ln and "req/s" in ln and "done=" in ln]
    assert len(progress) == 3  # 6 tasks, one line per 2
    assert "queue empty" in lines[-1]
    assert len(lines) <= 8


def test_failures_are_logged_with_the_task_identity(site, adapter, fetcher, queue):
    bad = make_task("results", "bad")
    site.add(bad.url, "gone", status=404)
    queue.enqueue([bad])
    summary = _run(adapter, fetcher, queue)
    text = Path(summary.log_path).read_text()
    assert "results/bad" in text and "404" in text


def test_expand_followups_is_offline_and_idempotent(small_site, fetcher, queue):
    old_parser = FakeAdapter(follow_horses=False)
    plan_backfill(SOURCE, START, END, adapter=old_parser, queue=queue)
    _run(old_parser, fetcher, queue)
    assert queue.counts() == {("fixture", "fetched"): 1, ("results", "fetched"): 2}
    hits = small_site.total_hits

    fixed = FakeAdapter()
    assert expand_followups(SOURCE, adapter=fixed, queue=queue, fetcher=fetcher) == 3
    assert expand_followups(SOURCE, adapter=fixed, queue=queue, fetcher=fetcher) == 0
    assert small_site.total_hits == hits
    assert queue.counts()[("horse", "pending")] == 3
    assert queue.counts()[("results", "fetched")] == 2


def test_expand_followups_page_type_filter_and_error_bookkeeping(site, fetcher, queue):
    adapter = FakeAdapter()
    site.add(f"{BASE}/fixture/2026-09", "MARKER\nmeeting:m1\n")
    site.add(f"{BASE}/results/m1", "MARKER BROKEN\nhorse:H1\n")
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    _run(adapter, fetcher, queue)
    assert queue.get("results", "m1")["error"].startswith("parse: ")

    # parser "fixed": the cached page now parses, the follow-up appears and the error is cleared
    adapter.parse = lambda page: {"runners": [{"horse_id": "H1"}]}
    assert (
        expand_followups(
            SOURCE, page_types=["fixture"], adapter=adapter, queue=queue, fetcher=fetcher
        )
        == 0
    )
    assert queue.get("horse", "H1") is None
    assert (
        expand_followups(
            SOURCE, page_types=["results"], adapter=adapter, queue=queue, fetcher=fetcher
        )
        == 1
    )
    row = queue.get("results", "m1")
    assert (row["status"], row["error"], row["attempts"]) == ("fetched", None, 1)


def test_expand_followups_survives_pages_that_still_do_not_parse(site, fetcher, queue):
    adapter = FakeAdapter()
    site.add(f"{BASE}/results/bad", "MARKER BROKEN")
    site.add(f"{BASE}/results/good", "MARKER\nhorse:H5\n")
    queue.enqueue(
        [make_task("results", "bad", priority=2), make_task("results", "good", priority=1)]
    )
    _run(FakeAdapter(follow_horses=False), fetcher, queue)

    assert expand_followups(SOURCE, adapter=adapter, queue=queue, fetcher=fetcher) == 1
    assert queue.get("results", "bad")["error"].startswith("parse: ")


# -------------------------------------------------------------- HKTrack test_ingest_adversarial.py


class CardAdapter(FakeAdapter):
    """Like a real adapter: the first race of a card lists the other races ('race:<key>' lines),
    and that must work when the first race itself was abandoned (the adapter decides what an
    abandoned page yields)."""

    def followup_tasks(self, parsed):
        races = [make_task("results", r, priority=40) for r in parsed.get("races", [])]
        if parsed.get("abandoned"):
            return races
        return races + super().followup_tasks(parsed)

    def parse(self, page):
        parsed = super().parse(page)
        lines = page.body.decode().splitlines()
        parsed["races"] = [ln.split(":", 1)[1] for ln in lines if ln.startswith("race:")]
        parsed["abandoned"] = "ABANDONED" in page.body.decode()
        return parsed


def test_abandoned_first_race_still_leads_to_the_rest_of_the_card(site, fetcher, queue):
    site.add(f"{BASE}/fixture/2026-09", "MARKER\nmeeting:D_R01\n")
    site.add(f"{BASE}/results/D_R01", "MARKER ABANDONED\nrace:D_R02\nhorse:H9\n")
    site.add(f"{BASE}/results/D_R02", "MARKER\nhorse:H1\n")
    site.add(f"{BASE}/horse/H1", "MARKER")
    adapter = CardAdapter()
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)

    summary = run_backfill(SOURCE, adapter=adapter, fetcher=fetcher, queue=queue)

    assert queue.get("results", "D_R01")["status"] == "abandoned"
    assert queue.get("results", "D_R02")["status"] == "fetched"
    assert queue.get("horse", "H1")["status"] == "fetched"
    assert queue.get("horse", "H9") is None  # the adapter, not the runner, decided to skip these
    assert (summary.abandoned, summary.failed, summary.stopped_reason) == (1, 0, "queue empty")


def test_parse_error_on_an_abandoned_page_keeps_the_page_and_the_status(site, fetcher, queue):
    site.add(f"{BASE}/fixture/2026-09", "MARKER\nmeeting:D_R01\n")
    site.add(f"{BASE}/results/D_R01", "MARKER ABANDONED BROKEN\n")
    adapter = CardAdapter()
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    summary = run_backfill(SOURCE, adapter=adapter, fetcher=fetcher, queue=queue)
    row = queue.get("results", "D_R01")
    assert row["status"] == "abandoned" and row["error"].startswith("parse: ")
    assert summary.stopped_reason == "queue empty"


def test_expand_also_walks_abandoned_pages(site, fetcher, queue):
    site.add(f"{BASE}/fixture/2026-09", "MARKER\nmeeting:D_R01\n")
    site.add(f"{BASE}/results/D_R01", "MARKER ABANDONED\nrace:D_R02\n")
    site.add(f"{BASE}/results/D_R02", "MARKER\n")
    blind = FakeAdapter(follow_horses=False)  # an adapter version that did not know 'race:' lines
    plan_backfill(SOURCE, START, END, adapter=blind, queue=queue)
    run_backfill(SOURCE, adapter=blind, fetcher=fetcher, queue=queue)
    assert queue.get("results", "D_R02") is None

    hits = site.total_hits
    assert expand_followups(SOURCE, adapter=CardAdapter(), queue=queue, fetcher=fetcher) == 1
    assert queue.get("results", "D_R02")["status"] == "pending"
    assert queue.get("results", "D_R01")["status"] == "abandoned"
    assert site.total_hits == hits


class RaceDayAdapter(FakeAdapter):
    """Meetings dated today (or later) are provisional, like a real adapter's
    `race_date >= today`."""

    def __init__(self) -> None:
        super().__init__()
        self.today = "2026-09-23"

    def fixture_tasks(self, start, end):
        return [dataclasses.replace(t, mutable=True) for t in super().fixture_tasks(start, end)]

    def meeting_tasks(self, fixture_pages):
        return [
            dataclasses.replace(t, mutable=t.key[:10] >= self.today)
            for t in super().meeting_tasks(fixture_pages)
        ]


def test_meeting_scraped_before_it_was_run_is_not_empty_forever(site, fetcher, queue):
    adapter = RaceDayAdapter()
    site.add(f"{BASE}/fixture/2026-09", "MARKER\nmeeting:2026-09-23_R1\n")
    site.add(f"{BASE}/results/2026-09-23_R1", "MARKER No information.")
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    run_backfill(SOURCE, adapter=adapter, fetcher=fetcher, queue=queue)
    assert queue.get("results", "2026-09-23_R1")["status"] == "empty"

    # next morning: the results are up, and the adapter no longer calls that meeting mutable
    adapter.today = "2026-09-24"
    site.add(f"{BASE}/results/2026-09-23_R1", "MARKER\nhorse:H1\n")
    site.add(f"{BASE}/horse/H1", "MARKER")
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    run_backfill(SOURCE, adapter=adapter, fetcher=fetcher, queue=queue)

    assert queue.get("results", "2026-09-23_R1")["status"] == "fetched"
    assert fetcher.read_cached("results", 2026, "2026-09-23_R1").body == b"MARKER\nhorse:H1\n"
    assert queue.get("horse", "H1")["status"] == "fetched"

    # and from then on it is immutable: a third day costs one fixture request and nothing else
    adapter.today = "2026-09-25"
    hits = site.total_hits
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    run_backfill(SOURCE, adapter=adapter, fetcher=fetcher, queue=queue)
    assert site.total_hits == hits + 1
    assert site.hits[f"{BASE}/results/2026-09-23_R1"] == 2


def test_manifest_error_while_enqueueing_followups_is_not_recorded_as_a_parse_error(
    small_site, fetcher, queue, monkeypatch
):
    adapter = FakeAdapter()
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)
    real_enqueue = queue.enqueue

    def locked(tasks):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(queue, "enqueue", locked)
    with pytest.raises(sqlite3.OperationalError):
        run_backfill(SOURCE, adapter=adapter, fetcher=fetcher, queue=queue)
    row = queue.get("fixture", "2026-09")
    # not 'fetched' with its follow-ups lost
    assert row["status"] == "pending" and row["error"] is None

    monkeypatch.setattr(queue, "enqueue", real_enqueue)
    hits = small_site.total_hits
    summary = run_backfill(SOURCE, adapter=adapter, fetcher=fetcher, queue=queue)
    assert summary.stopped_reason == "queue empty" and queue.counts()[("horse", "fetched")] == 3
    assert small_site.hits[f"{BASE}/fixture/2026-09"] == 1 and small_site.total_hits == hits + 5


# ---------------------------------------------------------------- HKTrack test_ingest_review2.py


def test_cache_hits_do_not_keep_the_marker_breaker_shut(site, fetcher, queue, http):
    """Every page the SITE serves is a block page. Cache hits in between say nothing about the
    site, so they must not reset the streak; otherwise the run grinds through the whole queue
    collecting block pages."""
    http["circuit_breaker"]["max_consecutive_marker_misses"] = 3
    tasks = [make_task("results", f"k{i:02d}") for i in range(20)]
    _warm_cache(site, fetcher, tasks[1::2])
    for t in tasks[0::2]:
        site.add(t.url, "<html>Please verify you are human</html>")
    queue.enqueue(tasks)
    before = site.total_hits

    summary = run_backfill(SOURCE, adapter=FakeAdapter(), fetcher=fetcher, queue=queue, http=http)

    assert summary.circuit_opened and "marker" in summary.stopped_reason
    assert site.total_hits - before == 3  # not 10
    assert queue.counts()[("results", "failed")] == 3


def test_cache_hits_do_not_keep_the_failure_breaker_shut(site, fetcher, queue, http, monkeypatch):
    """A local fault (disk full) fails every page that comes from the network: five of those stop
    the run even when cached pages are handled fine in between. Every such page cost the site a
    request for nothing."""
    tasks = [make_task("results", f"k{i:02d}") for i in range(30)]
    _warm_cache(site, fetcher, tasks[1::2])
    for t in tasks[0::2]:
        site.add(t.url, "MARKER fresh")
    queue.enqueue(tasks)
    before = site.total_hits

    def disk_full(path, body):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("horsetrack.core.ingest.fetcher._write_gzip_atomic", disk_full)
    summary = run_backfill(SOURCE, adapter=FakeAdapter(), fetcher=fetcher, queue=queue, http=http)

    assert summary.circuit_opened
    assert site.total_hits - before == 5  # not 15
    assert summary.failed == 5


def test_block_page_does_not_destroy_the_last_good_copy_of_a_mutable_page(site, fetcher, queue):
    card = make_task("racecard", "2026-09-23_R01", mutable=True)
    site.add(card.url, "MARKER card as of 18:00")
    queue.enqueue([card])
    run_backfill(SOURCE, adapter=FakeAdapter(), fetcher=fetcher, queue=queue)
    assert fetcher.read_cached("racecard", 2026, card.key).body == b"MARKER card as of 18:00"

    site.add(card.url, "<html>Access denied</html>")  # HTTP 200, no marker
    queue.enqueue([card])
    run_backfill(SOURCE, adapter=FakeAdapter(), fetcher=fetcher, queue=queue)

    row = queue.get("racecard", card.key)
    assert row["status"] == "failed" and row["error"].startswith("blocked")
    kept = fetcher.read_cached("racecard", 2026, card.key)
    assert kept is not None and kept.body == b"MARKER card as of 18:00"
    path = raw_page_path(SOURCE, "racecard", 2026, card.key)
    assert path.with_name(path.name + ".blocked").exists()  # the block page is kept for diagnosis

    # and the retry goes back to the site, not to the restored copy
    site.add(card.url, "MARKER card as of 19:00")
    queue.reset_failed()
    run_backfill(SOURCE, adapter=FakeAdapter(), fetcher=fetcher, queue=queue)
    assert fetcher.read_cached("racecard", 2026, card.key).body == b"MARKER card as of 19:00"
    assert site.hits[card.url] == 3


def test_a_blocked_first_fetch_still_leaves_nothing_in_the_cache(site, fetcher, queue):
    task = make_task("results", "k1")
    site.add(task.url, "captcha")
    queue.enqueue([task])
    run_backfill(SOURCE, adapter=FakeAdapter(), fetcher=fetcher, queue=queue)
    assert fetcher.read_cached("results", 2026, "k1") is None
    folder = raw_page_path(SOURCE, "results", 2026, "k1").parent
    assert sorted(p.name for p in folder.iterdir()) == ["k1.html.gz.blocked"]


class BadFollowupAdapter(FakeAdapter):
    def __init__(self, **overrides) -> None:
        super().__init__()
        self.overrides = overrides

    def followup_tasks(self, parsed):
        tasks = super().followup_tasks(parsed)
        if not tasks:
            return tasks
        return [dataclasses.replace(tasks[0], **self.overrides), *tasks[1:]]


@pytest.mark.parametrize(
    "overrides",
    [
        {"key": date(2026, 9, 20)},
        {"url": None},
        {"key": ""},
        {"url": ""},
        {"year": None},
        {"body": b"x"},
    ],
    ids=["key-not-str", "url-none", "key-empty", "url-empty", "year-none", "body-bytes"],
)
def test_unstorable_followup_is_a_parse_error_not_a_crash_or_a_silent_drop(
    small_site, fetcher, queue, overrides
):
    """A follow-up the manifest cannot store is an adapter bug: noted on the page like any other
    parse error, never a crash (the page would stay pending and crash every restart) nor a silent
    drop."""
    adapter = BadFollowupAdapter(**overrides)
    plan_backfill(SOURCE, START, END, adapter=adapter, queue=queue)

    summary = run_backfill(SOURCE, adapter=adapter, fetcher=fetcher, queue=queue)

    assert summary.stopped_reason == "queue empty"
    for key in ("2026-09-16_R1", "2026-09-20_R1"):
        row = queue.get("results", key)
        assert row["status"] == "fetched" and row["error"].startswith("parse: ")
    assert queue.count_parse_errors() == 2
    # all or nothing per page; expand brings the follow-ups back once the adapter is fixed
    assert ("horse", "fetched") not in queue.counts()
