"""The backfill loop, ported from HKTrack: queue -> fetcher -> adapter.classify -> manifest ->
follow-up tasks.

Built to run unattended for many hours: every outcome is committed per task, a broken parser never
stops the run or loses a raw page, and anything that looks like the site pushing back stops it for
good. The fetcher's circuit breaker is per host, but the first CircuitOpen from any host stops the
whole run. The run's own streaks (failed tasks, pages without the content marker) count across
the whole run, as in HKTrack.
"""

from __future__ import annotations

import signal
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any

import httpx

from horsetrack.core.ingest.adapters import SourceAdapter, get_adapter
from horsetrack.core.ingest.fetcher import CircuitOpen, Fetcher, FetchError
from horsetrack.core.ingest.queue import STATUS_FAILED, FetchQueue
from horsetrack.core.ingest.types import (
    STATUS_ABANDONED,
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FETCHED,
    FetchTask,
    RawPage,
)
from horsetrack.core.paths import logs_dir, manifest_path

try:  # POSIX only; without it the single-worker lock is skipped
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

FIXTURE_PAGE_TYPE = "fixture"  # pages that list meetings; every other page goes through parse()

REASON_QUEUE_EMPTY = "queue empty"
REASON_MAX_REQUESTS = "max_requests"
REASON_MAX_MINUTES = "max_minutes"
REASON_INTERRUPTED = "interrupted"
REASON_CIRCUIT = "circuit open"
REASON_ALREADY_RUNNING = "already running"
REASON_QUIET_HOURS = "quiet hours"

_MAX_PARSE_ERRORS_LOGGED = 20
# pages the adapter is asked for follow-ups
_FOLLOWED_STATUSES = (STATUS_FETCHED, STATUS_ABANDONED)


@dataclass
class RunSummary:
    """requested = tasks that went to the network; from_cache = tasks served by the raw cache
    instead. fetched/empty/abandoned/failed count outcomes over both."""

    requested: int = 0
    from_cache: int = 0
    fetched: int = 0
    empty: int = 0
    abandoned: int = 0
    failed: int = 0
    enqueued_followups: int = 0
    stopped_reason: str = ""
    elapsed_s: float = 0.0
    log_path: str = ""

    @property
    def processed(self) -> int:
        return self.requested + self.from_cache

    @property
    def circuit_opened(self) -> bool:
        return self.stopped_reason.startswith(REASON_CIRCUIT)


class _RunLog:
    """Append-only, line-buffered progress log under logs/<source>/: `tail -f` works and a hard
    kill loses nothing. Timestamps are UTC."""

    def __init__(self, source: str) -> None:
        folder = logs_dir(source)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"scrape-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    def write(self, message: str) -> None:
        self._fh.write(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {message}\n")

    def close(self) -> None:
        self._fh.close()


class _StopFlag:
    """First SIGINT/SIGTERM: finish the current page, then stop. A second one interrupts
    immediately."""

    def __init__(self) -> None:
        self.requested = False
        self._previous: dict[int, Any] = {}

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            # nohup / background launch: keep that protection
            if signal.getsignal(sig) is signal.SIG_IGN:
                continue
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except ValueError:  # not the main thread: the caller owns signal handling
                pass

    def restore(self) -> None:
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)
        self._previous.clear()


class _AlreadyRunning(RuntimeError):
    pass


@contextmanager
def _single_worker_lock(manifest: Path) -> Iterator[None]:
    """One backfill per manifest. Two overlapping runs would silently double the request rate."""
    if fcntl is None:  # pragma: no cover
        yield
        return
    lock_path = manifest.with_name(manifest.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise _AlreadyRunning(f"another run holds {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _breaker_limits(http: Mapping[str, Any] | None) -> tuple[int, int]:
    breaker = (http or {}).get("circuit_breaker") or {}
    return (
        int(breaker.get("max_consecutive_failures", 5)),
        int(breaker.get("max_consecutive_marker_misses", 10)),
    )


def _make_fetcher(source: str, http: Mapping[str, Any] | None) -> Fetcher:
    """The Fetcher a call builds for itself when none is passed: it needs the source's http
    mapping, which core cannot look up from a source name alone."""
    if http is None:
        raise ValueError(f"no fetcher for {source!r}: pass fetcher= or the source's http= mapping")
    return Fetcher(source, http=http)


def _followups(adapter: SourceAdapter, page: RawPage) -> list[FetchTask]:
    if page.page_type == FIXTURE_PAGE_TYPE:
        return list(adapter.meeting_tasks([page]))
    return list(adapter.followup_tasks(adapter.parse(page)))


def _one_line(exc: BaseException, limit: int = 300) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:limit]


def plan_backfill(
    source: str,
    start: date,
    end: date,
    *,
    adapter: SourceAdapter | None = None,
    queue: FetchQueue | None = None,
) -> int:
    """Enqueue the adapter's fixture tasks for [start, end]; returns how many are new."""
    adapter = adapter or get_adapter(source)
    own_queue = queue is None
    queue = queue or FetchQueue(manifest_path(source))
    try:
        return queue.enqueue(adapter.fixture_tasks(start, end))
    finally:
        if own_queue:
            queue.close()


def _host_of_url(url: str) -> str:
    """The host the Fetcher's per-host breaker files the task under (httpx's parse)."""
    try:
        return httpx.URL(url).host
    except httpx.InvalidURL:
        return ""


class _Run:
    """State of one run_backfill call; split out so the loop body stays readable."""

    def __init__(self, adapter: SourceAdapter, fetcher: Any, queue: FetchQueue,
                 http: Mapping[str, Any] | None, log: _RunLog, summary: RunSummary):  # fmt: skip
        self.adapter, self.fetcher, self.queue = adapter, fetcher, queue
        self.log, self.summary = log, summary
        self.max_failures, self.max_marker_misses = _breaker_limits(http)
        self.failure_streak = 0
        self.marker_miss_streaks: dict[str, int] = {}  # per URL host, like the Fetcher's breaker
        self.parse_errors = 0
        self.handled: set[tuple[str, str]] = set()

    def _fail(self, task: FetchTask, error: str, **outcome: Any) -> None:
        self.queue.mark(task, STATUS_FAILED, error=error, **outcome)
        self.summary.failed += 1
        self.log.write(f"FAILED {task.page_type}/{task.key} {error}")

    def _note_parse_error(self, task: FetchTask, error: str) -> None:
        self.parse_errors += 1
        if self.parse_errors <= _MAX_PARSE_ERRORS_LOGGED:
            self.log.write(f"PARSE {task.page_type}/{task.key} {error}")
        if self.parse_errors == _MAX_PARSE_ERRORS_LOGGED:
            self.log.write("PARSE further parse errors are not logged; see the manifest errors")

    def _new_followups(self, page: RawPage) -> list[FetchTask]:
        """Follow-ups, minus mutable tasks already handled in this run: enqueueing those would
        reset them to pending, and a page shared by many parents (a horse in 20 races) would be
        requested 20 times."""
        return [
            t
            for t in _followups(self.adapter, page)
            if not (t.mutable and (t.page_type, t.key) in self.handled)
        ]

    def process(self, task: FetchTask) -> str | None:
        """Handle one task; returns a stop reason when the circuit opens."""
        self.handled.add((task.page_type, task.key))
        try:
            page = self.fetcher.fetch(task)
        except CircuitOpen as exc:  # any host: the run stops, the other hosts are not worked alone
            self.summary.requested += 1
            self._fail(task, _one_line(exc), http_status=exc.http_status)
            return f"{REASON_CIRCUIT}: {_one_line(exc)}"
        except FetchError as exc:
            self.summary.requested += 1
            self._fail(task, _one_line(exc), http_status=exc.http_status, final_url=exc.final_url)
            return self._count_failure()
        except Exception as exc:  # local faults (disk, bad key): contain, never grind the queue
            # the network may have been touched: budgets must not under-count
            self.summary.requested += 1
            self._fail(task, f"{type(exc).__name__}: {_one_line(exc)}")
            return self._count_failure()

        from_cache = bool(page.meta.get("from_cache"))
        if from_cache:
            self.summary.from_cache += 1
        else:
            self.summary.requested += 1
        outcome = {
            "http_status": page.meta.get("http_status"),
            "final_url": page.meta.get("final_url"),
            "sha256": page.meta.get("sha256"),
            "n_bytes": len(page.body),
        }

        try:
            status = self.adapter.classify(task.page_type, page.body)
        except Exception as exc:
            status = f"classify raised {type(exc).__name__}: {_one_line(exc)}"

        if status == STATUS_BLOCKED:
            quarantine = getattr(self.fetcher, "quarantine", None)
            if quarantine is not None:
                quarantine(task)  # a block page must not be served from the cache on the retry
            self._fail(task, "blocked: expected content marker missing", **outcome)
            host = _host_of_url(task.url)
            streak = self.marker_miss_streaks[host] = self.marker_miss_streaks.get(host, 0) + 1
            if streak >= self.max_marker_misses:
                return (
                    f"{REASON_CIRCUIT}: {streak} consecutive pages without content marker"
                    f" from {host}"
                )
            return None
        if status not in (STATUS_FETCHED, STATUS_EMPTY, STATUS_ABANDONED):
            self._fail(task, f"unusable classification: {status}", **outcome)
            return self._count_failure()

        # a cache hit says nothing about the site (or the disk), so it must not reset a breaker
        if not from_cache:
            self.failure_streak = 0
            self.marker_miss_streaks[_host_of_url(task.url)] = 0
        error = None
        if status in _FOLLOWED_STATUSES:
            # An abandoned page is parsed too: what it still leads to (the rest of a card whose
            # first race was called off) is the adapter's decision, not ours.
            # Enqueued before the page is marked: a kill in between re-derives them from the cache.
            try:
                self.summary.enqueued_followups += self.queue.enqueue(self._new_followups(page))
            except sqlite3.Error:  # the manifest, not the parser: crash with the page pending
                raise
            except Exception as exc:
                error = f"parse: {_one_line(exc)}"
                self._note_parse_error(task, error)
        self.queue.mark(task, status, error=error, **outcome)
        setattr(self.summary, status, getattr(self.summary, status) + 1)
        return None

    def _count_failure(self) -> str | None:
        self.failure_streak += 1
        if self.failure_streak >= self.max_failures:
            return f"{REASON_CIRCUIT}: {self.failure_streak} consecutive failed tasks"
        return None


def run_backfill(
    source: str,
    *,
    max_requests: int | None = None,
    max_minutes: float | None = None,
    page_types: list[str] | None = None,
    adapter: SourceAdapter | None = None,
    fetcher: Any = None,
    queue: FetchQueue | None = None,
    log_every: int = 200,
    http: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
    quiet: Callable[[], str | None] | None = None,
) -> RunSummary:
    """Work through the pending queue until a budget, an empty queue, a signal, quiet hours or the
    circuit breaker stops.

    max_requests counts tasks that went to the network (cache hits are free). `quiet`, asked
    before every task, returns a reason while the site must not be contacted. `http` is the
    mapping the Fetcher takes; the run reads its own limits from its `circuit_breaker`
    (`max_consecutive_failures`, default 5; `max_consecutive_marker_misses`, default 10, counted
    per URL host), and
    builds the Fetcher from it when none is passed. Progress goes to
    logs/<source>/scrape-<UTC>.log under the data directory only; the caller prints the summary.
    """
    adapter = adapter or get_adapter(source)
    own_queue, own_fetcher = queue is None, fetcher is None
    queue = queue or FetchQueue(manifest_path(source))
    summary = RunSummary()
    started = clock()
    try:
        with _single_worker_lock(queue.path):
            fetcher = fetcher or _make_fetcher(source, http)
            log = _RunLog(source)
            summary.log_path = str(log.path)
            stop = _StopFlag()
            stop.install()
            try:
                run = _Run(adapter, fetcher, queue, http, log, summary)
                _loop(run, stop, clock, started, max_requests, max_minutes, page_types,
                      max(int(log_every), 1), quiet)  # fmt: skip
            except BaseException as exc:  # a bug or a local fault: leave a trace, then re-raise
                summary.stopped_reason = f"crashed: {type(exc).__name__}: {_one_line(exc)}"
                raise
            finally:
                stop.restore()
                summary.elapsed_s = clock() - started
                log.write(f"STOP {summary.stopped_reason} {_progress(summary, summary.elapsed_s)}")
                log.close()
                if own_fetcher:
                    fetcher.close()
    except _AlreadyRunning as exc:
        summary.stopped_reason = f"{REASON_ALREADY_RUNNING}: {exc}"
        summary.elapsed_s = clock() - started
    finally:
        if own_queue:
            queue.close()
    return summary


def _progress(summary: RunSummary, elapsed_s: float) -> str:
    rate = summary.requested / elapsed_s if elapsed_s > 0 else 0.0
    return (
        f"done={summary.processed} network={summary.requested} cache={summary.from_cache}"
        f" fetched={summary.fetched} empty={summary.empty} abandoned={summary.abandoned}"
        f" failed={summary.failed} followups={summary.enqueued_followups}"
        f" elapsed={elapsed_s:.0f}s {rate:.2f} req/s"
    )


def _loop(run: _Run, stop: _StopFlag, clock: Callable[[], float], started: float,
          max_requests: int | None, max_minutes: float | None, page_types: list[str] | None,
          log_every: int, quiet: Callable[[], str | None] | None = None) -> None:  # fmt: skip
    summary = run.summary
    pending = sum(n for (_, status), n in run.queue.counts().items() if status == "pending")
    run.log.write(f"START pending={pending} max_requests={max_requests} max_minutes={max_minutes}"
                  f" page_types={','.join(page_types) if page_types else 'all'}")  # fmt: skip
    try:
        while True:
            if stop.requested:
                summary.stopped_reason = REASON_INTERRUPTED
            elif max_requests is not None and summary.requested >= max_requests:
                summary.stopped_reason = REASON_MAX_REQUESTS
            elif max_minutes is not None and clock() - started >= max_minutes * 60.0:
                summary.stopped_reason = REASON_MAX_MINUTES
            elif quiet is not None and (why := quiet()):
                summary.stopped_reason = f"{REASON_QUIET_HOURS}: {why}"
            if summary.stopped_reason:
                return
            tasks = run.queue.next_pending(1, page_types)
            if not tasks:
                summary.stopped_reason = REASON_QUEUE_EMPTY
                return
            reason = run.process(tasks[0])
            if summary.processed and summary.processed % log_every == 0:
                run.log.write(f"PROGRESS {_progress(summary, clock() - started)}")
            if reason:
                summary.stopped_reason = reason
                return
    except KeyboardInterrupt:  # second Ctrl-C, or a caller without our handlers: task stays pending
        summary.stopped_reason = REASON_INTERRUPTED


def expand_followups(
    source: str,
    page_types: list[str] | None = None,
    adapter: SourceAdapter | None = None,
    queue: FetchQueue | None = None,
    fetcher: Any = None,
    *,
    http: Mapping[str, Any] | None = None,
) -> int:
    """Re-walk fetched pages from the raw cache and enqueue follow-ups that are missing (after a
    parser fix).

    Idempotent and offline: only `read_cached` is used. Parse-error notes in the manifest are
    refreshed. Without `fetcher`, one is built from `http` as in run_backfill.
    """
    adapter = adapter or get_adapter(source)
    own_queue, own_fetcher = queue is None, fetcher is None
    fetcher = fetcher or _make_fetcher(source, http)
    queue = queue or FetchQueue(manifest_path(source))
    added = 0
    try:
        for page_type in page_types or queue.page_types():
            for status in _FOLLOWED_STATUSES:
                for row in queue.iter_status(page_type, status):
                    page = fetcher.read_cached(page_type, row["year"], row["key"])
                    if page is None:
                        continue
                    try:
                        added += queue.enqueue(_followups(adapter, replace(page, url=row["url"])))
                        error = None
                    except sqlite3.Error:
                        raise
                    except Exception as exc:
                        error = f"parse: {_one_line(exc)}"
                    if error != row["error"]:
                        queue.set_error(page_type, row["key"], error)
    finally:
        if own_fetcher:
            fetcher.close()
        if own_queue:
            queue.close()
    return added
