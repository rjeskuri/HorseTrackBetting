"""Polite, cache-first HTTP fetcher, ported from HKTrack. Source-agnostic: it executes FetchTasks.

One client and one worker. Politeness is per host: each host has its own gap and jitter (the
source's `http.hosts[<host>]`, else `http.hosts.default`) and its own circuit breaker, so a request
waits only for the last request to the same host and a host that said stop shuts only itself. The
gap also covers retries and every redirect hop. Raw bodies are written gzip-compressed and
atomically to the cache; a cached immutable page is never requested again.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import os
import random
import time
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from horsetrack.core.env import get_setting
from horsetrack.core.ingest.types import FetchTask, RawPage
from horsetrack.core.paths import raw_page_path

USER_AGENT_VAR = "HORSETRACK_USER_AGENT"
_MAX_REDIRECTS = 5  # every hop is a rate-limited request of its own; a longer chain is a loop
_MAX_RETRY_AFTER_S = 300.0
_QUARANTINE_SUFFIX = ".blocked"
_PREVIOUS_SUFFIX = (
    ".prev"  # last copy of a refetchable page, so that a block page cannot destroy it
)
_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.9,*/*;q=0.8"


class CircuitOpen(RuntimeError):
    """A host told us to stop (or keeps failing). Nothing further goes to that host from this
    Fetcher; other hosts keep working."""

    def __init__(self, message: str, *, http_status: int | None = None, host: str | None = None):
        super().__init__(message)
        self.http_status = http_status
        self.host = host


class FetchError(RuntimeError):
    """One task could not be fetched (retries exhausted or a non-retryable status). The run may
    continue."""

    def __init__(
        self, message: str, *, http_status: int | None = None, final_url: str | None = None
    ):
        super().__init__(message)
        self.http_status = http_status
        self.final_url = final_url


@dataclass(frozen=True)
class _Rate:
    gap: float  # seconds between request starts, before jitter
    jitter: float  # each gap is gap * (1 +/- jitter)


@dataclass
class _HostState:
    rate: _Rate
    last_start: float | None = None
    next_gap: float = 0.0
    failures: int = 0  # consecutive failed tasks whose last request went to this host
    open_reason: str | None = None
    open_status: int | None = None


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _retry_after_seconds(response: httpx.Response) -> float:
    value = response.headers.get("retry-after", "")
    return min(float(value), _MAX_RETRY_AFTER_S) if value.isdigit() else 0.0


def _write_gzip_atomic(path: Path, body: bytes) -> None:
    """tmp file in the same directory + fsync + os.replace: readers see the old file or the new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                gz.write(body)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _keep_previous(path: Path) -> None:
    """Hard-link the current file to <name>.prev before it is overwritten. Best effort: never
    fails a fetch."""
    previous = path.with_name(path.name + _PREVIOUS_SUFFIX)
    tmp = previous.with_name(previous.name + ".tmp")
    try:
        tmp.unlink(missing_ok=True)
        os.link(path, tmp)
        os.replace(tmp, previous)
    except OSError:  # nothing to keep yet, or a file system without hard links
        pass


def _read_gzip(path: Path) -> bytes | None:
    """Cached body, or None when the file is missing or unreadable (treated as a cache miss)."""
    try:
        with gzip.open(path, "rb") as fh:
            return fh.read()
    except (OSError, EOFError, zlib.error):
        return None


def _user_agent(http: Mapping[str, Any]) -> str:
    """The HORSETRACK_USER_AGENT setting, else `http.user_agent`; blank counts as missing."""
    for value in (get_setting(USER_AGENT_VAR), http.get("user_agent")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"no User-Agent: set {USER_AGENT_VAR} or the source's http.user_agent")


def _rate(name: str, entry: Any) -> _Rate:
    where = f"http.hosts[{name!r}]"
    if not isinstance(entry, Mapping) or "requests_per_second" not in entry:
        raise ValueError(f"{where} needs requests_per_second")
    try:
        rps = float(entry["requests_per_second"])
        jitter = float(entry.get("jitter", 0.0))
    except (TypeError, ValueError):
        raise ValueError(f"{where}: requests_per_second and jitter must be numbers") from None
    if not (math.isfinite(rps) and rps > 0):
        raise ValueError(f"{where}.requests_per_second must be positive and finite, got {rps}")
    if not math.isfinite(jitter):
        raise ValueError(f"{where}.jitter must be finite, got {jitter}")
    return _Rate(gap=1.0 / rps, jitter=min(max(jitter, 0.0), 0.9))


def _rates(http: Mapping[str, Any]) -> dict[str, _Rate]:
    """host (lower case) -> rate, with the `default` entry every unlisted host falls back to."""
    hosts = http.get("hosts")
    if not isinstance(hosts, Mapping) or "default" not in hosts:
        raise ValueError("http.hosts.default is required: the rate for hosts without an entry")
    return {str(name).lower(): _rate(str(name), entry) for name, entry in hosts.items()}


def _host_of(exc: Exception, fallback: str) -> str:
    """The host of the request an httpx error belongs to (fallback when it carries none)."""
    try:
        return exc.request.url.host  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):  # no request attached (httpx raises RuntimeError)
        return fallback


class Fetcher:
    def __init__(
        self,
        source: str,
        *,
        http: Mapping[str, Any],
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ):
        self.source = source
        user_agent = _user_agent(http)
        self._rates = _rates(http)
        breaker = http.get("circuit_breaker") or {}
        self._max_retries = int(http.get("max_retries", 4))
        self._backoff_base = float(http.get("backoff_base_s", 2.0))
        self._stop_on_status = {int(s) for s in breaker.get("stop_on_status", [403, 429])}
        self._max_consecutive_failures = int(breaker.get("max_consecutive_failures", 5))

        self._sleep = sleep
        self._clock = clock
        self._rng = rng if rng is not None else random.Random()
        self._hosts: dict[str, _HostState] = {}
        self.requests_sent = 0

        self._client = self._build_client(
            transport,
            user_agent=user_agent,
            timeout_s=float(http.get("timeout_s", 30)),
            http2=bool(http.get("http2", False)),
        )

    def _build_client(
        self,
        transport: httpx.BaseTransport | None,
        *,
        user_agent: str,
        timeout_s: float,
        http2: bool,
    ) -> httpx.Client:
        kwargs: dict[str, Any] = {
            "headers": {
                "User-Agent": user_agent,
                "Accept": _ACCEPT,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip",
            },
            "timeout": httpx.Timeout(timeout_s),
            "follow_redirects": False,  # _send follows them itself, one rate-limited hop at a time
            "transport": transport,
        }
        if http2:
            try:
                return httpx.Client(http2=True, **kwargs)
            except ImportError:  # optional h2 package missing: HTTP/1.1 keep-alive is fine
                pass
        return httpx.Client(**kwargs)

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ network

    def _host(self, host: str) -> _HostState:
        state = self._hosts.get(host)
        if state is None:
            rate = self._rates.get(host, self._rates["default"])
            state = self._hosts[host] = _HostState(rate=rate, next_gap=rate.gap)
        return state

    def _check_open(self, host: str) -> None:
        state = self._hosts.get(host)
        if state is not None and state.open_reason is not None:
            raise CircuitOpen(
                f"circuit open for {host}: {state.open_reason}",
                http_status=state.open_status,
                host=host,
            )

    def _throttle(self, host: str) -> None:
        """Spacing between request starts to one host. No credit is carried over, so there is
        never a burst after a slow response."""
        state = self._host(host)
        if state.last_start is not None:
            wait = state.last_start + state.next_gap - self._clock()
            if wait > 0:
                self._sleep(wait)
        state.last_start = self._clock()
        jitter = state.rate.jitter
        state.next_gap = state.rate.gap * (1.0 + self._rng.uniform(-jitter, jitter))

    def _open(self, host: str, reason: str, http_status: int | None = None) -> CircuitOpen:
        state = self._host(host)
        state.open_reason, state.open_status = reason, http_status
        return CircuitOpen(reason, http_status=http_status, host=host)

    def _failure(
        self, host: str, message: str, http_status: int | None, final_url: str | None
    ) -> RuntimeError:
        """The exception for one failed task; the one that completes the host's failure streak
        opens that host's circuit."""
        state = self._host(host)
        state.failures += 1
        if state.failures >= self._max_consecutive_failures:
            return self._open(
                host,
                f"{state.failures} consecutive failed requests to {host} (last: {message})",
                http_status,
            )
        return FetchError(message, http_status=http_status, final_url=final_url)

    def _send(self, task: FetchTask) -> httpx.Response:
        """One attempt: the request plus its redirect hops. Every hop is a request to the host it
        goes to, so every hop is checked against that host's breaker and waits for its gap
        (httpx's own redirect following would fire them back to back)."""
        headers = {}
        content = task.body.encode("utf-8") if task.body is not None else None
        if content is not None and task.body.lstrip()[:1] in ("{", "["):
            headers["Content-Type"] = "application/json"
        elif content is not None:  # any other body is an HTML form post (key=value&...)
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = self._client.build_request(
            task.method, task.url, content=content, headers=headers
        )
        for _ in range(_MAX_REDIRECTS + 1):
            host = request.url.host
            self._check_open(host)
            self._throttle(host)
            self.requests_sent += 1
            response = self._client.send(request, follow_redirects=False)
            if response.next_request is None:
                return response
            sent, request = request, response.next_request
        raise httpx.TooManyRedirects(f"more than {_MAX_REDIRECTS} redirects", request=sent)

    def _task_host(self, task: FetchTask) -> str:
        try:
            url = httpx.URL(task.url)
        except httpx.InvalidURL as exc:
            raise FetchError(f"InvalidURL: {exc} for {task.page_type}/{task.key}") from None
        if url.scheme not in ("http", "https") or not url.host:
            raise FetchError(f"not an http(s) URL: {task.url!r} for {task.page_type}/{task.key}")
        return url.host

    def request(self, task: FetchTask) -> httpx.Response:
        """One rate-limited request with retries. No caching. Raises FetchError or CircuitOpen."""
        host = self._task_host(task)
        problem, http_status, final_url, extra_wait = "not attempted", None, None, 0.0
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(max(self._backoff_base * 2 ** (attempt - 1), extra_wait))
            try:
                response = self._send(task)
            except (httpx.TransportError, httpx.DecodingError) as exc:  # timeouts, resets, DNS
                problem, http_status, final_url = f"{type(exc).__name__}: {exc}", None, None
                extra_wait, host = 0.0, _host_of(exc, host)
                continue
            except (httpx.HTTPError, httpx.InvalidURL) as exc:  # redirect loops: no retry helps
                problem, host = f"{type(exc).__name__}: {exc}", _host_of(exc, host)
                break
            http_status, final_url = response.status_code, str(response.url)
            host = response.url.host  # the host that answered last
            if http_status in self._stop_on_status:
                raise self._open(
                    host,
                    f"HTTP {http_status} from {host} ({task.page_type}/{task.key})",
                    http_status,
                )
            if http_status >= 500:
                problem, extra_wait = f"HTTP {http_status}", _retry_after_seconds(response)
                continue
            if not response.is_success:
                problem = f"HTTP {http_status}"
                break
            self._host(host).failures = 0
            return response
        raise self._failure(
            host, f"{problem} for {task.page_type}/{task.key}", http_status, final_url
        )

    # ------------------------------------------------------------------ cache

    def _path(self, page_type: str, year: int, key: str) -> Path:
        for part in (page_type, key):
            if not part or part in (".", "..") or "/" in part or "\\" in part or "\x00" in part:
                raise ValueError(f"not usable as a cache file name: {part!r}")
        return raw_page_path(self.source, page_type, int(year), key)

    def _task_path(self, task: FetchTask) -> Path:
        if task.source != self.source:
            raise ValueError(f"task for {task.source!r} given to the {self.source!r} fetcher")
        return self._path(task.page_type, task.year, task.key)

    def _cached_page(self, path: Path, page_type: str, key: str, url: str) -> RawPage | None:
        body = _read_gzip(path)
        if body is None:
            return None
        written = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        meta = {"http_status": None, "final_url": None, "sha256": _sha256(body),
                "n_bytes": len(body), "from_cache": True}  # fmt: skip
        return RawPage(self.source, page_type, key, url, body, fetched_utc=written, meta=meta)

    def read_cached(self, page_type: str, year: int, key: str) -> RawPage | None:
        """The cached page or None. Never touches the network. The URL is not stored (url='')."""
        return self._cached_page(self._path(page_type, year, key), page_type, key, "")

    def fetch(self, task: FetchTask, *, force: bool = False) -> RawPage:
        """Cache-first. Only a mutable task or force=True goes to the network when a raw file
        exists; an open circuit still serves the cache."""
        path = self._task_path(task)
        if not (task.mutable or force):
            cached = self._cached_page(path, task.page_type, task.key, task.url)
            if cached is not None:
                return cached
        response = self.request(task)
        body = response.content
        if task.mutable:
            _keep_previous(path)
        _write_gzip_atomic(path, body)
        meta = {"http_status": response.status_code, "final_url": str(response.url),
                "sha256": _sha256(body), "n_bytes": len(body), "from_cache": False}  # fmt: skip
        now = datetime.now(timezone.utc)
        return RawPage(
            self.source, task.page_type, task.key, task.url, body, fetched_utc=now, meta=meta
        )

    def quarantine(self, task: FetchTask) -> Path | None:
        """Move a cached body that turned out to be a block or error page aside (kept for
        diagnosis), so the next attempt goes back to the site instead of being served the bad
        copy forever. If the block page had overwritten an earlier copy of a mutable page, that
        copy is put back."""
        path = self._task_path(task)
        if not path.exists():
            return None
        target = path.with_name(path.name + _QUARANTINE_SUFFIX)
        os.replace(path, target)
        # A refetchable page keeps its last good copy (a race card on race night). Only for
        # mutable tasks: they always go back to the network, so the restored copy can never stand
        # in for the retry.
        previous = path.with_name(path.name + _PREVIOUS_SUFFIX)
        if task.mutable and previous.exists():
            os.replace(previous, path)
        return target
