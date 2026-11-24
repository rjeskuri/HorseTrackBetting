"""Fetcher: cache-first, rate-limited per host, retrying, circuit-breaking per host. All offline.

Ported from HKTrack tests/unit/ingest/test_ingest_fetcher.py (every test), plus the fetcher-level
redirect and `.prev` tests of test_ingest_adversarial.py and test_ingest_review2.py, plus the
per-host tests of ticket #19.

Seams the implementation must honour (pinned here):

* `Fetcher(source, *, http, transport=None, sleep=time.sleep, clock=time.monotonic, rng=None)`,
  `http` a plain mapping: `{user_agent?, http2, timeout_s, max_retries, backoff_base_s,
  circuit_breaker: {stop_on_status, max_consecutive_failures},
  hosts: {default: {requests_per_second, jitter}, <host>: {...}}}`.
* Gap and jitter come from `http.hosts[<host>]`, else `http.hosts.default`; a request waits only
  for the last request to the same host. Every redirect hop is a request to the host it goes to.
* The breaker is per host: `CircuitOpen.host` names the host that is shut, the others keep working.
* User-Agent: `horsetrack.core.env.get_setting("HORSETRACK_USER_AGENT")`, else `http.user_agent`;
  neither (or a blank one) is a ValueError from the constructor.
* Raw pages live at `horsetrack.core.paths.raw_page_path(source, page_type, year, key)`.
* Tests never read the developer's real `.env` or environment (`isolated` below).
"""

from __future__ import annotations

import copy
import dataclasses
import gzip
import hashlib
import os
import random
import re
from pathlib import Path

import pytest

from horsetrack.core.paths import raw_page_path

# isort: split
# The module under test gets a block of its own, before httpx and the fakes: while it is missing,
# it is the import error a red run reports, and the order lints the same before and after it exists.
from horsetrack.core.ingest.fetcher import CircuitOpen, Fetcher, FetchError
from horsetrack.core.ingest.types import (
    STATUS_ABANDONED,
    STATUS_BLOCKED,
    STATUS_EMPTY,
    STATUS_FETCHED,
    FetchTask,
    ParsedPage,
    RawPage,
)

# isort: split
import httpx
from fakes import BASE, HTTP, SOURCE, USER_AGENT, FakeClock, FakeSite, make_task

UA_VAR = "HORSETRACK_USER_AGENT"
A, B = "https://a.test", "https://b.test"
JRA = "www.jra.go.jp"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch) -> Path:
    """Raw pages under tmp_path; no real `.env` or environment variable can supply a setting."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv(UA_VAR, raising=False)
    return root


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


def _gaps(times: list[float]) -> list[float]:
    return [round(b - a, 6) for a, b in zip(times, times[1:], strict=False)]


def _redirect(target: str, status: int = 302):
    return lambda request: httpx.Response(status, headers={"location": target})


def _record_starts(site: FakeSite, clock: FakeClock) -> list[tuple[str, float]]:
    """(host, seconds since now) of every request the site receives."""
    t0 = clock()
    starts: list[tuple[str, float]] = []
    site.on_request = lambda request: starts.append((request.url.host, round(clock() - t0, 6)))
    return starts


def _times_on(starts: list[tuple[str, float]], host: str) -> list[float]:
    return [t for h, t in starts if h == host]


# ---------------------------------------------------------------- types


def test_types_keep_the_hktrack_contract():
    task = FetchTask("jra", "results", "k1", "https://x.test/results/k1", 2025)
    assert (task.source, task.page_type, task.key, task.url, task.year) == (
        "jra", "results", "k1", "https://x.test/results/k1", 2025,
    )  # fmt: skip
    assert (task.priority, task.method, task.body, task.mutable) == (0, "GET", None, False)
    with pytest.raises(AttributeError):
        task.key = "other"

    page = RawPage("nar", "entries", "k2", "https://y.test/k2", b"body")
    assert (page.source, page.page_type, page.key, page.url, page.body) == (
        "nar", "entries", "k2", "https://y.test/k2", b"body",
    )  # fmt: skip
    assert page.fetched_utc is None and page.meta == {}
    assert RawPage("nar", "entries", "k3", "u", b"").meta is not page.meta

    parsed: ParsedPage = {"runners": [{"horse_id": "H1"}]}
    assert parsed["runners"][0]["horse_id"] == "H1"
    statuses = (STATUS_FETCHED, STATUS_EMPTY, STATUS_ABANDONED, STATUS_BLOCKED)
    assert statuses == ("fetched", "empty", "abandoned", "blocked")


# ---------------------------------------------------------------- cache


def test_fetch_writes_gzip_and_returns_raw_page(site, make_fetcher):
    body = "MARKER café 馬".encode() * 50
    task = make_task("results", "2026-09-16_R1")
    site.add(task.url, body)

    page = make_fetcher().fetch(task)

    assert (page.source, page.page_type, page.key, page.url, page.body) == (
        SOURCE, "results", "2026-09-16_R1", task.url, body,
    )  # fmt: skip
    assert page.fetched_utc is not None and page.fetched_utc.utcoffset().total_seconds() == 0
    assert page.meta["http_status"] == 200
    assert page.meta["final_url"] == task.url
    assert page.meta["sha256"] == hashlib.sha256(body).hexdigest()
    assert page.meta["from_cache"] is False

    path = raw_page_path(SOURCE, "results", 2026, "2026-09-16_R1")
    assert path.exists()
    assert gzip.decompress(path.read_bytes()) == body
    assert path.stat().st_size < len(body)


def test_cache_hit_makes_zero_requests(site, make_fetcher):
    task = make_task("results", "k1")
    site.add(task.url, "MARKER one")
    fetcher = make_fetcher()
    first = fetcher.fetch(task)
    assert site.total_hits == 1

    again = fetcher.fetch(task)
    other_instance = make_fetcher().fetch(task)

    assert site.total_hits == 1
    for page in (again, other_instance):
        assert page.body == first.body
        assert page.meta["from_cache"] is True
        assert page.meta["sha256"] == first.meta["sha256"]
        assert page.url == task.url


def test_cache_hit_needs_no_rate_limit_sleep(site, make_fetcher, clock):
    task = make_task("results", "k1")
    site.add(task.url, "MARKER one")
    fetcher = make_fetcher()
    fetcher.fetch(task)
    for _ in range(5):
        fetcher.fetch(task)
    assert clock.sleeps == []


def test_mutable_and_force_bypass_the_cache(site, make_fetcher):
    fixed = make_task("results", "k1")
    card = make_task("racecard", "k2", mutable=True)
    site.add(fixed.url, "MARKER v1")
    site.add(card.url, "MARKER card v1")
    fetcher = make_fetcher()
    fetcher.fetch(fixed)
    fetcher.fetch(card)

    site.add(fixed.url, "MARKER v2")
    site.add(card.url, "MARKER card v2")
    assert fetcher.fetch(fixed).body == b"MARKER v1"
    assert fetcher.fetch(card).body == b"MARKER card v2"
    assert fetcher.fetch(fixed, force=True).body == b"MARKER v2"
    assert site.hits[fixed.url] == 2 and site.hits[card.url] == 2
    assert fetcher.read_cached("racecard", 2026, "k2").body == b"MARKER card v2"
    assert fetcher.read_cached("results", 2026, "k1").body == b"MARKER v2"


def test_read_cached(site, make_fetcher):
    task = make_task("horse", "H1", year=2023)
    site.add(task.url, "MARKER horse")
    fetcher = make_fetcher()
    assert fetcher.read_cached("horse", 2023, "H1") is None
    fetcher.fetch(task)
    page = fetcher.read_cached("horse", 2023, "H1")
    assert (page.page_type, page.key, page.body, page.meta["from_cache"]) == (
        "horse",
        "H1",
        b"MARKER horse",
        True,
    )
    assert fetcher.read_cached("horse", 2024, "H1") is None
    assert site.total_hits == 1


def test_corrupt_cache_file_is_refetched(site, make_fetcher):
    task = make_task("results", "k1")
    site.add(task.url, "MARKER good")
    path = raw_page_path(SOURCE, "results", 2026, "k1")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x1f\x8b this is not a complete gzip stream")
    fetcher = make_fetcher()
    assert fetcher.read_cached("results", 2026, "k1") is None
    assert fetcher.fetch(task).body == b"MARKER good"
    assert gzip.decompress(path.read_bytes()) == b"MARKER good"


def test_atomic_write_leaves_no_temp_files(site, make_fetcher):
    task = make_task("results", "k1")
    site.add(task.url, "MARKER body")
    make_fetcher().fetch(task)
    folder = raw_page_path(SOURCE, "results", 2026, "k1").parent
    assert [p.name for p in folder.iterdir()] == ["k1.html.gz"]


def test_failed_replace_keeps_the_previous_file_intact(site, make_fetcher, monkeypatch):
    task = make_task("results", "k1")
    site.add(task.url, "MARKER v1")
    fetcher = make_fetcher()
    fetcher.fetch(task)
    path = raw_page_path(SOURCE, "results", 2026, "k1")
    before = path.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")

    site.add(task.url, "MARKER v2")
    with monkeypatch.context() as patch:  # never monkeypatch.undo(): it would drop `isolated` too
        patch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            fetcher.fetch(task, force=True)

    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == ["k1.html.gz"]


def test_failed_first_write_leaves_no_cache_entry(site, make_fetcher, monkeypatch, tmp_path):
    task = make_task("results", "k1")
    site.add(task.url, "MARKER v1")
    fetcher = make_fetcher()
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            fetcher.fetch(task)
    folder = raw_page_path(SOURCE, "results", 2026, "k1").parent
    assert folder.is_relative_to(tmp_path) and list(folder.iterdir()) == []
    assert fetcher.read_cached("results", 2026, "k1") is None


@pytest.mark.parametrize("key", ["../escape", "a/b", "", ".."])
def test_keys_that_are_not_plain_file_stems_are_rejected(site, make_fetcher, key):
    task = make_task("results", key, url=f"{BASE}/x")
    site.add(task.url, "MARKER")
    with pytest.raises(ValueError):
        make_fetcher().fetch(task)
    assert site.total_hits == 0


def test_task_for_another_source_is_rejected(site, make_fetcher):
    task = dataclasses.replace(make_task("results", "k1"), source="elsewhere")
    site.add(task.url, "MARKER")
    with pytest.raises(ValueError):
        make_fetcher().fetch(task)
    assert site.total_hits == 0


def test_a_block_page_does_not_destroy_the_last_good_copy_of_a_mutable_page(site, make_fetcher):
    card = make_task("racecard", "2026-09-23_R01", mutable=True)
    site.add(card.url, "MARKER card as of 18:00")
    fetcher = make_fetcher()
    fetcher.fetch(card)

    site.add(card.url, "<html>Access denied</html>")  # HTTP 200: only the caller can tell
    assert fetcher.fetch(card).body == b"<html>Access denied</html>"
    moved = fetcher.quarantine(card)
    assert moved is not None and moved.name == "2026-09-23_R01.html.gz.blocked"
    assert gzip.decompress(moved.read_bytes()) == b"<html>Access denied</html>"
    kept = fetcher.read_cached("racecard", 2026, card.key)
    assert kept is not None and kept.body == b"MARKER card as of 18:00"

    # and the retry goes back to the site, not to the restored copy
    site.add(card.url, "MARKER card as of 19:00")
    assert fetcher.fetch(card).body == b"MARKER card as of 19:00"
    assert fetcher.read_cached("racecard", 2026, card.key).body == b"MARKER card as of 19:00"
    assert site.hits[card.url] == 3


def test_successful_refetch_leaves_only_the_current_copy_readable(site, make_fetcher):
    card = make_task("racecard", "c1", mutable=True)
    fetcher = make_fetcher()
    for body in ("MARKER v1", "MARKER v2", "MARKER v3"):
        site.add(card.url, body)
        fetcher.fetch(card)
    assert fetcher.read_cached("racecard", 2026, "c1").body == b"MARKER v3"
    folder = raw_page_path(SOURCE, "racecard", 2026, "c1").parent
    assert not [p.name for p in folder.iterdir() if p.name.endswith((".tmp", ".blocked"))]
    assert len([p for p in folder.iterdir() if p.name.endswith(".html.gz")]) == 1


# ---------------------------------------------------------------- politeness (one host)


def test_rate_limiter_enforces_the_minimum_gap_without_real_sleeping(site, make_fetcher, clock):
    times: list[float] = []
    site.on_request = lambda request: times.append(clock())
    tasks = [make_task("results", f"k{i}") for i in range(30)]
    for t in tasks:
        site.add(t.url, "MARKER")
    fetcher = make_fetcher(rng=random.Random(7))
    for t in tasks:
        fetcher.fetch(t)

    gaps = _gaps(times)
    assert len(gaps) == 29
    assert min(gaps) >= 0.7 and max(gaps) <= 1.3
    assert len(set(gaps)) > 10  # jittered, not a fixed beat
    assert clock.sleeps and sum(clock.sleeps) == pytest.approx(times[-1] - times[0])


def test_rate_limiter_without_jitter_is_exactly_the_configured_rate(
    site, make_fetcher, clock, http
):
    http["hosts"]["default"].update(requests_per_second=0.5, jitter=0.0)
    times: list[float] = []
    site.on_request = lambda request: times.append(clock())
    fetcher = make_fetcher()
    for i in range(4):
        task = make_task("results", f"k{i}")
        site.add(task.url, "MARKER")
        fetcher.fetch(task)
    assert _gaps(times) == [2.0, 2.0, 2.0]


def test_slow_responses_are_not_followed_by_a_catch_up_burst(site, make_fetcher, clock, http):
    http["hosts"]["default"]["jitter"] = 0.0
    times: list[float] = []

    def slow_first(request):
        times.append(clock())
        if len(times) == 1:
            clock.advance(10.0)  # the first response takes 10 s

    site.on_request = slow_first
    fetcher = make_fetcher()
    for i in range(4):
        task = make_task("results", f"k{i}")
        site.add(task.url, "MARKER")
        fetcher.fetch(task)
    assert _gaps(times) == [10.0, 1.0, 1.0]


def test_sends_polite_headers(site, make_fetcher, http):
    task = make_task("results", "k1")
    site.add(task.url, "MARKER")
    make_fetcher().fetch(task)
    headers = site.requests[0].headers
    assert headers["user-agent"] == http["user_agent"] == USER_AGENT
    assert headers.get("accept-language", "").strip()
    assert headers["accept-encoding"] == "gzip"


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        ('{"query":"{x}"}', "application/json"),
        ("[1, 2]", "application/json"),
        ("cname=pw01k1&x=1", "application/x-www-form-urlencoded"),
    ],
)
def test_post_sends_the_body_with_its_content_type(site, make_fetcher, body, content_type):
    task = make_task("odds", "snap", url=f"{BASE}/graphql", method="POST", body=body, mutable=True)
    site.add(task.url, '{"data":{}}')
    response = make_fetcher().request(task)
    sent = site.requests[0]
    assert response.status_code == 200
    assert sent.method == "POST"
    assert sent.content == body.encode()
    assert sent.headers["content-type"] == content_type


def test_request_does_not_touch_the_cache(site, make_fetcher):
    task = make_task("results", "k1")
    site.add(task.url, "MARKER")
    fetcher = make_fetcher()
    fetcher.request(task)
    fetcher.request(task)
    assert site.total_hits == 2
    assert fetcher.read_cached("results", 2026, "k1") is None


def test_redirects_are_followed_and_the_final_url_recorded(site, make_fetcher):
    task = make_task("results", "old")
    target = f"{BASE}/archive/results/old"
    site.pages[task.url] = _redirect(target)
    site.add(target, "MARKER archived")
    page = make_fetcher().fetch(task)
    assert page.body == b"MARKER archived"
    assert page.meta["final_url"] == target
    assert page.url == task.url


def test_every_redirect_hop_is_rate_limited(site, make_fetcher, clock):
    times: list[float] = []
    site.on_request = lambda request: times.append(clock())
    tasks = [make_task("horse", f"H{i}") for i in range(6)]
    for t in tasks:
        target = f"{BASE}/otherhorse/{t.key}"
        site.pages[t.url] = _redirect(target)
        site.add(target, "MARKER retired")

    fetcher = make_fetcher()
    for t in tasks:
        page = fetcher.fetch(t)
        assert page.body == b"MARKER retired"
        assert page.meta["final_url"] == f"{BASE}/otherhorse/{t.key}"

    assert len(times) == 12 and fetcher.requests_sent == 12
    assert min(_gaps(times)) >= 0.7  # never two requests inside one rate-limit slot


def test_relative_redirect_and_cookies_survive_manual_redirect_handling(site, make_fetcher):
    task = make_task("results", "old")
    site.pages[task.url] = lambda request: httpx.Response(
        302, headers={"location": "/archive/results/old", "set-cookie": "session=abc; Path=/"}
    )
    site.add(f"{BASE}/archive/results/old", "MARKER archived")
    page = make_fetcher().fetch(task)
    assert page.meta["final_url"] == f"{BASE}/archive/results/old"
    assert site.requests[-1].headers.get("cookie") == "session=abc"
    assert site.requests[-1].headers["user-agent"] == USER_AGENT


def test_redirect_loop_is_bounded_rate_limited_and_not_retried(site, make_fetcher, clock):
    times: list[float] = []
    site.on_request = lambda request: times.append(clock())
    a, b = f"{BASE}/results/a", f"{BASE}/results/b"
    site.pages[a] = _redirect(b)
    site.pages[b] = _redirect(a)
    fetcher = make_fetcher()
    with pytest.raises(FetchError):
        fetcher.fetch(make_task("results", "a"))
    assert len(times) <= 6  # the first request + at most 5 hops, and no retry rounds on top
    assert min(_gaps(times)) >= 0.7
    assert fetcher.read_cached("results", 2026, "a") is None


def test_http2_config_never_breaks_construction(http):
    """The h2 package is optional: http2: true must silently fall back to HTTP/1.1 without it."""
    http["http2"] = True
    Fetcher(SOURCE, http=http).close()


def test_http2_falls_back_when_h2_is_missing(http, monkeypatch):
    real_client = httpx.Client
    seen: list[bool] = []

    def client_without_h2(*args, **kwargs):
        seen.append(bool(kwargs.get("http2")))
        if kwargs.get("http2"):
            raise ImportError("Using http2=True, but the 'h2' package is not installed.")
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_without_h2)
    Fetcher(SOURCE, http=http).close()
    assert seen == [True, False]


# ---------------------------------------------------------------- retries and circuit breaker


def test_retries_with_exponential_backoff_then_succeeds(site, make_fetcher, clock):
    task = make_task("results", "k1")
    responses = iter([500, 503, 200])
    times: list[float] = []

    def flaky(request):
        times.append(clock())
        return httpx.Response(next(responses), content=b"MARKER finally")

    site.pages[task.url] = flaky
    page = make_fetcher().fetch(task)
    assert page.body == b"MARKER finally"
    assert site.hits[task.url] == 3
    assert [s for s in clock.sleeps if s >= 2.0] == [2.0, 4.0]
    assert min(_gaps(times)) >= 0.7  # the rate limit holds across retries too


def test_gives_up_after_max_retries_and_caches_nothing(site, make_fetcher, clock):
    task = make_task("results", "k1")
    site.add(task.url, "oops", status=500)
    fetcher = make_fetcher()
    with pytest.raises(FetchError) as err:
        fetcher.fetch(task)
    assert err.value.http_status == 500
    assert site.hits[task.url] == 5  # 1 try + max_retries 4
    assert [s for s in clock.sleeps if s >= 2.0] == [2.0, 4.0, 8.0, 16.0]
    assert fetcher.read_cached("results", 2026, "k1") is None


@pytest.mark.parametrize(
    ("retry_after", "wait"),
    [
        ("30", 30.0),
        ("299", 299.0),
        ("301", 300.0),
        ("100000", 300.0),
        ("Wed, 21 Oct 2015 07:28:00 GMT", 2.0),
        (None, 2.0),
    ],
)
def test_retry_after_on_a_server_error_is_honoured_up_to_300_s(
    site, make_fetcher, clock, retry_after, wait
):
    task = make_task("results", "k1")
    answers = iter([503, 200])
    times: list[float] = []

    def busy_then_fine(request):
        times.append(clock())
        status = next(answers)
        headers = {"retry-after": retry_after} if status == 503 and retry_after else {}
        return httpx.Response(status, headers=headers, content=b"MARKER ok")

    site.pages[task.url] = busy_then_fine
    assert make_fetcher().fetch(task).body == b"MARKER ok"
    assert _gaps(times) == [wait]  # max(backoff 2 s, Retry-After capped at 300 s)


def test_timeouts_are_retried_then_failed(site, make_fetcher, http):
    http["max_retries"] = 2
    task = make_task("results", "k1")

    def timeout(request):
        raise httpx.ReadTimeout("timed out", request=request)

    site.pages[task.url] = timeout
    with pytest.raises(FetchError) as err:
        make_fetcher().fetch(task)
    assert site.hits[task.url] == 3
    assert err.value.http_status is None
    assert "ReadTimeout" in str(err.value)


def test_connection_errors_are_retried(site, make_fetcher):
    task = make_task("results", "k1")
    calls = iter([True, False])

    def flaky(request):
        if next(calls):
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, content=b"MARKER ok")

    site.pages[task.url] = flaky
    assert make_fetcher().fetch(task).body == b"MARKER ok"


def test_client_errors_fail_without_retry(site, make_fetcher):
    task = make_task("results", "gone")
    site.add(task.url, "not here", status=404)
    fetcher = make_fetcher()
    with pytest.raises(FetchError) as err:
        fetcher.fetch(task)
    assert err.value.http_status == 404
    assert site.hits[task.url] == 1
    assert fetcher.read_cached("results", 2026, "gone") is None


def test_server_error_behind_a_redirect_is_retried_with_backoff(site, make_fetcher, clock):
    task = make_task("results", "k1")
    target = f"{BASE}/archive/k1"
    site.pages[task.url] = _redirect(target)
    answers = iter([httpx.Response(503), httpx.Response(200, content=b"MARKER ok")])
    site.pages[target] = lambda request: next(answers)
    assert make_fetcher().fetch(task).body == b"MARKER ok"
    assert site.hits[target] == 2
    assert max(clock.sleeps) >= 2.0  # a backoff happened


@pytest.mark.parametrize("status", [403, 429])
def test_stop_status_opens_the_circuit_immediately(site, make_fetcher, status):
    blocked = make_task("results", "k1")
    fine = make_task("results", "k2")
    site.add(blocked.url, "denied", status=status)
    site.add(fine.url, "MARKER")
    fetcher = make_fetcher()

    with pytest.raises(CircuitOpen) as err:
        fetcher.fetch(blocked)
    assert str(status) in str(err.value) and err.value.http_status == status
    assert err.value.host == "site.test"
    assert site.hits[blocked.url] == 1  # no retry against a site that said stop
    assert fetcher.read_cached("results", 2026, "k1") is None

    # the breaker stays open: nothing else reaches that host from this fetcher
    with pytest.raises(CircuitOpen):
        fetcher.fetch(fine)
    with pytest.raises(CircuitOpen):
        fetcher.request(fine)
    assert site.total_hits == 1


@pytest.mark.parametrize("status", [403, 429])
def test_stop_status_behind_a_redirect_opens_the_circuit(site, make_fetcher, status):
    task = make_task("results", "k1")
    target = f"{BASE}/denied"
    site.pages[task.url] = _redirect(target)
    site.add(target, "go away", status=status)
    fetcher = make_fetcher()
    with pytest.raises(CircuitOpen) as excinfo:
        fetcher.fetch(task)
    assert excinfo.value.http_status == status
    assert site.total_hits == 2
    with pytest.raises(CircuitOpen):
        fetcher.fetch(make_task("results", "k2"))
    assert site.total_hits == 2


def test_open_circuit_still_serves_the_cache(site, make_fetcher):
    cached = make_task("results", "k1")
    denied = make_task("results", "k2")
    site.add(cached.url, "MARKER cached")
    site.add(denied.url, "denied", status=403)
    fetcher = make_fetcher()
    fetcher.fetch(cached)
    with pytest.raises(CircuitOpen):
        fetcher.fetch(denied)
    assert fetcher.fetch(cached).body == b"MARKER cached"


def test_circuit_opens_after_consecutive_failed_tasks(site, make_fetcher, http):
    http["max_retries"] = 0
    http["circuit_breaker"]["max_consecutive_failures"] = 3
    fetcher = make_fetcher()
    bad = [make_task("results", f"bad{i}") for i in range(4)]
    for t in bad:
        site.add(t.url, "oops", status=500)
    good = make_task("results", "good")
    site.add(good.url, "MARKER")

    for t in bad[:2]:
        with pytest.raises(FetchError):
            fetcher.fetch(t)
    fetcher.fetch(good)  # a success resets the streak
    for t in bad[:2]:
        with pytest.raises(FetchError):
            fetcher.fetch(t)
    with pytest.raises(CircuitOpen) as err:
        fetcher.fetch(bad[2])
    assert "3 consecutive" in str(err.value)
    with pytest.raises(CircuitOpen):
        fetcher.fetch(bad[3])
    assert site.hits[bad[3].url] == 0


def test_circuit_open_is_a_runtime_error_but_not_a_fetch_error():
    assert issubclass(CircuitOpen, RuntimeError)
    assert issubclass(FetchError, RuntimeError)
    assert not issubclass(CircuitOpen, FetchError)


def test_quarantine_moves_a_bad_page_out_of_the_cache(site, make_fetcher):
    task = make_task("results", "k1")
    site.add(task.url, "captcha page")
    fetcher = make_fetcher()
    fetcher.fetch(task)
    moved = fetcher.quarantine(task)
    assert moved is not None and moved.exists() and moved.name == "k1.html.gz.blocked"
    assert gzip.decompress(moved.read_bytes()) == b"captcha page"
    assert fetcher.read_cached("results", 2026, "k1") is None
    assert fetcher.quarantine(task) is None

    site.add(task.url, "MARKER real page")
    assert fetcher.fetch(task).body == b"MARKER real page"
    assert site.hits[task.url] == 2


# ---------------------------------------------------------------- per host (ticket #19)


def test_each_host_gets_its_own_configured_rate(site, make_fetcher, clock, http):
    http["hosts"] = {
        "default": {"requests_per_second": 1.0, "jitter": 0.5},
        "a.test": {"requests_per_second": 0.5, "jitter": 0.0},  # 2 s
        "b.test": {"requests_per_second": 2.0, "jitter": 0.0},  # 0.5 s
    }
    starts = _record_starts(site, clock)
    order = ["a", "b", "b", "b", "b"] * 2 + ["a"]
    fetcher = make_fetcher()
    for i, host in enumerate(order):
        task = make_task("results", f"k{i}", base=f"https://{host}.test")
        site.add(task.url, "MARKER")
        fetcher.fetch(task)

    # One worker: a request starts at max(now, last start on its own host + that host's gap).
    assert starts == [
        ("a.test", 0.0), ("b.test", 0.0), ("b.test", 0.5), ("b.test", 1.0), ("b.test", 1.5),
        ("a.test", 2.0), ("b.test", 2.0), ("b.test", 2.5), ("b.test", 3.0), ("b.test", 3.5),
        ("a.test", 4.0),
    ]  # fmt: skip
    assert _gaps(_times_on(starts, "a.test")) == [2.0, 2.0]
    assert _gaps(_times_on(starts, "b.test")) == [0.5] * 7
    assert fetcher.requests_sent == 11


def test_jitter_is_taken_from_each_host_entry(site, make_fetcher, clock, http):
    http["hosts"] = {
        "default": {"requests_per_second": 1.0, "jitter": 0.0},
        "j.test": {"requests_per_second": 0.5, "jitter": 0.3},  # 2 s +/- 30 %
    }
    starts = _record_starts(site, clock)
    fetcher = make_fetcher(rng=random.Random(11))
    for i in range(20):
        task = make_task("results", f"j{i}", base="https://j.test")
        site.add(task.url, "MARKER")
        fetcher.fetch(task)
    for i in range(5):
        task = make_task("results", f"p{i}", base="https://plain.test")
        site.add(task.url, "MARKER")
        fetcher.fetch(task)

    jittered = _gaps(_times_on(starts, "j.test"))
    assert len(jittered) == 19
    assert min(jittered) >= 1.4 and max(jittered) <= 2.6
    assert len(set(jittered)) > 8
    assert _gaps(_times_on(starts, "plain.test")) == [1.0] * 4


@pytest.mark.parametrize(("default_rps", "gap"), [(0.25, 4.0), (0.5, 2.0), (2.0, 0.5)])
def test_a_host_without_an_entry_uses_the_default_rate(
    site, make_fetcher, clock, http, default_rps, gap
):
    http["hosts"] = {
        "default": {"requests_per_second": default_rps, "jitter": 0.0},
        "listed.test": {"requests_per_second": 4.0, "jitter": 0.5},
    }
    starts = _record_starts(site, clock)
    fetcher = make_fetcher()
    for i in range(4):
        task = make_task("results", f"k{i}", base="https://unlisted.test")
        site.add(task.url, "MARKER")
        fetcher.fetch(task)
    assert _gaps(_times_on(starts, "unlisted.test")) == [gap] * 3


@pytest.mark.parametrize(
    ("default_rps", "jra_rps", "gap"), [(5.0, 1.0, 1.0), (0.25, 1.0, 1.0), (5.0, 0.5, 2.0)]
)
def test_the_source_rate_applies_to_www_jra_go_jp(
    site, make_fetcher, clock, http, default_rps, jra_rps, gap
):
    """Bug (e): HKTrack's live path paced JRA by the global rate. The host entry wins, for mutable
    page fetches and odds POSTs alike, whatever the default is."""
    http["hosts"] = {
        "default": {"requests_per_second": default_rps, "jitter": 0.0},
        JRA: {"requests_per_second": jra_rps, "jitter": 0.0},
    }
    starts = _record_starts(site, clock)
    post_url = f"https://{JRA}/odds"
    site.add(post_url, "MARKER odds")
    fetcher = make_fetcher()
    for i in range(3):
        page = make_task("odds", f"k{i}", url=f"https://{JRA}/odds/k{i}", mutable=True)
        site.add(page.url, "MARKER card")
        fetcher.fetch(page)
        post = make_task(
            "odds", f"p{i}", url=post_url, method="POST", body=f"cname=k{i}", mutable=True
        )
        assert fetcher.request(post).status_code == 200
    assert [h for h, _ in starts] == [JRA] * 6
    assert _gaps([t for _, t in starts]) == [gap] * 5


def test_a_redirect_hop_waits_for_the_gap_of_the_host_it_goes_to(site, make_fetcher, clock, http):
    http["hosts"] = {
        "default": {"requests_per_second": 1.0, "jitter": 0.0},
        "a.test": {"requests_per_second": 4.0, "jitter": 0.0},  # 0.25 s
        "b.test": {"requests_per_second": 0.25, "jitter": 0.0},  # 4 s
    }
    starts = _record_starts(site, clock)
    direct = make_task("results", "b1", base=B)
    moved = make_task("results", "m1", base=A)
    after = make_task("results", "a2", base=A)
    site.add(direct.url, "MARKER b1")
    site.pages[moved.url] = _redirect(f"{B}/results/m1")
    site.add(f"{B}/results/m1", "MARKER moved")
    site.add(after.url, "MARKER a2")
    fetcher = make_fetcher()

    fetcher.fetch(direct)
    page = fetcher.fetch(moved)
    fetcher.fetch(after)

    assert page.body == b"MARKER moved" and page.meta["final_url"] == f"{B}/results/m1"
    assert starts == [("b.test", 0.0), ("a.test", 0.0), ("b.test", 4.0), ("a.test", 4.0)]


@pytest.mark.parametrize("status", [403, 429])
def test_a_block_on_one_host_leaves_other_hosts_open(site, make_fetcher, status):
    a_pages = [make_task("results", f"a{i}", base=A) for i in range(3)]
    b_pages = [make_task("results", f"b{i}", base=B) for i in range(4)]
    site.add(a_pages[0].url, "denied", status=status)
    for t in a_pages[1:] + b_pages:
        site.add(t.url, f"MARKER {t.key}")
    fetcher = make_fetcher()
    assert fetcher.fetch(b_pages[0]).body == b"MARKER b0"

    with pytest.raises(CircuitOpen) as err:
        fetcher.fetch(a_pages[0])
    assert err.value.host == "a.test" and err.value.http_status == status

    # host a stays shut: neither fetch nor request reaches it
    for t in a_pages[1:]:
        with pytest.raises(CircuitOpen) as again:
            fetcher.fetch(t)
        assert again.value.host == "a.test"
    with pytest.raises(CircuitOpen):
        fetcher.request(a_pages[1])
    assert site.hits_on("a.test") == 1

    # host b is still served from the network
    for t in b_pages[1:]:
        page = fetcher.fetch(t)
        assert page.body == f"MARKER {t.key}".encode() and page.meta["from_cache"] is False
    assert fetcher.request(b_pages[0]).status_code == 200
    assert site.hits_on("b.test") == 5


def test_a_failure_streak_opens_only_that_host(site, make_fetcher, http):
    http["max_retries"] = 0
    http["circuit_breaker"]["max_consecutive_failures"] = 3
    a_bad = [make_task("results", f"a{i}", base=A) for i in range(4)]
    b_good = [make_task("results", f"b{i}", base=B) for i in range(3)]
    b_bad = [make_task("results", f"x{i}", base=B) for i in range(2)]
    for t in a_bad + b_bad:
        site.add(t.url, "oops", status=500)
    for t in b_good:
        site.add(t.url, f"MARKER {t.key}")
    fetcher = make_fetcher()

    for i in range(2):
        with pytest.raises(FetchError):
            fetcher.fetch(a_bad[i])
        fetcher.fetch(b_good[i])  # a success on b does not reset a's streak
    with pytest.raises(CircuitOpen) as err:
        fetcher.fetch(a_bad[2])
    assert err.value.host == "a.test"
    with pytest.raises(CircuitOpen):
        fetcher.fetch(a_bad[3])
    assert site.hits[a_bad[3].url] == 0

    # a's failures did not count toward b: two failures on b stay below the limit of 3
    for t in b_bad:
        with pytest.raises(FetchError):
            fetcher.fetch(t)
    assert fetcher.fetch(b_good[2]).body == b"MARKER b2"


def test_a_stop_status_behind_a_cross_host_redirect_shuts_the_host_that_sent_it(site, make_fetcher):
    moved = make_task("results", "r1", base=A)
    moved_too = make_task("results", "r2", base=A)
    plain_a = make_task("results", "p1", base=A)
    other_b = make_task("results", "x1", base=B)
    site.pages[moved.url] = _redirect(f"{B}/denied")
    site.pages[moved_too.url] = _redirect(f"{B}/elsewhere")
    site.add(f"{B}/denied", "go away", status=403)
    site.add(f"{B}/elsewhere", "MARKER never sent")
    site.add(plain_a.url, "MARKER a")
    site.add(other_b.url, "MARKER b")
    fetcher = make_fetcher()

    with pytest.raises(CircuitOpen) as err:
        fetcher.fetch(moved)
    assert err.value.host == "b.test" and err.value.http_status == 403
    assert fetcher.fetch(plain_a).body == b"MARKER a"  # a never said stop
    with pytest.raises(CircuitOpen):
        fetcher.fetch(other_b)
    with pytest.raises(CircuitOpen) as hop:  # a answers; the hop to the shut host is not sent
        fetcher.fetch(moved_too)
    assert hop.value.host == "b.test"
    assert site.hits_on("b.test") == 1


@pytest.mark.parametrize("where", ["default", JRA])
@pytest.mark.parametrize("rps", [0, -1.0])
def test_a_non_positive_rate_is_refused_at_construction(site, http, where, rps):
    http["hosts"][where] = {"requests_per_second": rps, "jitter": 0.0}
    with pytest.raises(ValueError):
        Fetcher(SOURCE, http=http, transport=site.transport)


# ---------------------------------------------------------------- User-Agent


@pytest.mark.parametrize(
    ("env_value", "dotenv_value", "config_value", "expected"),
    [
        ("horsetrack-env/2.0", None, USER_AGENT, "horsetrack-env/2.0"),
        (None, "horsetrack-dotenv/3.0", USER_AGENT, "horsetrack-dotenv/3.0"),
        ("horsetrack-env/2.0", "horsetrack-dotenv/3.0", USER_AGENT, "horsetrack-env/2.0"),
        ("horsetrack-env/2.0", None, None, "horsetrack-env/2.0"),
        (None, None, "horsetrack-config/4.0", "horsetrack-config/4.0"),
    ],
)
def test_user_agent_comes_from_the_setting_over_config(
    site, make_fetcher, http, monkeypatch, isolated, env_value, dotenv_value, config_value, expected
):
    if env_value is not None:
        monkeypatch.setenv(UA_VAR, env_value)
    if dotenv_value is not None:
        (isolated / ".env").write_text(f"OTHER_SETTING=1\n{UA_VAR}={dotenv_value}\n")
    if config_value is None:
        http.pop("user_agent")
    else:
        http["user_agent"] = config_value
    task = make_task("results", "k1")
    site.add(task.url, "MARKER")
    make_fetcher().fetch(task)
    assert site.requests[-1].headers["user-agent"] == expected


_ABSENT = object()


@pytest.mark.parametrize(
    ("env_value", "config_value"),
    [(None, _ABSENT), (None, None), (None, ""), (None, "   "), ("", _ABSENT)],
    ids=["absent", "none", "empty", "blank", "empty-setting"],
)
def test_no_user_agent_is_an_error_at_construction(
    site, http, monkeypatch, isolated, env_value, config_value
):
    (isolated / ".env").write_text("OTHER_SETTING=1\n")
    if env_value is not None:
        monkeypatch.setenv(UA_VAR, env_value)
    if config_value is _ABSENT:
        http.pop("user_agent")
    else:
        http["user_agent"] = config_value
    with pytest.raises(ValueError):
        Fetcher(SOURCE, http=http, transport=site.transport)


def test_no_user_agent_string_in_src():
    browser = re.compile(
        r"Mozilla/|AppleWebKit|Gecko/|Safari/|Chrome/|Firefox/|curl/|python-requests/|python-httpx/"
    )
    literal = re.compile(
        r"""["']User-Agent["']\s*:\s*["']|user_agent["']\s*,\s*["']|user_agent\s*=\s*["'][^"']"""
    )
    yaml_key = re.compile(r"^\s*user_agent\s*:", re.MULTILINE)
    offenders = []
    for path in sorted(SRC.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml", ".toml", ".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        checks = [browser, literal] + ([yaml_key] if path.suffix in {".yaml", ".yml"} else [])
        offenders += [f"{path.relative_to(SRC)}: {c.pattern}" for c in checks if c.search(text)]
    assert not offenders, offenders
