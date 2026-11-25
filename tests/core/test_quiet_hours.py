"""Per-host rates and quiet hours for `horsetrack scrape run` (ticket #45).

Ported from HKTrack tests/unit/ingest/test_jurisdiction_rate_and_quiet_hours.py, on synthetic
pages (the real JRA page fixtures and the real-adapter plan/run cases are not ported: the JRA
adapter comes in its own epic).

- Each request is paced by the source config's `http.hosts[<host>]`, else `http.default`.
- `http.quiet_hours: [{days, start, end, timezone, hosts}]` (start inclusive, end exclusive, in
  `timezone`): while a window is open, no request goes to the hosts it names. `scrape run` refuses
  to start inside one (exit 0, one line naming quiet hours, no request) and stops cleanly when one
  begins mid-run; hosts no window names keep running.
- jra.yaml carries HKTrack's window (Sat and Sun 09:00-17:00 Asia/Tokyo) for the netkeiba hosts,
  and never for www.jra.go.jp, the odds logger's host.

Seams pinned here (on top of those in test_scrape_cli.py): `horsetrack.core.ingest.cli._utc_now()`
returns the aware UTC wall clock the quiet check reads, looked up at call time, at the start of
`run` and before every request. No real time passes: every Fetcher sleeps on a FakeClock.
"""

from __future__ import annotations

import copy
import random
from datetime import datetime, time, timedelta, timezone
from statistics import mean
from zoneinfo import ZoneInfo

import pytest
from fakes import SOURCE, FakeClock, FakeSite
from scrape_fakes import PlainAdapter, lines, offline_env, register_adapter
from typer.testing import CliRunner

from horsetrack.core.config import load_country_yaml
from horsetrack.core.ingest.fetcher import Fetcher
from horsetrack.core.ingest.queue import FetchQueue
from horsetrack.core.ingest.types import FetchTask
from horsetrack.core.paths import manifest_path

# isort: split
# The module under test gets a block of its own: while it is missing, it is the import error a red
# run reports.
from horsetrack.core.ingest import cli as cli_module

# isort: split
from horsetrack.cli import app

cli = CliRunner()
UTC = timezone.utc
TOKYO = ZoneInfo("Asia/Tokyo")
LONDON = ZoneInfo("Europe/London")
LOGGER_HOST = "www.jra.go.jp"
NETKEIBA_HOSTS = ("db.netkeiba.com", "race.netkeiba.com")
# A Wednesday, 12:00 in Tokyo (03:00 in London): outside every quiet window used here.
WEEKDAY_NOON_UTC = datetime(2026, 11, 18, 3, 0, tzinfo=UTC)


def _yaml_http(country: str, source: str) -> dict:
    return load_country_yaml(country, source)["http"]


def _rate(http: dict, host: str) -> tuple[float, float]:
    """(gap in seconds, jitter) the config sets for `host`, read straight from the yaml."""
    entry = (http.get("hosts") or {}).get(host, http["default"])
    return 1.0 / float(entry["requests_per_second"]), float(entry.get("jitter", 0.0))


def _fake_config() -> dict:
    """A source with distinct exact rates per host and one quiet window on a.test only."""
    return {
        "source": SOURCE,
        "timezone": "Asia/Tokyo",
        "http": {
            "default": {"requests_per_second": 2.0, "jitter": 0.0},
            "hosts": {
                "a.test": {"requests_per_second": 4.0, "jitter": 0.0},
                "b.test": {"requests_per_second": 0.5, "jitter": 0.0},
            },
            "circuit_breaker": {"stop_on_status": [403, 429], "max_consecutive_failures": 5},
            "quiet_hours": [
                {
                    "days": ["wed"],
                    "start": "10:00",
                    "end": "11:30",
                    "timezone": "Europe/London",
                    "hosts": ["a.test"],
                }
            ],
        },
    }


@pytest.fixture
def site() -> FakeSite:
    return FakeSite()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake_config() -> dict:
    return _fake_config()


@pytest.fixture(autouse=True)
def starts(tmp_path, monkeypatch, site, clock, fake_config) -> list[tuple[str, float]]:
    """(host, fake-clock time) of every request start."""
    offline_env(monkeypatch, tmp_path, site, clock)
    for source in ("jra", "hkjc", SOURCE):
        register_adapter(monkeypatch, source, PlainAdapter(source))
    real = cli_module.source_config

    def fake_source_config(source: str):
        return copy.deepcopy(fake_config) if source == SOURCE else real(source)

    monkeypatch.setattr(cli_module, "source_config", fake_source_config)
    monkeypatch.setattr(cli_module, "_utc_now", lambda: WEEKDAY_NOON_UTC)
    record: list[tuple[str, float]] = []
    site.on_request = lambda request: record.append((request.url.host, clock()))
    return record


def _local(zone: ZoneInfo, y: int, mo: int, d: int, h: int, mi: int) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=zone).astimezone(UTC)


def _queue_pages(site: FakeSite, source: str, hosts: list[str], page_type: str = "results") -> None:
    """One synthetic page per entry of `hosts`, requested in that order."""
    tasks = [
        FetchTask(
            source,
            page_type,
            f"{page_type}{i:03d}",
            f"https://{host}/synthetic/{i:03d}",
            2025,
            priority=1000 - i,
        )
        for i, host in enumerate(hosts)
    ]
    for task in tasks:
        site.add(task.url, f"MARKER synthetic page {task.key}")
    with FetchQueue(manifest_path(source)) as queue:
        queue.enqueue(tasks)


def _counts(source: str) -> dict[tuple[str, str], int]:
    with FetchQueue(manifest_path(source)) as queue:
        return queue.counts()


def _pending(source: str) -> int:
    return sum(n for (_, status), n in _counts(source).items() if status == "pending")


def _run(source: str, *extra: str):
    return cli.invoke(app, ["scrape", "run", "-s", source, *extra])


def _gaps(starts: list[tuple[str, float]]) -> list[float]:
    times = [t for _, t in starts]
    return [b - a for a, b in zip(times, times[1:], strict=False)]


# ---------------------------------------------------------------- per-host rate


@pytest.mark.parametrize(("host", "gap"), [("a.test", 0.25), ("b.test", 2.0), ("c.test", 0.5)])
def test_rate_comes_from_the_host_entry(starts, site, host, gap):
    """a.test and b.test have their own entries, c.test falls back to http.default."""
    _queue_pages(site, SOURCE, [host] * 6)

    result = _run(SOURCE)

    assert result.exit_code == 0, result.output
    assert site.hits_on(host) == 6 and _pending(SOURCE) == 0
    assert _gaps(starts) == pytest.approx([gap] * 5, abs=1e-9)


JRA_HTTP = _yaml_http("japan", "jra")
HKJC_HTTP = _yaml_http("hongkong", "hkjc")
RATE_CASES = [
    *[("jra", host) for host in (JRA_HTTP.get("hosts") or {})],
    ("jra", "unlisted.test"),  # no entry: http.default
    ("hkjc", "racing.hkjc.com"),
]


JITTER_SEED = 232


@pytest.fixture
def seeded_jitter(monkeypatch, starts):
    """Every Fetcher the runner builds draws its jitter from random.Random(JITTER_SEED), so the
    gaps below are the same on every run (`starts` first: it installs the offline Fetcher)."""
    wired = Fetcher.__init__

    def seeded_init(self, *args, rng=None, **kwargs):
        wired(self, *args, rng=rng or random.Random(JITTER_SEED), **kwargs)

    monkeypatch.setattr(Fetcher, "__init__", seeded_init)


@pytest.mark.parametrize(("source", "host"), RATE_CASES)
def test_real_config_rates_per_host(seeded_jitter, starts, site, source, host):
    """40 requests to one host are paced by that host's yaml entry (else the default), jitter
    included: gaps stay within gap * (1 +/- jitter) and average out near the gap.

    The jitter RNG is seeded, so this is deterministic (an unseeded mean of 39 draws failed in CI
    by chance). The first gap is the exact gap, the other 38 are gap * (1 + U(-j, j)); with the
    fixed seed the mean sits well inside gap * jitter / 3. A wrong gap shifts every gap, so it
    breaks the min/max bounds or the mean bound (a 10 % wrong gap moves the mean by 0.1 * gap,
    above the allowed gap * jitter / 3 on every real host); jitter not applied collapses the
    spread to 0."""
    http = JRA_HTTP if source == "jra" else HKJC_HTTP
    gap, jitter = _rate(http, host)
    _queue_pages(site, source, [host] * 40)

    result = _run(source)

    assert result.exit_code == 0, result.output
    assert not any(ln.startswith("ERROR") for ln in lines(result))
    assert site.total_hits == 40 and len(starts) == 40
    gaps = _gaps(starts)
    assert min(gaps) >= gap * (1 - jitter) - 1e-9, (host, min(gaps))
    assert max(gaps) <= gap * (1 + jitter) + 1e-9, (host, max(gaps))
    assert abs(mean(gaps) - gap) <= gap * jitter / 3 + 1e-9, (host, mean(gaps))
    if jitter > 0:  # the jitter is applied, not just bounded: 39 uniform draws spread out
        assert max(gaps) - min(gaps) > gap * jitter, (host, gaps)


# ---------------------------------------------------------------- quiet hours, jra.yaml

# (UTC instant, quiet for the netkeiba hosts?). 2026-11-21 is a Saturday, 2026-11-22 a Sunday.
QUIET_TABLE = [
    (_local(TOKYO, 2026, 11, 21, 9, 0), True),  # Sat 09:00 JST = Sat 00:00 UTC: the window opens
    (_local(TOKYO, 2026, 11, 21, 8, 59), False),  # Sat 08:59 JST = Fri 23:59 UTC
    (_local(TOKYO, 2026, 11, 21, 12, 30), True),
    (_local(TOKYO, 2026, 11, 21, 16, 59), True),
    (_local(TOKYO, 2026, 11, 21, 17, 1), False),
    (_local(TOKYO, 2026, 11, 21, 21, 0), False),  # Sat 21:00 JST = Sat 12:00 UTC
    (_local(TOKYO, 2026, 11, 22, 9, 30), True),  # Sun 09:30 JST = Sun 00:30 UTC
    (_local(TOKYO, 2026, 11, 22, 15, 40), True),  # a Sunday afternoon post time
    (_local(TOKYO, 2026, 11, 22, 1, 0), False),  # Sun 01:00 JST = Sat 16:00 UTC
    (_local(TOKYO, 2026, 11, 20, 12, 0), False),  # Friday noon
    (_local(TOKYO, 2026, 11, 18, 12, 0), False),  # Wednesday noon
    (_local(TOKYO, 2026, 11, 28, 10, 0), True),  # the next Saturday
]


def _row_id(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return value[0]
    return str(value)


@pytest.mark.parametrize("hosts", [NETKEIBA_HOSTS, NETKEIBA_HOSTS[::-1]], ids=_row_id)
@pytest.mark.parametrize(("now_utc", "quiet"), QUIET_TABLE, ids=_row_id)
def test_jra_run_refuses_during_quiet_hours(monkeypatch, site, now_utc, quiet, hosts):
    monkeypatch.setattr(cli_module, "_utc_now", lambda: now_utc)
    _queue_pages(site, "jra", [hosts[0], hosts[1], hosts[0]])

    result = _run("jra")

    assert result.exit_code == 0, result.output
    out = lines(result)
    assert not any(ln.startswith("ERROR") for ln in out)
    if quiet:
        assert len(out) == 1 and "quiet" in out[0].lower(), out
        assert site.total_hits == 0
        assert _pending("jra") == 3
    else:
        assert site.total_hits == 3
        assert _pending("jra") == 0


def test_jra_run_stops_when_quiet_hours_begin_mid_run(starts, monkeypatch, site, clock):
    """A run started Sat 08:59:50 JST stops cleanly once 09:00 arrives: quiet hours are checked
    before every request (through `_utc_now`, which here follows the fetchers' fake clock), and
    the page in flight completes."""
    started_at = _local(TOKYO, 2026, 11, 21, 8, 59) + timedelta(seconds=50)
    clock_zero = clock()
    monkeypatch.setattr(
        cli_module, "_utc_now", lambda: started_at + timedelta(seconds=clock() - clock_zero)
    )
    _queue_pages(site, "jra", ["db.netkeiba.com"] * 30)
    gap, jitter = _rate(JRA_HTTP, "db.netkeiba.com")

    result = _run("jra")

    assert result.exit_code == 0, result.output
    out = lines(result)
    assert len(out) == 1 and not out[0].startswith("ERROR") and "quiet" in out[0].lower(), out
    window_opens = _local(TOKYO, 2026, 11, 21, 9, 0)
    sent_at = [started_at + timedelta(seconds=t - clock_zero) for _, t in starts]
    assert 3 <= len(sent_at) < 30  # it ran before 09:00 and stopped long before the queue emptied
    # one inter-request gap of slack: the check may precede the fetcher's own pacing wait
    slack = timedelta(seconds=gap * (1 + jitter) + 0.5)
    assert max(sent_at) < window_opens + slack, max(sent_at)
    counts = _counts("jra")
    assert counts.get(("results", "fetched"), 0) == len(sent_at)  # every page requested completed
    assert counts.get(("results", "failed"), 0) == 0
    assert counts[("results", "pending")] == 30 - len(sent_at)


@pytest.mark.parametrize(
    "now_utc",
    [
        _local(TOKYO, 2026, 11, 21, 9, 0),
        _local(TOKYO, 2026, 11, 21, 12, 30),
        _local(TOKYO, 2026, 11, 22, 15, 40),
    ],
    ids=_row_id,
)
def test_the_logger_host_keeps_running_in_quiet_hours(monkeypatch, site, now_utc):
    monkeypatch.setattr(cli_module, "_utc_now", lambda: now_utc)
    _queue_pages(site, "jra", [LOGGER_HOST] * 3)

    result = _run("jra")

    assert result.exit_code == 0, result.output
    assert not any(ln.startswith("ERROR") for ln in lines(result))
    assert site.hits_on(LOGGER_HOST) == 3 and _pending("jra") == 0


def test_a_mixed_queue_sends_nothing_to_a_quiet_host(monkeypatch, site):
    """Inside the window, whatever happens to the logger host's tasks, the netkeiba hosts get no
    request and their tasks stay pending."""
    monkeypatch.setattr(cli_module, "_utc_now", lambda: _local(TOKYO, 2026, 11, 21, 12, 30))
    _queue_pages(site, "jra", [LOGGER_HOST, LOGGER_HOST], page_type="official")
    _queue_pages(site, "jra", ["db.netkeiba.com"] * 2, page_type="db")
    _queue_pages(site, "jra", ["race.netkeiba.com"] * 2, page_type="race")

    result = _run("jra")

    assert result.exit_code == 0, result.output
    assert not any(ln.startswith("ERROR") for ln in lines(result))
    assert site.hits_on("db.netkeiba.com") == 0 and site.hits_on("race.netkeiba.com") == 0
    counts = _counts("jra")
    assert counts[("db", "pending")] == 2 and counts[("race", "pending")] == 2


def test_hkjc_has_no_quiet_hours(monkeypatch, site):
    monkeypatch.setattr(cli_module, "_utc_now", lambda: _local(TOKYO, 2026, 11, 21, 12, 0))
    _queue_pages(site, "hkjc", ["racing.hkjc.com"] * 2)
    result = _run("hkjc")
    assert result.exit_code == 0, result.output
    assert site.total_hits == 2 and _pending("hkjc") == 0


def test_quiet_hours_do_not_cover_the_logger_host():
    """jra.yaml: no window ever names www.jra.go.jp (every window names its hosts), and both
    netkeiba hosts are quiet Sat and Sun 09:00-17:00 Asia/Tokyo."""
    windows = JRA_HTTP.get("quiet_hours") or []
    assert windows, "jra.yaml http.quiet_hours is missing"
    for window in windows:
        hosts = [str(h).lower() for h in window.get("hosts") or []]
        assert hosts, f"a quiet_hours entry without hosts would cover every host: {window!r}"
        assert LOGGER_HOST not in hosts, window
    for host in NETKEIBA_HOSTS:
        covering = [w for w in windows if host in [str(h).lower() for h in w["hosts"]]]
        assert any(
            {str(d).lower()[:3] for d in w["days"]} == {"sat", "sun"}
            and time.fromisoformat(str(w["start"])) == time(9, 0)
            and time.fromisoformat(str(w["end"])) == time(17, 0)
            and str(w["timezone"]) == "Asia/Tokyo"
            for w in covering
        ), (host, covering)


# ---------------------------------------------------------------- quiet hours, the window rules

# (UTC instant, host, quiet?) for the fake source's window: Wed 10:00-11:30 Europe/London, a.test.
WINDOW_TABLE = [
    (_local(LONDON, 2026, 11, 18, 9, 59), "a.test", False),
    (_local(LONDON, 2026, 11, 18, 10, 0), "a.test", True),  # start inclusive
    (_local(LONDON, 2026, 11, 18, 11, 29), "a.test", True),
    (_local(LONDON, 2026, 11, 18, 11, 30), "a.test", False),  # end exclusive
    (_local(LONDON, 2026, 11, 17, 10, 30), "a.test", False),  # a Tuesday
    (_local(LONDON, 2026, 11, 19, 10, 30), "a.test", False),  # a Thursday
    (datetime(2026, 7, 15, 9, 15, tzinfo=UTC), "a.test", True),  # 10:15 BST
    (datetime(2026, 7, 15, 10, 45, tzinfo=UTC), "a.test", False),  # 11:45 BST
    (_local(LONDON, 2026, 11, 18, 10, 30), "b.test", False),  # not a host the window names
    (_local(LONDON, 2026, 11, 18, 10, 30), "c.test", False),
]


@pytest.mark.parametrize(("now_utc", "host", "quiet"), WINDOW_TABLE, ids=_row_id)
def test_quiet_windows_follow_the_config(monkeypatch, site, now_utc, host, quiet):
    monkeypatch.setattr(cli_module, "_utc_now", lambda: now_utc)
    _queue_pages(site, SOURCE, [host] * 2)

    result = _run(SOURCE)

    assert result.exit_code == 0, result.output
    out = lines(result)
    assert not any(ln.startswith("ERROR") for ln in out)
    if quiet:
        assert len(out) == 1 and "quiet" in out[0].lower(), out
        assert site.total_hits == 0 and _pending(SOURCE) == 2
    else:
        assert site.hits_on(host) == 2 and _pending(SOURCE) == 0


@pytest.mark.parametrize(
    "change",
    [
        {"start": "11:30", "end": "10:00"},  # start after end
        {"start": "10:00", "end": "10:00"},  # empty window
        {"days": ["funday"]},
        {"timezone": "Mars/Olympus"},
        {"start": "25:00"},
    ],
    ids=["reversed", "empty", "bad-day", "bad-zone", "bad-time"],
)
def test_a_bad_quiet_hours_entry_is_one_error_line(fake_config, site, change):
    fake_config["http"]["quiet_hours"][0].update(change)
    _queue_pages(site, SOURCE, ["a.test", "b.test"])

    result = _run(SOURCE)

    assert result.exit_code == 1, result.output
    assert [ln.startswith("ERROR") for ln in lines(result)] == [True]
    assert site.total_hits == 0 and _pending(SOURCE) == 2
