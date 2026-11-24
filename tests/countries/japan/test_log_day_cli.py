"""`horsetrack japan log-day` (ticket #25): the all-venue logger from one command, with a daily
health line and an alert when a venue logs nothing.

Contract pinned here (the coder follows it):

The command lives in `horsetrack.countries.japan.logger.log_day` and registers on the japan group as
`log-day`. Three module attributes are its seams, looked up at call time so tests can replace them:
- `utc_now() -> datetime`: the only clock (aware UTC). It gives the default date (today in
  Asia/Tokyo), and it is the `now` of run_day and of every capture (so snapshot_utc is its time).
- `sleep(seconds)`: the only wait, passed to run_day.
- `make_fetcher(...) -> Fetcher`: the ingest Fetcher every JRADB request goes through (any
  arguments; the tests' replacement ignores them and serves the JRADB double).

`log-day [--date YYYY-MM-DD] [--venues KYO,TOK] [--pools WIN,...] [--dry-run]`:
- wires the JRA site (discover_venues, read_post_times, capture_meeting with the pools asked, the
  default being logger.yaml's), the odds log (odds_snapshot_path('jra', date, venue)) and run_day
  with logger.yaml as its config;
- log records go to logs_dir('japan')/log-day-<date>.log; at most about 10 lines on the terminal;
- `--dry-run` prints the time zone, the windows, the stop rule, the pools, the rate and the venues
  given, exits 0 and sends no request;
- at the end it appends to logs_dir('japan')/health.log one line per venue
  `<date> <venue> snapshots=N rows=N failures=N wrong_meeting=N max_gap_t20_s=N stopped=<reason>`
  and `ALERT <date> <venue> <reason>` (also on stderr) for a venue that logged 0 rows, stopped
  early, or whose max_gap_t20_s > 120;
- exit 0 clean, 2 with any ALERT.

The JRADB double (jradb_double.py) serves a synthetic week through the real Fetcher; its post times
are moved 20 minutes later here so every race's T-20 window falls in logger.yaml's 20 s window
(a race off before 09:50 has 5-minute rounds in its window, a true gap alert). Nothing goes to the
network, and nothing really sleeps: the clock is fake and sleep moves it.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.countries.japan import odds_log

# isort: split
import jradb_double as dbl
from jradb_double import Meeting, World

UTC = timezone.utc
SAT = date(2026, 11, 28)
ALL_POOLS = ["WIN", "PLACE", "QUINELLA", "QUINELLA_PLACE"]
MODULE = "horsetrack.countries.japan.logger.log_day"
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
HEALTH = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<venue>\S+) snapshots=(?P<snapshots>\d+) rows=(?P<rows>\d+)"
    r" failures=(?P<failures>\d+) wrong_meeting=(?P<wrong_meeting>\d+)"
    r" max_gap_t20_s=(?P<gap>\S+) stopped=(?P<stopped>\S+)$"
)
ALERT = re.compile(r"^ALERT (?P<date>\d{4}-\d{2}-\d{2}) (?P<venue>\S+) (?P<reason>\S.*)$")
MAX_TERMINAL_LINES = 12  # "at most about 10"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    """The data directory under tmp_path; no real `.env` can supply a setting; no real network."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data"))
    monkeypatch.setenv("HORSETRACK_USER_AGENT", "horsetrack-tests/1.0")
    (tmp_path / "data").mkdir()

    def no_network(self, request):
        raise AssertionError(f"a real HTTP request was attempted: {request.method} {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    return tmp_path / "data"


@pytest.fixture
def late_posts(monkeypatch):
    """Every race of the double posts 20 minutes later than make_race says (first race 09:50+)."""
    original = dbl.make_race

    def make_race(day, venue, number):
        race = original(day, venue, number)
        return dataclasses.replace(race, post_minutes=race.post_minutes + 20)

    monkeypatch.setattr(dbl, "make_race", make_race)


class Clock:
    """now() is the fake time; sleep(s) moves it (never back) and really waits for nothing."""

    def __init__(self, start: datetime):
        self.t = start
        self.slept = 0.0

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        self.slept += seconds
        self.t += timedelta(seconds=seconds)


def run(monkeypatch, clock: Clock, world: World, *args: str):
    """Invoke `horsetrack japan log-day *args` with the seams on the fake clock and the double.
    Returns (result, site)."""
    mod = importlib.import_module(MODULE)  # the module under test; missing = the red state
    site = dbl.JradbSite(dict(world.pages))
    monkeypatch.setattr(mod, "utc_now", clock.now)
    monkeypatch.setattr(mod, "sleep", clock.sleep)
    monkeypatch.setattr(mod, "make_fetcher", lambda *a, **k: dbl.make_fetcher(site))
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "log-day", *args])
    return result, site


def plain(text: str) -> str:
    return ANSI.sub("", text)


def terminal_lines(result) -> list[str]:
    return [line for line in plain(result.output).splitlines() if line.strip()]


def health_file() -> Path:
    return paths.logs_dir("japan") / "health.log"


def read_health(day: str) -> tuple[dict[str, list[dict]], list[dict]]:
    """health.log's lines for `day`: ({venue: [fields, ...]}, [ALERT fields, ...])."""
    health: dict[str, list[dict]] = {}
    alerts = []
    for line in health_file().read_text(encoding="utf-8").splitlines():
        if line.startswith("ALERT"):
            m = ALERT.match(line)
            assert m, f"malformed ALERT line: {line!r}"
            if m["date"] == day:
                alerts.append(m.groupdict())
            continue
        m = HEALTH.match(line)
        assert m, f"malformed health line: {line!r}"
        if m["date"] == day:
            health.setdefault(m["venue"], []).append(m.groupdict())
    return health, alerts


def snapshots_of(race_date: date, venue: str) -> dict[str, list[datetime]]:
    """race_id -> the snapshot instants the venue's log holds for it."""
    path = paths.odds_snapshot_path("jra", race_date.isoformat(), venue)
    out: dict[str, set[datetime]] = {}
    for row in odds_log.read_rows(path):
        out.setdefault(row["race_id"], set()).add(datetime.fromisoformat(row["snapshot_utc"]))
    return {rid: sorted(times) for rid, times in out.items()}


def longest_gap(times: list[datetime], post: datetime, within_min: int = 20) -> float:
    """The #34 criterion from first principles: the longest gap between captures inside
    [post - within_min, post], the first counted from the window start."""
    start = post - timedelta(minutes=within_min)
    inside = [t for t in times if start <= t <= post]
    assert inside, "no capture inside T-20"
    edges = [start, *inside]
    return max((b - a).total_seconds() for a, b in zip(edges, edges[1:], strict=False))


# ---------------------------------------------------------------- the dry run


@pytest.mark.parametrize(
    "args, venues, pools",
    [
        (["--date", "2026-11-28", "--venues", "KYO,TOK"], ["KYO", "TOK"], ALL_POOLS),
        (
            ["--date", "2026-11-29", "--venues", "NAK", "--pools", "WIN,QUINELLA"],
            ["NAK"],
            ["WIN", "QUINELLA"],
        ),
    ],
    ids=["kyo-tok-all-pools", "nak-two-pools"],
)
def test_dry_run_prints_the_plan_and_sends_nothing(monkeypatch, data_dir, args, venues, pools):
    world = World({SAT: [Meeting("KYO", 2), Meeting("TOK", 2)]})
    clock = Clock(datetime(2026, 11, 27, 23, 0, tzinfo=UTC))
    result, site = run(monkeypatch, clock, world, "--dry-run", *args)

    assert result.exit_code == 0, plain(result.output)
    assert site.seen == [], "the dry run sent a request"
    assert not (data_dir / "odds_snapshots").exists()

    out = plain(result.output)
    lines = terminal_lines(result)
    assert 0 < len(lines) <= MAX_TERMINAL_LINES, out
    config = load_country_yaml("japan", "logger")
    assert "Asia/Tokyo" in out
    for window in config["windows"]:
        assert str(window["from"]) in out
        assert re.search(rf"(?<![\d.]){window['every_s']}(?![\d.])", out), window
    assert str(config["stop"]) in out
    assert re.search(rf"(?<![\d.]){config['after_last_race_min']}(?![\d.])", out)
    for venue in venues:
        assert venue in out
    for venue in {"KYO", "TOK", "NAK"} - set(venues):
        assert venue not in out
    for pool in pools:
        assert pool in out
    if "PLACE" not in pools:
        assert "PLACE" not in out  # neither PLACE nor QUINELLA_PLACE was asked for
    rps = load_country_yaml("japan", "jra")["http"]["hosts"][dbl.HOST]["requests_per_second"]
    number = (
        rf"(?<![\d.:-]){int(rps)}(\.0+)?(?![\d.:-])" if rps == int(rps) else re.escape(str(rps))
    )
    assert any(re.search(r"rate|req", line, re.I) and re.search(number, line) for line in lines), (
        f"no line gives the rate ({rps} requests a second): {out}"
    )


# ---------------------------------------------------------------- a run


@pytest.mark.parametrize(
    "venues, meetings, pools",
    [
        ("TOK,KYO", [Meeting("TOK", 1), Meeting("KYO", 1, nichi=3)], None),
        ("FKS", [Meeting("FKS", 1, 3, 1), Meeting("TOK", 1)], "WIN,QUINELLA"),
    ],
    ids=["tok-kyo-all-pools", "fks-two-pools"],
)
def test_log_day_writes_one_health_line_per_venue(
    monkeypatch, data_dir, late_posts, venues, meetings, pools
):
    world = World({SAT: meetings})
    health_file().parent.mkdir(parents=True, exist_ok=True)
    earlier = (
        "2026-11-21 TOK snapshots=3 rows=30 failures=0 wrong_meeting=0"
        " max_gap_t20_s=20 stopped=ended"
    )
    health_file().write_text(earlier + "\n", encoding="utf-8")
    clock = Clock(datetime(2026, 11, 27, 21, 0, tzinfo=UTC))  # 06:00 in Tokyo on the race day
    args = ["--date", SAT.isoformat(), "--venues", venues]
    if pools:
        args += ["--pools", pools]

    result, site = run(monkeypatch, clock, world, *args)

    out = plain(result.output)
    assert result.exit_code == 0, out
    assert len(terminal_lines(result)) <= MAX_TERMINAL_LINES, out
    assert "ALERT" not in out
    assert site.seen and all(s.served for s in site.seen)
    assert (paths.logs_dir("japan") / f"log-day-{SAT.isoformat()}.log").is_file()
    assert health_file().read_text(encoding="utf-8").splitlines()[0] == earlier  # appended

    health, alerts = read_health(SAT.isoformat())
    asked = venues.split(",")
    assert alerts == []
    assert sorted(health) == sorted(asked)
    wanted_pools = set(pools.split(",")) if pools else set(ALL_POOLS)
    for venue in asked:
        assert len(health[venue]) == 1, venue
        line = health[venue][0]
        log_path = paths.odds_snapshot_path("jra", SAT.isoformat(), venue)
        members = list(odds_log.iter_members(log_path))
        rows = [row for member in members for row in member]
        assert rows, venue
        assert {row["pool"] for row in rows} == wanted_pools, venue
        assert int(line["snapshots"]) == len(members), venue
        assert int(line["rows"]) == len(rows), venue
        assert (int(line["failures"]), int(line["wrong_meeting"])) == (0, 0), venue
        # The gap, from the log's own snapshot times and the double's post times.
        snaps = snapshots_of(SAT, venue)
        races = [world.race(SAT, venue, n) for n in range(1, world.meeting(SAT, venue).races + 1)]
        gaps = [longest_gap(snaps[race.race_id], race.post_utc) for race in races]
        assert float(line["gap"]) == pytest.approx(max(gaps), abs=1.0), venue
        assert float(line["gap"]) <= 120, venue
    for venue in {m.venue for m in meetings} - set(asked):
        assert not paths.odds_snapshot_path("jra", SAT.isoformat(), venue).exists()


@pytest.mark.parametrize(
    "venues, empty",
    [("TOK,NAK", {"NAK"}), ("HSN,TOK,SAP", {"HSN", "SAP"})],
    ids=["one-empty", "two-empty"],
)
def test_a_venue_that_logged_nothing_raises_an_alert_and_exit_2(
    monkeypatch, data_dir, late_posts, venues, empty
):
    world = World({SAT: [Meeting("TOK", 1), Meeting("KYO", 1, nichi=3)]})  # only TOK of these
    clock = Clock(datetime(2026, 11, 27, 21, 0, tzinfo=UTC))

    result, _ = run(monkeypatch, clock, world, "--date", SAT.isoformat(), "--venues", venues)

    assert result.exit_code == 2, plain(result.output)
    assert len(terminal_lines(result)) <= MAX_TERMINAL_LINES, plain(result.output)
    health, alerts = read_health(SAT.isoformat())
    asked = venues.split(",")
    assert sorted(health) == sorted(asked)
    for venue in asked:
        assert len(health[venue]) == 1, venue
        logged = int(health[venue][0]["rows"])
        assert (logged == 0) is (venue in empty), venue
    assert {a["venue"] for a in alerts} == empty
    stderr_alerts = [line for line in plain(result.stderr).splitlines() if "ALERT" in line]
    assert {word for line in stderr_alerts for word in line.split()} >= empty
    assert not any("TOK" in line.split() for line in stderr_alerts)
    # The details went to the day's log, not the terminal.
    day_log = paths.logs_dir("japan") / f"log-day-{SAT.isoformat()}.log"
    text = day_log.read_text(encoding="utf-8")
    assert all(venue in text for venue in empty)


@pytest.mark.parametrize(
    "now, today",
    [
        (datetime(2026, 11, 27, 16, 0, tzinfo=UTC), date(2026, 11, 28)),  # 01:00 in Tokyo
        (datetime(2026, 11, 27, 14, 59, 59, tzinfo=UTC), date(2026, 11, 27)),  # 23:59:59
        (datetime(2026, 11, 28, 15, 0, tzinfo=UTC), date(2026, 11, 29)),  # midnight in Tokyo
    ],
    ids=["01h00-next-day", "23h59-same-day", "midnight"],
)
def test_the_default_date_is_today_in_tokyo(monkeypatch, data_dir, late_posts, now, today):
    # TOK races on the UTC date, the Tokyo date and the day after alike: only the date in the
    # health log tells which day was logged.
    around = sorted({now.date(), today, today + timedelta(days=1)})
    world = World({d: [Meeting("TOK", 1, nichi=i)] for i, d in enumerate(around, start=5)})

    result, _ = run(monkeypatch, Clock(now), world, "--venues", "TOK")

    assert result.exit_code in (0, 2), plain(result.output)
    lines = health_file().read_text(encoding="utf-8").splitlines()
    assert lines
    dates = {line.split()[1] if line.startswith("ALERT") else line.split()[0] for line in lines}
    assert dates == {today.isoformat()}
    assert (paths.logs_dir("japan") / f"log-day-{today.isoformat()}.log").is_file()
