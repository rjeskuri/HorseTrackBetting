"""`horsetrack japan log-day --unattended` and its LaunchAgent (ticket #27): the logger started by
launchd every 30 minutes on the Mac, with no restart loop against the site after a block (HKTrack's
bug f: its supervisor relaunched 30 s after every exit).

Contract pinned here (the coder follows it):

`log-day --unattended` (module horsetrack.countries.japan.logger.log_day, with the seams of
test_log_day_cli.py: `utc_now`, `sleep` and `make_fetcher`, looked up at call time):
- No --date: the race date is today in Asia/Tokyo from utc_now(), whatever the process's local
  time zone (TZ). "Today's stop" is logger.yaml's `stop`, a Tokyo wall time on that date.
- It exits 0 at once (no request, no sleep) when any of these holds:
  - the block marker logs_dir('japan')/jra-block.txt has a line `<host> <race date> ...` for the
    JRA official host (jra.yaml hosts.official) and today: the line schedule.run_day appends when
    a 403/429 opens the breaker;
  - another open file holds fcntl.flock(LOCK_EX) on logs_dir('japan')/log-day.lock: another
    --unattended start or an attended log-day (below). A start that goes on takes
    fcntl.flock(fd, LOCK_EX | LOCK_NB) on that file (never a blocking wait) and keeps it until it
    exits; the file existing blocks nothing;
  - the no-meeting marker logs_dir('japan')/jra-no-meeting.txt has a line whose first field is
    today's date (ISO). Its lines are `<race date> at=<UTC instant>`, appended;
  - the day-done marker logs_dir('japan')/jra-day-done.txt has a line whose first field is today's
    date (ISO). Its lines are `<race date> at=<UTC instant>`, appended;
  - the time is past today's stop.
  The markers are read under the lock.
- Otherwise it runs the day as log-day does (run_day with logger.yaml and the block marker above);
  as logger.yaml says, nothing is requested before the first window. The venues come from one
  discovery request; when it answers with no venue, the start appends today's no-meeting line and
  exits 0 (a blocked discovery is a block, not a day without racing: no no-meeting line).
- Ruling: a start whose run of the day finished appends today's day-done line, `at=` being
  utc_now() once run_day has returned. Finished means run_day returned, with at least one venue,
  and the day was not stopped by a block (DaySummary.stopped is not 'circuit_open': a block has its
  own marker, and clearing that line by hand resumes the day). A run that raised, or that was
  terminated before run_day returned, writes no day-done line, so the next start resumes the day.
  (Without it a start after the last race but before the stop rediscovers, rereads post times
  and appends a health block of 'logged 0 rows' alerts.)
- So after a block, on a day without racing, or once the day's run has finished, every later start
  that day sends nothing and appends no health line; a line for another day, or for another host,
  blocks nothing.

Ruling: an attended `log-day` (no --unattended; any --date, --venues, --pools) takes the same
fcntl.flock(fd, LOCK_EX | LOCK_NB) on logs_dir('japan')/log-day.lock for its whole run, so a manual
run next to the agent never doubles the per-host rate to the JRA site. When another open file holds
the lock it sends no request, does not sleep, appends no health line and exits 1 (the module's error
code: the owner at the terminal sees it), with one `ERROR` line on stderr that names the lock file
(`log-day.lock`). `--dry-run` takes no lock: it prints its plan and exits 0 while the lock is held.

`launchd` (module horsetrack.countries.japan.logger.launchd, registered on the japan group as
`launchd`): `horsetrack japan launchd --print | --install | --uninstall`.
- Seam `launchctl(args: list[str]) -> int`: runs /bin/launchctl with `args` (the arguments only,
  not the program) and returns its exit code; looked up at call time. The tests replace it with a
  fake launchd and start no process.
- The LaunchAgents directory is Path.home() / "Library" / "LaunchAgents", resolved at call time
  (never at import), as are horsetrack.core.env.repo_root() and logs_dir().
- The plist: Label `horsetrack.japan.logger`; ProgramArguments `[/usr/bin/caffeinate, -i,
  <venv bin>/horsetrack, japan, log-day, --unattended]`, the venv bin being the running
  interpreter's (Path(sys.executable).parent, which is sys.prefix/bin in the venv); StartInterval
  1800; RunAtLoad true; KeepAlive false; WorkingDirectory repo_root(); StandardOutPath and
  StandardErrorPath under logs_dir('japan'). Nothing secret: the job reads the data dir and the
  user agent from the repo's .env, so no HORSETRACK_* variable, user agent, token or data dir goes
  in EnvironmentVariables, and no setting's value appears anywhere in the plist.
- --print writes the plist XML to stdout; nothing else is written and launchctl is not called.
- --install creates logs_dir('japan') (launchd creates no log folder), writes
  <LaunchAgents>/horsetrack.japan.logger.plist and runs `launchctl bootstrap gui/<uid> <that path>`.
  A job already loaded is booted out first, so a re-install loads the new plist. It exits non-zero
  when the bootstrap fails.
- --uninstall runs `launchctl bootout gui/<uid>/horsetrack.japan.logger` (or `bootout gui/<uid>
  <plist path>`) and removes the plist file.

The JRADB double (jradb_double.py) serves a synthetic week through the real Fetcher. Nothing goes
to the network and nothing really sleeps (the clock is fake and sleep moves it). In the launchd
tests HOME is under tmp_path, starting any process is an AssertionError, and the real user's
LaunchAgents plist is checked unchanged.
"""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import plistlib
import pwd
import re
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
TOKYO = ZoneInfo("Asia/Tokyo")
TUE, WED, THU, FRI, SAT, SUN, MON = (date(2026, 11, d) for d in range(24, 31))
LOG_DAY = "horsetrack.countries.japan.logger.log_day"
LAUNCHD = "horsetrack.countries.japan.logger.launchd"
LABEL = "horsetrack.japan.logger"
HOST = dbl.HOST  # jra.yaml hosts.official
TESTS_DIR = Path(__file__).resolve().parent
SRC = Path(__file__).resolve().parents[3] / "src"
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

_STOP_H, _STOP_M = (int(x) for x in str(load_country_yaml("japan", "logger")["stop"]).split(":"))
TO_LAST_SECOND_S = 24 * 3600 - 1 - (_STOP_H * 3600 + _STOP_M * 60)  # stop -> 23:59:59


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


# ---------------------------------------------------------------- the clock and the site


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


@dataclass
class Site(dbl.JradbSite):
    """The JRADB double, noting the fake time of every request. With `block_at`, the n-th request
    (1-based) and every later one is answered `status` (403/429: the breaker opens)."""

    clock: Clock | None = None
    block_at: int | None = None
    status: int = 403
    at: list[datetime] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.clock is not None:
            self.at.append(self.clock.now())
        if self.block_at is not None and len(self.seen) + 1 >= self.block_at:
            url = request.url
            self.seen.append(dbl.Seen(request.method, url.host, url.path, None, None, False))
            return httpx.Response(self.status, content=b"", headers={"Content-Type": "text/html"})
        return super().handler(request)


@dataclass
class Start:
    result: Any
    site: Site
    waited: float  # seconds the start slept

    @property
    def code(self) -> int:
        return self.result.exit_code

    @property
    def requests(self) -> int:
        return len(self.site.seen)

    @property
    def output(self) -> str:
        return plain(self.result.output)

    @property
    def stderr(self) -> str:
        return plain(self.result.stderr)


def plain(text: str) -> str:
    return ANSI.sub("", text)


def start(
    monkeypatch,
    clock: Clock,
    world: World,
    *,
    site: Site | None = None,
    sleep: Callable[[float], None] | None = None,
    args: Sequence[str] = ("--unattended",),
) -> Start:
    """One `horsetrack japan log-day --unattended` (or log-day with `args`), as launchd starts it:
    the seams on the fake clock, a fresh JRADB double over `world` (or `site`)."""
    mod = importlib.import_module(LOG_DAY)
    site = site if site is not None else Site(dict(world.pages), clock=clock)
    monkeypatch.setattr(mod, "utc_now", clock.now)
    monkeypatch.setattr(mod, "sleep", sleep or clock.sleep)
    monkeypatch.setattr(mod, "make_fetcher", lambda *a, **k: dbl.make_fetcher(site))
    from horsetrack.cli import app

    before = clock.slept
    result = CliRunner().invoke(app, ["japan", "log-day", *args])
    out = plain(result.output)
    assert "No such option" not in out and "Usage:" not in out, f"{list(args)} refused: {out}"
    return Start(result, site, clock.slept - before)


def at(day: date, hh: int, mm: int, ss: int = 0) -> datetime:
    """The UTC instant of a Tokyo wall time on `day`."""
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=TOKYO).astimezone(UTC)


def wall(day: date, hhmm: str) -> datetime:
    hh, mm = (int(x) for x in str(hhmm).split(":"))
    return at(day, hh, mm)


def first_window(day: date) -> datetime:
    return min(wall(day, w["from"]) for w in load_country_yaml("japan", "logger")["windows"])


def stop(day: date) -> datetime:
    return wall(day, load_country_yaml("japan", "logger")["stop"])


def week() -> World:
    """A JRA week: racing on Saturday and Sunday only, one race a day."""
    return World({SAT: [Meeting("TOK", 1)], SUN: [Meeting("KYO", 1, nichi=8)]})


def logs() -> Path:
    return paths.logs_dir("japan")


def no_meeting_days() -> list[str]:
    """The first field of every line of the no-meeting marker."""
    path = logs() / "jra-no-meeting.txt"
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split()[0] for line in lines if line.strip()]


def block_lines() -> list[list[str]]:
    """[host, race date] of every line of the block marker."""
    path = logs() / "jra-block.txt"
    if not path.is_file():
        return []
    return [line.split()[:2] for line in path.read_text(encoding="utf-8").splitlines()]


def write_block_lines(*lines: tuple[str, date, int]) -> None:
    """Block lines as schedule.run_day appends them: `<host> <race date> status=<s> at=<UTC>`."""
    path = logs() / "jra-block.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        f"{host} {day.isoformat()} status={status} at={at(day, 9, 41).isoformat()}\n"
        for host, day, status in lines
    )
    path.write_text(text, encoding="utf-8")


def day_done_lines() -> list[tuple[str, datetime | None]]:
    """(race date, the at= instant or None) of every line of the day-done marker."""
    path = logs() / "jra-day-done.txt"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        stamps = [f[3:] for f in fields[1:] if f.startswith("at=")]
        instant = None
        if stamps:
            text = stamps[0][:-1] + "+00:00" if stamps[0].endswith("Z") else stamps[0]
            instant = datetime.fromisoformat(text)
        out.append((fields[0], instant))
    return out


def day_done_days() -> list[str]:
    return [day for day, _ in day_done_lines()]


def write_day_done_lines(*days: date) -> None:
    """Day-done lines as a finished start appends them: `<race date> at=<UTC instant>`."""
    path = logs() / "jra-day-done.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{day.isoformat()} at={at(day, 10, 31).isoformat()}\n" for day in days)
    path.write_text(text, encoding="utf-8")


def health_bytes() -> bytes | None:
    path = logs() / "health.log"
    return path.read_bytes() if path.is_file() else None


def day_end(world: World, day: date) -> datetime:
    """When the day's last venue ends: its last post + after_last_race_min."""
    after = timedelta(minutes=load_country_yaml("japan", "logger")["after_last_race_min"])
    posts = [t for m in world.days[day] for t in world.expected_post_times(day, m.venue).values()]
    return max(posts) + after


def logged_rows(day: date, venue: str) -> list[dict]:
    path = paths.odds_snapshot_path("jra", day.isoformat(), venue)
    return list(odds_log.read_rows(path)) if path.exists() else []


# ---------------------------------------------------------------- the lock


def lock_path() -> Path:
    return logs() / "log-day.lock"


def held_elsewhere(path: Path) -> bool:
    """Whether some other open file holds an exclusive flock on `path` (the probe keeps nothing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False


@contextmanager
def holding(path: Path) -> Iterator[None]:
    """Another holder of the lock: an exclusive flock on its own open file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# A second `log-day --unattended` (or log-day with the arguments after `out`) in a process of its
# own, on the same data dir: the seams on a fake clock from `now`, a JRADB double with no racing
# that day, no network. It writes what it did as JSON to `out`.
SECOND_START = textwrap.dedent(
    """
    import importlib
    import json
    import sys
    from datetime import datetime, timedelta
    from pathlib import Path

    tests_dir, repo, now, out = sys.argv[1:5]
    args = sys.argv[5:]
    sys.path.insert(0, tests_dir)

    import httpx


    def no_network(self, request):
        raise AssertionError(f"a real HTTP request was attempted: {request.url}")


    httpx.HTTPTransport.handle_request = no_network

    import horsetrack.core.env as env

    env.repo_root = lambda: Path(repo)

    import jradb_double as dbl
    from typer.testing import CliRunner

    mod = importlib.import_module("horsetrack.countries.japan.logger.log_day")
    clock = {"t": datetime.fromisoformat(now), "slept": 0.0}


    def sleep(seconds):
        seconds = max(0.0, float(seconds))
        clock["slept"] += seconds
        clock["t"] += timedelta(seconds=seconds)


    today = clock["t"].astimezone(dbl.JST).date()
    world = dbl.World({today + timedelta(days=7): [dbl.Meeting("NAK", 1)]})
    site = dbl.JradbSite(dict(world.pages))
    mod.utc_now = lambda: clock["t"]
    mod.sleep = sleep
    mod.make_fetcher = lambda *a, **k: dbl.make_fetcher(site)

    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "log-day", *args])
    done = {
        "exit_code": result.exit_code,
        "requests": len(site.seen),
        "slept": clock["slept"],
        "output": result.output,
        "stderr": result.stderr,
    }
    Path(out).write_text(json.dumps(done))
    """
)


def start_in_another_process(
    tmp_path: Path, now: datetime, args: Sequence[str] = ("--unattended",)
) -> dict:
    driver = tmp_path / "second_start.py"
    driver.write_text(SECOND_START, encoding="utf-8")
    out = tmp_path / "second_start.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC), env.get("PYTHONPATH", "")) if p)
    argv = [sys.executable, str(driver), str(TESTS_DIR), str(tmp_path / "repo"), now.isoformat()]
    # A blocking lock would wait for the first start, which waits for this: the timeout ends it.
    proc = subprocess.run(
        [*argv, str(out), *args], env=env, capture_output=True, text=True, timeout=30, check=False
    )
    if proc.returncode != 0 or not out.is_file():
        return {"error": f"exit {proc.returncode}: {proc.stderr[-3000:]}"}
    return json.loads(out.read_text(encoding="utf-8"))


def test_a_running_logger_holds_the_lock_and_a_second_start_exits_0(monkeypatch, tmp_path):
    lock = lock_path()
    clock = Clock(at(SAT, 9, 25))  # Saturday's one race is near: the first start logs it
    held: list[bool] = []
    second: dict = {}

    def sleep(seconds: float) -> None:
        # The running logger waits: is its lock held, and what does a second start do meanwhile?
        held.append(held_elsewhere(lock))
        if not second:
            try:
                second.update(start_in_another_process(tmp_path, clock.now()))
            except Exception as exc:  # e.g. a blocking lock: the second start waits for the first
                second["error"] = repr(exc)
        clock.sleep(seconds)

    first = start(monkeypatch, clock, week(), sleep=sleep)

    assert first.code in (0, 2), first.output
    assert logged_rows(SAT, "TOK"), f"the first start did not log the day: {first.output}"
    assert held, "the running logger never waited"
    assert all(held), f"the lock was free at {held.count(False)} of {len(held)} waits of the run"
    assert "error" not in second, second.get("error")
    got = (second["exit_code"], second["requests"], second["slept"])
    assert got == (0, 0, 0.0), f"second start while one runs: {second}"
    assert not held_elsewhere(lock), "the lock outlived the run"

    # The flock, not the file: another holder blocks a start; once it lets go, the file is still
    # there and the next start runs (Monday, no racing: its one discovery).
    monday = week()
    with holding(lock):
        blocked = start(monkeypatch, Clock(at(MON, 5, 0)), monday)
    assert (blocked.code, blocked.requests, blocked.waited) == (0, 0, 0.0), blocked.output
    assert lock.is_file()
    after = start(monkeypatch, Clock(at(MON, 5, 0)), monday)
    assert after.code == 0, after.output
    assert after.requests == 1, f"{after.requests} requests once the lock was free"


def one_error_naming_the_lock(stderr: str) -> list[str]:
    lines = [line for line in plain(stderr).splitlines() if line.strip()]
    ok = len(lines) == 1 and lines[0].startswith("ERROR") and "log-day.lock" in lines[0]
    return [] if ok else lines


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--date", SAT.isoformat()],
        ["--date", MON.isoformat()],
        ["--date", SAT.isoformat(), "--venues", "TOK", "--pools", "WIN"],
    ],
    ids=["today", "a-racing-day", "a-day-without-racing", "venues-and-pools"],
)
def test_an_attended_log_day_takes_the_same_lock(monkeypatch, args):
    """The owner's `log-day` while the agent (here: the test, a genuine flock holder) holds the
    lock: refused before any request, an error the owner sees. --dry-run is not refused."""
    lock, world = lock_path(), week()
    with holding(lock):
        refused = start(monkeypatch, Clock(at(SAT, 9, 25)), world, args=args)
        dry = start(monkeypatch, Clock(at(SAT, 9, 25)), world, args=[*args, "--dry-run"])

    assert refused.code == 1, f"exit {refused.code} while the lock was held: {refused.output}"
    assert refused.requests == 0, f"{refused.requests} request(s) while the lock was held"
    assert refused.waited == 0.0, f"it waited {refused.waited} s instead of exiting"
    assert not one_error_naming_the_lock(refused.result.stderr), (
        f"stderr is not one ERROR line naming log-day.lock: {refused.stderr!r}"
    )
    assert health_bytes() is None, "a refused log-day appended health lines"
    assert not logged_rows(SAT, "TOK")

    assert dry.code == 0, f"--dry-run refused while the lock was held: {dry.output}"
    assert dry.requests == 0

    # The flock, not the file: once the holder lets go, the same kind of run goes on.
    after = start(monkeypatch, Clock(at(MON, 5, 0)), world, args=["--date", MON.isoformat()])
    assert after.code in (0, 2), after.output
    assert after.requests == 1, f"{after.requests} requests once the lock was free"


@pytest.mark.parametrize(
    "second_args, second_code",
    [(["--unattended"], 0), ([], 1)],
    ids=["then-an-unattended-start", "then-another-attended-log-day"],
)
def test_a_running_attended_log_day_holds_the_lock_against_a_second_start(
    monkeypatch, tmp_path, second_args, second_code
):
    """The reverse order: the owner's log-day runs, and a start in another process (launchd's, or
    a second terminal's) sends nothing meanwhile."""
    lock = lock_path()
    clock = Clock(at(SAT, 9, 25))
    held: list[bool] = []
    second: dict = {}

    def sleep(seconds: float) -> None:
        held.append(held_elsewhere(lock))
        if not second:
            try:
                second.update(start_in_another_process(tmp_path, clock.now(), second_args))
            except Exception as exc:  # e.g. a blocking lock: the second start waits for the first
                second["error"] = repr(exc)
        clock.sleep(seconds)

    first = start(monkeypatch, clock, week(), sleep=sleep, args=[])

    assert first.code in (0, 2), first.output
    assert logged_rows(SAT, "TOK"), f"the attended log-day did not log the day: {first.output}"
    assert held, "the attended log-day never waited"
    assert all(held), f"the lock was free at {held.count(False)} of {len(held)} waits of the run"
    assert "error" not in second, second.get("error")
    got = (second["exit_code"], second["requests"], second["slept"])
    assert got == (second_code, 0, 0.0), f"{second_args} while log-day runs: {second}"
    if second_code:
        assert not one_error_naming_the_lock(second["stderr"]), second["stderr"]
    assert not held_elsewhere(lock), "the lock outlived the run"


# ---------------------------------------------------------------- after a block


@pytest.mark.parametrize(
    "today, first, later",
    [
        (SAT, (at(SAT, 6, 0), 1, 403), [at(SAT, 7, 30), at(SAT, 12, 0)]),
        (SUN, (at(SUN, 9, 20), 6, 429), [at(SUN, 9, 50), at(SUN, 16, 0)]),
        (SAT, None, [at(SAT, 5, 0), at(SAT, 13, 0)]),
    ],
    ids=["403-on-discovery", "429-mid-capture", "marker-left-by-an-earlier-start"],
)
def test_a_block_marker_for_today_exits_0_without_a_request(monkeypatch, today, first, later):
    world = week()
    if first is None:
        # Today's line, and lines that must block nothing: another day, another host.
        write_block_lines((HOST, today, 403), (HOST, FRI, 429), ("odds.example.invalid", MON, 403))
    else:
        t0, block_at, status = first
        clock = Clock(t0)
        site = Site(dict(world.pages), clock=clock, block_at=block_at, status=status)
        blocked = start(monkeypatch, clock, world, site=site)
        assert blocked.code in (0, 2), blocked.output
        assert blocked.requests == block_at, (
            f"{blocked.requests} requests: the start went on after the {status}"
        )
        assert [HOST, today.isoformat()] in block_lines(), "no block line for today"
        assert today.isoformat() not in no_meeting_days(), "a block is not a day without racing"
        assert today.isoformat() not in day_done_days(), "a block is not a finished day"

    for t in later:
        s = start(monkeypatch, Clock(t), world)
        assert s.code == 0, s.output
        assert s.requests == 0, f"a start at {t:%H:%M}Z sent {s.requests} request(s) after a block"
        assert s.waited == 0.0, f"a start at {t:%H:%M}Z waited {s.waited} s instead of exiting"

    # The block is for today and that host only: Monday, without racing, has its one discovery.
    s = start(monkeypatch, Clock(at(MON, 5, 0)), world)
    assert s.code == 0, s.output
    assert s.requests == 1, f"Monday sent {s.requests} requests"


# ---------------------------------------------------------------- a day without racing


@pytest.mark.parametrize(
    "today, first, later",
    [
        (TUE, (1, 30), [(7, 30), (12, 0), (16, 40)]),
        (THU, (9, 10), [(9, 40), (16, 44)]),
    ],
    ids=["first-start-before-the-window", "first-start-mid-morning"],
)
def test_a_day_without_racing_costs_one_request_per_day(monkeypatch, today, first, later):
    world = week()
    s = start(monkeypatch, Clock(at(today, *first)), world)
    assert s.code == 0, s.output
    assert s.requests == 1, f"the first start sent {s.requests} requests, not one discovery"
    assert min(s.site.at) >= first_window(today), "a request before the first window"
    assert no_meeting_days() == [today.isoformat()]
    total = s.requests

    for hh, mm in later:
        s = start(monkeypatch, Clock(at(today, hh, mm)), world)
        assert s.code == 0, s.output
        assert s.requests == 0, f"a start at {hh:02d}:{mm:02d} sent {s.requests} request(s)"
        assert s.waited == 0.0, f"a start at {hh:02d}:{mm:02d} waited instead of exiting"
        total += s.requests
    assert total == 1

    # The next day is a day of its own: one discovery again, then nothing.
    nxt = today + timedelta(days=1)
    s = start(monkeypatch, Clock(at(nxt, 0, 10)), world)
    assert s.code == 0, s.output
    assert s.requests == 1, f"the next day's first start sent {s.requests} requests"
    assert sorted(no_meeting_days()) == [today.isoformat(), nxt.isoformat()]
    s = start(monkeypatch, Clock(at(nxt, 11, 0)), world)
    assert (s.code, s.requests, s.waited) == (0, 0, 0.0), s.output


# ---------------------------------------------------------------- once the day's run has finished


TWO_VENUE_SUN = {SUN: [Meeting("KYO", 1, nichi=8), Meeting("NAK", 1, nichi=8)]}


@pytest.mark.parametrize(
    "today, days, first, later",
    [
        # TOK's one race is off at 09:35 and ends at 09:45; 10:15 is the reviewer's case.
        (SAT, None, (9, 25), [(10, 15), (13, 0), (16, 44, 59)]),
        # Both venues end at 09:45; the first start comes in the 5-minute window.
        (SUN, TWO_VENUE_SUN, (9, 0), [(9, 46), (12, 30), (16, 44, 59)]),
        # Today's line from an earlier start; lines for other days block nothing.
        (SAT, None, None, [(9, 25), (12, 0)]),
    ],
    ids=["one-venue-ended-mid-morning", "two-venues", "marker-left-earlier"],
)
def test_a_start_after_the_day_has_finished_exits_0_without_a_request(
    monkeypatch, today, days, first, later
):
    world = week() if days is None else World(days)
    venues = [m.venue for m in world.days[today]]
    if first is None:
        write_day_done_lines(FRI, today, date(2026, 12, 5))
    else:
        s = start(monkeypatch, Clock(at(today, *first)), world)
        assert s.code in (0, 2), s.output
        for venue in venues:
            assert logged_rows(today, venue), f"{today} {venue} logged nothing: {s.output}"
        assert day_done_days() == [today.isoformat()], (
            f"day-done lines {day_done_lines()} after the day's run finished"
        )
        [(_, finished)] = day_done_lines()
        assert finished is not None and finished.utcoffset() == timedelta(0), day_done_lines()
        assert finished >= max(s.site.at), "the day-done line predates the run's last request"
        assert finished >= day_end(world, today), "the day was marked done before it ended"
    health = health_bytes()

    for hms in later:
        t = at(today, *hms)
        assert (first is None or t > day_end(world, today)) and t < stop(today)
        s = start(monkeypatch, Clock(t), world)
        when = t.astimezone(TOKYO).strftime("%H:%M:%S")
        assert s.code == 0, f"a start at {when} once the day was done: {s.output}"
        assert s.requests == 0, f"a start at {when} sent {s.requests} request(s): the day was done"
        assert s.waited == 0.0, f"a start at {when} waited {s.waited} s instead of exiting"
        assert health_bytes() == health, f"a start at {when} appended health lines"
    assert day_done_days().count(today.isoformat()) == 1, day_done_lines()

    # The day-done line is for today only: Monday, without racing, has its one discovery.
    s = start(monkeypatch, Clock(at(MON, 5, 0)), world)
    assert s.code == 0, s.output
    assert s.requests == 1, f"Monday sent {s.requests} requests"


class Crash(Exception):
    """An error from inside the day's run."""


@pytest.mark.parametrize(
    "death, code",
    [(lambda: Crash("synthetic failure mid-day"), 1), (lambda: SystemExit(143), 143)],
    ids=["an-error", "terminated"],
)
def test_a_run_interrupted_mid_day_is_resumed_by_the_next_start(monkeypatch, death, code):
    """A run that died before run_day returned is not a finished day: the next start goes on."""
    # Posts 09:35 and 10:10, so the day ends at 10:20: the run dies between the races, and the
    # next start comes in race 2's hot window.
    world = World({SAT: [Meeting("TOK", 2)]})
    clock = Clock(at(SAT, 9, 41))
    dies_at = at(SAT, 9, 47)

    def sleep(seconds: float) -> None:
        if clock.now() >= dies_at:
            raise death()
        clock.sleep(seconds)

    first = start(monkeypatch, clock, world, sleep=sleep)
    assert first.code == code, f"exit {first.code}: {first.output}"
    assert logged_rows(SAT, "TOK"), "the run logged nothing before it died"
    assert SAT.isoformat() not in day_done_days(), "a run that died counted as the day done"

    resume_at = at(SAT, 10, 12)
    resumed = start(monkeypatch, Clock(resume_at), world)
    assert resumed.code in (0, 2), resumed.output
    assert resumed.requests >= 1, "the next start did not resume the day"
    new = {
        r["race_id"]
        for r in logged_rows(SAT, "TOK")
        if datetime.fromisoformat(r["snapshot_utc"]) >= resume_at
    }
    assert f"jra:{SAT.isoformat()}:TOK:02" in new, f"race 2 not logged after the resume: {new}"
    assert day_done_days() == [SAT.isoformat()], "the resumed run finished: no day-done line"

    s = start(monkeypatch, Clock(at(SAT, 10, 45)), world)
    assert (s.code, s.requests, s.waited) == (0, 0, 0.0), s.output


# ---------------------------------------------------------------- the stop


@pytest.mark.parametrize(
    "day, after_stop_s, runs",
    [
        (SAT, 1, False),
        (SAT, 95 * 60, False),
        (TUE, TO_LAST_SECOND_S, False),
        (SUN, -60, True),
        (TUE, -60, True),
    ],
    ids=[
        "racing-day-a-second-past",
        "racing-day-95-min-past",
        "no-racing-last-second",
        "racing-day-a-minute-before",
        "no-racing-a-minute-before",
    ],
)
def test_after_the_stop_time_it_exits_0(monkeypatch, day, after_stop_s, runs):
    now = stop(day) + timedelta(seconds=after_stop_s)
    assert now.astimezone(TOKYO).date() == day
    s = start(monkeypatch, Clock(now), week())
    if not runs:
        assert s.code == 0, s.output
        assert s.requests == 0, f"{s.requests} request(s) after the stop"
        assert s.waited == 0.0, "it waited instead of exiting"
        return
    # Before the stop the day still runs: its discovery at least.
    assert s.code in (0, 2), s.output
    assert s.requests >= 1, "nothing was sent before the stop"
    if day not in week().days:
        assert s.requests == 1
        assert no_meeting_days() == [day.isoformat()]


# ---------------------------------------------------------------- the race date


@pytest.fixture
def mac_time_zone() -> Iterator[Callable[[str], None]]:
    """Set the process's local time zone (TZ + tzset), restored afterwards."""
    saved = os.environ.get("TZ")

    def use(name: str) -> None:
        os.environ["TZ"] = name
        time.tzset()

    yield use
    if saved is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = saved
    time.tzset()


NO_RACING_SAT = {FRI: [("KYO", 1)], SUN: [("NAK", 1)]}
RACING_SAT = {FRI: [("KYO", 1)], SAT: [("TOK", 1)]}


@pytest.mark.parametrize(
    "tz, now, days, expect",
    [
        # Tokyo Sat 01:30; Friday in UTC and in Los Angeles
        ("America/Los_Angeles", datetime(2026, 11, 27, 16, 30, tzinfo=UTC), NO_RACING_SAT, "none"),
        # Tokyo Sat 10:00; Los Angeles Fri 17:00, past 16:45 there
        ("America/Los_Angeles", datetime(2026, 11, 28, 1, 0, tzinfo=UTC), NO_RACING_SAT, "none"),
        # Tokyo Sat 20:30, past the stop; Auckland Sun 00:30, the day Sunday's racing
        ("Pacific/Auckland", datetime(2026, 11, 28, 11, 30, tzinfo=UTC), NO_RACING_SAT, "stop"),
        # Tokyo Sat 08:55, racing at TOK; Friday (racing at KYO) in UTC and in Los Angeles
        ("America/Los_Angeles", datetime(2026, 11, 27, 23, 55, tzinfo=UTC), RACING_SAT, "TOK"),
    ],
    ids=["la-tokyo-ahead-a-day", "la-past-its-local-stop", "auckland-ahead-of-tokyo", "la-racing"],
)
def test_the_race_date_is_tokyo_today_whatever_the_mac_time_zone(
    monkeypatch, mac_time_zone, tz, now, days, expect
):
    mac_time_zone(tz)
    today = now.astimezone(TOKYO).date()
    assert datetime.fromtimestamp(now.timestamp()).date() != today, "the Mac's date is Tokyo's"
    world = World({d: [Meeting(v, n) for v, n in meetings] for d, meetings in days.items()})

    s = start(monkeypatch, Clock(now), world)

    others = [(d, v) for d, meetings in days.items() for v, _ in meetings if d != today]
    for d, venue in others:
        assert not logged_rows(d, venue), f"{d} {venue} logged: the race date was not {today}"
    if expect == "none":
        assert s.code == 0, s.output
        assert s.requests == 1, f"{s.requests} requests, not one discovery"
        assert no_meeting_days() == [today.isoformat()]
    elif expect == "stop":
        assert (s.code, s.requests, s.waited) == (0, 0, 0.0), s.output
        assert no_meeting_days() == []
    else:
        assert s.code in (0, 2), s.output
        rows = logged_rows(today, expect)
        assert rows, f"{today} {expect} logged nothing"
        assert {r["race_id"].rsplit(":", 1)[0] for r in rows} == {f"jra:{today}:{expect}"}
        assert no_meeting_days() == []


# ---------------------------------------------------------------- launchd


class FakeLaunchd:
    """launchctl as the tests' launchd: every call recorded; `loaded` is what launchd would run,
    label -> the plist it read at bootstrap. Exit codes as launchctl's: 5 for a bootstrap of a
    loaded job or one it cannot read, 3 for a bootout of a job not loaded, 64 for anything else.
    `refuse` makes every bootstrap fail with that code."""

    def __init__(self) -> None:
        self.domain = f"gui/{os.getuid()}"
        self.loaded: dict[str, dict] = {}
        self.calls: list[list[str]] = []
        self.refuse: int | None = None

    @staticmethod
    def _read(path: str) -> dict | None:
        p = Path(path)
        if not p.is_absolute() or not p.is_file():
            return None
        try:
            return plistlib.loads(p.read_bytes())
        except Exception:
            return None

    def __call__(self, args) -> int:
        args = [str(a) for a in args]
        self.calls.append(args)
        verb, rest = (args[0], args[1:]) if args else ("", [])
        if verb == "bootstrap" and len(rest) == 2 and rest[0] == self.domain:
            plist = self._read(rest[1])
            if plist is None or plist.get("Label") in self.loaded:
                return 5
            if self.refuse is not None:
                return self.refuse
            self.loaded[plist["Label"]] = plist
            return 0
        if verb == "bootout":
            if len(rest) == 1 and rest[0].startswith(self.domain + "/"):
                label = rest[0][len(self.domain) + 1 :]
            elif len(rest) == 2 and rest[0] == self.domain and self._read(rest[1]) is not None:
                label = self._read(rest[1]).get("Label")
            else:
                return 64
            if label not in self.loaded:
                return 3
            del self.loaded[label]
            return 0
        return 64


def _state(path: Path) -> tuple[bool, bytes | None]:
    try:
        return True, path.read_bytes()
    except FileNotFoundError:
        return False, None


@pytest.fixture
def home(tmp_path, monkeypatch) -> Iterator[Path]:
    """HOME under tmp_path, and no process can start: the real launchctl and the real
    LaunchAgents directory stay out of reach (and the real plist is checked unchanged)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    real = Path(pwd.getpwuid(os.getuid()).pw_dir) / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    before = _state(real)

    def no_process(*args, **kwargs):
        raise AssertionError(f"a real process was started: {args!r}")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    for name in ("system", "posix_spawn", "posix_spawnp", "fork", "execv", "execve", "execvp"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, no_process)
    yield home
    assert _state(real) == before, f"the test changed the real {real.name}"


def load_launchd(monkeypatch) -> FakeLaunchd:
    """The launchd module with a fake launchctl. Reloaded under the fake HOME, so even a path
    computed at import would land under tmp_path."""
    mod = importlib.reload(importlib.import_module(LAUNCHD))  # missing: the red state
    fake = FakeLaunchd()
    monkeypatch.setattr(mod, "launchctl", fake)
    return fake


def launchd_cli(*args: str):
    from horsetrack.cli import app

    return CliRunner().invoke(app, ["japan", "launchd", *args])


def printed_plist() -> dict:
    result = launchd_cli("--print")
    assert result.exit_code == 0, plain(result.output)
    assert result.stdout.startswith("<?xml"), result.stdout[:200]
    return plistlib.loads(result.stdout.encode())


def venv_horsetrack() -> set[Path]:
    """The running interpreter's bin/horsetrack (the same file, named either way)."""
    return {Path(sys.executable).parent / "horsetrack", Path(sys.prefix) / "bin" / "horsetrack"}


@pytest.mark.parametrize(
    "repo, data, agent",
    [
        ("repo", "data", "horsetrack-tests/1.0 (+synthetic-contact-a)"),
        ("checkout-b", "store-b", "odds-research/2.3 (+synthetic-contact-b)"),
    ],
    ids=["default-layout", "other-checkout"],
)
def test_plist_runs_log_day_unattended_every_30_minutes_without_keepalive(
    monkeypatch, tmp_path, home, repo, data, agent
):
    root, data_path = tmp_path / repo, tmp_path / data
    root.mkdir(exist_ok=True)
    data_path.mkdir(exist_ok=True)
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data_path))
    monkeypatch.setenv("HORSETRACK_USER_AGENT", agent)
    dotenv_agent, token = f"dotenv-{agent}", f"synthetic-token-{repo}-0123456789abcdef"
    (root / ".env").write_text(
        f"HORSETRACK_USER_AGENT={dotenv_agent}\nHORSETRACK_SERVICE_TOKEN={token}\n",
        encoding="utf-8",
    )
    launchd = load_launchd(monkeypatch)

    result = launchd_cli("--print")
    assert result.exit_code == 0, plain(result.output)
    text = result.stdout
    assert text.startswith("<?xml"), text[:200]
    plist = plistlib.loads(text.encode())

    assert plist["Label"] == LABEL
    args = plist["ProgramArguments"]
    assert args[:2] == ["/usr/bin/caffeinate", "-i"], args
    assert Path(args[2]) in venv_horsetrack(), args
    assert args[3:] == ["japan", "log-day", "--unattended"], args
    assert plist["StartInterval"] == 1800 and type(plist["StartInterval"]) is int
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is False
    assert Path(plist["WorkingDirectory"]).resolve() == root.resolve()
    log_dir = paths.logs_dir("japan")
    assert log_dir.is_relative_to(data_path.resolve())
    for key in ("StandardOutPath", "StandardErrorPath"):
        assert Path(plist[key]).is_absolute(), plist[key]
        assert Path(plist[key]).is_relative_to(log_dir), (key, plist[key])

    # Nothing secret: the job reads its settings from the repo's .env.
    env = plist.get("EnvironmentVariables", {})
    assert not [k for k in env if k.startswith("HORSETRACK")], env
    assert not [v for v in env.values() if str(data_path) in str(v)], env
    for secret in (agent, dotenv_agent, token):
        assert secret not in text, f"{secret!r} is in the plist"

    # Printing installs nothing.
    assert launchd.calls == []
    assert not (home / "Library" / "LaunchAgents" / f"{LABEL}.plist").exists()


def test_install_and_uninstall_call_launchctl_with_the_plist(monkeypatch, tmp_path, home):
    launchd = load_launchd(monkeypatch)
    agent_plist = home / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    printed = printed_plist()

    result = launchd_cli("--install")
    assert result.exit_code == 0, (plain(result.output), launchd.calls)
    assert agent_plist.is_file()
    assert plistlib.loads(agent_plist.read_bytes()) == printed
    assert launchd.loaded == {LABEL: printed}, launchd.calls
    assert paths.logs_dir("japan").is_dir(), "launchd will not create the log folder"

    # A re-install (the repo moved) replaces the loaded job with the new plist.
    moved = tmp_path / "repo-moved"
    moved.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: moved)
    again = printed_plist()
    assert again["WorkingDirectory"] != printed["WorkingDirectory"]
    result = launchd_cli("--install")
    assert result.exit_code == 0, (plain(result.output), launchd.calls)
    assert plistlib.loads(agent_plist.read_bytes()) == again
    assert launchd.loaded == {LABEL: again}, launchd.calls

    result = launchd_cli("--uninstall")
    assert result.exit_code == 0, (plain(result.output), launchd.calls)
    assert launchd.loaded == {}, launchd.calls
    assert not agent_plist.exists()

    # launchctl refusing the job is a failed install, not a quiet success.
    launchd.refuse = 5
    result = launchd_cli("--install")
    assert result.exit_code != 0, plain(result.output)
    assert launchd.loaded == {}
