"""`pull-vm --every SECONDS`: the VM's odds log pulled live during racing (ticket #236).

Contract pinned here (module `horsetrack.countries.japan.logger.pull_vm`):
- `pull_vm_loop(*, host, every, race_date, until=None, remote_dir=DEFAULT_REMOTE_DIR,
  dry_run=False, runner=None, sleep=time.sleep, now=utc_now)`, all keyword arguments: `race_date`
  is a `datetime.date`, `until` an aware datetime (None: run until Ctrl-C), `every` a positive int
  of seconds. A round is one rsync of the day's odds folder only:
  `rsync -a --partial [--dry-run] <host>:<remote>/odds_snapshots/jra/<DATE>/ <data_dir>/vm/<host>/
  odds_snapshots/jra/<DATE>` (destination folder created first unless dry run), never `--delete*`.
  The first round runs at once, then `sleep(every)` between rounds; no round starts after `until`.
  A round whose runner returns non-zero prints an `error` line to stderr and the loop goes on.
  `runner=None` is the module attribute `run_rsync`, looked up at call time. A host that is not an
  ssh alias raises ValueError before anything runs.
- Module attributes looked up at call time by the CLI: `utc_now()` (the clock: the default date is
  today in Asia/Tokyo and `--until HH:MM` is Tokyo time on the pull's date) and `time.sleep`
  (`import time`, called as `time.sleep(...)`; tests patch `time.sleep` itself).
- CLI `horsetrack japan pull-vm --host H --every N [--date YYYY-MM-DD] [--until HH:MM]`. Without
  `--every` one nightly pull of three subtrees, as before. N must be a positive integer.
  `--every` with `--print-launchd` is refused. Ctrl-C ends the loop with exit 0 and no traceback.

All fixtures are synthetic and built in tmp_path; no real rsync, ssh, sleep or clock.
"""

from __future__ import annotations

import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from horsetrack.core import paths

UTC = timezone.utc
DEFAULT_REMOTE = "/var/lib/horsetrack"
START = datetime(2026, 11, 22, 0, 0, tzinfo=UTC)  # 09:00 in Tokyo
DAY = date(2026, 11, 22)
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    return tmp_path / "data"


class Clock:
    """now() is the fake time; sleep(s) moves it. `fail_sleep_at` raises on that sleep call."""

    def __init__(self, start: datetime = START, interrupt_at: int | None = None):
        self.t = start
        self.sleeps: list[float] = []
        self.interrupt_at = interrupt_at

    def now(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if len(self.sleeps) > 5000:
            raise RuntimeError("the loop slept 5000 times")
        if self.interrupt_at is not None and len(self.sleeps) >= self.interrupt_at:
            raise KeyboardInterrupt
        self.t += timedelta(seconds=seconds)


class Recorder:
    """A fake rsync runner: argv, the time of the round and whether the destination existed."""

    def __init__(self, clock: Clock, codes: dict[int, int] | None = None, default: int = 0):
        self.clock, self.codes, self.default = clock, codes or {}, default
        self.calls: list[list[str]] = []
        self.times: list[datetime] = []
        self.dest_existed: list[bool] = []

    def __call__(self, argv):
        n = len(self.calls)
        self.calls.append(list(argv))
        self.times.append(self.clock.now())
        self.dest_existed.append(Path(argv[-1]).is_dir())
        return self.codes.get(n, self.default)


def loop(clock: Clock, runner, **kwargs):
    from horsetrack.countries.japan.logger.pull_vm import pull_vm_loop

    options = dict(host="vm1", every=60, race_date=DAY, sleep=clock.sleep, now=clock.now)
    options.update(kwargs)
    return pull_vm_loop(runner=runner, **options)


def expected_argv(host, race_date, data_dir, remote=DEFAULT_REMOTE, dry_run=False):
    base = remote.rstrip("/")
    day = race_date.isoformat()
    argv = ["rsync", "-a", "--partial"]
    if dry_run:
        argv.append("--dry-run")
    return argv + [
        f"{host}:{base}/odds_snapshots/jra/{day}/",
        str(data_dir / "vm" / host / "odds_snapshots" / "jra" / day),
    ]


def norm(argv):
    """argv with the destination as a Path (a trailing slash or not is not pinned)."""
    return [*argv[:-1], str(Path(argv[-1]))]


def round_count_ok(n: int, every: int, window_s: int) -> bool:
    """No round after `until`: ceil(window/every) rounds, or one more at the stop time itself."""
    return n in {math.ceil(window_s / every), window_s // every + 1}


# ------------------------------------------------------------------ the library loop

LOOP_CASES = [
    # host, every, window (s), date, remote
    ("vm1", 60, 300, DAY, DEFAULT_REMOTE),
    ("vm1", 30, 600, DAY, DEFAULT_REMOTE),
    ("horse-trial", 90, 600, date(2026, 12, 6), "/srv/ht/"),
    ("trial_vm_2", 45, 400, date(2027, 1, 9), "/mnt/data/horsetrack"),
]


@pytest.mark.parametrize("host, every, window, race_date, remote", LOOP_CASES)
def test_the_loop_pulls_the_days_odds_folder_every_n_seconds(
    data_dir, host, every, window, race_date, remote
):
    clock = Clock()
    rec = Recorder(clock)
    until = START + timedelta(seconds=window)

    loop(clock, rec, host=host, every=every, race_date=race_date, until=until, remote_dir=remote)

    assert round_count_ok(len(rec.calls), every, window), len(rec.calls)
    assert rec.times[0] == START  # the first round is at once
    assert all(t <= until for t in rec.times)  # no round after the stop time
    gaps = {(b - a).total_seconds() for a, b in zip(rec.times, rec.times[1:], strict=False)}
    assert gaps == {float(every)}
    assert set(clock.sleeps) == {every}
    want = norm(expected_argv(host, race_date, data_dir, remote))
    assert [norm(a) for a in rec.calls] == [want] * len(rec.calls)
    assert all(rec.dest_existed)
    # only the VM's own day folder is created; the Mac's own logs are never touched
    assert {p.name for p in data_dir.iterdir()} == {"vm"}
    assert (data_dir / "vm" / host / "odds_snapshots" / "jra" / race_date.isoformat()).is_dir()
    for argv in rec.calls:
        assert not [a for a in argv if a.startswith("--delete") or a == "--remove-source-files"]


def test_the_loop_passes_dry_run_through(data_dir):
    clock = Clock()
    rec = Recorder(clock)
    loop(clock, rec, until=START + timedelta(seconds=180), dry_run=True)
    assert len(rec.calls) >= 3
    for argv in rec.calls:
        assert argv.count("--dry-run") == 1
        plain = norm([a for a in argv if a != "--dry-run"])
        assert plain == norm(expected_argv("vm1", DAY, data_dir))
        assert not [a for a in argv if a.startswith("--delete")]


def test_the_loop_runs_until_interrupted_without_an_until(data_dir):
    clock = Clock(interrupt_at=3)  # the third sleep is Ctrl-C
    rec = Recorder(clock)
    try:
        loop(clock, rec, every=20)
    except KeyboardInterrupt:  # the library may let it through; the CLI is what ends with exit 0
        pass
    assert len(rec.calls) == 3
    assert clock.sleeps == [20, 20, 20]
    assert [t - START for t in rec.times] == [
        timedelta(0),
        timedelta(seconds=20),
        timedelta(seconds=40),
    ]


def test_a_failing_round_prints_an_error_and_the_loop_goes_on(data_dir, capsys):
    clock = Clock()
    rec = Recorder(clock, codes={1: 23})  # the second round fails, as does nothing else
    until = START + timedelta(seconds=300)

    loop(clock, rec, until=until)

    assert round_count_ok(len(rec.calls), 60, 300)
    assert rec.times[2] == START + timedelta(seconds=120)  # the round after the failure ran
    err = capsys.readouterr().err.lower()
    assert "error" in err and "23" in err
    assert err.count("error") == 1  # one line for the one failed round


def test_every_round_failing_still_keeps_looping(data_dir, capsys):
    clock = Clock()
    rec = Recorder(clock, default=12)
    loop(clock, rec, every=100, until=START + timedelta(seconds=500))
    assert round_count_ok(len(rec.calls), 100, 500)
    assert capsys.readouterr().err.lower().count("error") == len(rec.calls)


def test_an_until_already_past_starts_no_loop(data_dir):
    clock = Clock()
    rec = Recorder(clock)
    loop(clock, rec, until=START - timedelta(minutes=5))
    assert len(rec.calls) == 0 and clock.sleeps == []


@pytest.mark.parametrize("host", ["1.2.3.4", "user@1.2.3.4", "vm1:/etc", ""])
def test_the_loop_refuses_an_address(data_dir, host):
    clock = Clock()
    rec = Recorder(clock)
    with pytest.raises(ValueError):
        loop(clock, rec, host=host, until=START + timedelta(minutes=3))
    assert rec.calls == [] and not (data_dir / "vm").exists()


# ------------------------------------------------------------------ the CLI


@pytest.fixture
def cli(monkeypatch):
    """Run `japan pull-vm` with a fake rsync, clock and sleep; returns (invoke, clock, recorder)."""
    from horsetrack.countries.japan.logger import pull_vm as mod

    clock = Clock()
    rec = Recorder(clock)
    monkeypatch.setattr(mod, "run_rsync", rec)
    monkeypatch.setattr(mod, "utc_now", clock.now, raising=False)
    monkeypatch.setattr(time, "sleep", clock.sleep)

    def invoke(*args):
        from horsetrack.cli import app

        return CliRunner().invoke(app, ["japan", "pull-vm", *args])

    return invoke, clock, rec


def test_cli_loops_until_the_stop_time_in_tokyo(cli, data_dir):
    invoke, clock, rec = cli

    args = ["--host", "vm1", "--every", "60", "--date", "2026-11-22", "--until", "09:04"]
    result = invoke(*args, "--remote-dir", "/srv/ht/")

    assert result.exit_code == 0, ANSI.sub("", result.output)
    until = datetime(2026, 11, 22, 9, 4, tzinfo=timezone(timedelta(hours=9)))
    assert round_count_ok(len(rec.calls), 60, 240)
    assert all(t <= until for t in rec.times)
    want = norm(expected_argv("vm1", DAY, data_dir, "/srv/ht/"))
    assert [norm(a) for a in rec.calls] == [want] * len(rec.calls)
    assert set(clock.sleeps) == {60}


def test_cli_defaults_the_date_to_today_in_tokyo_and_passes_dry_run(cli, data_dir):
    invoke, clock, rec = cli
    clock.t = datetime(2026, 11, 28, 16, 0, tzinfo=UTC)  # 01:00 on the 29th in Tokyo

    result = invoke("--host", "vm2", "--every", "30", "--until", "01:03", "--dry-run")

    assert result.exit_code == 0, ANSI.sub("", result.output)
    assert round_count_ok(len(rec.calls), 30, 180)
    want = norm(expected_argv("vm2", date(2026, 11, 29), data_dir, dry_run=True))
    assert [norm(a) for a in rec.calls] == [want] * len(rec.calls)


def test_cli_ctrl_c_ends_the_loop_with_exit_0_and_no_traceback(cli, data_dir):
    invoke, clock, rec = cli
    clock.interrupt_at = 3

    result = invoke("--host", "vm1", "--every", "60", "--date", "2026-11-22")

    out = ANSI.sub("", result.output)
    assert result.exit_code == 0, out
    assert len(rec.calls) == 3
    assert "Traceback" not in out and "Aborted" not in out
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert out.strip(), "a short message tells the loop was stopped"
    assert len(out.strip().splitlines()) <= 3


def test_cli_ctrl_c_during_a_pull_also_ends_with_exit_0(cli, monkeypatch):
    invoke, clock, rec = cli
    from horsetrack.countries.japan.logger import pull_vm as mod

    def interrupted(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(mod, "run_rsync", interrupted)

    result = invoke("--host", "vm1", "--every", "60", "--date", "2026-11-22")

    assert result.exit_code == 0, ANSI.sub("", result.output)
    assert "Traceback" not in result.output


def test_cli_a_failing_round_does_not_stop_the_loop(cli, monkeypatch):
    invoke, clock, rec = cli
    rec.codes = {0: 23}

    result = invoke("--host", "vm1", "--every", "60", "--date", "2026-11-22", "--until", "09:03")

    assert result.exit_code == 0, ANSI.sub("", result.output)
    assert round_count_ok(len(rec.calls), 60, 180)
    assert "error" in ANSI.sub("", result.output).lower()  # stderr is mixed into the output


@pytest.mark.parametrize("every", ["0", "-1", "-60", "abc", "1.5", ""])
def test_cli_every_must_be_a_positive_integer(cli, data_dir, every):
    invoke, clock, rec = cli
    result = invoke("--host", "vm1", "--every", every, "--date", "2026-11-22", "--until", "09:03")
    assert result.exit_code != 0
    assert "no such option" not in ANSI.sub("", result.output).lower()
    assert rec.calls == [] and clock.sleeps == []
    assert not (data_dir / "vm").exists()


def test_cli_every_needs_a_value(cli):
    invoke, clock, rec = cli
    result = invoke("--host", "vm1", "--every")
    assert result.exit_code != 0
    assert "no such option" not in ANSI.sub("", result.output).lower()
    assert rec.calls == []


def test_cli_every_with_print_launchd_is_refused(cli, data_dir):
    invoke, clock, rec = cli
    result = invoke("--host", "vm1", "--every", "60", "--print-launchd")
    assert result.exit_code != 0
    assert "no such option" not in ANSI.sub("", result.output).lower()
    assert "<plist" not in result.stdout
    assert rec.calls == [] and not (data_dir / "vm").exists()


@pytest.mark.parametrize("bad", ["2026-13-45", "yesterday", "22/11/2026", "2026-11-31"])
def test_cli_a_bad_date_is_refused(cli, data_dir, bad):
    invoke, clock, rec = cli
    result = invoke("--host", "vm1", "--every", "60", "--date", bad, "--until", "09:03")
    assert result.exit_code != 0
    assert "no such option" not in ANSI.sub("", result.output).lower()
    assert rec.calls == [] and not (data_dir / "vm").exists()


@pytest.mark.parametrize("bad", ["25:99", "9am", "09:3x", ""])
def test_cli_a_bad_stop_time_is_refused(cli, data_dir, bad):
    invoke, clock, rec = cli
    result = invoke("--host", "vm1", "--every", "60", "--date", "2026-11-22", "--until", bad)
    assert result.exit_code != 0
    assert "no such option" not in ANSI.sub("", result.output).lower()
    assert rec.calls == []


@pytest.mark.parametrize("host", ["1.2.3.4", "user@1.2.3.4"])
def test_cli_the_loop_refuses_an_address(cli, data_dir, host):
    invoke, clock, rec = cli
    result = invoke("--host", host, "--every", "60", "--date", "2026-11-22", "--until", "09:03")
    assert result.exit_code != 0
    assert "no such option" not in ANSI.sub("", result.output).lower()
    assert rec.calls == [] and not (data_dir / "vm").exists()
