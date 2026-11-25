"""`horsetrack japan systemd --print` (ticket #33): the systemd unit and timer that run the odds
logger unattended on the Linux VM, the twin of `horsetrack japan launchd --print` (#27).

Contract pinned here (the coder follows it):

* Module horsetrack.countries.japan.logger.systemd, its typer command registered on the japan
  group as `systemd`: `horsetrack japan systemd --print [--repo-dir DIR] [--data-dir DIR]
  [--user NAME]`. --print writes the two files to stdout and changes nothing: it writes no file
  and starts no process.
* Layout of the output: each file follows a header line of its own, exactly
  `# horsetrack-japan-logger.service` or `# horsetrack-japan-logger.timer` (trailing spaces
  aside); a file's text runs from after its header to the next header or the end. Each header
  appears once, and only blank lines come before the first. scripts/vm_setup.sh splits the output
  on these lines into /etc/systemd/system/<name>.
* The service, in [Service]: one ExecStart, exactly the argv
  `<repo>/.venv/bin/horsetrack japan log-day --unattended` (no `-`, `+`, `!` or `@` prefix);
  WorkingDirectory=<repo>; Environment=HORSETRACK_DATA_DIR=<data>, the only HORSETRACK_* variable
  it sets; EnvironmentFile=-<repo>/.env (optional; the owner writes the VM's .env, which holds
  the user agent); one Restart=no and no other Restart* setting; User=<name> with --user and no
  User= line without it. Neither [Unit] OnFailure= nor OnSuccess= names the service or the timer.
* Defaults: --repo-dir /opt/horsetrack (where vm_setup.sh clones) and --data-dir
  pull_vm.DEFAULT_REMOTE_DIR (/var/lib/horsetrack, where `horsetrack japan pull-vm` pulls from).
  They never come from the local environment (its data dir, its repo root, its .env): the unit
  is for the VM.
* The timer: its OnCalendar values (one or more) fire every 30 minutes, at two minutes of every
  hour 30 apart, every day, each value ending in the Asia/Tokyo time zone (systemd >= 245);
  OnCalendar is its only trigger (no OnBootSec, OnUnitActiveSec, OnUnitInactiveSec, ...);
  Persistent=true; it starts the service (no Unit=, or Unit=horsetrack-japan-logger.service);
  [Install] WantedBy=timers.target, so `systemctl enable --now` works.
* Nothing secret or identifying in either file: no user agent, token, URL, IP address,
  user@host, home path, local data dir or local repo path.

The files are read as systemd reads them: `#` and `;` comment lines, `[Section]` headers,
`Key=Value` with spaces allowed around `=`, a trailing backslash continuing a line, and an empty
assignment resetting a list setting (OnCalendar=, Environment=, EnvironmentFile=, ExecStart=).
Every other non-blank line is an error.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import os
import pwd
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from horsetrack.core import paths
from horsetrack.countries.japan.logger.pull_vm import DEFAULT_REMOTE_DIR

REPO = Path(__file__).resolve().parents[3]
SERVICE = "horsetrack-japan-logger.service"
TIMER = "horsetrack-japan-logger.timer"
HEADERS = {f"# {SERVICE}": SERVICE, f"# {TIMER}": TIMER}
DEFAULT_REPO_DIR = "/opt/horsetrack"
TOKYO = "Asia/Tokyo"
DAY_S = 24 * 3600

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def plain(text: str) -> str:
    """Strip ANSI escapes: Typer forces a terminal under GITHUB_ACTIONS / FORCE_COLOR."""
    return ANSI.sub("", text)


# ------------------------------------------------------------------ reading the output


def systemd_cli(*args: str):
    from horsetrack.cli import app

    return CliRunner().invoke(app, ["japan", "systemd", *args])


def printed(*args: str) -> tuple[str, dict[str, str]]:
    """`systemd --print *args`: the whole stdout, and each file's text by its name."""
    result = systemd_cli("--print", *args)
    assert result.exit_code == 0, plain(result.output)
    text = result.stdout
    files: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        name = HEADERS.get(line.rstrip())
        if name is not None:
            assert name not in files, f"the header of {name} is printed twice:\n{text}"
            files[name] = []
            current = name
            continue
        if current is None:
            assert not line.strip(), f"text before the first header: {line!r}\n{text}"
            continue
        files[current].append(line)
    assert set(files) == {SERVICE, TIMER}, f"headers found: {sorted(files)}\n{text}"
    return text, {name: "\n".join(lines) for name, lines in files.items()}


def entries(text: str) -> list[tuple[str, str, str]]:
    """(section, key, value) of every assignment, in order, as systemd reads the file."""
    logical: list[str] = []
    pending: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line[:1] in ("#", ";") or (not line and pending is None):
            continue
        if pending is not None:
            line = f"{pending} {line}".strip()
            pending = None
        if line.endswith("\\"):
            pending = line[:-1].rstrip()
            continue
        logical.append(line)
    if pending:
        logical.append(pending)
    out: list[tuple[str, str, str]] = []
    section = None
    for line in logical:
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        key, sep, value = line.partition("=")
        assert sep, f"not a [Section] or Key=Value line: {line!r}\n{text}"
        assert section is not None, f"{line!r} comes before any [Section]\n{text}"
        out.append((section, key.strip(), value.strip()))
    return out


def raw_values(es, section: str, key: str) -> list[str]:
    """Every assignment of `key` in `section`, empty ones included."""
    return [v for s, k, v in es if s == section and k == key]


def values(es, section: str, key: str) -> list[str]:
    """The list setting's values after systemd's rule: an empty assignment resets the list."""
    out: list[str] = []
    for v in raw_values(es, section, key):
        out = [] if v == "" else [*out, v]
    return out


def last(es, section: str, key: str) -> str | None:
    """A single-valued setting: the last assignment wins."""
    found = raw_values(es, section, key)
    return found[-1] if found else None


def environment(es) -> dict[str, str]:
    """The [Service] Environment= assignments, as systemd parses them (quotes allowed)."""
    env: dict[str, str] = {}
    for value in raw_values(es, "Service", "Environment"):
        if value == "":
            env = {}
            continue
        for word in shlex.split(value):
            name, sep, val = word.partition("=")
            assert sep, f"Environment= word without '=': {word!r}"
            env[name] = val
    return env


# ------------------------------------------------------------------ OnCalendar, read by hand


def _field(text: str, top: int, spec: str) -> set[int]:
    """One calendar field: `*`, `a`, `a,b`, `a..b`, `a/step`, `*/step`, `a..b/step`."""
    out: set[int] = set()
    for item in text.split(","):
        base, slash, step = item.partition("/")
        if base == "*":
            lo, hi = 0, top
        elif ".." in base:
            a, b = base.split("..", 1)
            lo, hi = int(a), int(b)
        else:
            lo = int(base)
            hi = top if slash else lo
        stride = int(step) if slash else 1
        assert stride > 0 and 0 <= lo <= hi <= top, f"{spec!r}: field {item!r} out of range"
        out.update(range(lo, hi + 1, stride))
    return out


def calendar_seconds(spec: str) -> set[int]:
    """The seconds of the day (in the value's own zone) at which one OnCalendar value fires. The
    value must fire every day and end in the Asia/Tokyo zone; anything else is an assertion."""
    parts = spec.split()
    assert len(parts) >= 2 and parts[-1] == TOKYO, (
        f"OnCalendar={spec!r} is not in Tokyo time: it must end in {TOKYO}"
    )
    parts = parts[:-1]
    if len(parts) == 2:
        assert parts[0] == "*-*-*", f"OnCalendar={spec!r}: not every day"
        parts = parts[1:]
    assert len(parts) == 1, f"OnCalendar={spec!r}: expected [*-*-*] H:M[:S] {TOKYO}"
    fields = parts[0].split(":")
    assert len(fields) in (2, 3), f"OnCalendar={spec!r}: expected H:M[:S]"
    hours = _field(fields[0], 23, spec)
    minutes = _field(fields[1], 59, spec)
    seconds = _field(fields[2], 59, spec) if len(fields) == 3 else {0}
    return {h * 3600 + m * 60 + s for h in hours for m in minutes for s in seconds}


def fires_every_30_minutes(specs: list[str]) -> bool:
    """True when the union of the values fires 48 times a day, 30 minutes apart (wrapping)."""
    times = sorted(set().union(*(calendar_seconds(s) for s in specs)))
    if len(times) != 48:
        return False
    gaps = [b - a for a, b in zip(times[:-1], times[1:], strict=True)]
    gaps.append(times[0] + DAY_S - times[-1])
    return all(g == 1800 for g in gaps)


# ------------------------------------------------------------------ addresses, hosts, secrets


def addresses(text: str) -> list[str]:
    """Every IPv4 or IPv6 address in `text` (times such as 00:30:00 are not addresses)."""
    found = []
    for m in re.finditer(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?![\w.])", text):
        try:
            ipaddress.ip_address(m.group())
        except ValueError:
            continue
        found.append(m.group())
    for m in re.finditer(r"(?<![\w:.])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![\w:])", text):
        try:
            ipaddress.ip_address(m.group())
        except ValueError:
            continue
        found.append(m.group())
    return found


USER_AT_HOST = re.compile(r"[A-Za-z0-9._%+-]+[@][A-Za-z0-9][A-Za-z0-9.-]*")
URL = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://\S+")


def public_tree():
    """scripts/check_public_tree.py loaded as a module (scripts/ is not a package)."""
    name = "check_public_tree"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# ------------------------------------------------------------------ fixtures


def _tree(root: Path) -> list[tuple[str, int]]:
    return sorted(
        (str(p.relative_to(root)), p.stat().st_mtime_ns if p.is_file() else 0)
        for p in root.rglob("*")
    )


@pytest.fixture
def local(tmp_path, monkeypatch):
    """A local environment the unit must ignore: a data dir and a user agent in the process
    environment, a repo root whose .env holds another agent and a token, HOME under tmp_path. No
    process can start. After the test, nothing under tmp_path has been written."""
    data, root, home = tmp_path / "local-data", tmp_path / "local-repo", tmp_path / "home"
    for d in (data, root, home):
        d.mkdir()
    agent = "horsetrack-tests/1.0 (+synthetic-contact-local)"
    dotenv_agent, token = "dotenv-odds-research/2.3", "synthetic-token-0123456789abcdef"
    (root / ".env").write_text(
        f"HORSETRACK_USER_AGENT={dotenv_agent}\nHORSETRACK_SERVICE_TOKEN={token}\n"
        f"{paths.DATA_DIR_VAR}={data}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    monkeypatch.setenv("HORSETRACK_USER_AGENT", agent)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)

    def no_process(*args, **kwargs):
        raise AssertionError(f"a process was started: {args!r}")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    for name in ("system", "posix_spawn", "posix_spawnp", "fork", "execv", "execve", "execvp"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, no_process)

    before = _tree(tmp_path)
    yield {
        "secrets": (agent, dotenv_agent, token),
        "paths": (str(tmp_path), str(data), str(root), str(home)),
    }
    assert _tree(tmp_path) == before, "--print wrote a file"


# ------------------------------------------------------------------ the unit


@pytest.mark.parametrize(
    "args, repo, data, user",
    [
        pytest.param([], DEFAULT_REPO_DIR, DEFAULT_REMOTE_DIR, None, id="defaults"),
        pytest.param(
            ["--repo-dir", "/srv/odds-logger", "--data-dir", "/data/horsetrack-b"]
            + ["--user", "logger"],
            "/srv/odds-logger",
            "/data/horsetrack-b",
            "logger",
            id="all-flags",
        ),
        pytest.param(
            ["--user", "svc-ht", "--data-dir", "/mnt/block/ht"],
            DEFAULT_REPO_DIR,
            "/mnt/block/ht",
            "svc-ht",
            id="data-dir-and-user",
        ),
        pytest.param(
            ["--repo-dir", "/opt/ht-trial"], "/opt/ht-trial", DEFAULT_REMOTE_DIR, None, id="repo"
        ),
    ],
)
def test_unit_runs_log_day_unattended_with_the_data_dir(local, args, repo, data, user):
    _, files = printed(*args)
    es = entries(files[SERVICE])

    starts = raw_values(es, "Service", "ExecStart")
    assert len(starts) == 1, starts
    assert shlex.split(starts[0]) == [
        f"{repo}/.venv/bin/horsetrack",
        "japan",
        "log-day",
        "--unattended",
    ], starts[0]
    assert raw_values(es, "Service", "WorkingDirectory") == [repo]

    env = environment(es)
    assert env.get(paths.DATA_DIR_VAR) == data, env
    assert [k for k in env if k.startswith("HORSETRACK")] == [paths.DATA_DIR_VAR], env
    env_files = values(es, "Service", "EnvironmentFile")
    assert f"-{repo}/.env" in env_files, f"EnvironmentFile={env_files}"

    if user is None:
        assert raw_values(es, "Service", "User") == [], "User= without --user"
    else:
        assert last(es, "Service", "User") == user

    # The local environment never leaks in: its data dir, repo root and home are not the VM's.
    for path in local["paths"]:
        assert path not in files[SERVICE], f"{path} (local) is in the unit"


@pytest.mark.parametrize(
    "args",
    [
        pytest.param([], id="defaults"),
        pytest.param(["--user", "logger"], id="user"),
        pytest.param(
            ["--repo-dir", "/srv/odds-b", "--data-dir", "/srv/data-b", "--user", "svc-b"],
            id="all-flags",
        ),
    ],
)
def test_unit_never_restarts(local, args):
    _, files = printed(*args)
    service, timer = entries(files[SERVICE]), entries(files[TIMER])

    restarts = [(s, v) for s, k, v in service if k == "Restart"]
    assert restarts == [("Service", "no")], restarts
    others = sorted({k for _, k, _ in service if k.startswith("Restart") and k != "Restart"})
    assert others == [], f"restart settings besides Restart=no: {others}"

    # Nothing starts it again when it exits: no unit hook names it, no timer re-arms on its exit.
    for key in ("OnFailure", "OnSuccess"):
        hooked = " ".join(raw_values(service, "Unit", key))
        assert SERVICE not in hooked and TIMER not in hooked, f"{key}={hooked}"
    rearm = [k for _, k, _ in timer if k in ("OnUnitInactiveSec", "OnUnitActiveSec")]
    assert rearm == [], f"the timer re-arms on the service: {rearm}"


# ------------------------------------------------------------------ the timer


@pytest.mark.parametrize(
    "args",
    [
        pytest.param([], id="defaults"),
        pytest.param(
            ["--repo-dir", "/srv/odds-b", "--data-dir", "/srv/data-b", "--user", "svc-b"],
            id="all-flags",
        ),
    ],
)
def test_timer_fires_every_30_minutes_in_tokyo_time(local, args):
    _, files = printed(*args)
    es = entries(files[TIMER])

    calendar = values(es, "Timer", "OnCalendar")
    assert calendar, f"no OnCalendar in the timer:\n{files[TIMER]}"
    assert fires_every_30_minutes(calendar), (
        f"OnCalendar={calendar} does not fire every 30 minutes, every hour of every day"
    )
    triggers = sorted({k for s, k, _ in es if s == "Timer" and k.startswith("On")})
    assert triggers == ["OnCalendar"], f"triggers besides the calendar: {triggers}"

    persistent = last(es, "Timer", "Persistent")
    assert persistent is not None and persistent.lower() in ("true", "yes", "on", "1"), persistent
    assert last(es, "Timer", "Unit") in (None, SERVICE), last(es, "Timer", "Unit")
    wanted = " ".join(values(es, "Install", "WantedBy")).split()
    assert "timers.target" in wanted, f"[Install] WantedBy={wanted}: enable --now would fail"


# ------------------------------------------------------------------ nothing secret


@pytest.mark.parametrize(
    "args",
    [
        pytest.param([], id="defaults"),
        pytest.param(["--user", "svc-ht", "--data-dir", "/var/lib/ht-b"], id="flags"),
    ],
)
def test_unit_and_timer_hold_no_secret_address_or_host_name(local, args):
    text, _ = printed(*args)

    for secret in local["secrets"]:
        assert secret not in text, f"{secret!r} is in the printed files"
    real_home = pwd.getpwuid(os.getuid()).pw_dir
    for path in (*local["paths"], real_home, str(REPO)):
        assert path not in text, f"{path} is in the printed files"
    assert public_tree().check(["systemd-print"], lambda _: text.encode()) == []
    assert addresses(text) == [], addresses(text)
    assert USER_AT_HOST.findall(text) == [], USER_AT_HOST.findall(text)
    assert URL.findall(text) == [], URL.findall(text)
