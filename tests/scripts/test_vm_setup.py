"""scripts/vm_setup.sh (ticket #33): set the Japan odds logger up on the Linux VM.

    scripts/vm_setup.sh --repo-url URL --tag TAG [--dry-run]

Contract pinned here (the coder follows it):

* A bash script. The tests run it as `bash scripts/vm_setup.sh ...` with the `bash` on the test
  process's PATH (macOS gives bash 3.2, CI gives 5.x), so it keeps to bash 3.2: no associative
  arrays, `mapfile` or `${x,,}`, and no `"${arr[@]}"` of an empty array under `set -u`.
* It calls every tool by name (`sudo`, `mkdir`, `systemctl`, ...), never by an absolute path such
  as /usr/bin/sudo, so a PATH of stubs intercepts all of them.
* Arguments come first, before any other check (a root check included): a missing or empty
  --tag, or --tag with no value, exits non-zero with a message naming `--tag`, prints no plan and
  runs nothing, with or without --dry-run. The arguments may come in any order.
* The real run: install uv if it is missing (uv's own installer, which picks the aarch64 build on
  the Oracle Ampere VM), clone or fetch --repo-url at --tag into /opt/horsetrack, run
  `uv sync --frozen --no-dev` there, write /etc/systemd/system/horsetrack-japan-logger.service
  and /etc/systemd/system/horsetrack-japan-logger.timer from
  `<that checkout's venv>/horsetrack japan systemd --print`, `systemctl daemon-reload`, then
  `systemctl enable --now horsetrack-japan-logger.timer`. It never writes a .env: the owner
  writes the VM's .env on the VM.
* --dry-run prints every command of the real run on stdout, one per line, with the values filled
  in (the URL, the tag, /opt/horsetrack, both unit paths), in the order they would run, and exits
  0. It runs none of them and creates no file or directory anywhere: the only commands it may
  run are probes that change nothing (`command -v`, `[ -d ... ]`, `uname`, `arch`,
  `dpkg --print-architecture`, `id`), and nothing it runs prints "command not found" or "No such
  file or directory". It works on any host (the tests run it on macOS and Linux); here `uname`
  reports Linux on x86_64 or aarch64. A `uv` on PATH counts as installed; with none on PATH (and
  none under $HOME) the plan installs it.
* Nothing secret or identifying in the script: no IP address, user@host, home path, token or
  email, and no URL but uv's installer's (astral.sh, github.com/astral-sh/...): the repo URL comes
  only from --repo-url.

Each run gets a PATH of stub executables under tmp_path, then /usr/bin:/bin. Every tool that
changes state (git, uv, curl, sh, sudo, systemctl, install, tee, mkdir, cp, ...) records its call
and exits 97; the probes answer and record. HOME and TMPDIR are under tmp_path, the environment
is built from scratch (no HORSETRACK_* variable passes through), and /opt/horsetrack, the two unit
paths, /var/lib/horsetrack, /usr/local/bin/uv and the repo's own .env are checked unchanged.
"""

from __future__ import annotations

import importlib.util
import ipaddress
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "vm_setup.sh"
BASH = shutil.which("bash")

INSTALL_DIR = "/opt/horsetrack"
UNIT_DIR = "/etc/systemd/system"
SERVICE = "horsetrack-japan-logger.service"
TIMER = "horsetrack-japan-logger.timer"

# Tools that change state: a call to any of them during a dry run fails the test. `uv` joins
# them when a test puts uv on PATH; otherwise no uv is reachable at all.
MUTATORS = (
    "git",
    "uvx",
    "curl",
    "wget",
    "sh",
    "dash",
    "sudo",
    "su",
    "doas",
    "systemctl",
    "systemd-analyze",
    "loginctl",
    "timedatectl",
    "install",
    "tee",
    "mkdir",
    "cp",
    "mv",
    "rm",
    "rmdir",
    "ln",
    "touch",
    "chown",
    "chmod",
    "chgrp",
    "mktemp",
    "dd",
    "truncate",
    "tar",
    "unzip",
    "csplit",
    "useradd",
    "adduser",
    "usermod",
    "groupadd",
    "apt",
    "apt-get",
    "snap",
    "pip",
    "pip3",
    "pipx",
    "python",
    "python3",
    "horsetrack",
    "rsync",
    "scp",
    "ssh",
)
WATCHED = (
    Path(INSTALL_DIR),
    Path(UNIT_DIR) / SERVICE,
    Path(UNIT_DIR) / TIMER,
    Path("/var/lib/horsetrack"),
    Path("/usr/local/bin/uv"),
    REPO / ".env",
)
DEB_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}
OTHER_ARCH = {
    "x86_64": re.compile(r"aarch64|arm64", re.I),
    "aarch64": re.compile(r"x86[_-]?64|amd64|i[3-6]86", re.I),
}

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def plain(text: str) -> str:
    return ANSI.sub("", text)


# ------------------------------------------------------------------ running the script


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _state(path: Path):
    try:
        st = path.lstat()
    except FileNotFoundError:
        return None
    except PermissionError:
        return "unreadable"
    return (st.st_mode, st.st_size, st.st_mtime_ns)


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


TOOL_BY_PATH = re.compile(
    r"(?<![\w.$/-])/(?:usr/)?(?:local/)?s?bin/"
    r"(sudo|su|doas|systemctl|mkdir|install|tee|cp|mv|rm|ln|chown|chmod|git|curl|wget|sh|dash)"
    r"(?![\w.-])"
)


def script_text() -> str:
    if not SCRIPT.is_file():
        pytest.fail(f"{SCRIPT.relative_to(REPO)} does not exist yet", pytrace=False)
    return SCRIPT.read_text(encoding="utf-8")


@dataclass
class Run:
    code: int
    stdout: str
    stderr: str
    forbidden: list[str]  # calls to the failing stubs, one line each: "<tool> <args>"
    changed: list[str]  # paths created, removed or modified
    scratch: Path  # this run's HOME, TMPDIR and cwd live under it

    @property
    def lines(self) -> list[str]:
        return [ln.rstrip() for ln in plain(self.stdout).splitlines() if ln.strip()]


def run_setup(tmp_path: Path, *args: str, arch: str = "x86_64", uv: bool = False) -> Run:
    """`bash scripts/vm_setup.sh *args` on a PATH of stubs; `uv` puts a (failing) uv on it."""
    text = script_text()
    by_path = sorted({m.group(1) for m in TOOL_BY_PATH.finditer(text)})
    assert by_path == [], f"the script calls tools by absolute path, past the stubs: {by_path}"
    assert BASH, "no bash on PATH"
    if not uv:
        real = [p for p in ("/usr/bin/uv", "/bin/uv") if Path(p).exists()]
        if real:
            pytest.skip(f"a real uv at {real[0]} is on the test PATH; cannot test uv missing")

    scratch = Path(tempfile.mkdtemp(dir=tmp_path))
    stubs, home, tmp, cwd = (scratch / d for d in ("stubs", "home", "tmp", "cwd"))
    for d in (stubs, home, tmp, cwd):
        d.mkdir()
    forbidden_log, probe_log = scratch / "forbidden.log", scratch / "probes.log"
    forbidden_log.touch()
    probe_log.touch()

    fail = 'printf "%s %s\\n" "{name}" "$*" >> "{log}"\nexit 97\n'
    for name in (*MUTATORS, "uv") if uv else MUTATORS:
        _stub(stubs / name, fail.format(name=name, log=forbidden_log))
    record = f'printf "%s %s\\n" "$(basename "$0")" "$*" >> "{probe_log}"\n'
    _stub(
        stubs / "uname",
        record
        + f"""case "$*" in
  -m|-p|-i) echo {arch} ;;
  -r) echo 6.8.0-synthetic ;;
  -n) echo synthetic-vm ;;
  -o) echo GNU/Linux ;;
  -sm|"-s -m") echo "Linux {arch}" ;;
  -a) echo "Linux synthetic-vm 6.8.0-synthetic #1 SMP {arch} GNU/Linux" ;;
  *) echo Linux ;;
esac
""",
    )
    _stub(stubs / "arch", record + f"echo {arch}\n")
    _stub(
        stubs / "dpkg",
        record
        + f"""if [ "$*" = "--print-architecture" ]; then echo {DEB_ARCH[arch]}; exit 0; fi
{fail.format(name="dpkg", log=forbidden_log)}""",
    )

    env = {
        "PATH": f"{stubs}:/usr/bin:/bin",
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "USER": "vmuser",
        "LOGNAME": "vmuser",
        "LANG": "C",
        "LC_ALL": "C",
    }
    before = {p: _state(p) for p in WATCHED}
    proc = subprocess.run(
        [BASH, str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    changed = [str(p) for p in WATCHED if _state(p) != before[p]]
    for d in (home, tmp, cwd):
        changed += [str(d / rel) for rel in _tree(d)]
    forbidden = [ln for ln in forbidden_log.read_text().splitlines() if ln.strip()]
    return Run(proc.returncode, proc.stdout, proc.stderr, forbidden, changed, scratch)


# ------------------------------------------------------------------ reading the plan

TOKEN = re.compile(r"[^\s'\"`|;&()<>]+")
DIR = re.compile(r"(?<![\w./-])" + re.escape(INSTALL_DIR) + r"(?![\w.-])")
UV_INSTALL = re.compile(
    r"astral\.sh/uv|/astral-sh/uv/releases|\binstall\s+(?:-\S+\s+)*(?:astral-)?uv(?![\w.-])"
)


def names(line: str) -> set[str]:
    """The words of a planned line, each path cut to its last part."""
    return {t.rstrip("/").rsplit("/", 1)[-1] for t in TOKEN.findall(line)}


def first(lines: list[str], what: str, pred) -> int:
    for i, line in enumerate(lines):
        if pred(line):
            return i
    raise AssertionError(f"the plan has no {what}:\n" + "\n".join(lines))


def in_install_dir(lines: list[str], i: int) -> bool:
    """Line i names /opt/horsetrack, or an earlier planned `cd /opt/horsetrack` does."""
    if DIR.search(lines[i]):
        return True
    return any("cd" in names(ln) and DIR.search(ln) for ln in lines[:i])


def word(value: str) -> re.Pattern:
    return re.compile(r"(?<![\w.-])" + re.escape(value) + r"(?![\w.-])")


# ------------------------------------------------------------------ the dry run


@pytest.mark.parametrize(
    "arch, uv, url, tag, order",
    [
        pytest.param(
            "x86_64",
            False,
            "https://example.invalid/owner-a/odds-logger.git",
            "v0.4.0",
            "url-first",
            id="x86_64-without-uv",
        ),
        pytest.param(
            "aarch64",
            False,
            "https://example.invalid/team-b/horsetrack.git",
            "trial-2026-11-29",
            "dry-run-first",
            id="aarch64-without-uv",
        ),
        pytest.param(
            "aarch64",
            True,
            "https://example.invalid/c/horse-track-c",
            "v1.2.3-rc1",
            "tag-first",
            id="aarch64-with-uv",
        ),
    ],
)
def test_dry_run_prints_the_planned_commands_and_runs_none(tmp_path, arch, uv, url, tag, order):
    args = {
        "url-first": ["--repo-url", url, "--tag", tag, "--dry-run"],
        "dry-run-first": ["--dry-run", "--repo-url", url, "--tag", tag],
        "tag-first": ["--tag", tag, "--dry-run", "--repo-url", url],
    }[order]
    run = run_setup(tmp_path, *args, arch=arch, uv=uv)

    # Runs none: no failing stub was called, nothing appeared or changed, and exit 0.
    assert run.forbidden == [], f"the dry run ran: {run.forbidden}\n{run.stdout}"
    assert run.changed == [], f"the dry run changed: {run.changed}"
    assert run.code == 0, f"exit {run.code}\n{run.stdout}\n{run.stderr}"
    for complaint in ("command not found", "No such file or directory"):
        assert complaint not in run.stderr, run.stderr

    lines = run.lines
    gits = [i for i, ln in enumerate(lines) if "git" in names(ln)]
    assert gits, "the plan runs no git:\n" + "\n".join(lines)
    fetch = first(lines, f"git command with {url}", lambda ln: "git" in names(ln) and url in ln)
    at_tag = first(
        lines, f"git command at {tag}", lambda ln: "git" in names(ln) and word(tag).search(ln)
    )
    assert any(DIR.search(lines[i]) for i in gits), f"no git command into {INSTALL_DIR}"

    sync = first(
        lines,
        "uv sync --frozen --no-dev",
        lambda ln: {"uv", "sync", "--frozen", "--no-dev"} <= names(ln),
    )
    assert in_install_dir(lines, sync), f"uv sync does not run in {INSTALL_DIR}: {lines[sync]}"
    assert fetch < sync and at_tag < sync, "uv sync is planned before the checkout"

    installs = [i for i, ln in enumerate(lines) if UV_INSTALL.search(ln)]
    if uv:
        assert installs == [], f"uv is on PATH, yet the plan installs it: {lines[installs[0]]}"
    else:
        assert installs, "uv is missing, yet the plan does not install it:\n" + "\n".join(lines)
        assert installs[0] < sync, "uv sync is planned before uv is installed"

    show = first(
        lines,
        "`horsetrack japan systemd --print`",
        lambda ln: {"horsetrack", "japan", "systemd", "--print"} <= names(ln),
    )
    assert in_install_dir(lines, show), f"the unit text is not from {INSTALL_DIR}: {lines[show]}"
    assert sync < show, "the unit is printed before the venv exists"
    written = [
        first(lines, f"write of {UNIT_DIR}/{name}", lambda ln, n=name: f"{UNIT_DIR}/{n}" in ln)
        for name in (SERVICE, TIMER)
    ]
    reload = first(
        lines, "systemctl daemon-reload", lambda ln: {"systemctl", "daemon-reload"} <= names(ln)
    )
    enable = first(
        lines,
        f"systemctl enable --now {TIMER}",
        lambda ln: {"systemctl", "enable", "--now", TIMER} <= names(ln),
    )
    assert show < reload and max(written) < reload < enable, (
        "the plan must write the units, then daemon-reload, then enable --now the timer:\n"
        + "\n".join(lines)
    )

    # aarch64 (the Oracle Ampere VM) gets nothing built for x86, and the reverse.
    plan = run.stdout.replace(str(run.scratch), "<scratch>")
    stray = OTHER_ARCH[arch].findall(plan)
    assert stray == [], f"an {arch} plan names another architecture: {stray}\n{plan}"


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--repo-url", "https://example.invalid/a/r.git", "--dry-run"], id="dry"),
        pytest.param(["--repo-url", "https://example.invalid/b/r.git"], id="real-run"),
        pytest.param(
            ["--repo-url", "https://example.invalid/c/r.git", "--dry-run", "--tag"],
            id="no-value",
        ),
        pytest.param(
            ["--tag", "", "--repo-url", "https://example.invalid/d/r.git", "--dry-run"],
            id="empty",
        ),
    ],
)
def test_missing_tag_is_an_error(tmp_path, args):
    run = run_setup(tmp_path, *args, arch="aarch64")

    assert run.code != 0, f"exit 0 without a tag\n{run.stdout}"
    assert "--tag" in plain(run.stdout + run.stderr), (run.stdout, run.stderr)
    assert run.forbidden == [], f"ran without a tag: {run.forbidden}"
    assert run.changed == [], f"changed without a tag: {run.changed}"
    enables = [ln for ln in run.lines if {"systemctl", "enable"} <= names(ln)]
    assert enables == [], f"printed a plan without a tag: {enables}"


# ------------------------------------------------------------------ nothing secret


def public_tree():
    """scripts/check_public_tree.py loaded as a module (scripts/ is not a package)."""
    name = "check_public_tree"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def addresses(text: str) -> list[str]:
    """Every IPv4 or IPv6 address in `text`."""
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
URL = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://([^/\s\"'<>)}]+)([^\s\"'<>)}]*)")
ENV_FILE = r"[\"']?[^\s\"';|&<>()]*\.env[\"']?(?![\w.-])"
ENV_WRITES = (
    re.compile(r">>?\s*" + ENV_FILE),
    re.compile(r"\btee\b(?:\s+-\S+)*\s+" + ENV_FILE),
)
COPIERS = {"cp", "mv", "install", "ln", "touch", "tee", "dd"}


def allowed_url(host: str, path: str) -> bool:
    """uv's installer only: astral.sh, or uv's releases on github.com."""
    host = host.lower()
    return (
        host == "astral.sh"
        or host.endswith(".astral.sh")
        or (host == "github.com" and path.startswith("/astral-sh/"))
    )


def env_writes(lines: list[str], *, copies: bool) -> list[str]:
    hits = [ln for ln in lines if any(p.search(ln) for p in ENV_WRITES)]
    if copies:
        for ln in lines:
            tokens = TOKEN.findall(ln)
            if tokens and COPIERS & names(ln) and tokens[-1].rsplit("/", 1)[-1] == ".env":
                hits.append(ln)
    return hits


def test_the_script_holds_no_secret_address_or_host_name(tmp_path):
    text = script_text()

    assert public_tree().check(["scripts/vm_setup.sh"], lambda _: text.encode()) == []
    assert addresses(text) == [], addresses(text)
    assert USER_AT_HOST.findall(text) == [], USER_AT_HOST.findall(text)
    urls = [m.group(0) for m in URL.finditer(text) if not allowed_url(m.group(1), m.group(2))]
    assert urls == [], f"URLs besides uv's installer (the repo URL is --repo-url's): {urls}"
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert env_writes(code, copies=False) == [], "the script writes a .env"

    run = run_setup(
        tmp_path, "--repo-url", "https://example.invalid/e/r.git", "--tag", "v9.9.9", "--dry-run"
    )
    assert run.code == 0, f"exit {run.code}\n{run.stdout}\n{run.stderr}"
    assert env_writes(run.lines, copies=True) == [], "the plan writes a .env"
    assert run.forbidden == [] and run.changed == []
