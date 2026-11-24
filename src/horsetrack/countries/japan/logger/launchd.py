"""`horsetrack japan launchd --print | --install | --uninstall` (ticket #27): the LaunchAgent that
starts `horsetrack japan log-day --unattended` every 30 minutes on the Mac.

The job is `/usr/bin/caffeinate -i <venv bin>/horsetrack japan log-day --unattended` (no idle
sleep while it runs), with StartInterval 1800, RunAtLoad true and KeepAlive false: launchd never
relaunches it on exit, and log-day's lock and markers make each start that has nothing to do exit
at once. The working directory is the repo root; the job reads the data dir and the user agent from
the repo's .env, so nothing secret goes in the plist. Its stdout and stderr go under
logs_dir('japan').

Seam, looked up at call time: `launchctl(args) -> int` runs /bin/launchctl with `args`. The
LaunchAgents directory, the repo root and logs_dir() are resolved at call time.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

from horsetrack.core import env, paths

LABEL = "horsetrack.japan.logger"
START_INTERVAL_S = 1800
EXIT_ERROR = 1


def launchctl(args: list[str]) -> int:
    """Run /bin/launchctl with `args` (the arguments only); its exit code."""
    return subprocess.run(["/bin/launchctl", *args], check=False).returncode


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def job() -> dict[str, Any]:
    """The LaunchAgent's plist, as a dict."""
    logs = paths.logs_dir("japan")
    horsetrack = Path(sys.executable).parent / "horsetrack"
    return {
        "Label": LABEL,
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-i",
            str(horsetrack),
            "japan",
            "log-day",
            "--unattended",
        ],
        "StartInterval": START_INTERVAL_S,
        "RunAtLoad": True,
        "KeepAlive": False,
        "WorkingDirectory": str(env.repo_root()),
        "StandardOutPath": str(logs / "launchd.out.log"),
        "StandardErrorPath": str(logs / "launchd.err.log"),
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _install() -> int:
    data = plistlib.dumps(job())
    paths.logs_dir("japan").mkdir(parents=True, exist_ok=True)  # launchd creates no log folder
    path = plist_path()
    # A loaded job first, so a re-install loads the new plist; not loaded is fine.
    launchctl(["bootout", f"{_domain()}/{LABEL}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    code = launchctl(["bootstrap", _domain(), str(path)])
    if code != 0:
        typer.echo(f"ERROR launchctl bootstrap {_domain()} {path} exited {code}", err=True)
        return EXIT_ERROR
    typer.echo(f"installed {LABEL}: {path}")
    return 0


def _uninstall() -> int:
    code = launchctl(["bootout", f"{_domain()}/{LABEL}"])
    if code != 0:
        typer.echo(f"launchctl bootout exited {code} (the job was not loaded?)")
    path = plist_path()
    path.unlink(missing_ok=True)
    typer.echo(f"uninstalled {LABEL}: removed {path}")
    return 0


def launchd_command(
    show: bool = typer.Option(False, "--print", help="Print the plist; change nothing."),
    install: bool = typer.Option(
        False, "--install", help="Write the plist to ~/Library/LaunchAgents and load it."
    ),
    uninstall: bool = typer.Option(False, "--uninstall", help="Unload the job, remove its plist."),
) -> None:
    """The LaunchAgent that runs `log-day --unattended` every 30 minutes."""
    try:
        if [show, install, uninstall].count(True) != 1:
            raise ValueError("give exactly one of --print, --install, --uninstall")
        if show:
            typer.echo(plistlib.dumps(job()).decode("utf-8"), nl=False)
            return
        code = _install() if install else _uninstall()
    except Exception as exc:
        typer.echo(f"ERROR {' '.join(str(exc).split())[:300] or type(exc).__name__}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    if code:
        raise typer.Exit(code)
