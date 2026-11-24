"""Pull the VM's logs back to the Mac (ticket #34).

The VM runs the odds logger for the trial; its logs are compared with the Mac's own, never merged
into them. `pull_vm` copies `odds_snapshots/`, `raw/jra/official_odds/` and `logs/` from the VM into
`$HORSETRACK_DATA_DIR/vm/<ALIAS>/` with `rsync -a --partial` over plain ssh: never `--delete`, never
a destination outside that VM's folder.

The VM is named by an `~/.ssh/config` alias only. The key, user, port and address stay in the ssh
config, so a host that looks like an address (`1.2.3.4`, `user@host`, IPv6, `host:path`) is refused.
`--print-launchd` prints a LaunchAgent that runs the pull daily at 22:00 local. launchd jobs do
not inherit the shell environment, so the plist pins `HORSETRACK_DATA_DIR` and sends the job's
stdout and stderr to `<data_dir>/vm/<ALIAS>/pull-vm.{out,err}.log`, so a failed night leaves a
trace. launchd does not create that folder: run the pull by hand once first, which creates
`vm/<ALIAS>/`. Printing creates nothing, and fails if the data dir is not set.
"""

from __future__ import annotations

import plistlib
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import typer

from horsetrack.core import paths

DEFAULT_REMOTE_DIR = "/var/lib/horsetrack"
SUBTREES = ("odds_snapshots", "raw/jra/official_odds", "logs")

# An alias is a plain word: letters, digits, `_` and `-`, not starting with `-`. No dots, so a
# dotted-quad address cannot pass; no `@`, `:`, `/` or whitespace either.
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

Runner = Callable[[list[str]], int]


def run_rsync(argv: list[str]) -> int:
    """Run rsync (no shell) and return its exit code."""
    try:
        return subprocess.run(argv, check=False).returncode
    except FileNotFoundError:
        return 127


def check_host(host: str) -> str:
    """Return the host if it is an ssh alias; raise ValueError for an address or odd characters."""
    if not _ALIAS.fullmatch(host):
        raise ValueError(
            f"host {host!r} is not an ssh alias: use a name from ~/.ssh/config "
            "(letters, digits, '_' and '-'), not an address or user@host"
        )
    return host


def pull_vm(
    host: str,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    dry_run: bool = False,
    runner: Runner | None = None,
) -> int:
    """Pull each subtree of the VM into `<data_dir>/vm/<host>/`; 0 only if every rsync returned 0.

    Raises ValueError for a host that is not an alias, before anything runs or is created."""
    check_host(host)
    run = runner if runner is not None else run_rsync
    remote = remote_dir.rstrip("/")
    vm_root = paths.data_dir() / "vm" / host
    codes = []
    for subtree in SUBTREES:
        dest = vm_root / subtree
        argv = ["rsync", "-a", "--partial"]
        if dry_run:
            argv.append("--dry-run")
        else:
            dest.mkdir(parents=True, exist_ok=True)  # rsync only creates the last component
        argv += [f"{host}:{remote}/{subtree}/", str(dest)]
        codes.append(run(argv))
    return next((code for code in codes if code != 0), 0)


def _horsetrack_executable() -> str:
    beside = Path(sys.executable).parent / "horsetrack"
    if beside.exists():
        return str(beside)
    return shutil.which("horsetrack") or "horsetrack"


def launchd_plist(host: str, remote_dir: str = DEFAULT_REMOTE_DIR) -> str:
    """LaunchAgent plist text that runs the pull for `host` daily at 22:00 local.

    Raises ValueError for a host that is not an alias, and `paths.DataDirNotSet` (or
    `DataDirInsideRepo`) when the data dir does not resolve: the job needs it in its environment."""
    check_host(host)
    data_dir = paths.data_dir()
    log_dir = data_dir / "vm" / host
    args = [_horsetrack_executable(), "japan", "pull-vm", "--host", host]
    if remote_dir != DEFAULT_REMOTE_DIR:
        args += ["--remote-dir", remote_dir]
    plist = {
        "Label": f"horsetrack.pull-vm.{host}",
        "ProgramArguments": args,
        "EnvironmentVariables": {paths.DATA_DIR_VAR: str(data_dir)},
        "StandardOutPath": str(log_dir / "pull-vm.out.log"),
        "StandardErrorPath": str(log_dir / "pull-vm.err.log"),
        "StartCalendarInterval": {"Hour": 22, "Minute": 0},
    }
    return plistlib.dumps(plist).decode()


# ---------------------------------------------------------------- the CLI


def pull_vm_command(
    host: str = typer.Option(..., "--host", help="ssh alias of the VM, from ~/.ssh/config."),
    remote_dir: str = typer.Option(DEFAULT_REMOTE_DIR, "--remote-dir", help="Data dir on the VM."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Pass --dry-run to rsync."),
    print_launchd: bool = typer.Option(
        False, "--print-launchd", help="Print a LaunchAgent plist (daily 22:00) and exit."
    ),
) -> None:
    """Pull the VM's logs into $HORSETRACK_DATA_DIR/vm/<host>/ (rsync over ssh, never --delete)."""
    try:
        if print_launchd:
            typer.echo(launchd_plist(host, remote_dir=remote_dir), nl=False)
            return
        code = pull_vm(host, remote_dir=remote_dir, dry_run=dry_run)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--host") from exc
    except (paths.DataDirNotSet, paths.DataDirInsideRepo) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if code != 0:
        typer.echo(f"error: rsync exited with {code}", err=True)
        raise typer.Exit(code)
