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
trace. launchd does not create the folder of those logs (a job without it fails to start, exit 78),
so the CLI's `--print-launchd` creates `vm/<ALIAS>/` (and only that, never wiping anything) once
the host and data dir are validated, then prints. No rsync runs. It fails, creating nothing, if
the host is refused or the data dir is not set. `launchd_plist` itself creates nothing.

Live pull (ticket #236): `--every SECONDS` loops during racing, each round one `rsync -a --partial`
of the day's `odds_snapshots/jra/<DATE>/` only into `vm/<ALIAS>/odds_snapshots/jra/<DATE>` (the same
layout as the nightly pull, never `--delete`), the first round at once, until `--until HH:MM` (Tokyo
time on `--date`, default today in Tokyo) or Ctrl-C (exit 0). A failed round prints an error line to
stderr and the loop goes on. It cannot be combined with `--print-launchd`. `shadow-day` and
`board-serve` read the pulled folder with `--from-vm ALIAS` (`vm_day_dir`); the Mac's own
`odds_snapshots/` is never written. Seams looked up at call time: `utc_now`, `run_rsync`,
`time.sleep`.
"""

from __future__ import annotations

import plistlib
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import typer

from horsetrack.core import paths

DEFAULT_REMOTE_DIR = "/var/lib/horsetrack"
SUBTREES = ("odds_snapshots", "raw/jra/official_odds", "logs")

# An alias is a plain word: letters, digits, `_` and `-`, not starting with `-`. No dots, so a
# dotted-quad address cannot pass; no `@`, `:`, `/` or whitespace either.
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

JST = timezone(timedelta(hours=9))  # Japan keeps no daylight saving time
_STOP_TIME = re.compile(r"(\d\d):(\d\d)")

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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def vm_day_dir(host: str, race_date: date) -> Path:
    """The pulled copy of one day's JRA odds logs: `<data_dir>/vm/<host>/odds_snapshots/jra/<date>`,
    the same layout the nightly pull keeps. `shadow-day` and `board-serve --from-vm` read it.

    Raises ValueError for a host that is not an alias."""
    check_host(host)
    return paths.data_dir() / "vm" / host / "odds_snapshots" / "jra" / race_date.isoformat()


def pull_vm_loop(
    *,
    host: str,
    every: int,
    race_date: date,
    until: datetime | None = None,
    remote_dir: str = DEFAULT_REMOTE_DIR,
    dry_run: bool = False,
    runner: Runner | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = utc_now,
) -> None:
    """Pull the VM's odds folder of `race_date` into `vm_day_dir(host, race_date)` every `every`
    seconds, the first round at once, until `until` (an aware time) or until interrupted.

    A round is `rsync -a --partial` of that one day folder, never `--delete`. A round that fails
    prints an `error` line to stderr and the loop goes on. No round starts after `until`. Raises
    ValueError for a host that is not an alias, before anything runs or is created; a
    KeyboardInterrupt passes through."""
    dest = vm_day_dir(host, race_date)
    if every < 1:
        raise ValueError(f"every must be a positive number of seconds, got {every!r}")
    run = runner if runner is not None else run_rsync
    argv = ["rsync", "-a", "--partial"]
    if dry_run:
        argv.append("--dry-run")
    argv += [
        f"{host}:{remote_dir.rstrip('/')}/odds_snapshots/jra/{race_date.isoformat()}/",
        str(dest),
    ]
    step = timedelta(seconds=every)
    while until is None or now() <= until:
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)  # rsync only creates the last component
        code = run(list(argv))
        if code != 0:
            typer.echo(f"error: rsync exited with {code} pulling {host} {race_date}", err=True)
        if until is not None and now() + step > until:
            return  # the next round would start after the stop time
        sleep(every)


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


def _loop_args(
    race_date: str | None, until: str | None, every: int
) -> tuple[date, datetime | None]:
    """The pull's date (default: today in Asia/Tokyo) and stop time (`HH:MM` Tokyo time on that
    date), or a BadParameter."""
    if race_date is None:
        day = utc_now().astimezone(JST).date()
    else:
        try:
            day = date.fromisoformat(race_date)
        except ValueError as exc:
            raise typer.BadParameter(
                f"{race_date!r} is not a date (YYYY-MM-DD)", param_hint="--date"
            ) from exc
    if until is None:
        return day, None
    match = _STOP_TIME.fullmatch(until)
    try:
        if match is None:
            raise ValueError
        stop = datetime(day.year, day.month, day.day, int(match[1]), int(match[2]), tzinfo=JST)
    except ValueError as exc:
        raise typer.BadParameter(
            f"{until!r} is not a time (HH:MM, Tokyo time)", param_hint="--until"
        ) from exc
    return day, stop


def pull_vm_command(
    host: str = typer.Option(..., "--host", help="ssh alias of the VM, from ~/.ssh/config."),
    remote_dir: str = typer.Option(DEFAULT_REMOTE_DIR, "--remote-dir", help="Data dir on the VM."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Pass --dry-run to rsync."),
    print_launchd: bool = typer.Option(
        False,
        "--print-launchd",
        help="Print a LaunchAgent plist (daily 22:00) and exit; creates the job log folder.",
    ),
    every: int | None = typer.Option(
        None,
        "--every",
        min=1,
        help="Pull the day's odds folder every SECONDS (60 is a good value) until --until or "
        "Ctrl-C, into vm/<host>/odds_snapshots/jra/<date>/ (read it with --from-vm). "
        "Without it, one nightly pull of all subtrees.",
    ),
    race_date: str | None = typer.Option(
        None, "--date", help="With --every: race day (YYYY-MM-DD); default today in Asia/Tokyo."
    ),
    until: str | None = typer.Option(
        None, "--until", help="With --every: stop time HH:MM (Tokyo time); default until Ctrl-C."
    ),
) -> None:
    """Pull the VM's logs into $HORSETRACK_DATA_DIR/vm/<host>/ (rsync over ssh, never --delete)."""
    if every is None:
        for option, value in (("--date", race_date), ("--until", until)):
            if value is not None:
                raise typer.BadParameter("needs --every", param_hint=option)
    elif print_launchd:
        raise typer.BadParameter("cannot be combined with --print-launchd", param_hint="--every")
    try:
        if print_launchd:
            text = launchd_plist(host, remote_dir=remote_dir)  # validates host and data dir
            # launchd will not create the log folder, and a job without it never starts
            (paths.data_dir() / "vm" / host).mkdir(parents=True, exist_ok=True)
            typer.echo(text, nl=False)
            return
        if every is not None:
            _run_loop(host, every, race_date, until, remote_dir, dry_run)
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


def _run_loop(
    host: str,
    every: int,
    race_date: str | None,
    until: str | None,
    remote_dir: str,
    dry_run: bool,
) -> None:
    """The `--every` loop of the command; Ctrl-C ends it with a short message and exit 0."""
    day, stop = _loop_args(race_date, until, every)
    vm_day_dir(host, day)  # a host that is not an alias is refused before anything runs
    typer.echo(f"pulling {host} {day} every {every}s until {until or 'Ctrl-C'}")
    try:
        pull_vm_loop(
            host=host,
            every=every,
            race_date=day,
            until=stop,
            remote_dir=remote_dir,
            dry_run=dry_run,
            sleep=time.sleep,  # looked up now: the seam tests (and callers) patch
            now=utc_now,
        )
    except KeyboardInterrupt:
        typer.echo("stopped")
        return
    typer.echo("reached the stop time")
