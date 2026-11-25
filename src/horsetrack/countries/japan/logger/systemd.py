"""`horsetrack japan systemd --print` (ticket #33): the systemd service and timer that start
`horsetrack japan log-day --unattended` every 30 minutes on the Linux VM, the twin of `launchd.py`.

The service runs `<repo>/.venv/bin/horsetrack japan log-day --unattended` with Restart=no: systemd
never relaunches it on exit, and log-day's lock and markers make each start that has nothing to do
exit at once. The timer starts it at :00 and :30 of every hour, Tokyo time (Persistent=true, so a
start missed while the VM was down runs at boot); a start while the service still runs is a no-op.

The unit sets only HORSETRACK_DATA_DIR and reads the VM's `<repo>/.env` (optional), which the owner
writes on the VM and which holds the user agent; nothing secret goes in the unit. That .env must not
set HORSETRACK_DATA_DIR: systemd lets EnvironmentFile= values override Environment= ones whatever
their order, so a data dir there (say .env.example's `~/Data/horsetrack`, which is
/root/Data/horsetrack for a root service) sends the logs away from the unit's data dir, where
`pull-vm` looks; vm_setup.sh refuses to run while the VM's .env names another one. The defaults are
the VM's paths (`scripts/vm_setup.sh` clones into /opt/horsetrack; `pull-vm` pulls from
/var/lib/horsetrack), never this machine's environment. --print writes the two files to stdout,
each after a header line `# <file name>` that vm_setup.sh splits on; it writes no file.
"""

from __future__ import annotations

import re

import typer

from horsetrack.core import paths
from horsetrack.countries.japan.logger.pull_vm import DEFAULT_REMOTE_DIR

SERVICE = "horsetrack-japan-logger.service"
TIMER = "horsetrack-japan-logger.timer"
DEFAULT_REPO_DIR = "/opt/horsetrack"
ON_CALENDAR = "*-*-* *:00,30:00 Asia/Tokyo"
EXIT_ERROR = 1

# Values go into unit files verbatim, so only plain absolute paths and user names: no whitespace,
# quotes, `%` specifiers, `$` expansions or newlines that would change how systemd reads a line.
_PATH = re.compile(r"/[A-Za-z0-9._+/-]+")
_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,31}")


def _abs_path(value: str, option: str) -> str:
    if not _PATH.fullmatch(value) or not value.strip("/"):
        raise ValueError(f"{option} {value!r}: give an absolute path of letters, digits, ._+-/")
    return value.rstrip("/")


def service_unit(
    repo_dir: str = DEFAULT_REPO_DIR, data_dir: str = DEFAULT_REMOTE_DIR, user: str | None = None
) -> str:
    """The text of horsetrack-japan-logger.service."""
    repo = _abs_path(repo_dir, "--repo-dir")
    data = _abs_path(data_dir, "--data-dir")
    if user is not None and not _USER.fullmatch(user):
        raise ValueError(f"--user {user!r}: not a user name")
    user_line = f"User={user}\n" if user is not None else ""
    return (
        "[Unit]\n"
        "Description=HorseTrackBetting Japan odds logger (log-day --unattended)\n"
        "Wants=network-online.target\n"
        "After=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"{user_line}"
        f"WorkingDirectory={repo}\n"
        f"Environment={paths.DATA_DIR_VAR}={data}\n"
        f"EnvironmentFile=-{repo}/.env\n"
        f"ExecStart={repo}/.venv/bin/horsetrack japan log-day --unattended\n"
        "Restart=no\n"
    )


def timer_unit() -> str:
    """The text of horsetrack-japan-logger.timer."""
    return (
        "[Unit]\n"
        "Description=Start the Japan odds logger every 30 minutes, Tokyo time\n"
        "\n"
        "[Timer]\n"
        f"OnCalendar={ON_CALENDAR}\n"
        "Persistent=true\n"
        f"Unit={SERVICE}\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )


def printed(
    repo_dir: str = DEFAULT_REPO_DIR, data_dir: str = DEFAULT_REMOTE_DIR, user: str | None = None
) -> str:
    """Both files, each after its `# <file name>` header line."""
    service = service_unit(repo_dir, data_dir, user)
    return f"# {SERVICE}\n{service}\n# {TIMER}\n{timer_unit()}"


def systemd_command(
    show: bool = typer.Option(False, "--print", help="Print the unit and timer; change nothing."),
    repo_dir: str = typer.Option(DEFAULT_REPO_DIR, "--repo-dir", help="The checkout on the VM."),
    data_dir: str = typer.Option(
        DEFAULT_REMOTE_DIR, "--data-dir", help="HORSETRACK_DATA_DIR on the VM."
    ),
    user: str | None = typer.Option(None, "--user", help="Run as this user (default root)."),
) -> None:
    """The systemd service and timer that run `log-day --unattended` every 30 minutes on the VM."""
    try:
        if not show:
            raise ValueError("give --print (the unit is installed by scripts/vm_setup.sh)")
        text = printed(repo_dir, data_dir, user)
    except ValueError as exc:
        typer.echo(f"ERROR {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    typer.echo(text, nl=False)
