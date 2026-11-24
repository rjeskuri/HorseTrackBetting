"""The `horsetrack` CLI root. Commands live in their own modules and register on a group."""

import typer

from horsetrack import __version__
from horsetrack.core.ingest.cli import scrape_app
from horsetrack.countries.japan.board.server import board_serve_command
from horsetrack.countries.japan.logger.compact import compact
from horsetrack.countries.japan.logger.compare import compare_logs_command
from horsetrack.countries.japan.logger.log_day import log_day
from horsetrack.countries.japan.logger.pull_vm import pull_vm_command

app = typer.Typer(no_args_is_help=True)
japan = typer.Typer(no_args_is_help=True, help="Japan commands.")
app.add_typer(japan, name="japan")
app.add_typer(scrape_app, name="scrape")
japan.command("compact")(compact)
japan.command("log-day")(log_day)
japan.command("pull-vm")(pull_vm_command)
japan.command("board-serve")(board_serve_command)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the package version and exit.",
    ),
) -> None:
    """HorseTrackBetting research and tooling."""


japan.command("compare-logs")(compare_logs_command)
