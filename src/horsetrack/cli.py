"""The `horsetrack` CLI root. Commands live in their own modules and register on a group."""

import typer

from horsetrack import __version__

app = typer.Typer(no_args_is_help=True)
japan = typer.Typer(no_args_is_help=True, help="Japan commands.")
app.add_typer(japan, name="japan")


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
