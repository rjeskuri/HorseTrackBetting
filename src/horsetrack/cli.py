"""The `horsetrack` CLI root. Commands live in their own modules and register on a group."""

import typer

from horsetrack import __version__
from horsetrack.core.ingest.cli import scrape_app
from horsetrack.core.store.cli import store_app
from horsetrack.countries.japan.bets.cli import bets_app
from horsetrack.countries.japan.board.server import board_serve_command
from horsetrack.countries.japan.handicap.workouts_jravan import (
    workouts_command as jravan_workouts_command,
)
from horsetrack.countries.japan.jra.going_fetch import fetch_going_command
from horsetrack.countries.japan.jra.win_sales_fetch import fetch_win_sales_command
from horsetrack.countries.japan.jravan.bloodlines import (
    bloodlines_command as jravan_bloodlines_command,
)
from horsetrack.countries.japan.jravan.card_versions import (
    card_versions_command as jravan_card_versions_command,
)
from horsetrack.countries.japan.jravan.check import check_command as jravan_check_command
from horsetrack.countries.japan.jravan.dividends import (
    dividends_command as jravan_dividends_command,
)
from horsetrack.countries.japan.jravan.dump import dump_command as jravan_dump_command
from horsetrack.countries.japan.jravan.horse_runs import (
    horse_runs_command as jravan_horse_runs_command,
)
from horsetrack.countries.japan.jravan.masters import masters_command as jravan_masters_command
from horsetrack.countries.japan.jravan.odds import final_odds_command as jravan_final_odds_command
from horsetrack.countries.japan.jravan.races import races_command as jravan_races_command
from horsetrack.countries.japan.jravan.reconcile import (
    reconcile_command as jravan_reconcile_command,
)
from horsetrack.countries.japan.jravan.runners import runners_command as jravan_runners_command
from horsetrack.countries.japan.jravan.timeseries import (
    timeseries_command as jravan_timeseries_command,
)
from horsetrack.countries.japan.jravan.votes import votes_command as jravan_votes_command
from horsetrack.countries.japan.jrdb.tables import build_command as jrdb_build_command
from horsetrack.countries.japan.logger.compact import compact
from horsetrack.countries.japan.logger.compare import compare_logs_command
from horsetrack.countries.japan.logger.launchd import launchd_command
from horsetrack.countries.japan.logger.log_day import log_day
from horsetrack.countries.japan.logger.pull_vm import pull_vm_command
from horsetrack.countries.japan.logger.systemd import systemd_command
from horsetrack.countries.japan.models.v0.nowcast_store import nowcast_fit_command
from horsetrack.countries.japan.research.coverage import coverage_command
from horsetrack.countries.japan.research.exacta_ceiling import exacta_ceiling_app
from horsetrack.countries.japan.research.exacta_impact import exacta_impact_command
from horsetrack.countries.japan.research.exacta_rules import exacta_rules_command
from horsetrack.countries.japan.research.joined_coverage import joined_coverage_command
from horsetrack.countries.japan.research.leaderboard_build import leaderboard_command
from horsetrack.countries.japan.research.pool_depth import pool_depth_app
from horsetrack.countries.japan.research.spread_scan import spread_scan_app
from horsetrack.countries.japan.research.win5_backfill import win5_history_command
from horsetrack.countries.japan.research.win5_study import win5_study_app
from horsetrack.countries.japan.shadow.day import shadow_day_command
from horsetrack.countries.japan.studies.ability import ability_study_command
from horsetrack.countries.japan.studies.debut_members import debut_members_command
from horsetrack.countries.japan.studies.nowcast import nowcast_study_command

app = typer.Typer(no_args_is_help=True)
japan = typer.Typer(no_args_is_help=True, help="Japan commands.")
app.add_typer(japan, name="japan")
app.add_typer(scrape_app, name="scrape")
app.add_typer(store_app, name="store")
japan.command("compact")(compact)
japan.command("log-day")(log_day)
japan.command("launchd")(launchd_command)
japan.command("systemd")(systemd_command)
japan.command("pull-vm")(pull_vm_command)
japan.command("board-serve")(board_serve_command)
japan.command("shadow-day")(shadow_day_command)
japan.command("win5-history")(win5_history_command)
japan.add_typer(win5_study_app, name="win5-study")
japan.command("fetch-going")(fetch_going_command)
japan.command("fetch-win-sales")(fetch_win_sales_command)
japan.command("exacta-impact")(exacta_impact_command)
japan.command("exacta-rules")(exacta_rules_command)
japan.add_typer(exacta_ceiling_app, name="exacta-ceiling")
japan.add_typer(pool_depth_app, name="pool-depth")
japan.command("nowcast-study")(nowcast_study_command)
japan.command("nowcast-fit")(nowcast_fit_command)
japan.command("ability-study")(ability_study_command)
japan.command("debut-members")(debut_members_command)
japan.command("coverage")(coverage_command)
japan.command("joined-coverage")(joined_coverage_command)
japan.command("leaderboard")(leaderboard_command)
japan.add_typer(spread_scan_app, name="spread-scan")
japan.add_typer(bets_app, name="bets")
jrdb = typer.Typer(no_args_is_help=True, help="JRDB commands.")
japan.add_typer(jrdb, name="jrdb")
jrdb.command("build")(jrdb_build_command)
jravan = typer.Typer(no_args_is_help=True, help="JRA-VAN DataLab (JV-Data) commands.")
japan.add_typer(jravan, name="jravan")
jravan.command("bloodlines")(jravan_bloodlines_command)
jravan.command("card-versions")(jravan_card_versions_command)
jravan.command("check")(jravan_check_command)
jravan.command("dividends")(jravan_dividends_command)
jravan.command("dump")(jravan_dump_command)
jravan.command("final-odds")(jravan_final_odds_command)
jravan.command("horse-runs")(jravan_horse_runs_command)
jravan.command("masters")(jravan_masters_command)
jravan.command("races")(jravan_races_command)
jravan.command("reconcile")(jravan_reconcile_command)
jravan.command("runners")(jravan_runners_command)
jravan.command("timeseries")(jravan_timeseries_command)
jravan.command("votes")(jravan_votes_command)
jravan.command("workouts")(jravan_workouts_command)


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
