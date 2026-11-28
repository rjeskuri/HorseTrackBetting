"""`horsetrack japan bets`: check a stake against the day cap, record a placed bet, settle a day.

`check` and `record` go through the day's `BetLog` (#453), which holds the cap arithmetic and the
append; `settle` joins the day's bets to `settle.day_dividends` and writes
`$HORSETRACK_DATA_DIR/bets/japan/<day>.settled.json` beside the log (#454); a holdout day's bets
are written 'holdout', without a read of the store. Exit codes: 0 fits or recorded within the cap;
1 the stake does not fit (`check`) or the recorded bet is over the cap (`record`, the bet is kept);
2 refused or failed, nothing written. Everything printed stays in the terminal and every file
under the data dir: the bet log is the owner's financial record.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any, NoReturn

import pandas as pd
import typer

from horsetrack.core import env
from horsetrack.core.eval.registry import default_registry
from horsetrack.core.eval.reports import is_inside
from horsetrack.countries.japan.bets.log import (
    BetLog,
    BetLogInRepo,
    CapCheck,
    Limits,
    bet_log_path,
    load_limits,
)
from horsetrack.countries.japan.bets.settle import (
    day_dividends,
    played_by_option,
    settle_bets,
    totals,
)
from horsetrack.countries.japan.bets.shortlist import shortlist, shortlist_text
from horsetrack.countries.japan.research import leaderboard_build

EXIT_OVER_CAP = 1
EXIT_REFUSED = 2
# BetRefused, BetLogTampered, a moved data dir (ValueError); BetLogInRepo, DataDirNotSet
# (RuntimeError); a store file not found (OSError)
_ERRORS = (ValueError, RuntimeError, OSError)

DATE_HELP = "The race day, YYYY-MM-DD (its JST date)."
bets_app = typer.Typer(
    no_args_is_help=True,
    help="The manual bet log: check the day cap, record a bet, settle a day, list the top options.",
)


def _refused(exc: Exception, what: str = "refused") -> NoReturn:
    typer.echo(f"{what}: {exc}", err=True)
    raise typer.Exit(EXIT_REFUSED)


def _day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"--date {value!r} is not a YYYY-MM-DD date") from None


def _log(day: date, limits: Limits) -> BetLog:
    return BetLog(bet_log_path(day), day=day, limits=limits)


def _yen(amount: float) -> str:
    return f"¥{amount:,.0f}" if float(amount).is_integer() else f"¥{amount:,.2f}"


def _left(limits: Limits, staked: int) -> str:
    remaining = limits.day_cap_yen - staked
    if remaining < 0:
        return f"¥0 left, {_yen(-remaining)} past the cap"
    return f"{_yen(remaining)} left"


@bets_app.command("check")
def check_command(
    race_date: Annotated[str, typer.Option("--date", help=DATE_HELP)],
    stake: Annotated[int, typer.Option("--stake", help="The stake in yen.")],
) -> None:
    """Whether a stake fits the day's cap; writes nothing. Exit 0 it fits, 1 it does not."""
    try:
        day = _day(race_date)
        limits = load_limits()
        answer = _log(day, limits).check(stake)
    except _ERRORS as exc:
        _refused(exc)
    typer.echo(
        f"{day.isoformat()}: staked {_yen(answer.staked_yen)} of the "
        f"{_yen(limits.day_cap_yen)} day cap, {_left(limits, answer.staked_yen)}"
    )
    if answer.off_flat:
        typer.echo(f"{_yen(stake)} is off the flat stake of {_yen(limits.flat_stake_yen)}")
    if not answer.fits:
        typer.echo(f"{_yen(stake)} does not fit: it would take the day over the cap")
        raise typer.Exit(EXIT_OVER_CAP)
    typer.echo(f"{_yen(stake)} fits")


@bets_app.command("record")
def record_command(
    race_date: Annotated[str, typer.Option("--date", help=DATE_HELP)],
    race_id: Annotated[
        str, typer.Option("--race-id", help="jra:<YYYY-MM-DD>:<VENUE>:<NN>, or jra:<day>:WIN5.")
    ],
    pool: Annotated[str, typer.Option("--pool", help="WIN, PLACE, QUINELLA, ... WIN5.")],
    combination: Annotated[
        str, typer.Option("--combination", help="Horse numbers joined by ',', e.g. 4,7.")
    ],
    stake: Annotated[int, typer.Option("--stake", help="The stake in yen.")],
    option_id: Annotated[
        str | None, typer.Option("--option-id", help="The leaderboard option bet on.")
    ] = None,
    placed_utc: Annotated[
        str | None,
        typer.Option("--placed-utc", help="When it was placed, ISO with an offset; default now."),
    ] = None,
    note: Annotated[str | None, typer.Option("--note", help="A free-text note.")] = None,
) -> None:
    """Record a bet placed by hand. A bet over the cap is kept and flagged: exit 1."""
    bet: dict[str, Any] = {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "stake_yen": stake,
        "placed_utc": datetime.now(timezone.utc).isoformat() if placed_utc is None else placed_utc,
    }
    if option_id is not None:
        bet["option_id"] = option_id
    if note is not None:
        bet["note"] = note
    try:
        day = _day(race_date)
        limits = load_limits()
        log = _log(day, limits)
        stored = log.record(bet)
    except _ERRORS as exc:
        _refused(exc)
    typer.echo(
        f"recorded bet {stored['seq']}: {stored['race_id']} {stored['pool']} "
        f"{stored['combination']} {_yen(stored['stake_yen'])} ({stored['day_status']})"
    )
    try:  # the bet is written: a failed re-read must not read as a refusal
        staked = log.staked_yen()
        typer.echo(
            f"the day staked {_yen(staked)} of the {_yen(limits.day_cap_yen)} day cap, "
            f"{_left(limits, staked)}"
        )
    except _ERRORS as exc:
        typer.echo(f"the bet is recorded; the day's total could not be read: {exc}", err=True)
    if stored["off_flat"]:
        typer.echo(f"off_flat: not the flat stake of {_yen(limits.flat_stake_yen)}")
    if stored["over_cap"]:
        typer.echo("over_cap: this bet takes the day over the day's cap; it is kept")
        raise typer.Exit(EXIT_OVER_CAP)


def settled_path(day: date) -> Path:
    """``data_dir()/bets/japan/<YYYY-MM-DD>.settled.json``, beside the day's bet log."""
    return bet_log_path(day).with_name(f"{day.isoformat()}.settled.json")


def _write_settled(day: date, settled: pd.DataFrame) -> Path:
    """The settled bets as JSON records, replaced whole (the file is derived, the log is not)."""
    path = settled_path(day)
    if is_inside(path, env.repo_root()):
        raise BetLogInRepo(f"{path} is inside the repository: settled bets stay outside it")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = settled.to_json(orient="records", force_ascii=False, indent=1) + "\n"
    # a temp file of this run's own, so two runs never write one partial file
    fd, partial = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(partial, path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise
    return path


@bets_app.command("settle")
def settle_command(race_date: Annotated[str, typer.Option("--date", help=DATE_HELP)]) -> None:
    """Settle the day's bets against its final dividends and write the settled file."""
    try:
        day = _day(race_date)
        bets = _log(day, load_limits()).read()
        if not bets:
            typer.echo(f"{day.isoformat()}: no bets logged, nothing settled")
            return
        settled = settle_bets(pd.DataFrame(bets), day_dividends(day))
        path = _write_settled(day, settled)
    except _ERRORS as exc:
        _refused(exc, "not settled, the settled file unchanged")
    for row in settled.to_dict("records"):
        paid = "" if pd.isna(row["paid_yen"]) else f" paid {_yen(row['paid_yen'])}"
        typer.echo(
            f"  {row['seq']:>3} {row['race_id']} {row['pool']} {row['combination']} "
            f"{_yen(row['stake_yen'])}: {row['outcome']}{paid}"
        )
    day_total = totals(settled)
    per_100 = day_total["return_per_100"]
    per_100 = "n/a" if per_100 is None else f"{per_100:.1f}"
    typer.echo(
        f"{day.isoformat()}: staked {_yen(day_total['staked_yen'])}, paid "
        f"{_yen(day_total['paid_yen'])}, return per ¥100 {per_100} over {day_total['n_bets']} "
        f"won or lost bets; {day_total['n_refunded']} refunded, {day_total['n_pending']} pending"
    )
    typer.echo(f"wrote {path}")


def _all_settled(day: date) -> pd.DataFrame:
    """Every ``<day>.settled.json`` beside ``day``'s bet log as one frame; reads only."""
    records: list[dict[str, Any]] = []
    for path in sorted(settled_path(day).parent.glob("*.settled.json")):
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, list):
            raise ValueError(f"{path} is not a list of settled bets")
        records.extend(content)
    return pd.DataFrame(records)


@bets_app.command("shortlist")
def shortlist_command(
    race_date: Annotated[str, typer.Option("--date", help=DATE_HELP)],
    top: Annotated[int, typer.Option("--top", min=1, help="How many options to list.")] = 10,
) -> None:
    """The top-ranked options of the private leaderboard beside the day's cap and what has been
    played on each. Terminal only, writes nothing: the figures are withheld (D21)."""
    try:
        day = _day(race_date)
        limits = load_limits()
        staked = _log(day, limits).staked_yen()
        # the head needs the staked total and the cap left; no stake is being asked about
        check = CapCheck(
            fits=staked <= limits.day_cap_yen,
            staked_yen=staked,
            remaining_yen=limits.day_cap_yen - staked,
            off_flat=False,
        )
        # a reader: the registry folder of a fresh data root is not created
        rows = leaderboard_build.build_rows(registry=default_registry("japan", create=False))
        ranked = shortlist(rows, top=top)
        settled = _all_settled(day)
        played = played_by_option(settled) if len(settled) else None
    except _ERRORS as exc:
        _refused(exc)
    typer.echo(shortlist_text(ranked, day=day, check=check, limits=limits, played=played))
