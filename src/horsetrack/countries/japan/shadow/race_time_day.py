"""`horsetrack japan race-time` (ticket #414): the race-time forecast of every race of a day.

- `run_race_time_day(race_date, *, model, model_version, races, now, sleep, replay=False,
  day_dir=None) -> RaceTimeRun`: each race of the card `races()` (re-read on every pass) is
  recorded once through `race_time.record_race` and appended to the day's log as it is made. Post
  times come from each venue's `<VENUE>.races.json` sidecar and first sightings (#225) from its
  `<VENUE>.results.json` sidecar in `jra_day_dir(race_date, day_dir)`, re-read on every pass;
  `latest_post` re-reads the races sidecar after each computation. Prospective (replay=False): a
  race with a post is recorded once the clock is LEAD before its input cutoff (post - primary),
  waiting with `sleep` (at most POLL_S at a time, so a moved post is seen); a race whose cutoff
  has passed is recorded at once; races without a post are recorded last, once no race with a
  post is left, so a post that appears meanwhile still gives a forecast. replay=True records every
  race at once with replay=True and never sleeps. card_version hashes the card inputs (the race
  rows without their results, and the posts).
- `race_time_log_path(race_date)`: `data_dir()/race_time/japan/<YYYY-MM-DD>.jsonl`, append-only.
- `load_speed_model(race_date)`: #305's model (`studies.speed_run`: its race table, its keys and
  its primary t5 error) fitted on the store's races strictly before race_date, holdout days
  dropped as speed_run drops them; the holdout is never opened. `load_day_races(race_date)`: the
  store's races of the day with the model's keys and their top-3 time (NaN until resulted).
- D14: on a Japan holdout day `run_race_time_day`, `load_day_races` and the command raise
  `HoldoutDayError` before any read (card, sidecar or store) and write no log.

Seams for the command, looked up at call time: `utc_now`, `sleep`, `load_speed_model`,
`load_day_races`. Nothing is adopted (D16).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.store.tables import read_table
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board.snapshot import jra_day_dir, load_decision_grid
from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec
from horsetrack.countries.japan.models.v0.nowcast import HoldoutDayError
from horsetrack.countries.japan.shadow.race_time import coverage, record_race
from horsetrack.countries.japan.studies import speed_run

COUNTRY = "japan"
SOURCE = "jra"
LEAD = timedelta(seconds=60)  # recorded this long before the cutoff (within the 5-minute window)
POLL_S = 60.0  # the longest single wait, so a post moved earlier is seen in time
EXIT_ERROR = 1

__all__ = [
    "RaceTimeRun",
    "load_day_races",
    "load_speed_model",
    "race_time_log_path",
    "run_race_time_day",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sleep(seconds: float) -> None:
    time.sleep(seconds)


@dataclass(frozen=True)
class RaceTimeRun:
    records: list[dict[str, Any]]
    coverage: dict[str, dict[str, int]]
    path: Path


def race_time_log_path(race_date: date) -> Path:
    """``data_dir()/race_time/japan/<YYYY-MM-DD>.jsonl``, the data dir read on every call."""
    if not isinstance(race_date, date) or isinstance(race_date, datetime):
        raise TypeError(f"race_date must be a date, got {race_date!r}")
    return paths.data_dir() / "race_time" / COUNTRY / f"{race_date.isoformat()}.jsonl"


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _race_no(race_id: str) -> int:
    return int(str(race_id).rsplit(":", 1)[1])


def _card(frame: pd.DataFrame, folder: Path) -> pd.DataFrame:
    """The day's races with post_utc and first_seen_utc from the venues' sidecars."""
    card = frame.drop_duplicates("race_id").reset_index(drop=True)
    venues = set(card["venue"])
    posts = {v: odds_log.read_post_times(folder / f"{v}.races.json") for v in venues}
    seen = {v: odds_log.read_published(folder / f"{v}.results.json") for v in venues}
    keys = list(zip(card["venue"], card["race_id"].map(_race_no), strict=True))
    card["post_utc"] = pd.to_datetime([posts[v].get(n) for v, n in keys], utc=True)
    card["first_seen_utc"] = pd.to_datetime([seen[v].get(n) for v, n in keys], utc=True)
    return card


def _card_version(race_date: date, card: pd.DataFrame) -> str:
    inputs = card.drop(columns=["top3_time_s", "first_seen_utc"], errors="ignore")
    rows = sorted(inputs.astype(object).where(inputs.notna(), None).to_dict("records"), key=repr)
    text = json.dumps(rows, sort_keys=True, default=str)
    return f"card:{race_date.isoformat()}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def _refuse_holdout(race_date: date) -> None:
    """D14: a holdout day is fully locked, outcomes and odds alike; refuse before any read."""
    if is_holdout_day(COUNTRY, race_date):
        raise HoldoutDayError(f"{race_date.isoformat()} is a {COUNTRY} holdout day (D14)")


def run_race_time_day(
    race_date: date,
    *,
    model: SpeedModel,
    model_version: str,
    races: Callable[[], pd.DataFrame],
    now: Callable[[], datetime],
    sleep: Callable[[float], None],
    replay: bool = False,
    day_dir: Path | None = None,
) -> RaceTimeRun:
    """Record every race of the day once (see the module docstring)."""
    _refuse_holdout(race_date)
    folder = jra_day_dir(race_date, day_dir)
    path = race_time_log_path(race_date)
    primary = timedelta(minutes=load_decision_grid().primary)
    records: list[dict[str, Any]] = []
    done: set[str] = set()
    while True:
        card = _card(races(), folder)
        version = _card_version(race_date, card)
        todo = [r for r in card.to_dict("records") if r["race_id"] not in done]
        if not todo:
            break
        timed = [r for r in todo if pd.notna(r["post_utc"])]
        if replay or not timed:
            due = todo
        else:
            clock = pd.Timestamp(now()).tz_convert("UTC")
            wake = {r["race_id"]: r["post_utc"] - primary - LEAD for r in timed}
            due = sorted(
                (r for r in timed if wake[r["race_id"]] <= clock), key=lambda r: r["post_utc"]
            )
            if not due:
                sleep(min((min(wake.values()) - clock).total_seconds(), POLL_S))
                continue
        for race in due:
            sidecar, no = folder / f"{race['venue']}.races.json", _race_no(race["race_id"])
            rec = record_race(
                race,
                model=model,
                results=card,
                model_version=model_version,
                card_version=version,
                now=now,
                replay=replay,
                latest_post=lambda s=sidecar, n=no: odds_log.read_post_times(s).get(n),
            )
            _append(path, rec)
            records.append(rec)
            done.add(race["race_id"])
    latest = _card(races(), folder)
    return RaceTimeRun(records=records, coverage=coverage(records, latest), path=path)


# ---------------------------------------------------------------- the store seams


def load_speed_model(race_date: date) -> tuple[SpeedModel, str]:
    """#305's model fitted on the store's development races strictly before race_date."""
    races, runners = speed_run.read_inputs()  # holdout days dropped, as the study drops them
    table, _ = speed_run.build_race_table(races, runners, estimand=speed_run.ESTIMANDS[0])
    table = table.loc[(table["race_date"] < race_date).to_numpy()].reset_index(drop=True)
    spec = SpeedSpec(
        par_keys=speed_run.PAR_KEYS,
        variant_keys=speed_run.VARIANT_KEYS,
        same_day_variant=True,
        t_df=speed_run.ERROR_MODELS["t5"],
    )
    model = SpeedModel(spec).fit(table)
    about = {
        "spec": asdict(spec),
        "estimand": speed_run.ESTIMANDS[0],
        "keys": speed_run.KEYS,
        "builds": speed_run.store_builds(),
    }
    digest = hashlib.sha256(json.dumps(about, sort_keys=True).encode()).hexdigest()[:12]
    last = max(table["race_date"]).isoformat()
    return model, f"{speed_run.FAMILY_ID}:t5:fit-to-{last}:n{len(table)}:{digest}"


def load_day_races(race_date: date) -> pd.DataFrame:
    """The store's races of the day: race_id, venue, the model's keys and top3_time_s."""
    _refuse_holdout(race_date)
    races = read_table(SOURCE, "races", columns=list(speed_run.RACE_COLUMNS))
    races = races.loc[(speed_run._race_days(races["race_date"]) == race_date).to_numpy()]
    if races.empty:
        raise LookupError(f"the {SOURCE} store has no race on {race_date.isoformat()}")
    top3 = pd.Series(dtype=float)
    runners = read_table(SOURCE, "runners", columns=list(speed_run.RUNNER_COLUMNS))
    runners = runners.loc[runners["race_id"].isin(races["race_id"])]
    if not runners.empty:
        top3, _ = speed_run._mean_placed_1_3(runners)
    out = pd.DataFrame({"race_id": races["race_id"].to_numpy(), "venue": races["venue"].to_numpy()})
    for key, column in speed_run.KEYS.items():
        values = races[column].astype(object)
        out[key] = values.where(values.notna(), speed_run.UNKNOWN).to_numpy()
    out["top3_time_s"] = out["race_id"].map(top3).astype(float)
    return out


# ---------------------------------------------------------------- the command


def race_time_command(
    race_date: str = typer.Option(..., "--date", help="Race date (YYYY-MM-DD)."),
    replay: bool = typer.Option(False, "--replay", help="Record every race now, as replays."),
    day_dir: str | None = typer.Option(
        None, "--day-dir", help="The day's odds-log folder (default: odds_snapshots/jra/<date>/)."
    ),
) -> None:
    """Record the race-time forecast of every race of a day and print its coverage by venue."""
    try:
        day = date.fromisoformat(race_date)
    except ValueError as exc:
        typer.echo(f"ERROR --date {race_date!r} is not a YYYY-MM-DD date", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    try:
        _refuse_holdout(day)
        model, version = load_speed_model(day)
        typer.echo(f"race-time {day.isoformat()} model={version} replay={replay}")
        run = run_race_time_day(
            day,
            model=model,
            model_version=version,
            races=lambda: load_day_races(day),
            now=utc_now,
            sleep=sleep,
            replay=replay,
            day_dir=None if day_dir is None else Path(day_dir),
        )
    except Exception as exc:
        typer.echo(f"ERROR {' '.join(str(exc).split())[:300] or type(exc).__name__}", err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    for venue, counts in run.coverage.items():
        typer.echo(f"{venue}: " + " ".join(f"{name}={n}" for name, n in counts.items()))
    typer.echo(f"log: {run.path}")
