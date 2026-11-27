"""`horsetrack japan race-time` (ticket #414): the race-time forecast of every race of a day.

- `run_race_time_day(race_date, *, model, model_version, races, now, sleep, replay=False,
  day_dir=None) -> RaceTimeRun`: each race of the card `races()` (re-read on every pass) is
  recorded once through `race_time.record_race` and appended to the day's log as it is made. Post
  times come from each venue's `<VENUE>.races.json` sidecar, else the card's own post_utc, and
  first sightings (#225) from its `<VENUE>.results.json` sidecar in `jra_day_dir(race_date,
  day_dir)`, re-read on every pass. Each pass reads the clock before it reads the card, store and
  sidecars, and its records take that instant as their input snapshot (`inputs_utc`), so a
  sighting logged after the read is never counted as seen by it. `latest_post` re-reads the races
  sidecar after each computation and, for a race the sidecar lacks, the card (`races()`), so a
  card post moved during the computation still moves the cutoff it is judged against.
  Prospective (replay=False): a
  race with a post is recorded once the clock is LEAD before its input cutoff (post - primary),
  waiting with `sleep` (at most POLL_S at a time, so a moved post is seen); a race whose cutoff
  has passed is recorded at once; races without a post are recorded last, once no race with a
  post is left, so a post that appears meanwhile still gives a forecast. replay=True records every
  race at once with replay=True and never sleeps. card_version hashes the card inputs (the race
  rows without their results, and the posts); each version's card is appended, before the first
  record made under it, to `card_snapshots_path(race_date)` and read back by
  `load_card_snapshots(race_date) -> {card_version: card}`, to verify each record against.
- `race_time_log_path(race_date)`: `data_dir()/race_time/japan/<YYYY-MM-DD>.jsonl`, append-only.
- `load_speed_model(race_date)`: #305's model (`studies.speed_run`: its race table, its keys and
  its primary t5 error) fitted on the store's races strictly before race_date, holdout days
  dropped as speed_run drops them; the holdout is never opened. Its version hashes the spec, the
  estimand, the keys and the training rows, so later store rows leave it unchanged.
- `load_day_races(race_date)`: every race of the upcoming cards
  (`processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet`; venue from meeting_id, post_utc
  from post_time_local), then the store's races of the day no card lists; the model's keys
  ("unknown" when missing), post_utc and the store's top-3 time (NaN until resulted). The card
  stays the schedule; a turf race without a rail (the card never has one) takes the latest
  rail of its venue and day in the store's `track_condition` table published by its cutoff,
  else the rail of the store's own row of the race, else stays "unknown".
- D14: on a Japan holdout day `run_race_time_day`, `load_day_races`, `load_card_snapshots` and
  the command raise
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
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.store.tables import read_table, table_path
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
    "card_snapshots_path",
    "load_card_snapshots",
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
    """The day's races with first_seen_utc from the venues' sidecars and post_utc from their
    sidecar where it has the race, else the card's own post_utc (NaT when the card has none)."""
    card = frame.drop_duplicates("race_id").reset_index(drop=True)
    venues = set(card["venue"])
    posts = {v: odds_log.read_post_times(folder / f"{v}.races.json") for v in venues}
    seen = {v: odds_log.read_published(folder / f"{v}.results.json") for v in venues}
    keys = list(zip(card["venue"], card["race_id"].map(_race_no), strict=True))
    own = pd.to_datetime(card["post_utc"], utc=True) if "post_utc" in card else pd.NaT
    side = pd.Series(pd.to_datetime([posts[v].get(n) for v, n in keys], utc=True))
    card["post_utc"] = side.where(side.notna(), own)
    card["first_seen_utc"] = pd.to_datetime([seen[v].get(n) for v, n in keys], utc=True)
    return card


def _latest_post(race: dict, folder: Path, races: Callable[[], pd.DataFrame]) -> Any:
    """The race's post read now: its sidecar's, else the re-read card's (None when neither)."""
    side = odds_log.read_post_times(folder / f"{race['venue']}.races.json")
    if _race_no(race["race_id"]) in side:
        return side[_race_no(race["race_id"])]
    frame = races()
    if "post_utc" not in frame:
        return None
    posts = pd.to_datetime(frame.loc[frame["race_id"] == race["race_id"], "post_utc"], utc=True)
    posts = posts.dropna()
    return None if posts.empty else posts.iloc[0]


def _card_inputs(card: pd.DataFrame) -> pd.DataFrame:
    """The card without its results: what the card version hashes and its snapshot keeps."""
    return card.drop(columns=["top3_time_s", "first_seen_utc"], errors="ignore")


def _card_version(race_date: date, card: pd.DataFrame) -> str:
    inputs = _card_inputs(card)
    rows = sorted(inputs.astype(object).where(inputs.notna(), None).to_dict("records"), key=repr)
    text = json.dumps(rows, sort_keys=True, default=str)
    return f"card:{race_date.isoformat()}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def card_snapshots_path(race_date: date) -> Path:
    """``data_dir()/race_time/japan/<YYYY-MM-DD>.cards.jsonl``, beside the log, append-only."""
    return race_time_log_path(race_date).with_suffix(".cards.jsonl")


def load_card_snapshots(race_date: date) -> dict[str, pd.DataFrame]:
    """Each saved card version of the day -> its card (race_id, venue, post_utc, the par keys)."""
    _refuse_holdout(race_date)
    path = card_snapshots_path(race_date)
    out: dict[str, pd.DataFrame] = {}
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for line in filter(None, lines):
        saved = json.loads(line)
        card = pd.DataFrame(saved["races"])
        card["post_utc"] = pd.to_datetime(card["post_utc"], utc=True)
        out[saved["card_version"]] = card
    return out


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
    saved: set[str] = set()
    while True:
        read_at = pd.Timestamp(now()).tz_convert("UTC")  # before the card, store and sidecars
        card = _card(races(), folder)
        version = _card_version(race_date, card)
        todo = [r for r in card.to_dict("records") if r["race_id"] not in done]
        if not todo:
            break
        timed = [r for r in todo if pd.notna(r["post_utc"])]
        if replay or not timed:
            due = todo
        else:
            clock = read_at
            wake = {r["race_id"]: r["post_utc"] - primary - LEAD for r in timed}
            due = sorted(
                (r for r in timed if wake[r["race_id"]] <= clock), key=lambda r: r["post_utc"]
            )
            if not due:
                sleep(min((min(wake.values()) - clock).total_seconds(), POLL_S))
                continue
        if version not in saved:  # the card each record's card_version names, for verify_replay
            inputs = json.loads(_card_inputs(card).to_json(orient="records", date_format="iso"))
            _append(card_snapshots_path(race_date), {"card_version": version, "races": inputs})
            saved.add(version)
        for race in due:
            rec = record_race(
                race,
                model=model,
                results=card,
                model_version=model_version,
                card_version=version,
                now=now,
                replay=replay,
                latest_post=lambda r=race: _latest_post(r, folder, races),
                inputs_utc=read_at,  # a sighting written after the read was not among them
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
    rows = table.sort_values("race_id", kind="stable").to_csv(index=False)  # the training rows
    about = {
        "spec": asdict(spec),
        "estimand": speed_run.ESTIMANDS[0],
        "keys": speed_run.KEYS,
        "table": hashlib.sha256(rows.encode()).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(about, sort_keys=True).encode()).hexdigest()[:12]
    last = max(table["race_date"]).isoformat()
    return model, f"{speed_run.FAMILY_ID}:t5:fit-to-{last}:n{len(table)}:{digest}"


def _card_post_utc(value: Any) -> pd.Timestamp:
    """An upcoming card's post_time_local (ISO, +09:00; a naive one is read as JST) as UTC."""
    if value is None or pd.isna(value):
        return pd.NaT
    local = pd.Timestamp(value)
    return (local.tz_localize(speed_run.JST) if local.tzinfo is None else local).tz_convert("UTC")


def _upcoming(race_date: date) -> pd.DataFrame:
    """Every venue's upcoming card of the day, with venue from meeting_id and post_utc."""
    root = paths.processed_dir(SOURCE) / "upcoming"
    files = sorted(root.glob(f"jra_{race_date.isoformat()}_*/races.parquet"))
    if not files:
        return pd.DataFrame()
    card = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    card["venue"] = card["meeting_id"].astype(str).str.rsplit(":", n=1).str[-1]
    card["post_utc"] = pd.to_datetime(card["post_time_local"].map(_card_post_utc), utc=True)
    return card


def _keyed(races: pd.DataFrame) -> pd.DataFrame:
    """race_id, venue, the model's keys ("unknown" for a missing value or column) and post_utc."""
    out = races.loc[:, ["race_id", "venue", "post_utc"]].reset_index(drop=True)
    for key, column in speed_run.KEYS.items():
        values = races[column].astype(object) if column in races else pd.Series([None] * len(out))
        out[key] = values.where(values.notna(), speed_run.UNKNOWN).to_numpy()
    return out


def _course_rails(race_date: date) -> pd.DataFrame:
    """The day's turf rail publications in the store's track_condition table (#90): venue, rail
    and available_utc, ordered by availability; empty when the table is not built."""
    columns = ["venue", "race_date", "surface", "rail", "available_utc", "ingested_utc"]
    if not table_path(SOURCE, "track_condition").exists():
        return pd.DataFrame(columns=["venue", "rail", "available_utc"])
    rows = read_table(SOURCE, "track_condition", columns=columns)
    keep = (speed_run._race_days(rows["race_date"]) == race_date).to_numpy()
    rows = rows.loc[keep & (rows["surface"] == "turf").to_numpy() & rows["rail"].notna().to_numpy()]
    rows = rows.assign(available_utc=pd.to_datetime(rows["available_utc"], utc=True))
    return rows.sort_values(["available_utc", "ingested_utc"], kind="mergesort")


def _known_rails(out: pd.DataFrame, race_date: date, store: pd.DataFrame) -> list:
    """Each row's rail: a turf race the card gives no rail takes the latest track_condition rail of
    its venue published by its cutoff (post - primary), else the rail the store's own row of the
    race has (from the going pages, `going.attach_rail_setting`), else stays unknown."""
    primary = pd.Timedelta(minutes=load_decision_grid().primary)
    course = _course_rails(race_date)
    own = {
        rid: rail
        for rid, rail in zip(store["race_id"], store["rail_setting"], strict=True)
        if rail is not None and pd.notna(rail)
    }
    rails = []
    for row in out.to_dict("records"):
        rail = row["rail"]
        if rail == speed_run.UNKNOWN and row["surface"] == "turf":
            post = pd.Timestamp(row["post_utc"]) if pd.notna(row["post_utc"]) else None
            known = course.loc[(course["venue"] == row["venue"]).to_numpy()]
            if post is not None:
                known = known.loc[(known["available_utc"] <= post - primary).to_numpy()]
            if post is not None and not known.empty:
                rail = known["rail"].iloc[-1]
            else:
                rail = own.get(row["race_id"], rail)
        rails.append(rail)
    return rails


def load_day_races(race_date: date) -> pd.DataFrame:
    """The day's races: every upcoming card race, then the store's races of the day no card lists;
    race_id, venue, the model's keys, post_utc and top3_time_s (NaN until resulted)."""
    _refuse_holdout(race_date)
    card = _upcoming(race_date)
    races = read_table(SOURCE, "races", columns=list(speed_run.RACE_COLUMNS))
    races = races.loc[(speed_run._race_days(races["race_date"]) == race_date).to_numpy()]
    days = pd.Series([race_date] * len(races), index=races.index, dtype=object)
    races = races.assign(post_utc=speed_run._post_utc(days, races["post_time_local"]))
    listed = set() if card.empty else set(card["race_id"])
    parts = [
        _keyed(frame)
        for frame in (card, races.loc[~races["race_id"].isin(listed).to_numpy()])
        if not frame.empty
    ]
    if not parts:
        raise LookupError(f"no {SOURCE} card or store race on {race_date.isoformat()}")
    out = pd.concat(parts, ignore_index=True).drop_duplicates("race_id", ignore_index=True)
    out["rail"] = _known_rails(out, race_date, races)
    top3 = pd.Series(dtype=float)
    runners = read_table(SOURCE, "runners", columns=list(speed_run.RUNNER_COLUMNS))
    runners = runners.loc[runners["race_id"].isin(out["race_id"])]
    if not runners.empty:
        top3, _ = speed_run._mean_placed_1_3(runners)
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
