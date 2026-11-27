"""A synthetic JRA store and #87 member file under a tmp HORSETRACK_DATA_DIR (ticket #418).

Shared by the joined-coverage CLI test and the meta study's D9 tests. Every value is made up: the
venue code `QQV` matches no real one, horse ids are generated, prices come from a seeded RNG.

The store: two races on the 10th and the 24th of every month from 2014-06-10 (inside D10's price
era) to 2024-06-24, six runners each, every runner priced and starting, one winner per race. The
first race day of each year is 10 January, so walk-forward's 1-week embargo never reaches back
past 31 December: a member trained through the season before is out of sample for 2024-Q1.
Each runner is a horse's only start, so its first observed start is that race.

The member file (`studies.meta.MEMBER_KEYS` and the three member columns of `studies.meta.MEMBERS`)
holds every runner from `start` on, its `fold_season` the race's calendar year (a season
walk-forward: trained on the seasons before), less the runners in `drop`.

`add_race_day` adds a day of races (round 2: a late-December day the priced frame drops, an early
January day that pulls walk-forward's first 2024 train_end before 31 December).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from horsetrack.core import paths

VENUE = "QQV"
FIRST_DAY = date(2014, 6, 10)
LAST_DAY = date(2024, 6, 24)
RACES_PER_DAY = 2
RUNNERS = 6
CLASSES = ("DEBUT", "MAIDEN", "C1", "C2", "C3", "OP", "G3")
ODDS = (2.4, 3.8, 6.5, 11.0, 19.0, 42.0)
MEMBER_COLUMNS = ("p_fundamental_oof", "p_lgbm_no_jrdb_oof", "p_clogit_no_jrdb_oof")


def race_days(first: date = FIRST_DAY, last: date = LAST_DAY) -> list[date]:
    days = []
    for year in range(first.year, last.year + 1):
        for month in range(1, 13):
            for dom in (10, 24):
                d = date(year, month, dom)
                if first <= d <= last:
                    days.append(d)
    return days


def race_id(day: date, n: int) -> str:
    return f"jra:{day.isoformat()}:{VENUE}:{n:02d}"


@dataclass(frozen=True)
class Store:
    runners: pd.DataFrame
    races: pd.DataFrame

    @property
    def race_ids(self) -> list[str]:
        return list(self.races["race_id"])


def build_store(seed: int = 418) -> Store:
    rng = np.random.default_rng(seed)
    runner_rows, race_rows = [], []
    k = 0
    for day in race_days():
        for n in range(1, RACES_PER_DAY + 1):
            rid = race_id(day, n)
            cls = CLASSES[k % len(CLASSES)]
            k += 1
            race_rows.append(
                {
                    "race_id": rid,
                    "race_date": day,
                    "meeting_id": f"{day.isoformat()}:{VENUE}",
                    "status": "run",
                    "race_class": cls,
                }
            )
            odds = rng.permutation(ODDS)
            inv = 1.0 / odds
            winner = int(rng.choice(RUNNERS, p=inv / inv.sum()))
            order = [winner] + [i for i in rng.permutation(RUNNERS) if i != winner]
            finish = {runner: place + 1 for place, runner in enumerate(order)}
            for i in range(RUNNERS):
                runner_rows.append(
                    {
                        "race_id": rid,
                        "race_date": day,
                        "horse_id": f"h{k:05d}{i}",
                        "horse_no": i + 1,
                        "win_odds": float(odds[i]),
                        "finish_pos": finish[i],
                        "is_scratched": False,
                    }
                )
    runners = pd.DataFrame(runner_rows)
    runners["finish_pos"] = runners["finish_pos"].astype("Int64")
    return Store(runners=runners, races=pd.DataFrame(race_rows))


RUNNER_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("horse_id", pa.string()),
        ("horse_no", pa.int64()),
        ("win_odds", pa.float64()),
        ("finish_pos", pa.int64()),
        ("is_scratched", pa.bool_()),
    ]
)
RACE_SCHEMA = pa.schema(
    [
        ("race_id", pa.string()),
        ("race_date", pa.date32()),
        ("meeting_id", pa.string()),
        ("status", pa.string()),
        ("race_class", pa.string()),
    ]
)


def write_store(store: Store) -> Path:
    """The store's runners and races under `processed_dir('jra')` of the current data dir."""
    out = paths.processed_dir("jra")
    out.mkdir(parents=True, exist_ok=True)
    for name, frame, schema in (
        ("runners", store.runners, RUNNER_SCHEMA),
        ("races", store.races, RACE_SCHEMA),
    ):
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        pq.write_table(table, out / f"{name}.parquet")
    return out


def member_rows(
    store: Store, start: date, *, drop: frozenset[tuple[str, str]] = frozenset(), seed: int = 87
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    runners = store.runners.loc[pd.to_datetime(store.runners["race_date"]) >= pd.Timestamp(start)]
    keys = list(zip(runners["race_id"], runners["horse_id"], strict=True))
    keep = np.array([key not in drop for key in keys], dtype=bool)
    runners = runners.loc[keep]
    days = pd.to_datetime(runners["race_date"])
    q = 1.0 / runners["win_odds"].to_numpy()
    frame = pd.DataFrame(
        {
            "race_id": runners["race_id"].to_numpy(),
            "horse_id": runners["horse_id"].to_numpy(),
            "race_date": days.to_numpy(),
            "fold_season": days.dt.year.astype(str).to_numpy(),
        }
    )
    for col in MEMBER_COLUMNS:
        frame[col] = q * np.exp(rng.normal(0.0, 0.3, len(frame)))
    return frame


def write_members(path: Path, members: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(members, preserve_index=False), path)
    return path


def add_race_day(store: Store, day: date, *, status: str = "run", priced: bool = True) -> Store:
    """The store with `RACES_PER_DAY` more races on `day`, six runners each with horse ids of their
    own and the favourite (the first runner) winning. `status` is the races' status (a race not
    `run` has no finishing positions); `priced=False` leaves every Win price missing."""
    runner_rows, race_rows = [], []
    for n in range(1, RACES_PER_DAY + 1):
        rid = race_id(day, n)
        race_rows.append(
            {
                "race_id": rid,
                "race_date": day,
                "meeting_id": f"{day.isoformat()}:{VENUE}",
                "status": status,
                "race_class": "C1",
            }
        )
        for i in range(RUNNERS):
            runner_rows.append(
                {
                    "race_id": rid,
                    "race_date": day,
                    "horse_id": f"x{day:%Y%m%d}{n}{i}",
                    "horse_no": i + 1,
                    "win_odds": float(ODDS[i]) if priced else float("nan"),
                    "finish_pos": i + 1 if status == "run" else pd.NA,
                    "is_scratched": False,
                }
            )
    added = pd.DataFrame(runner_rows)
    added["finish_pos"] = added["finish_pos"].astype("Int64")
    return Store(
        runners=pd.concat([store.runners, added], ignore_index=True),
        races=pd.concat([store.races, pd.DataFrame(race_rows)], ignore_index=True),
    )
