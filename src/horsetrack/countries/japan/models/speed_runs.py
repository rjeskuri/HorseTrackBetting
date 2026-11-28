"""The horse-runs frame and race cards for #97's `HorseSpeed`, from the store (#346, epic #73).

C3's per-horse model (`models/speed_horse.py`, #97) fits on past runs and predicts a card; this
module builds both from the store's races and runners only (no JRDB table, D21).

- `build_runs(races, runners, *, decision_min)`: one row per run, a non-scratched runner of a kept
  race, with race_id, race_utc (= post_utc: post_time_local on the Asia/Tokyo wall clock of
  race_date, as-of.md, #227), horse_id, style, early_pos_mean3, weight_kg (actual_wt_kg, the exact
  carried kg), draw, #305's par keys (track, surface, distance, layout, rail, class_level; a null
  key is "unknown"), finish_time_s (null when the runner has none: unplaced runners are kept) and
  result_available_utc = post_utc + 30 min (the same-day dividends rule, availability.yaml).
  Races are left out and counted, in this order: holdout days (`is_holdout_day("japan", d)`,
  never read: their runs are in no row and feed no style or early position), jump races (the
  owner on #96: a jump start feeds no flat horse's window either), races without a post time.
  Scratched runners of kept races are no run and are counted. The counts: n_races, n_holdout,
  n_jump, n_no_post_time, n_kept, n_scratched, n_runs.
- style is #129's first-corner rule (`handicap.factors._style_code` / `_finish_style`): the most
  frequent style over the horse's newest 5 starts that have a first-corner position, ties to the
  more forward style, null when there is none. A run's window holds the horse's runs in other
  kept races whose result_available_utc is strictly before its decision time, race_utc -
  decision_min: never the run's own race, never a race published at or after that time.
- early_pos_mean3 is #129's (`handicap.factors._race_late` / `_pace`), #97's pace input (the owner
  on #97, 2026-11-26): a start's early position is (p - 1) / (n - 1), p its first-corner position
  (the first entry of running_positions) and n the race's field_size, null when n <= 1 or p is
  missing; the mean over the horse's newest 3 starts that have one, in the style's window; null
  when there is none.
- `build_card(races, runners, race_id, *, at_utc)`: the target race's non-scratched runners with
  the frame's columns minus the result; each style and early_pos_mean3 read the horse's starts in
  other kept races published strictly before at_utc. A card after the off (at_utc > post_utc),
  for a race on a holdout day, a jump race or a race without a post time is refused (ValueError).
- `read_inputs(*, start, end, source="jra")`: the store copy's races and runners (the JRA store
  by default; `source="jravan"` the JRA-VAN copy, #404) with start <= race_date <= end
  through #114's scoped reader (`models.v0.store.read_store_table`): only the columns above (no
  price column), the date range pushed into the scan, holdout days never returned.

The unseen-cell drop (a run whose cell has no par in the fitted `SpeedModel`) is not counted
here: a builder without the fitted model cannot know it, so it is #97's `HorseSpeed.fit`
`n_unseen`.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.handicap.factors import STYLES, _finish_style, _style_code
from horsetrack.countries.japan.models.v0.store import read_store_table
from horsetrack.countries.japan.studies.speed_run import (
    JUMP,
    KEYS,
    RESULT_LAG,
    UNKNOWN,
    _post_utc,
    _race_days,
)

COUNTRY = "japan"
SOURCE = "jra"  # read_inputs' default: the store's JRA results (netkeiba); no JRDB
STYLE_WINDOW = 5  # #129: the newest 5 starts with a first-corner position
EARLY_WINDOW = 3  # #129: the newest 3 starts with an early position
RACE_COLUMNS = (
    "race_id",
    "race_date",
    "venue",
    "surface",
    "distance_m",
    "course_config",
    "rail_setting",
    "race_class",
    "post_time_local",
    "field_size",
)
RUNNER_COLUMNS = (
    "race_id",
    "race_date",
    "horse_id",
    "draw",
    "actual_wt_kg",
    "running_positions",
    "finish_time_s",
    "is_scratched",
)
PAR_KEYS = tuple(KEYS)
CARD_COLUMNS = (
    "race_id",
    "race_utc",
    "horse_id",
    "style",
    "early_pos_mean3",
    "weight_kg",
    "draw",
    *PAR_KEYS,
)
RESULT_COLUMNS = ("finish_time_s", "result_available_utc")
RUN_COLUMNS = (*CARD_COLUMNS, *RESULT_COLUMNS)


def _check(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks column(s): {', '.join(missing)}")


def _first_corner(positions: pd.Series) -> list[int | None]:
    """The first-corner position of each start (the list's first entry), None when there is none."""
    out: list[int | None] = []
    for value in positions:
        first = value[0] if isinstance(value, (list, tuple, np.ndarray)) and len(value) else None
        out.append(None if first is None or pd.isna(first) else int(first))
    return out


def _starts(races: pd.DataFrame, runners: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Every non-scratched runner of a kept race with its race's columns, its style code, early
    position and result_available_utc, and the counts of what was left out."""
    _check(races, RACE_COLUMNS, "races")
    _check(runners, RUNNER_COLUMNS, "runners")
    races = races.loc[:, list(RACE_COLUMNS)].reset_index(drop=True)
    if not races["race_id"].is_unique:
        raise ValueError("races has duplicate race ids")
    counts = {"n_races": len(races)}

    days = _race_days(races["race_date"])
    held_days = {d for d in set(days) if is_holdout_day(COUNTRY, d)}
    held = days.isin(held_days).to_numpy(dtype=bool)
    counts["n_holdout"] = int(held.sum())
    races, days = races.loc[~held], days[~held]

    jump = races["surface"].eq(JUMP).fillna(False).to_numpy(dtype=bool)
    counts["n_jump"] = int(jump.sum())
    races, days = races.loc[~jump], days[~jump]

    post = _post_utc(days, races["post_time_local"])
    timed = post.notna().to_numpy(dtype=bool)
    counts["n_no_post_time"] = int((~timed).sum())
    races = races.loc[timed].assign(race_utc=pd.to_datetime(post[timed], utc=True))
    counts["n_kept"] = len(races)

    runners = runners.loc[runners["race_id"].isin(races["race_id"]), list(RUNNER_COLUMNS)]
    scratched = runners["is_scratched"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    counts["n_scratched"] = int(scratched.sum())
    runners = runners.loc[~scratched].drop(columns=["race_date", "is_scratched"])

    frame = runners.merge(races.drop(columns=["race_date"]), on="race_id", how="left")
    frame["result_available_utc"] = frame["race_utc"] + RESULT_LAG
    frame["weight_kg"] = frame["actual_wt_kg"].astype(float)
    frame["finish_time_s"] = frame["finish_time_s"].astype(float)
    for key, column in KEYS.items():
        values = frame[column].astype(object)
        frame[key] = values.where(values.notna(), UNKNOWN)
    corner, size = pl.col("p"), pl.col("n")
    coded = pl.DataFrame(
        {
            "p": pl.Series(_first_corner(frame["running_positions"]), dtype=pl.Int64),
            "n": pl.Series(
                [None if pd.isna(n) else int(n) for n in frame["field_size"]], dtype=pl.Int64
            ),
        }
    ).select(
        _style_code(corner, size).alias("code"),
        pl.when(size > 1).then((corner - 1) / (size - 1)).alias("early"),  # #129's `_race_late`
    )
    for column, name in (("code", "style_code"), ("early", "early")):
        frame[name] = coded[column].to_pandas(use_pyarrow_extension_array=False).to_numpy()
    return frame.reset_index(drop=True), counts


def _ns(stamps) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).as_unit("ns").asi8


def _windows(history: pd.DataFrame, column: str, horse: pd.Series, cutoff: np.ndarray, k: int):
    """`history`'s non-null `column` values sorted by (horse, result_available_utc), and each
    query's window [lo, hi) there: its horse's newest k values published strictly before its
    cutoff (ns, UTC)."""
    valued = history.loc[history[column].notna()]
    hist_t = _ns(valued["result_available_utc"])
    ranks = np.unique(np.concatenate([hist_t, cutoff]))
    base = len(ranks) + 1
    ids, _ = pd.factorize(pd.concat([valued["horse_id"], horse], ignore_index=True))
    hist_h, query_h = ids[: len(valued)].astype(np.int64), ids[len(valued) :].astype(np.int64)
    # one sort key per start: the horse, then the publication instant (dense rank, exact)
    hist_key = hist_h * base + np.searchsorted(ranks, hist_t)
    order = np.argsort(hist_key, kind="stable")
    hist_key = hist_key[order]
    values = valued[column].to_numpy(dtype=float)[order]
    hi = np.searchsorted(hist_key, query_h * base + np.searchsorted(ranks, cutoff), side="left")
    first = np.searchsorted(hist_key, query_h * base, side="left")
    return values, np.maximum(first, hi - k), hi


def _styles(history: pd.DataFrame, horse: pd.Series, cutoff: np.ndarray) -> list[str | None]:
    """Each query's style: #129's rule over the newest STYLE_WINDOW starts of `history` by that
    horse with a style code and result_available_utc strictly before its cutoff (ns, UTC)."""
    values, lo, hi = _windows(history, "style_code", horse, cutoff, STYLE_WINDOW)
    codes = values.astype(np.int64)
    onehot = np.zeros((len(codes) + 1, len(STYLES)), dtype=np.int64)
    onehot[np.arange(1, len(codes) + 1), codes] = 1
    cum = onehot.cumsum(axis=0)
    tally = cum[hi] - cum[lo]
    styled = pl.DataFrame({f"_style{k}": tally[:, k] for k in range(len(STYLES))})
    return styled.select(_finish_style())["style"].to_list()


def _early(history: pd.DataFrame, horse: pd.Series, cutoff: np.ndarray) -> np.ndarray:
    """Each query's early_pos_mean3: the mean early position over the newest EARLY_WINDOW starts
    of `history` by that horse with one, published strictly before its cutoff; NaN with none."""
    values, lo, hi = _windows(history, "early", horse, cutoff, EARLY_WINDOW)
    total, count = np.zeros(len(hi)), np.zeros(len(hi))
    for back in range(1, EARLY_WINDOW + 1):  # newest first, each value exactly once
        idx = hi - back
        ok = idx >= lo
        total[ok] += values[idx[ok]]
        count[ok] += 1
    mean = np.full(len(hi), np.nan)
    return np.divide(total, count, out=mean, where=count > 0)


def build_runs(
    races: pd.DataFrame, runners: pd.DataFrame, *, decision_min: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per run (RUN_COLUMNS) and the counts of what was left out (module docstring).
    Each run's style and early_pos_mean3 read the horse's other starts published before race_utc
    - decision_min."""
    if not decision_min >= 0:
        raise ValueError(f"decision_min must be >= 0 (a time before the off), got {decision_min}")
    starts, counts = _starts(races, runners)
    cutoff = _ns(starts["race_utc"]) - int(pd.Timedelta(minutes=decision_min).value)
    # decision_min >= 0 puts a run's own race (published 30 min after its off) past its cutoff
    starts["style"] = _styles(starts, starts["horse_id"], cutoff)
    starts["early_pos_mean3"] = _early(starts, starts["horse_id"], cutoff)
    frame = starts.loc[:, list(RUN_COLUMNS)].reset_index(drop=True)
    counts["n_runs"] = len(frame)
    return frame, counts


def build_card(races: pd.DataFrame, runners: pd.DataFrame, race_id: str, *, at_utc) -> pd.DataFrame:
    """The target race's non-scratched runners with CARD_COLUMNS; each style and early_pos_mean3
    read the horse's starts in other kept races published strictly before `at_utc` (module
    docstring)."""
    at = pd.Timestamp(at_utc)
    if at.tzinfo is None:
        raise ValueError("at_utc must be timezone-aware")
    at = at.tz_convert("UTC")
    _check(races, RACE_COLUMNS, "races")
    target = races.loc[races["race_id"] == race_id]
    if len(target) != 1:
        raise ValueError(f"race {race_id!r} is not in races exactly once ({len(target)} rows)")
    starts, _ = _starts(races, runners)
    card = starts.loc[starts["race_id"] == race_id]
    if not (starts["race_id"] == race_id).any():
        day = _race_days(target["race_date"]).iloc[0]
        post = _post_utc(pd.Series([day]), target["post_time_local"].reset_index(drop=True))
        if is_holdout_day(COUNTRY, day):
            raise ValueError(f"race {race_id!r} is on a holdout day")
        if target["surface"].eq(JUMP).fillna(False).iloc[0]:
            raise ValueError(f"race {race_id!r} is a jump race")
        if post.isna().iloc[0]:
            raise ValueError(f"race {race_id!r} has no post time")
    elif at > card["race_utc"].iloc[0]:
        raise ValueError(f"at_utc {at} is after the off of race {race_id!r}")
    history = starts.loc[starts["race_id"] != race_id]
    card = card.reset_index(drop=True)
    cutoff = np.full(len(card), int(at.as_unit("ns").value), dtype=np.int64)
    card["style"] = _styles(history, card["horse_id"], cutoff)
    card["early_pos_mean3"] = _early(history, card["horse_id"], cutoff)
    return card.loc[:, list(CARD_COLUMNS)]


def read_inputs(
    *, start: date, end: date, source: str = SOURCE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The store copy `source`'s races (RACE_COLUMNS) and runners (RUNNER_COLUMNS) with start <=
    race_date <= end, column- and date-scoped, holdout days never returned. No price column is
    read. `source` defaults to the JRA store; 'jravan' reads the JRA-VAN copy."""
    races, runners = (
        read_store_table(
            name, columns=columns, start=start, end=end, touches_price=False, source=source
        ).to_pandas()
        for name, columns in (("races", RACE_COLUMNS), ("runners", RUNNER_COLUMNS))
    )
    return races, runners
