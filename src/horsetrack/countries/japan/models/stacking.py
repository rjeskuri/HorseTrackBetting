"""Stacked-fold eligibility after the member joins (ticket #418, on #98's level 2).

The combined model trains on the level-2 rows that carry a prediction of every member, so the
joins can shorten a fold's training window below D9's minimum even when the store alone meets it
(#87's documented failure). `join_members` builds those rows and records each member prediction's
provenance; `stacked_oof` refuses, before any fit, a fold whose joined training window is under
`D9_MIN_SEASONS` (`IneligibleFold`) or a used row whose member prediction comes from a model
trained on it or carries no provenance (`InvalidProvenance`), then delegates to
`models.meta.oof_predictions`. A row a fold scores is held to that fold's `train_end` as well: its
member must have trained no later than the fold's own training data, or a member refitted inside
the test window would score it. `check_level2` runs the checks alone, for a caller that fits its
own way (the #87 study, `studies.meta`).

- `join_members(frame, folds, predictions, *, members)`: `frame` is the store's level-2 rows
  (`race_id`, `horse_id`, `race_date`, ...) on a unique index; `folds` are `core.eval.folds.Fold`s
  as positions into it. `predictions` is long, one row per (race_id, horse_id, member) with
  `logp`, `member_fold` (the member fold it is out of) and `member_train_end` (the last day the
  member model trained on), and optionally `p`, the probability itself, which the provenance
  carries beside the row's key for a caller that normalises the file's own values. A race is kept
  only when every row of it has every member's `logp`.
- `fold_windows(level2)`: per fold, the first and last race day of its joined training rows, the
  seasons between them as `studies.meta` counts D9's, the training races and whether D9 is met.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from horsetrack.core.eval.folds import Fold
from horsetrack.countries.japan.models.meta import oof_predictions
from horsetrack.countries.japan.studies.meta import D9_MIN_SEASONS, season_years

KEY = ["race_id", "horse_id"]
PROVENANCE = ("member_fold", "member_train_end")
ROW = "row"  # the provenance column holding the frame's index label
CARRIED = ("p",)  # prediction columns the provenance carries when given


class IneligibleFold(ValueError):
    """A fold's joined training window is shorter than D9's minimum history."""


class InvalidProvenance(ValueError):
    """A member prediction a fold uses is in-sample for its row, or carries no provenance."""


@dataclass(frozen=True, eq=False)
class Level2:
    """The joined level-2 rows (one `logp` column per member), the folds as positions into them,
    and one provenance row per kept (row, member), with the prediction's `p` when it had one."""

    frame: pd.DataFrame
    folds: list[Fold]
    provenance: pd.DataFrame
    members: tuple[str, ...]


@dataclass(frozen=True)
class FoldWindow:
    test_start: date
    test_end: date
    train_end: date
    train_from: date | None
    train_to: date | None
    years: float
    train_races: int
    eligible: bool


def _require(frame: pd.DataFrame, columns: Sequence[str], what: str, error: type) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise error(f"{what} has no column(s): {', '.join(missing)}")


def join_members(
    frame: pd.DataFrame,
    folds: Sequence[Fold],
    predictions: pd.DataFrame,
    *,
    members: Sequence[str],
) -> Level2:
    """The level-2 rows with every member's prediction (see the module doc)."""
    members = tuple(members)
    if not members or len(set(members)) != len(members):
        raise ValueError(f"members must be distinct and at least one, got {members!r}")
    _require(frame, [*KEY, "race_date"], "the frame", KeyError)
    _require(predictions, [*KEY, "member", "logp"], "the predictions", KeyError)
    _require(predictions, PROVENANCE, "the predictions", InvalidProvenance)
    if not frame.index.is_unique:
        raise ValueError("the frame's index has duplicate labels")
    if frame.duplicated(KEY).any():
        raise ValueError("the frame repeats a (race_id, horse_id)")
    clash = sorted(set(members) & {*frame.columns, ROW})
    if clash:
        raise ValueError(f"the frame already has column(s) named for members: {clash}")

    preds = predictions.loc[predictions["member"].isin(members)]
    if preds.duplicated([*KEY, "member"]).any():
        raise ValueError("a (race_id, horse_id, member) prediction occurs more than once")
    logp = preds.pivot(index=KEY, columns="member", values="logp").reindex(columns=list(members))
    joined = frame.join(logp, on=KEY)
    complete = joined[list(members)].notna().all(axis=1)
    kept_races = complete.groupby(joined["race_id"]).transform("all").to_numpy()
    level2 = joined.loc[kept_races]

    # old positions -> new positions; a dropped row maps to -1 and leaves the fold
    new_pos = np.full(len(frame), -1)
    new_pos[np.flatnonzero(kept_races)] = np.arange(int(kept_races.sum()))
    remapped = [
        Fold(
            train_idx=_remap(new_pos, f.train_idx),
            test_idx=_remap(new_pos, f.test_idx),
            train_end=f.train_end,
            test_start=f.test_start,
            test_end=f.test_end,
        )
        for f in folds
    ]
    rows = level2[KEY].assign(**{ROW: level2.index})
    carried = [c for c in CARRIED if c in preds.columns]
    provenance = rows.merge(preds[[*KEY, "member", *PROVENANCE, *carried]], on=KEY, how="inner")
    provenance = provenance[[ROW, *KEY, "member", *PROVENANCE, *carried]].reset_index(drop=True)
    return Level2(frame=level2, folds=remapped, provenance=provenance, members=members)


def _remap(new_pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    moved = new_pos[np.asarray(idx, dtype=int)]
    return moved[moved >= 0]


def fold_windows(level2: Level2, *, min_seasons: float = D9_MIN_SEASONS) -> list[FoldWindow]:
    """Each fold's joined training window against `min_seasons`; never raises on a short one."""
    out = []
    for fold in level2.folds:
        train = level2.frame.iloc[fold.train_idx]
        days = pd.to_datetime(train["race_date"])
        first, last = (None, None) if train.empty else (days.min().date(), days.max().date())
        years = 0.0 if first is None else season_years(first, last)
        out.append(
            FoldWindow(
                test_start=fold.test_start,
                test_end=fold.test_end,
                train_end=fold.train_end,
                train_from=first,
                train_to=last,
                years=years,
                train_races=int(train["race_id"].nunique()),
                eligible=years >= min_seasons,
            )
        )
    return out


def _check_provenance(level2: Level2) -> None:
    """Every member prediction of a row a fold uses has provenance and was made by a model
    trained only on days before the row's race day; a row a fold scores, by a model trained no
    later than that fold's train_end."""
    _require(level2.provenance, [ROW, "member", *PROVENANCE], "the provenance", InvalidProvenance)
    used = np.unique(
        np.concatenate(
            [np.asarray(idx, dtype=int) for f in level2.folds for idx in (f.train_idx, f.test_idx)]
            or [np.array([], dtype=int)]
        )
    )
    labels = level2.frame.index[used]
    prov = level2.provenance.loc[level2.provenance[ROW].isin(labels)]
    counts = prov.groupby(ROW)["member"].nunique().reindex(labels, fill_value=0)
    if (counts < len(level2.members)).any():
        raise InvalidProvenance(
            f"{int((counts < len(level2.members)).sum())} used rows lack a member's provenance"
        )
    race_day = pd.to_datetime(level2.frame.loc[prov[ROW], "race_date"]).dt.normalize().to_numpy()
    trained_to = pd.to_datetime(prov["member_train_end"]).dt.normalize().to_numpy()
    fold_name = prov["member_fold"]
    no_fold = (fold_name.isna() | fold_name.astype(str).str.strip().eq("")).to_numpy()
    no_end = pd.isna(trained_to)
    in_sample = ~no_end & (trained_to >= race_day)
    if no_fold.any() or no_end.any():
        raise InvalidProvenance(
            f"{int(no_fold.sum())} used member predictions have no member_fold and "
            f"{int(no_end.sum())} no member_train_end"
        )
    if in_sample.any():
        bad = prov.loc[in_sample].iloc[0]
        raise InvalidProvenance(
            f"{int(in_sample.sum())} used member predictions come from a model trained on their "
            f"row, e.g. member {bad['member']!r} on {bad['race_id']} {bad['horse_id']} "
            f"(trained to {pd.Timestamp(bad['member_train_end']).date()})"
        )
    rows = prov[ROW].to_numpy()
    for fold in level2.folds:
        scored = np.isin(rows, level2.frame.index[np.asarray(fold.test_idx, dtype=int)])
        late = scored & (trained_to > np.datetime64(fold.train_end, "ns"))
        if late.any():
            bad = prov.loc[late].iloc[0]
            raise InvalidProvenance(
                f"{int(late.sum())} member predictions scored by fold "
                f"{fold.test_start.isoformat()} come from a model trained past its train_end "
                f"{fold.train_end.isoformat()}, e.g. member {bad['member']!r} on "
                f"{bad['race_id']} {bad['horse_id']} "
                f"(trained to {pd.Timestamp(bad['member_train_end']).date()})"
            )


def check_level2(level2: Level2, *, min_seasons: float = D9_MIN_SEASONS) -> list[FoldWindow]:
    """`stacked_oof`'s checks without the fit: invalid member provenance (`InvalidProvenance`),
    then a joined training window under `min_seasons` (`IneligibleFold`). Returns the windows."""
    _check_provenance(level2)
    windows = fold_windows(level2, min_seasons=min_seasons)
    short = [w for w in windows if not w.eligible]
    if short:
        detail = "; ".join(
            f"{w.test_start.isoformat()} trains on {w.years:.2f} seasons "
            f"({w.train_from} to {w.train_to})"
            for w in short
        )
        raise IneligibleFold(
            f"{len(short)} of {len(windows)} folds train on fewer than {min_seasons} seasons "
            f"after the member joins (D9): {detail}"
        )
    return windows


def stacked_oof(
    level2: Level2,
    fit: Callable[[pd.DataFrame], Any],
    predict: Callable[[Any, pd.DataFrame], Any],
    *,
    min_seasons: float = D9_MIN_SEASONS,
) -> tuple[pd.Series, list[FoldWindow]]:
    """`models.meta.oof_predictions` over `level2.folds`, after every fold is checked: invalid
    member provenance (`InvalidProvenance`) or a joined training window under `min_seasons`
    (`IneligibleFold`) refuses before any fit."""
    windows = check_level2(level2, min_seasons=min_seasons)
    return oof_predictions(level2.frame, level2.folds, fit, predict), windows
