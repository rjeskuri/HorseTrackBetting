"""Handicap factors per runner as of decision time, and the in-fold handicap score (#129).

`handicap_factors(race_id, as_of=..., train_end=None)` gives one row per declared runner of a race:
the factors a handicapper weighs, built from #128's PP lines (newest 10 starts before the cut) and
today's card. The cut is the last race day before the race whose result is published before
`as_of` (#128's ruling). The in-fold pieces, #96's speed model behind the speed figure and the draw
bias, are fitted on the non-holdout race days on or before `train_end` (default: the cut). Every
read goes through #114's scoped reader, so a holdout day is never read. Every value column has a
Boolean `<column>_missing` companion; nothing is imputed.

Factor families, one function each (the rulings are pinned in tests/countries/japan/handicap/):

1. `_speed`: figure = #96's expected top-3 time of the start's race - finish time (#219, replacing
   #129's interim class par); best, last, mean of the newest 3, count. The model is #96's
   `SpeedModel` (par per six-key cell, day variant), fitted on the in-fold days of
   `speed_run.build_race_table`; days after `train_end` up to the cut are observed, and a start's
   variant is read 2 minutes before its race's post (D2), so no later race and not its own counts.
   The table builder gives every past start the figure as of its own race day (`_own_day_expected`:
   the model fitted on the days before it, its day's earlier results observed).
2. `_pace`: late figure = the race's median last 600 m - own; early position and late gain from
   the corner positions; means over the newest 3 starts that have each.
3. `_style`: front / stalker / midpack / closer from the first corner, the most frequent over the
   newest 5 starts (ties forward); today's front-runner count, share and rivals.
4. `_class`: level DEBUT/MAIDEN 0 .. G1 8; today's level against the newest 3, best figure at
   today's level or higher.
5. `_form`: slope of the newest 3 figures, days since the last start, runs since a 60-day layoff,
   starts in 90 days.
6. `_fit`: record and shrunk figure (m / (m + 2)) at today's distance band, surface, going band,
   venue and course.
7. `_condition`, `_connection_rates`: weight changes, jockey change, and the jockey's and trainer's
   strike rates over the 365 days to the cut, shrunk 20 runs toward the field rate.
8. `_draw_bias`: wins against 1/n by draw third at today's course and distance, shrunk by 2.

`fit_handicap_score` fits a ridge conditional logit on the winner over standardised
`SCORE_FEATURES` (a null contributes the training mean; the `_missing` flags are features of their
own); `oof_handicap_score` scores each #70 fold's test rows with the fit on its training rows; both
refuse any holdout-day row (D14). `build_handicap_table` writes one Parquet file per non-holdout
race day, `features/jra/handicap-96-v2/<YYYY-MM-DD>.parquet` (#129's own table, built with the
interim par, stays in `features/jra/handicap/`), built one day at a time, then adds the
out-of-fold `handicap_score` (null on a day no fold scores) and a completion marker.

No layer here is adopted for staking (D16); these are features and a scaffold score.
"""

from __future__ import annotations

import functools
import json
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
from scipy.optimize import minimize

from horsetrack.core.eval.folds import Fold, FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.paths import data_dir
from horsetrack.countries.japan.handicap.pp import (
    DEFAULT_N,
    HISTORY_START,
    _is_in,
    _lines_for,
    _read,
    last_published_day,
    race_day_of,
)
from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec
from horsetrack.countries.japan.models.v0.calibration import _group_softmax, _nll, _segments
from horsetrack.countries.japan.studies import speed_run

COUNTRY = "japan"
STYLES = ("front", "stalker", "midpack", "closer")  # 逃げ / 先行 / 差し / 追込
CLASS_LEVELS = {"DEBUT": 0, "MAIDEN": 0, "C1": 1, "C2": 2, "C3": 3, "OP": 4, "L": 5, "G3": 6,
                "G2": 7, "G1": 8}  # fmt: skip
FIRM_GOING = ("GOOD", "GOOD TO YIELDING")
SOFT_GOING = ("SOFT", "HEAVY")
LAYOFF_DAYS = 60
RECENT_DAYS = 90
RATE_WINDOW_DAYS = 365
RATE_PRIOR_RUNS = 20
FIT_PRIOR = 2
DRAW_PRIOR = 2
FIT_DIMS = ("dist", "surface", "going", "venue", "course")
WHO = ("jockey", "trainer")

# #96's model as #305's study fits it: six-key par, (track, surface) day variant, Student-t error
SPEED_SPEC = SpeedSpec(speed_run.PAR_KEYS, speed_run.VARIANT_KEYS, same_day_variant=True, t_df=5.0)
SPEED_ESTIMAND = "mean_placed_1_3"
SPEED_WARMUP_DAYS = 365  # as of its own day, a start in the table's first year has no figure

CARD_RACE = ("venue", "surface", "distance_m", "course_config", "going", "race_class")
CARD_RUNNER = ("horse_no", "horse_id", "draw", "actual_wt_kg", "horse_wt_change_kg",
               "jockey_key", "trainer_key")  # fmt: skip
KEYS = ("race_id", "race_date", "horse_no", "horse_id")

_F, _I = pl.Float64, pl.Int64
FACTOR_SCHEMA: dict[str, pl.DataType] = {
    "speed_best": _F, "speed_last": _F, "speed_mean3": _F, "speed_n": _I,
    "late_mean3": _F, "early_pos_mean3": _F, "late_gain_mean3": _F,
    "style": pl.String, "field_front_n": _I, "field_front_share": _F, "front_rivals": _I,
    "class_today": _I, "class_change": _F, "speed_best_at_class": _F,
    "speed_slope3": _F, "days_since_last": _I, "runs_since_layoff": _I, "starts_90d": _I,
    **{f"fit_{d}{s}": (_F if s == "" else _I) for d in FIT_DIMS for s in ("_n", "_top3", "")},
    "body_wt_change": _I, "wt_carried_change": _F, "jockey_change": pl.Boolean,
    "jockey_rate": _F, "trainer_rate": _F, "draw_bias": _F,
}  # fmt: skip
FACTOR_COLUMNS = tuple(FACTOR_SCHEMA)
SCHEMA: dict[str, pl.DataType] = {
    "race_id": pl.String, "race_date": pl.Date, "horse_no": _I, "horse_id": pl.String,
    **FACTOR_SCHEMA, **{f"{c}_missing": pl.Boolean for c in FACTOR_SCHEMA},
}  # fmt: skip
_COUNTS = ("speed_n", "starts_90d", *[f"fit_{d}_{s}" for d in FIT_DIMS for s in ("n", "top3")])

# The score's inputs: the factors that vary within a race (race-constant ones cancel in a
# conditional logit; the style label is categorical), plus one flag per distinct null pattern.
SCORE_FEATURES: tuple[str, ...] = (
    "speed_best", "speed_last", "speed_mean3", "speed_n", "late_mean3", "early_pos_mean3",
    "late_gain_mean3", "front_rivals", "class_change", "speed_best_at_class", "speed_slope3",
    "days_since_last", "runs_since_layoff", "starts_90d",
    *[f"fit_{d}{s}" for d in FIT_DIMS for s in ("_n", "_top3", "")],
    "body_wt_change", "wt_carried_change", "jockey_change", "jockey_rate", "trainer_rate",
    "draw_bias",
    *[f"{c}_missing" for c in ("speed_best", "late_mean3", "early_pos_mean3", "late_gain_mean3",
                               "class_change", "speed_best_at_class", "speed_slope3",
                               "days_since_last", "body_wt_change", "wt_carried_change",
                               "draw_bias")],
)  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------------------------


def handicap_factors(
    race_id: str, *, as_of: datetime, train_end: date | None = None
) -> pl.DataFrame:
    """The handicap factors of every declared runner of `race_id` as of `as_of`."""
    day = race_day_of(race_id)
    cut = min(day - timedelta(days=1), last_published_day(as_of))
    if train_end is None:
        train_end = cut
    elif train_end > cut:
        raise ValueError(
            f"train_end {train_end.isoformat()} is after the as-of cut {cut.isoformat()}"
        )
    return _factors_for(day, [race_id], cut=cut, train_end=train_end)


def _factors_for(
    day: date,
    race_ids: Sequence[str],
    *,
    cut: date,
    train_end: date,
    race_table: pd.DataFrame | None = None,
    own_day: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """The factors of the named races of one race day: history to `cut`, fits to `train_end`.
    `race_table` is #96's race table holding every race to `cut` (read when not given). `own_day`
    (`_own_day_expected`) gives each past start's expected time as of its own race day instead of
    the model fitted to `train_end`."""
    in_races = _is_in("race_id", race_ids)
    card = _read("races", ["race_id", *CARD_RACE], day, day, where=in_races)
    entries = (
        _read("runners", ["race_id", *CARD_RUNNER, "is_scratched"], day, day, where=in_races)
        .filter(~pl.col("is_scratched").fill_null(False))
        .drop("is_scratched")
        .with_columns(pl.col("horse_no").cast(pl.Int64))
        .join(card, on="race_id", how="left")
        .with_columns(
            pl.lit(day).alias("race_date"),
            _level(pl.col("race_class")).alias("class_today"),
            _dist_band(pl.col("distance_m")).alias("t_dist"),
            _going_band(pl.col("going")).alias("t_going"),
        )
    )
    lines = _lines_for(entries.select("race_id", "horse_no", "horse_id"), last_day=cut, n=DEFAULT_N)
    lines = lines.filter(pl.col("line_no").is_not_null())
    if own_day is None:
        race_ids = lines.get_column("pp_race_id").unique().to_list()
        if race_table is None:
            race_table = _race_table(cut)
        own_day = _expected_times(race_ids, race_table, cut=cut, train_end=train_end)
    lines = _speed_figures(lines, own_day)
    lines = _race_late(lines).join(
        entries.select(
            "race_id", "horse_no", "class_today", "t_dist", "t_going",
            pl.col("surface").alias("t_surface"), pl.col("venue").alias("t_venue"),
            pl.col("course_config").alias("t_course"),
        ),
        on=["race_id", "horse_no"], how="left",
    ).sort(["race_id", "horse_no", "line_no"])  # fmt: skip

    per_start = (_speed(), _pace(), _style(), _class(), _form(day), _fit(), _last_start())
    stats = lines.group_by(["race_id", "horse_no"]).agg(*[e for fam in per_start for e in fam])
    out = (
        entries.join(stats, on=["race_id", "horse_no"], how="left")
        .with_columns(pl.col(c).fill_null(0) for c in _COUNTS)
        .with_columns(*_finish_form(day), *_finish_fit(), _finish_style(), *_condition())
        .with_columns(*_pace_scenario())
    )
    out = _connection_rates(out, cut=cut)
    out = _draw_bias(out, train_end=train_end)
    out = out.select(
        *[pl.col(c).cast(SCHEMA[c]) for c in KEYS],
        *[pl.col(c).cast(t) for c, t in FACTOR_SCHEMA.items()],
    ).with_columns(pl.col(c).is_null().alias(f"{c}_missing") for c in FACTOR_SCHEMA)
    return out.sort(["race_id", "horse_no"])


def _level(cls: pl.Expr) -> pl.Expr:
    return cls.replace_strict(CLASS_LEVELS, default=None, return_dtype=pl.Int64)


def _dist_band(m: pl.Expr) -> pl.Expr:
    return (
        pl.when(m <= 1400).then(1).when(m <= 1800).then(2).when(m <= 2200).then(3)
        .when(m.is_not_null()).then(4)
    )  # fmt: skip


def _going_band(g: pl.Expr) -> pl.Expr:
    return (
        pl.when(g.is_in(FIRM_GOING))
        .then(pl.lit("firm"))
        .when(g.is_in(SOFT_GOING))
        .then(pl.lit("soft"))
    )


def _race_table(cut: date) -> pd.DataFrame:
    """#96's race table over every race to `cut`: the store's races and runners through the scoped
    reader (holdout days dropped), no JRDB table (D21)."""
    races = _read("races", list(speed_run.RACE_COLUMNS), HISTORY_START, cut).to_pandas()
    runners = _read("runners", list(speed_run.RUNNER_COLUMNS), HISTORY_START, cut).to_pandas()
    table = speed_run.build_race_table(races, runners, estimand=SPEED_ESTIMAND)[0]
    # A fit with a cell or a day alone in its group is not identified and ends where the row order
    # takes it, so the order is fixed here: the figure must not depend on how the store is laid out.
    return table.sort_values(["race_date", "race_id"], kind="stable", ignore_index=True)


def _speed_model(race_table: pd.DataFrame, *, cut: date, train_end: date) -> SpeedModel | None:
    """#96's model fitted on the races to `train_end`, having observed the results of the days
    after it to `cut`; None when no race is in the fold."""
    days = race_table["race_date"]
    fit = race_table[days <= train_end]
    if fit.empty:
        return None
    model = SpeedModel(SPEED_SPEC).fit(fit)
    later = race_table[(days > train_end) & (days <= cut)]
    return model.observe(later) if len(later) else model


def _expected_times(
    race_ids: Sequence[str], race_table: pd.DataFrame, *, cut: date, train_end: date
) -> pl.DataFrame:
    """#96's expected top-3 time of each named race at its own decision time (D2's primary, the
    post time less `speed_run.decision_min()`); no row for a race outside the table or in a cell
    the fit never saw."""
    rows = []
    race_table = race_table[race_table["race_date"] <= cut]
    model = _speed_model(race_table, cut=cut, train_end=train_end) if len(race_ids) else None
    if model is not None:
        lead = pd.Timedelta(minutes=speed_run.decision_min())
        for race in race_table[race_table["race_id"].isin(race_ids)].to_dict("records"):
            try:
                expected = model.expected_top3_time(race, at_utc=race["post_utc"] - lead)
            except KeyError:  # no par: the cell is not in the fold
                continue
            rows.append((race["race_id"], expected))
    return pl.DataFrame(
        rows, schema={"pp_race_id": pl.String, "expected": pl.Float64}, orient="row"
    )


def _own_day_expected(race_table: pd.DataFrame) -> pl.DataFrame:
    """#96's expected top-3 time of every race of `race_table` as of its own race day: the model
    fitted on the races of the days before it, having observed the results of its own day, read at
    post - `speed_run.decision_min()` minutes, so an earlier result of the meeting counts (the day
    variant) and neither the race itself nor a later one does. The figure as of a day is fixed, so
    it is computed once per race day (one fit each). No row for a race in the first
    `SPEED_WARMUP_DAYS` days of the table (a fit on a few weeks of races has one-race cells and no
    day variance to speak of) or in a cell with no earlier race. `race_table` is in `_race_table`'s
    order."""
    lead = pd.Timedelta(minutes=speed_run.decision_min())
    days = race_table["race_date"].to_numpy()
    starts = np.r_[np.flatnonzero(np.r_[True, days[1:] != days[:-1]]), len(race_table)]
    warm = days[0] + timedelta(days=SPEED_WARMUP_DAYS) if len(days) else None
    ids, expected = [], []
    for first, stop in zip(starts[1:-1], starts[2:], strict=True):  # day 0 has no earlier race
        if days[first] < warm:
            continue
        today = race_table.iloc[first:stop]
        model = SpeedModel(SPEED_SPEC).fit(race_table.iloc[:first]).observe(today)
        for race in today.to_dict("records"):
            try:
                expected.append(model.expected_top3_time(race, at_utc=race["post_utc"] - lead))
            except KeyError:  # no par: no earlier race of its cell
                continue
            ids.append(race["race_id"])
    return pl.DataFrame(
        {"pp_race_id": ids, "expected": expected},
        schema={"pp_race_id": pl.String, "expected": pl.Float64},
    )


def _speed_figures(lines: pl.DataFrame, expected: pl.DataFrame) -> pl.DataFrame:
    """Each start's figure, `fig` = #96's expected top-3 time of its race - finish time."""
    return lines.join(expected, on="pp_race_id", how="left").with_columns(
        (pl.col("expected") - pl.col("finish_time_s")).alias("fig")
    )


def _race_late(lines: pl.DataFrame) -> pl.DataFrame:
    """Each start's late figure (the race's median last 600 m - own), early position, late gain
    and running style code (0 front .. 3 closer)."""
    race_l3f = pl.DataFrame(schema={"pp_race_id": pl.String, "race_l3f": pl.Float64})
    if lines.height:
        race_ids = lines.get_column("pp_race_id").unique().to_list()
        first, last = lines.get_column("pp_date").min(), lines.get_column("pp_date").max()
        race_l3f = (
            _read(
                "runners",
                ["race_id", "last_3f_s", "is_scratched"],
                first,
                last,
                where=_is_in("race_id", race_ids),
            )  # fmt: skip
            .filter(~pl.col("is_scratched").fill_null(False) & pl.col("last_3f_s").is_not_null())
            .group_by("race_id")
            .agg(pl.col("last_3f_s").median().alias("race_l3f"))
            .rename({"race_id": "pp_race_id"})
        )
    p = pl.col("running_positions").list.get(0, null_on_oob=True)
    n = pl.col("field_size")
    return lines.join(race_l3f, on="pp_race_id", how="left").with_columns(
        (pl.col("race_l3f") - pl.col("last_3f_s")).alias("late"),
        pl.when(n > 1).then((p - 1) / (n - 1)).alias("early"),
        (pl.col("running_positions").list.get(-1, null_on_oob=True) - pl.col("finish_pos")).alias(
            "gain"
        ),
        _style_code(p, n).alias("style_code"),
        _level(pl.col("race_class")).alias("level"),
    )  # fmt: skip


def _style_code(p: pl.Expr, n: pl.Expr) -> pl.Expr:
    """Running style code (0 front .. 3 closer) from the first-corner position p in a field of n."""
    return (
        pl.when(p.is_null() | n.is_null()).then(None).when(p == 1).then(0)
        .when(3 * (p - 1) <= n - 1).then(1).when(3 * (p - 1) <= 2 * (n - 1)).then(2)
        .otherwise(3)
    )  # fmt: skip


def _newest(col: str, k: int) -> pl.Expr:
    """The newest k values of a start column that are not null (lines sorted newest first)."""
    return pl.col(col).drop_nulls().head(k)


def _speed() -> list[pl.Expr]:
    """1. Speed: best, last, mean of the newest 3 figures, and their count."""
    fig = pl.col("fig")
    return [
        fig.max().alias("speed_best"),
        _newest("fig", 1).first().alias("speed_last"),
        _newest("fig", 3).mean().alias("speed_mean3"),
        fig.count().alias("speed_n"),
    ]


def _pace() -> list[pl.Expr]:
    """2. Pace and late speed: means over the newest 3 starts that have each value."""
    return [
        _newest("late", 3).mean().alias("late_mean3"),
        _newest("early", 3).mean().alias("early_pos_mean3"),
        _newest("gain", 3).mean().alias("late_gain_mean3"),
    ]


def _style() -> list[pl.Expr]:
    """3. Running style: start counts per style over the newest 5 with corner positions."""
    return [(_newest("style_code", 5) == k).sum().alias(f"_style{k}") for k in range(len(STYLES))]


def _finish_style() -> pl.Expr:
    counts = [pl.col(f"_style{k}") for k in range(len(STYLES))]
    best = pl.max_horizontal(counts)
    expr = pl.when(best.is_null() | (best == 0)).then(None)
    for k, name in enumerate(STYLES):  # ties go to the more forward style
        expr = expr.when(counts[k] == best).then(pl.lit(name))
    return expr.alias("style")


def _pace_scenario() -> list[pl.Expr]:
    """3. Today's pace scenario: front-runners in the field, their share, this runner's rivals."""
    front = (pl.col("style") == "front").fill_null(False).cast(pl.Int64)
    n_front = front.sum().over("race_id")
    return [
        n_front.alias("field_front_n"),
        (n_front / pl.len().over("race_id")).alias("field_front_share"),
        (n_front - front).alias("front_rivals"),
    ]


def _class() -> list[pl.Expr]:
    """4. Class: today's level against the newest 3 levels, best figure at today's or higher."""
    return [
        (pl.col("class_today").first() - _newest("level", 3).mean()).alias("class_change"),
        pl.col("fig").filter(pl.col("level") >= pl.col("class_today")).max()
        .alias("speed_best_at_class"),
    ]  # fmt: skip


def _form(day: date) -> list[pl.Expr]:
    """5. Form cycle: the newest 3 figures, the last start, the latest layoff, recent starts."""
    layoff = pl.col("days_since_prev").is_null() | (pl.col("days_since_prev") >= LAYOFF_DAYS)
    ago = (pl.lit(day) - pl.col("pp_date")).dt.total_days()
    return [
        _newest("fig", 3).alias("_fig3"),
        pl.col("pp_date").first().alias("_last_date"),
        pl.col("line_no").filter(layoff).min().alias("_layoff_line"),
        pl.len().alias("_n_lines"),
        (ago <= RECENT_DAYS).sum().alias("starts_90d"),
    ]


def _finish_form(day: date) -> list[pl.Expr]:
    figs = pl.col("_fig3")
    k = figs.list.len()
    days = (pl.lit(day) - pl.col("_last_date")).dt.total_days()
    return [
        # least squares against start order, oldest to newest (the list is newest first)
        pl.when(k == 3).then((figs.list.get(0) - figs.list.get(2)) / 2)
        .when(k == 2).then(figs.list.get(0) - figs.list.get(1)).alias("speed_slope3"),
        days.alias("days_since_last"),
        pl.when(pl.col("_last_date").is_null()).then(None).when(days >= LAYOFF_DAYS).then(0)
        .otherwise(pl.coalesce("_layoff_line", "_n_lines")).alias("runs_since_layoff"),
    ]  # fmt: skip


_FIT_LINE = {"dist": _dist_band(pl.col("distance_m")), "surface": pl.col("surface"),
             "going": _going_band(pl.col("going")), "venue": pl.col("venue"),
             "course": pl.col("course_config")}  # fmt: skip


def _fit() -> list[pl.Expr]:
    """6. Fit: per dimension, starts matching today, their top-3 count and figures."""
    exprs = [pl.col("fig").mean().alias("_fig_mean")]
    for d in FIT_DIMS:
        match = (_FIT_LINE[d] == pl.col(f"t_{d}")).fill_null(False)
        exprs += [
            match.sum().alias(f"fit_{d}_n"),
            (match & (pl.col("finish_pos") <= 3).fill_null(False)).sum().alias(f"fit_{d}_top3"),
            (match & pl.col("fig").is_not_null()).sum().alias(f"_fit_{d}_m"),
            pl.col("fig").filter(match).mean().alias(f"_fit_{d}_mean"),
        ]
    return exprs


def _finish_fit() -> list[pl.Expr]:
    exprs = []
    for d in FIT_DIMS:
        m = pl.col(f"_fit_{d}_m")
        shrunk = m / (m + FIT_PRIOR) * (pl.col(f"_fit_{d}_mean") - pl.col("_fig_mean"))
        exprs.append(
            pl.when(pl.col("speed_n") == 0).then(None).when(m == 0).then(0.0).otherwise(shrunk)
            .alias(f"fit_{d}")
        )  # fmt: skip
    return exprs


def _last_start() -> list[pl.Expr]:
    return [pl.col("actual_wt_kg").first().alias("_last_wt"),
            pl.col("jockey_key").first().alias("_last_jockey")]  # fmt: skip


def _condition() -> list[pl.Expr]:
    """7. Condition: body-weight change; weight carried and jockey against the last start."""
    return [
        pl.col("horse_wt_change_kg").alias("body_wt_change"),
        (pl.col("actual_wt_kg") - pl.col("_last_wt")).alias("wt_carried_change"),
        (pl.col("jockey_key") != pl.col("_last_jockey")).alias("jockey_change"),
    ]


def _connection_rates(out: pl.DataFrame, *, cut: date) -> pl.DataFrame:
    """7. Connections: jockey and trainer win rates over the declared runners on the race days
    cut - 364 .. cut, shrunk by RATE_PRIOR_RUNS runs toward that window's overall rate."""
    window = _read(
        "runners", ["jockey_key", "trainer_key", "finish_pos", "is_scratched"],
        cut - timedelta(days=RATE_WINDOW_DAYS - 1), cut,
    ).filter(~pl.col("is_scratched").fill_null(False))  # fmt: skip
    won = (pl.col("finish_pos") == 1).fill_null(False)
    if window.height == 0:
        return out.with_columns(pl.lit(None, pl.Float64).alias(f"{w}_rate") for w in WHO)
    p0 = float(window.select(won.mean()).item())
    for who in WHO:
        key = f"{who}_key"
        rides = window.group_by(key).agg(pl.len().alias("_r"), won.sum().alias("_w"))
        rate = (pl.col("_w").fill_null(0) + RATE_PRIOR_RUNS * p0) / (
            pl.col("_r").fill_null(0) + RATE_PRIOR_RUNS
        )
        out = (
            out.join(rides, on=key, how="left")
            .with_columns(
                pl.when(pl.col(key).is_null()).then(None).otherwise(rate)
                .alias(f"{who}_rate")
            )
            .drop("_r", "_w")
        )  # fmt: skip
    return out


def _draw_group(draw: pl.Expr, n: pl.Expr) -> pl.Expr:
    return (
        pl.when(3 * (draw - 1) <= n - 1)
        .then(0)
        .when(3 * (draw - 1) <= 2 * (n - 1))
        .then(1)
        .when(draw.is_not_null())
        .then(2)
    )


def _draw_bias(out: pl.DataFrame, *, train_end: date) -> pl.DataFrame:
    """8. Draw: (wins + 2) / (sum of 1/n + 2) for today's draw third at today's (venue, surface,
    distance) over the in-fold runners there; null with none."""
    course = ["venue", "surface", "distance_m"]
    out = out.with_columns(_draw_group(pl.col("draw"), pl.len().over("race_id")).alias("_group"))
    todays = out.select(course).unique().drop_nulls().rows()
    bias = pl.DataFrame(
        schema={
            "venue": pl.String,
            "surface": pl.String,
            "distance_m": pl.Int64,
            "_group": pl.Int32,
            "draw_bias": pl.Float64,
        }  # fmt: skip
    )
    if todays:
        where = functools.reduce(operator.or_, [
            (ds.field("venue") == v) & (ds.field("surface") == s) & (ds.field("distance_m") == d)
            for v, s, d in todays
        ])  # fmt: skip
        races = _read("races", ["race_id", "race_date", *course], HISTORY_START, train_end,
                      where=where)  # fmt: skip
        if races.height:
            runners = _read(
                "runners", ["race_id", "draw", "finish_pos", "is_scratched"],
                races.get_column("race_date").min(), train_end,
                where=_is_in("race_id", races.get_column("race_id").unique().to_list()),
            ).filter(~pl.col("is_scratched").fill_null(False))  # fmt: skip
            n = pl.len().over("race_id")
            bias = (
                runners.with_columns(
                    _draw_group(pl.col("draw"), n).alias("_group"), (1 / n).alias("_e")
                )
                .join(races.drop("race_date"), on="race_id")
                .drop_nulls("_group")
                .group_by([*course, "_group"])
                .agg(
                    ((pl.col("finish_pos") == 1).fill_null(False).sum() + DRAW_PRIOR)
                    .truediv(pl.col("_e").sum() + DRAW_PRIOR)
                    .alias("draw_bias")
                )
            )  # fmt: skip
    cast = [pl.col(c).cast(out.schema[c]) for c in course] + [pl.col("_group").cast(pl.Int32)]
    return out.with_columns(pl.col("_group").cast(pl.Int32)).join(
        bias.with_columns(cast), on=[*course, "_group"], how="left"
    )


# ---------------------------------------------------------------------------------------------
# The handicap score: a ridge conditional logit on the winner, fitted in fold only
# ---------------------------------------------------------------------------------------------

GRAD_TOL = 1e-6  # on the mean gradient per race
HESSIAN_CHUNK_ROWS = 50_000


@dataclass(frozen=True, eq=False)
class HandicapScore:
    """A fitted score: P(i wins) proportional to exp(sum_f coef[f] * (x_f - center_f)) within a
    race, a null x_f contributing 0 (the training mean)."""

    features: tuple[str, ...]
    coef: dict[str, float]
    center: dict[str, float]
    alpha: float
    train_start: date
    train_end: date
    n_races: int

    def predict(self, frame: pl.DataFrame) -> pl.Series:
        """In-race win probabilities, one per row of `frame` (grouped by its race_id)."""
        codes = pd.factorize(frame.get_column("race_id").to_numpy())[0]
        return pl.Series("handicap_score", self._predict(_design(frame, self.features), codes))

    def _predict(self, x: np.ndarray, codes: np.ndarray) -> np.ndarray:
        center = np.array([self.center[f] for f in self.features])
        beta = np.array([self.coef[f] for f in self.features])
        z = x - center
        np.nan_to_num(z, copy=False, nan=0.0)
        return _group_softmax(z @ beta, codes) if len(codes) else np.empty(0)


def _design(
    frame: pl.DataFrame | pl.LazyFrame, features: Sequence[str], n: int | None = None
) -> np.ndarray:
    """The feature matrix as float64, a null as NaN; Booleans as 0/1. A LazyFrame (of `n` rows)
    is collected one column at a time, so only the matrix is held."""
    lazy = isinstance(frame, pl.LazyFrame)
    x = np.empty((frame.height if n is None else n, len(features)))
    for j, f in enumerate(features):
        col = frame.select(f).collect().to_series() if lazy else frame.get_column(f)
        if not (col.dtype.is_numeric() or col.dtype == pl.Boolean):
            raise TypeError(f"feature {f} is {col.dtype}, not numeric or Boolean")
        x[:, j] = col.cast(pl.Float64).fill_null(np.nan).to_numpy()
    return x


def _refuse_holdout(race_dates: pl.Series) -> None:
    """A row on a holdout day (D14) is a ValueError."""
    held = [d for d in race_dates.unique().to_list() if is_holdout_day(COUNTRY, d)]
    if held:
        raise ValueError(
            f"{len(held)} holdout race day(s) offered for fitting or scoring, the first "
            f"{min(held).isoformat()} (D14)"
        )


def _arrays(frame: pl.DataFrame, features: Sequence[str]):
    codes = pd.factorize(frame.get_column("race_id").to_numpy())[0]
    won = frame.get_column("won").fill_null(False).to_numpy().astype(bool)
    dates = frame.get_column("race_date").to_numpy()
    return _design(frame, features), won, codes, dates


def fit_handicap_score(
    frame: pl.DataFrame, *, features: Sequence[str] | None = None, alpha: float = 1.0
) -> HandicapScore:
    """The ridge conditional logit on `won` over `features` (default SCORE_FEATURES): the summed
    negative log-likelihood of each race's winner plus alpha / 2 times the squared coefficients
    of the standardised features. Races without a winner are skipped."""
    features = tuple(SCORE_FEATURES if features is None else features)
    _refuse_holdout(frame.get_column("race_date"))
    x, won, codes, dates = _arrays(frame, features)
    return _fit_rows(x, won, codes, dates, np.arange(frame.height), features, alpha)


def _fit_rows(x, won, codes, dates, rows, features, alpha) -> HandicapScore:
    """The fit on the given rows of the flat arrays (see fit_handicap_score)."""
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    rows = np.asarray(rows)
    winners = np.bincount(codes[rows], weights=won[rows], minlength=codes.max(initial=-1) + 1)
    rows = rows[winners[codes[rows]] > 0]
    if len(rows) == 0:
        raise ValueError("no race with a winner to fit")
    rows = rows[np.argsort(codes[rows], kind="stable")]
    c, y = codes[rows], won[rows]
    starts, sizes = _segments(c)
    n_races = len(starts)
    w = winners[c[starts]]
    w_rows = np.repeat(w, sizes)

    z = x[rows]
    with np.errstate(invalid="ignore"):
        center = np.nanmean(z, axis=0) if len(z) else np.zeros(z.shape[1])
        scale = np.nanstd(z, axis=0)
    center = np.nan_to_num(center, nan=0.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    z -= center
    z /= scale
    np.nan_to_num(z, copy=False, nan=0.0)

    def fun(beta):
        value, grad, _ = _nll(beta, z, y, w, w_rows, starts, sizes, 1)
        return value + 0.5 * alpha * beta @ beta, grad + alpha * beta

    def hess(beta):
        _, _, p = _nll(beta, z, y, w, w_rows, starts, sizes, 1)
        h = alpha * np.eye(z.shape[1])
        per = max(1, HESSIAN_CHUNK_ROWS // max(1, int(sizes.mean())))
        for a in range(0, n_races, per):
            b = min(a + per, n_races)
            r0, r1 = starts[a], (starts[b] if b < n_races else len(z))
            zc = z[r0:r1]
            zp = zc * p[r0:r1, None]
            h += (zp * w_rows[r0:r1, None]).T @ zc
            m = np.add.reduceat(zp, starts[a:b] - r0, axis=0)
            h -= (m * w[a:b, None]).T @ m
        return h

    result = minimize(
        fun, np.zeros(z.shape[1]), jac=True, hess=hess, method="trust-exact",
        options={"gtol": 1e-9},
    )  # fmt: skip
    converged = result.success or np.max(np.abs(result.jac), initial=0.0) < GRAD_TOL * n_races
    if not converged or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"the handicap score fit did not converge: {result.message}")
    beta = result.x / scale
    stamps = pd.to_datetime(dates[rows])
    return HandicapScore(
        features=tuple(features),
        coef={f: float(b) for f, b in zip(features, beta, strict=True)},
        center={f: float(m) for f, m in zip(features, center, strict=True)},
        alpha=float(alpha),
        train_start=stamps.min().date(),
        train_end=stamps.max().date(),
        n_races=int(n_races),
    )


def oof_handicap_score(
    frame: pl.DataFrame,
    *,
    folds: Sequence[Fold],
    features: Sequence[str] | None = None,
    alpha: float = 1.0,
) -> pl.DataFrame:
    """Out-of-fold `handicap_score`: fold k's test rows (positions into `frame`) through the fit on
    its training rows; null for a fold whose training rows hold no race with a winner."""
    features = tuple(SCORE_FEATURES if features is None else features)
    _refuse_holdout(frame.get_column("race_date"))
    x, won, codes, dates = _arrays(frame, features)
    keys = frame.select("race_id", pl.col("horse_no").cast(pl.Int64))
    return _oof_rows(x, won, codes, dates, keys, folds, features, alpha)


def _oof_rows(x, won, codes, dates, keys: pl.DataFrame, folds, features, alpha) -> pl.DataFrame:
    """oof_handicap_score on flat arrays; `keys` holds race_id and horse_no row for row."""
    parts = []
    for k, fold in enumerate(folds):
        test = np.asarray(fold.test_idx)
        if len(test) == 0:
            continue
        try:
            model = _fit_rows(x, won, codes, dates, fold.train_idx, features, alpha)
        except ValueError:
            score = np.full(len(test), np.nan)
        else:
            score = model._predict(x[test], pd.factorize(codes[test])[0])
        parts.append(
            keys[test].with_columns(
                pl.lit(k, pl.Int64).alias("fold"),
                pl.Series("handicap_score", score).fill_nan(None),
            )
        )
    if not parts:
        return pl.DataFrame(
            schema={
                "race_id": pl.String,
                "horse_no": pl.Int64,
                "fold": pl.Int64,
                "handicap_score": pl.Float64,
            }  # fmt: skip
        )
    return pl.concat(parts)


# ---------------------------------------------------------------------------------------------
# The per-day table
# ---------------------------------------------------------------------------------------------


def handicap_dir() -> Path:
    """#129's table (the interim class par), kept; the builder no longer writes here."""
    return data_dir() / "features" / "jra" / "handicap"


def handicap_96_dir() -> Path:
    """The builder's default destination: the table with #96's speed figure (#219), beside #129's
    (`handicap_dir`, which `track_bias_report` and `same_day_bias` read) and not over it."""
    return data_dir() / "features" / "jra" / "handicap-96-v2"


COMPLETE = "_complete.json"  # written after the score pass; a table without it is not finished


def table_complete(
    out_dir: Path, date_from: date, date_to: date, *, speed_at_start_day: bool = False
) -> bool:
    """Whether `out_dir` holds a finished build of exactly this range and speed mode (its marker
    names both) with every file it counted."""
    try:
        marker = json.loads((out_dir / COMPLETE).read_text())
    except (OSError, ValueError):
        return False
    want = {
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "speed_at_start_day": speed_at_start_day,
    }
    return all(marker.get(k) == v for k, v in want.items()) and marker.get("days") == len(
        list(out_dir.glob("*.parquet"))
    )


def _write(frame: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def build_handicap_table(
    date_from: date,
    date_to: date,
    *,
    scheme: FoldScheme | None = None,
    out_dir: Path | None = None,
    speed_at_start_day: bool = False,
) -> dict[date, Path]:
    """One Parquet file of handicap factors plus the out-of-fold `handicap_score` per non-holdout
    race day in the inclusive range, factors built one day at a time as of the race day, written
    under `out_dir` (default `handicap_96_dir()`, never #129's `handicap_dir()`). By default the
    factors are `handicap_factors(race, as_of=race day)`: #96's model fitted to the day's cut, so
    every past start is in the fit and has no day variant. With `speed_at_start_day` each past
    start's figure is #96's as of its own race day, day variant included (`_own_day_expected`).
    `COMPLETE`, written last after the score pass, names the range and the mode."""
    races = _read("races", ["race_id", "race_date"], date_from, date_to)
    by_day = races.group_by("race_date").agg(pl.col("race_id").sort())
    out_dir = handicap_96_dir() if out_dir is None else out_dir
    (out_dir / COMPLETE).unlink(missing_ok=True)  # a rebuild is not finished until it says so
    race_table = _race_table(date_to)  # #96's table once; each day reads the races up to its cut
    own_day = _own_day_expected(race_table) if speed_at_start_day else None
    written: dict[date, Path] = {}
    for day, race_ids in sorted(by_day.iter_rows()):
        if is_holdout_day(COUNTRY, day):
            continue
        cut = day - timedelta(days=1)
        frame = _factors_for(
            day, race_ids, cut=cut, train_end=cut, race_table=race_table, own_day=own_day
        ).with_columns(pl.lit(None, pl.Float64).alias("handicap_score"))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{day.isoformat()}.parquet"
        _write(frame, path)
        written[day] = path
        del frame
    if not written:
        return written

    scores = _oof_over(written, date_from, date_to, scheme or load_country_scheme(COUNTRY))
    for path in written.values():
        part = pl.read_parquet(path).drop("handicap_score")
        part = part.join(scores, on=["race_id", "horse_no"], how="left", maintain_order="left")
        _write(part, path)
        del part
    marker = {
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "days": len(written),
        "speed_at_start_day": speed_at_start_day,
    }
    tmp = out_dir / (COMPLETE + ".tmp")
    tmp.write_text(json.dumps(marker))
    tmp.replace(out_dir / COMPLETE)
    return written


def _oof_over(
    written: dict[date, Path], date_from: date, date_to: date, scheme: FoldScheme
) -> pl.DataFrame:
    """The out-of-fold score of every written row over `scheme`'s folds of their race days, the
    feature matrix read from the written files one column at a time."""
    scan = pl.scan_parquet([str(p) for _, p in sorted(written.items())])
    keys = scan.select("race_id", "race_date", pl.col("horse_no").cast(pl.Int64)).collect()
    won = _read("runners", ["race_id", "horse_no", "finish_pos"], date_from, date_to).select(
        "race_id", pl.col("horse_no").cast(pl.Int64), (pl.col("finish_pos") == 1).alias("won")
    )
    keys = keys.join(won, on=["race_id", "horse_no"], how="left", maintain_order="left")
    _refuse_holdout(keys.get_column("race_date"))
    x = _design(scan, SCORE_FEATURES, n=keys.height)
    _, won_rows, codes, dates = _arrays(keys, ())
    folds = walk_forward(keys.get_column("race_date").to_list(), scheme=scheme)
    oof = _oof_rows(
        x, won_rows, codes, dates, keys.select("race_id", "horse_no"), folds, SCORE_FEATURES, 1.0
    )
    return oof.select("race_id", "horse_no", "handicap_score")
