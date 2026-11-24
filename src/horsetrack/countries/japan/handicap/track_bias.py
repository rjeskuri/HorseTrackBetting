"""Within-day track bias from the races already run that day (#138).

Every function takes store-shaped polars frames (`races`, `runners`; docs/specs/store.md) and reads
nothing else: the caller scopes the store. The rulings are pinned in
tests/countries/japan/handicap/test_track_bias.py.

- Score: a finisher in place `pos` among `nf` finishers (nf >= 2) scores (nf - pos) / (nf - 1) -
  0.5, the share of rivals beaten minus one half; 0 means no bias. Each finisher is one draw
  observation at its draw third and, with a first-corner position, one style observation, both by
  #129's thirds rules over the race's declared runners.
- `fit_bias_prior`: each level's mean score per cell (venue, surface, distance, course_config,
  going, rail_setting) on the race days on or before `train_end`, shrunk by PRIOR_RACES
  pseudo-races toward (venue, surface, distance), that toward (venue, surface), and that toward 0;
  a cell with no history takes its coarsest known parent, and no history at the venue and surface
  gives 0. `rail_setting` (#222, JRA's A/B/C/D rail position, the store's races column) is part of
  the full cell only: a null one is a cell of its own, so a race with a letter reaches null-rail
  history only through the (venue, surface, distance) parent, and a races frame without the
  column is read as all null. The
  observation variance and (when not given) `day_sd`, the sd of a day's deviation from the
  prior, are moment estimates over the history's (day, venue, surface) groups; the day variance
  is measured against each group's variance with no day effect, a race's places shuffled among
  its finishers (so a level's scores in one race are drawn without replacement).
- `within_day_posterior`: today's deviation at each level, N(0, day_sd^2), updated conjugately by
  the scores of the races of the target's day, venue and surface with a smaller race_no whose
  results were published strictly before `decision_time`, each score taken against the prior mean
  of its own race's cell. The level's posterior mean is the target cell's prior mean plus the
  deviation's posterior mean.
- `track_bias_features`: the posterior per declared runner of the target race.

No layer here is adopted (D16); these are features and a measurement scaffold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.handicap.factors import COUNTRY, STYLES, _draw_group, _style_code

DRAW_LEVELS = ("inner", "middle", "outer")
LEVELS = tuple([("draw", lv) for lv in DRAW_LEVELS] + [("style", lv) for lv in STYLES])
CELL = ("venue", "surface", "distance_m", "course_config", "going", "rail_setting")
DEPTHS = (2, 3, len(CELL))  # (venue, surface) -> + distance -> the full cell
PRIOR_RACES = 10  # pseudo-races shrinking a cell's mean toward its parent's
MIN_DAY_SD = 0.01  # floor of an estimated day_sd
UNIFORM_VAR = 1 / 12  # the observation variance when the history has none to estimate it from


@dataclass(frozen=True, eq=False)
class BiasPrior:
    """A fitted history prior; opaque to callers."""

    train_end: date
    day_sd: float
    noise_var: float
    means: dict[tuple, float]  # (cell prefix, factor, level) -> shrunk mean, every depth

    def mean(self, cell: tuple, factor: str, level: str) -> float:
        for depth in reversed(DEPTHS):
            got = self.means.get((cell[:depth], factor, level))
            if got is not None:
                return got
        return 0.0


def _with_rail(races: pl.DataFrame) -> pl.DataFrame:
    """`races` with a `rail_setting` column, all null when it has none."""
    if "rail_setting" in races.columns:
        return races
    return races.with_columns(pl.lit(None, pl.String).alias("rail_setting"))


def _observations(runners: pl.DataFrame) -> pl.DataFrame:
    """One row per (finisher, factor) with its level and score y; race_id kept, with the race's
    finisher count nf and the population variance s2 of its finishers' scores."""
    n = pl.len().over("race_id")
    nf = pl.col("finish_pos").is_not_null().sum().over("race_id")
    scored = (
        runners.filter(~pl.col("is_scratched").fill_null(False))
        .with_columns(
            _draw_group(pl.col("draw"), n).alias("_d"),
            _style_code(pl.col("running_positions").list.get(0, null_on_oob=True), n).alias("_s"),
            pl.when(nf >= 2).then((nf - pl.col("finish_pos")) / (nf - 1) - 0.5).alias("y"),
            nf.alias("nf"),
        )
        .with_columns(pl.col("y").var(ddof=0).over("race_id").alias("s2"))
        .filter(pl.col("y").is_not_null())
    )
    parts = [
        scored.select(
            "race_id",
            pl.lit(factor).alias("factor"),
            pl.col(code)
            .replace_strict(dict(enumerate(names)), return_dtype=pl.String)
            .alias("level"),
            "y",
            "nf",
            "s2",
        ).drop_nulls("level")
        for factor, code, names in (("draw", "_d", DRAW_LEVELS), ("style", "_s", STYLES))
    ]
    return pl.concat(parts)


def fit_bias_prior(
    races: pl.DataFrame, runners: pl.DataFrame, *, train_end: date, day_sd: float | None = None
) -> BiasPrior:
    """The history prior on the race days on or before `train_end` (see the module doc)."""
    if day_sd is not None and not day_sd > 0:
        raise ValueError(f"day_sd must be positive, got {day_sd}")
    races = _with_rail(races).filter(pl.col("race_date") <= train_end)
    runners = runners.filter(pl.col("race_date") <= train_end)
    days = set(races.get_column("race_date").to_list()) | set(runners.get_column("race_date"))
    held = sorted(d for d in days if d is not None and is_holdout_day(COUNTRY, d))
    if held:
        raise ValueError(
            f"{len(held)} holdout race day(s) on or before train_end, the first "
            f"{held[0].isoformat()} (D14)"
        )
    obs = _observations(runners).join(
        races.select("race_id", "race_date", *CELL), on="race_id", how="inner"
    )
    means: dict[tuple, float] = {}
    up = None  # the parent depth
    for depth in DEPTHS:
        stats = obs.group_by([*CELL[:depth], "factor", "level"]).agg(
            pl.col("y").mean(), pl.col("race_id").n_unique().alias("n")
        )
        for *cell, factor, level, ybar, n in stats.iter_rows():
            cell = tuple(cell)
            parent = 0.0 if up is None else means[(cell[:up], factor, level)]
            means[(cell, factor, level)] = (n * ybar + PRIOR_RACES * parent) / (n + PRIOR_RACES)
        up = depth
    noise_var, tau = _day_spread(obs, means)
    return BiasPrior(
        train_end=train_end,
        day_sd=float(day_sd) if day_sd is not None else tau,
        noise_var=noise_var,
        means=means,
    )


def _day_spread(obs: pl.DataFrame, means: dict[tuple, float]) -> tuple[float, float]:
    """(observation variance, day sd) of each score's deviation e from its cell's prior mean over
    the history's (day, venue, surface, level) groups, by moments. The observation variance is
    the pooled within-group variance of e. The day variance is the mean over groups of d^2 - v,
    d the group's mean e and v its variance with no day effect. A level's scores in one race are
    not independent (a race's places are a permutation), so v is built from races: a race's mean
    e varies around d with variance lam * w0, w0 = s2 (nf - c) / ((nf - 1) c) for its c scores at
    the level among nf finishers (the variance of c places drawn without replacement, s2 the
    race's score variance), and lam, one per level, is fitted from how a group's race means
    spread around d, which a day effect shared by the group's races leaves untouched."""
    if obs.height == 0:
        return UNIFORM_VAR, MIN_DAY_SD
    prior = [
        means[(tuple(r[: len(CELL)]), r[len(CELL)], r[len(CELL) + 1])]
        for r in obs.select(*CELL, "factor", "level").iter_rows()
    ]
    key = ["race_date", "venue", "surface", "factor", "level"]
    obs = obs.with_columns((pl.col("y") - pl.Series(prior)).alias("e"))
    groups = obs.group_by(key).agg(
        pl.len().alias("m"),
        pl.col("e").mean().alias("d"),
        ((pl.col("e") - pl.col("e").mean()) ** 2).sum().alias("ss"),
    )
    dof = (groups.get_column("m") - 1).sum()
    noise_var = groups.get_column("ss").sum() / dof if dof > 0 else 0.0
    if not noise_var > 0:
        noise_var = UNIFORM_VAR

    a, w0 = pl.col("c") / pl.col("m"), pl.col("w0")
    races = (
        obs.group_by(*key, "race_id")
        .agg(pl.len().alias("c"), pl.col("e").mean(), pl.col("nf").first(), pl.col("s2").first())
        .with_columns(
            (
                pl.col("s2") * (pl.col("nf") - pl.col("c")) / ((pl.col("nf") - 1) * pl.col("c"))
            ).alias("w0")
        )
        .join(groups.select(*key, "m", "d"), on=key)
    )
    by_group = races.group_by(key).agg(
        ((pl.col("e") - pl.col("d")) ** 2).sum().alias("q"),
        (w0 * (1 - 2 * a)).sum().alias("q0"),
        pl.len().alias("r"),
        (a**2 * w0).sum().alias("v0"),
        pl.col("d").first(),
    )
    by_level = by_group.group_by("factor", "level").agg(
        pl.col("q").sum(), (pl.col("q0") + pl.col("r") * pl.col("v0")).sum().alias("eq")
    )
    lam = by_level.select(
        "factor",
        "level",
        pl.when(pl.col("eq") > 0).then(pl.col("q") / pl.col("eq")).otherwise(1.0).alias("lam"),
    )
    tau2 = (
        by_group.join(lam, on=["factor", "level"])
        .select((pl.col("d") ** 2 - pl.col("lam") * pl.col("v0")).mean())
        .item()
    )
    return float(noise_var), max(math.sqrt(max(tau2, 0.0)), MIN_DAY_SD)


def _aware(t: datetime, what: str) -> None:
    if not isinstance(t, datetime) or t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"{what} must be a timezone-aware datetime, got {t!r}")


def _posterior(
    prior: BiasPrior,
    races: pl.DataFrame,
    runners: pl.DataFrame,
    race_id: str,
    decision_time: datetime,
    published: Mapping[str, datetime],
) -> pl.DataFrame:
    _aware(decision_time, "decision_time")
    for rid, when in published.items():
        _aware(when, f"the publication time of {rid}")
    races = _with_rail(races)
    target = races.filter(pl.col("race_id") == race_id)
    if target.height != 1:
        raise ValueError(f"{race_id} is not exactly one row of races ({target.height})")
    row = target.row(0, named=True)
    day = row["race_date"]
    if is_holdout_day(COUNTRY, day):
        raise ValueError(f"{race_id} is on a holdout race day, {day.isoformat()} (D14)")
    if prior.train_end >= day:
        raise ValueError(
            f"the prior is fitted through {prior.train_end.isoformat()}, on or after the target's "
            f"race day {day.isoformat()}"
        )
    cell = tuple(row[c] for c in CELL)

    same_course = races.filter(
        (pl.col("race_date") == day)
        & (pl.col("venue") == row["venue"])
        & (pl.col("surface") == row["surface"])
        & (pl.col("race_no") < row["race_no"])
    )
    entered = [
        r
        for r in same_course.select("race_id", *CELL).iter_rows()
        if r[0] in published and published[r[0]] < decision_time
    ]
    cell_of = {r[0]: tuple(r[1:]) for r in entered}
    obs = _observations(runners.filter(pl.col("race_id").is_in(list(cell_of))))
    evidence: dict[tuple[str, str], list[float]] = {key: [] for key in LEVELS}
    obs = obs.select("race_id", "factor", "level", "y").sort("race_id", "factor", "level", "y")
    for rid, factor, level, y in obs.iter_rows():
        evidence[(factor, level)].append(y - prior.mean(cell_of[rid], factor, level))

    tau, var = prior.day_sd, prior.noise_var
    rows = []
    for factor, level in LEVELS:
        e = evidence[(factor, level)]
        prior_mean = prior.mean(cell, factor, level)
        if e:
            precision = 1 / tau**2 + len(e) / var
            mean, sd = prior_mean + sum(e) / var / precision, math.sqrt(1 / precision)
        else:
            mean, sd = prior_mean, tau
        rows.append((factor, level, prior_mean, tau, mean, sd, len(entered), len(e)))
    return pl.DataFrame(
        rows,
        schema={
            "factor": pl.String,
            "level": pl.String,
            "prior_mean": pl.Float64,
            "prior_sd": pl.Float64,
            "mean": pl.Float64,
            "sd": pl.Float64,
            "n_races": pl.Int64,
            "n_obs": pl.Int64,
        },  # fmt: skip
        orient="row",
    )


def within_day_posterior(
    prior: BiasPrior,
    races: pl.DataFrame,
    runners: pl.DataFrame,
    *,
    race_id: str,
    decision_time: datetime,
    published: Mapping[str, datetime],
) -> pl.DataFrame:
    """Seven rows, one per (factor, level): today's bias at `race_id` as of `decision_time`."""
    return _posterior(prior, races, runners, race_id, decision_time, published)


def track_bias_features(
    prior: BiasPrior,
    races: pl.DataFrame,
    runners: pl.DataFrame,
    *,
    race_id: str,
    decision_time: datetime,
    published: Mapping[str, datetime],
) -> pl.DataFrame:
    """One row per declared runner of `race_id`: its draw group's adjustment and the four style
    adjustments, each with its sd, and the number of races the update used."""
    post = _posterior(prior, races, runners, race_id, decision_time, published)
    by = {(f, lv): (m, sd) for f, lv, m, sd in post.select("factor", "level", "mean", "sd").rows()}
    n_races = post.get_column("n_races").item(0)
    draw = pl.DataFrame(
        [(lv, *by[("draw", lv)]) for lv in DRAW_LEVELS],
        schema={"draw_group": pl.String, "draw_adj": pl.Float64, "draw_adj_sd": pl.Float64},
        orient="row",
    )
    declared = runners.filter(
        (pl.col("race_id") == race_id) & ~pl.col("is_scratched").fill_null(False)
    )
    group = _draw_group(pl.col("draw"), pl.len())
    return (
        declared.select(
            "race_id",
            pl.col("horse_no").cast(pl.Int64),
            "horse_id",
            group.replace_strict(dict(enumerate(DRAW_LEVELS)), return_dtype=pl.String).alias(
                "draw_group"
            ),
        )
        .join(draw, on="draw_group", how="left", maintain_order="left")
        .with_columns(
            *[
                pl.lit(by[("style", lv)][k], pl.Float64).alias(f"style_adj_{lv}{sfx}")
                for lv in STYLES
                for k, sfx in ((0, ""), (1, "_sd"))
            ],
            pl.lit(n_races, pl.Int64).alias("bias_races_n"),
        )
        .sort("horse_no")
    )
