"""Development-data measurement for #138: does the within-day update beat the prior alone?

`track_bias_report(date_from, date_to)` scores each #70 fold's races out of fold. The bias prior is
fitted on the days up to the fold's `train_end`; each runner gets two feature pairs, the prior
alone (nothing published) and prior + within-day update: `draw_adj` and `style_adj`, the
adjustment of its #129 running style (null without one). Both pairs go through the same form, a
ridge conditional logit on the winner (#129's `fit_handicap_score`, alpha 1), fitted for fold k on
the out-of-fold rows of folds before k that are among fold k's training days, so outside its
embargo (fold 0 only trains). Scored as #69 defines it: winner log
loss per race, a dead heat's k winners each weighted 1/k, and a paired bootstrap over meetings.

A race's result counts as published at its logged `result_published_utc` (the odds logger's first
sighting, ticket #225) where the store has one, else at the proxy, its local post time +
PUBLISH_LAG (`published_times`); `publication_lag_summary` measures that proxy against the logged
times whose bracket (`result_unpublished_utc`, the last poll that did not show the result) is
narrow, leaving out late sightings and wide brackets. The decision time is the target's post
time - DECISION_LEAD. Holdout days are never read (#114's scoped reader); only aggregates are
returned and nothing is written. Measurement only: nothing here is adopted (D16).

`versus=` (#256) compares two day_sd modes head to head: each fold also fits a prior with
`day_sd=versus` on the same `train_end`, the update variant built from it is scored through its
own form (fitted on the same training rows) on the same races, and the per-race difference
against `day_sd`'s update gets the same meeting bootstrap.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.paths import processed_dir
from horsetrack.countries.japan.handicap.factors import (
    COUNTRY,
    STYLES,
    _read,
    fit_handicap_score,
    handicap_dir,
)
from horsetrack.countries.japan.handicap.pp import JST
from horsetrack.countries.japan.handicap.track_bias import (
    CELL,
    DaySd,
    fit_bias_prior,
    track_bias_features,
)
from horsetrack.countries.japan.models.v0.store import SOURCE

PUBLISH_LAG = timedelta(minutes=30)
DECISION_LEAD = timedelta(minutes=2)
LOGGED = "result_published_utc"
BOUND = "result_unpublished_utc"  # the last logger poll that did not show the result
RACE_COLUMNS = ("race_id", "race_date", "meeting_id", "race_no", "post_time_local", LOGGED, BOUND,
                *CELL)  # fmt: skip
RUNNER_COLUMNS = ("race_id", "race_date", "horse_no", "horse_id", "draw", "finish_pos",
                  "running_positions", "is_scratched")  # fmt: skip
OPTIONAL_RACE_COLUMNS = (
    "rail_setting",
    LOGGED,
    BOUND,
)  # a store built before #222 / #225 lacks them
VARIANTS = ("prior", "update")
VERSUS_VARIANT = "update_versus"
NO_VERSUS = object()  # `versus`'s default: None already means the pooled estimate
FEATURES = ("draw_adj", "style_adj")


def _read_races(date_from: date, date_to: date) -> pl.DataFrame:
    """The races, leaving out the optional columns the store was built without (#253): the rest
    of the module reads a missing column as null (`_column`, `track_bias._with_rail`)."""
    have = set(pq.read_schema(processed_dir(SOURCE) / "races.parquet").names)
    columns = [c for c in RACE_COLUMNS if c in have or c not in OPTIONAL_RACE_COLUMNS]
    return _read("races", columns, date_from, date_to)


def _post(day: date, hhmm: str | None) -> datetime | None:
    if not hhmm:
        return None
    hh, mm = hhmm.split(":")
    return datetime.combine(day, time(int(hh), int(mm)), tzinfo=JST)


def _column(races: pl.DataFrame, name: str) -> list:
    """The column's values, all None when the frame has no such column (an older store)."""
    return races.get_column(name).to_list() if name in races.columns else [None] * races.height


def _posts_and_logged(races: pl.DataFrame) -> list[tuple[str, datetime | None, datetime | None]]:
    """(race_id, local post time, logged publication time) per race; a frame without the logged
    column (a store built before #225) has none logged."""
    return [
        (rid, _post(day, hhmm), at)
        for (rid, day, hhmm), at in zip(
            races.select("race_id", "race_date", "post_time_local").rows(),
            _column(races, LOGGED),
            strict=True,
        )
    ]


def published_times(races: pl.DataFrame) -> dict[str, datetime]:
    """race_id -> when its result counts as published: the logged `result_published_utc` where
    there is one (it stands without a post time), else post + PUBLISH_LAG; a race with neither
    is left out."""
    out = {}
    for rid, post, logged in _posts_and_logged(races):
        if logged is not None:
            out[rid] = logged
        elif post is not None:
            out[rid] = post + PUBLISH_LAG
    return out


def default_max_bracket_s() -> float:
    """Twice logger.yaml's `results.every_s`: a healthy bracket is every_s plus up to one round
    (results polls ride on the logging rounds); a wider one spans a failed poll."""
    return 2.0 * float(load_country_yaml(COUNTRY, "logger")["results"]["every_s"])


def publication_lag_summary(races: pl.DataFrame, max_bracket_s: float | None = None) -> dict:
    """Aggregates only of logged publication time - local post time, in minutes, over the races
    having both and a measured bracket: count, min, median, p90, max and the share within
    PUBLISH_LAG (the proxy's claim). A race is measured when its bracket, result_published_utc -
    result_unpublished_utc, is at most `max_bracket_s` seconds (default: default_max_bracket_s);
    `excluded_late` counts those with no lower bound (a late sighting, or a frame without the
    column), `excluded_wide` those with a wider bracket. None measured: `races` 0, the rest None."""
    limit = default_max_bracket_s() if max_bracket_s is None else float(max_bracket_s)
    lags: list[float] = []
    late = wide = 0
    for (_, post, logged), bound in zip(
        _posts_and_logged(races), _column(races, BOUND), strict=True
    ):
        if post is None or logged is None:
            continue
        if bound is None:
            late += 1
        elif (logged - bound).total_seconds() > limit:
            wide += 1
        else:
            lags.append((logged - post).total_seconds() / 60.0)
    counts = {"excluded_late": late, "excluded_wide": wide}
    proxy = PUBLISH_LAG.total_seconds() / 60.0
    if not lags:
        stats = dict.fromkeys(("min_minutes", "median_minutes", "p90_minutes", "max_minutes",
                               "share_within_30_minutes"))  # fmt: skip
        return {"races": 0, **stats, **counts}
    lags = np.array(lags)
    return {
        **counts,
        "races": int(lags.size),
        "min_minutes": float(lags.min()),
        "median_minutes": float(np.median(lags)),
        "p90_minutes": float(np.quantile(lags, 0.9)),
        "max_minutes": float(lags.max()),
        "share_within_30_minutes": float((lags <= proxy).mean()),
    }


def _runner_style(day: date) -> pl.DataFrame:
    path = handicap_dir() / f"{day.isoformat()}.parquet"
    if not path.exists():
        return pl.DataFrame(schema={"race_id": pl.String, "horse_no": pl.Int64, "style": pl.String})
    return pl.read_parquet(path, columns=["race_id", "horse_no", "style"]).with_columns(
        pl.col("horse_no").cast(pl.Int64)
    )


def _pick_style(suffix: str) -> pl.Expr:
    """The adjustment of the runner's own style; null without one."""
    first, *rest = STYLES
    expr = pl.when(pl.col("style") == first).then(pl.col(f"style_adj_{first}"))
    for lv in rest:
        expr = expr.when(pl.col("style") == lv).then(pl.col(f"style_adj_{lv}"))
    return expr.alias(f"style_adj_{suffix}")


def _day_rows(
    prior, races: pl.DataFrame, runners: pl.DataFrame, day: date, versus_prior=None
) -> list[pl.DataFrame]:
    """Both variants' features, the winner flag and the update's race count per declared runner of
    the day's races that have a post time; with `versus_prior`, also the update built from it
    (`update_versus`)."""
    posts = {
        rid: _post(day, hhmm) for rid, hhmm in races.select("race_id", "post_time_local").rows()
    }
    published = published_times(races)
    style = _runner_style(day)
    out = []
    for rid, meeting in races.select("race_id", "meeting_id").sort("race_id").rows():
        if posts[rid] is None:
            continue
        kw = dict(race_id=rid, decision_time=posts[rid] - DECISION_LEAD)
        cols = []
        builds = [("prior", prior, {}), ("update", prior, published)]
        if versus_prior is not None:
            builds.append((VERSUS_VARIANT, versus_prior, published))
        for variant, source, pub in builds:
            f = track_bias_features(source, races, runners, **kw, published=pub).join(
                style, on=["race_id", "horse_no"], how="left", maintain_order="left"
            )
            cols += [f.get_column("draw_adj").alias(f"draw_adj_{variant}")]
            cols += [f.select(_pick_style(variant)).to_series()]
        won = f.join(
            runners.select("race_id", pl.col("horse_no").cast(pl.Int64), "finish_pos"),
            on=["race_id", "horse_no"], how="left", maintain_order="left",
        ).get_column("finish_pos") == 1  # fmt: skip
        out.append(
            f.select("race_id", "horse_no", "bias_races_n").with_columns(
                pl.lit(day).alias("race_date"),
                pl.lit(meeting).alias("meeting_id"),
                won.fill_null(False).alias("won"),
                *cols,
            )  # fmt: skip
        )
    return out


def _by_level(sds: dict[tuple[str, str], float]) -> dict[str, float]:
    return {f"{factor}:{level}": sd for (factor, level), sd in sds.items()}


def _bootstrap(gain: np.ndarray, meeting: np.ndarray, n_boot: int, seed: int) -> dict:
    codes, inv = np.unique(meeting, return_inverse=True)
    sums = np.bincount(inv, weights=gain, minlength=len(codes))
    counts = np.bincount(inv, minlength=len(codes))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(codes), size=(n_boot, len(codes)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "mean": float(gain.mean()),
        "lower_95_one_sided": float(np.quantile(means, 0.05)),
        "ci_95": (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))),
        "races": int(len(gain)),
        "meetings": int(len(codes)),
    }


def track_bias_report(
    date_from: date,
    date_to: date,
    *,
    scheme: FoldScheme | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    day_sd: DaySd = None,
    versus: DaySd | object = NO_VERSUS,
) -> dict:
    """Aggregates only: per-variant log loss and pseudo-R² against equal chances, the update's
    gain over the prior with meeting-bootstrap bounds overall and by races entered, and per fold
    the priors' day sd per level ("factor:level"), the unfloored estimate behind it (when
    estimated) and the observation variance. `day_sd` (None or "pooled": the pooled estimate;
    "per_level"; a float or a per-level mapping, #224) goes to every fold's prior. `versus` (any
    `day_sd` value; the default NO_VERSUS adds nothing) adds the paired comparison of #256:
    `ll_update_versus`, `gain_update_vs_versus` (the bootstrap of ll_update_versus - ll_update per
    race, positive = `day_sd`'s update is better) and `day_sd_versus` (per fold, as `day_sd`);
    every other key is unchanged."""
    compare = versus is not NO_VERSUS
    variants = (*VARIANTS, VERSUS_VARIANT) if compare else VARIANTS
    races = _read_races(date_from, date_to)
    runners = _read("runners", RUNNER_COLUMNS, date_from, date_to)
    folds = walk_forward(races.get_column("race_date").to_list(),
                         scheme=scheme or load_country_scheme(COUNTRY))  # fmt: skip
    dates = races.get_column("race_date")
    parts, priors, versus_priors = [], [], []
    for k, fold in enumerate(folds):
        prior = fit_bias_prior(races, runners, train_end=fold.train_end, day_sd=day_sd)
        priors.append(prior)
        versus_prior = None
        if compare:
            versus_prior = fit_bias_prior(races, runners, train_end=fold.train_end, day_sd=versus)
            versus_priors.append(versus_prior)
        for day in sorted(dates.gather(fold.test_idx).unique().to_list()):
            day_races = races.filter(pl.col("race_date") == day)
            day_runners = runners.filter(pl.col("race_date") == day)
            day_rows = _day_rows(prior, day_races, day_runners, day, versus_prior)
            parts += [p.with_columns(pl.lit(k).alias("fold")) for p in day_rows]
    rows = pl.concat(parts)

    scored = []
    for k in range(1, len(folds)):
        # fold k's own training days (#70's embargo), as #129's oof_handicap_score uses them
        train_days = dates.gather(folds[k].train_idx).unique()
        train = rows.filter((pl.col("fold") < k) & pl.col("race_date").is_in(train_days))
        test = rows.filter(pl.col("fold") == k)
        if test.height == 0:
            continue
        for variant in variants:
            names = {f"{f}_{variant}": f for f in FEATURES}
            model = fit_handicap_score(train.rename(names), features=FEATURES)
            test = test.with_columns(model.predict(test.rename(names)).alias(f"p_{variant}"))
        scored.append(test)
    frame = pl.concat(scored)

    per_race = (
        frame.with_columns(pl.len().over("race_id").alias("n"))
        .group_by("race_id", "meeting_id", "bias_races_n", "n")
        .agg(
            pl.col("won").sum().alias("k"),
            *[
                (-pl.col(f"p_{v}").filter(pl.col("won")).log().mean()).alias(f"ll_{v}")
                for v in variants
            ],  # fmt: skip
        )
        .filter(pl.col("k") > 0)
        .with_columns(pl.col("n").cast(pl.Float64).log().alias("ll_equal"))
        .sort("race_id")
    )
    meeting = per_race.get_column("meeting_id").to_numpy()
    eq = per_race.get_column("ll_equal").to_numpy()
    out: dict = {
        "races": per_race.height,
        "meetings": int(len(np.unique(meeting))),
        "scored_from": frame.get_column("race_date").min(),
        "scored_to": frame.get_column("race_date").max(),
        "day_sd": [_by_level(p.day_sd) for p in priors],
        "day_sd_estimate": [
            None
            if p.estimate is None
            else {"per_level": _by_level(p.estimate.per_level), "pooled": p.estimate.pooled}
            for p in priors
        ],
        "noise_var": [p.noise_var for p in priors],
    }
    for v in VARIANTS:
        ll = per_race.get_column(f"ll_{v}").to_numpy()
        out[f"ll_{v}"] = float(ll.mean())
        out[f"pseudo_r2_{v}"] = float(1 - ll.sum() / eq.sum())
        out[f"gain_{v}_vs_equal"] = _bootstrap(eq - ll, meeting, n_boot, seed)
    gain = (per_race.get_column("ll_prior") - per_race.get_column("ll_update")).to_numpy()
    entered = per_race.get_column("bias_races_n").to_numpy()
    out["gain_update_vs_prior"] = _bootstrap(gain, meeting, n_boot, seed)
    if compare:
        ll_versus = per_race.get_column(f"ll_{VERSUS_VARIANT}").to_numpy()
        out["ll_update_versus"] = float(ll_versus.mean())
        out["gain_update_vs_versus"] = _bootstrap(
            ll_versus - per_race.get_column("ll_update").to_numpy(), meeting, n_boot, seed
        )
        out["day_sd_versus"] = [_by_level(p.day_sd) for p in versus_priors]
    for name, mask in (("0", entered == 0), ("1-3", (entered >= 1) & (entered <= 3)),
                       ("4+", entered >= 4)):  # fmt: skip
        if mask.any():
            out[f"gain_update_vs_prior_entered_{name}"] = _bootstrap(
                gain[mask], meeting[mask], n_boot, seed
            )
    return out
