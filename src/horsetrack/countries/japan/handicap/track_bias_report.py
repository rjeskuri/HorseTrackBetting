"""Development-data measurement for #138: does the within-day update beat the prior alone?

`track_bias_report(date_from, date_to)` scores each #70 fold's races out of fold. The bias prior is
fitted on the days up to the fold's `train_end`; each runner gets two feature pairs, the prior
alone (nothing published) and prior + within-day update: `draw_adj` and `style_adj`, the
adjustment of its #129 running style (null without one). Both pairs go through the same form, a
ridge conditional logit on the winner (#129's `fit_handicap_score`, alpha 1), fitted for fold k on
the out-of-fold rows of folds before k that are among fold k's training days, so outside its
embargo (fold 0 only trains). Scored as #69 defines it: winner log
loss per race, a dead heat's k winners each weighted 1/k, and a paired bootstrap over meetings.

The store holds no publication time, so a proxy stands in: a race's result counts as published
at its local post time + PUBLISH_LAG, and the decision time is the target's post time -
DECISION_LEAD. Holdout days are never read (#114's scoped reader); only aggregates are returned
and nothing is written. Measurement only: nothing here is adopted (D16).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl

from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.countries.japan.handicap.factors import (
    COUNTRY,
    STYLES,
    _read,
    fit_handicap_score,
    handicap_dir,
)
from horsetrack.countries.japan.handicap.pp import JST
from horsetrack.countries.japan.handicap.track_bias import CELL, fit_bias_prior, track_bias_features

PUBLISH_LAG = timedelta(minutes=30)
DECISION_LEAD = timedelta(minutes=2)
RACE_COLUMNS = ("race_id", "race_date", "meeting_id", "race_no", "post_time_local", *CELL)
RUNNER_COLUMNS = ("race_id", "race_date", "horse_no", "horse_id", "draw", "finish_pos",
                  "running_positions", "is_scratched")  # fmt: skip
VARIANTS = ("prior", "update")
FEATURES = ("draw_adj", "style_adj")


def _post(day: date, hhmm: str | None) -> datetime | None:
    if not hhmm:
        return None
    hh, mm = hhmm.split(":")
    return datetime.combine(day, time(int(hh), int(mm)), tzinfo=JST)


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


def _day_rows(prior, races: pl.DataFrame, runners: pl.DataFrame, day: date) -> list[pl.DataFrame]:
    """Both variants' features, the winner flag and the update's race count per declared runner of
    the day's races that have a post time."""
    posts = {
        rid: _post(day, hhmm) for rid, hhmm in races.select("race_id", "post_time_local").rows()
    }
    published = {rid: t + PUBLISH_LAG for rid, t in posts.items() if t is not None}
    style = _runner_style(day)
    out = []
    for rid, meeting in races.select("race_id", "meeting_id").sort("race_id").rows():
        if posts[rid] is None:
            continue
        kw = dict(race_id=rid, decision_time=posts[rid] - DECISION_LEAD)
        cols = []
        for variant, pub in (("prior", {}), ("update", published)):
            f = track_bias_features(prior, races, runners, **kw, published=pub).join(
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
    day_sd: float | None = None,
) -> dict:
    """Aggregates only: per-variant log loss and pseudo-R² against equal chances, the update's
    gain over the prior with meeting-bootstrap bounds overall and by races entered, and the
    fitted priors' day sd and observation variance. `day_sd` (None: estimated) goes to every
    fold's prior."""
    races = _read("races", RACE_COLUMNS, date_from, date_to)
    runners = _read("runners", RUNNER_COLUMNS, date_from, date_to)
    folds = walk_forward(races.get_column("race_date").to_list(),
                         scheme=scheme or load_country_scheme(COUNTRY))  # fmt: skip
    dates = races.get_column("race_date")
    parts, priors = [], []
    for k, fold in enumerate(folds):
        prior = fit_bias_prior(races, runners, train_end=fold.train_end, day_sd=day_sd)
        priors.append((prior.day_sd, prior.noise_var))
        for day in sorted(dates.gather(fold.test_idx).unique().to_list()):
            day_races = races.filter(pl.col("race_date") == day)
            day_runners = runners.filter(pl.col("race_date") == day)
            parts += [p.with_columns(pl.lit(k).alias("fold"))
                      for p in _day_rows(prior, day_races, day_runners, day)]  # fmt: skip
    rows = pl.concat(parts)

    scored = []
    for k in range(1, len(folds)):
        # fold k's own training days (#70's embargo), as #129's oof_handicap_score uses them
        train_days = dates.gather(folds[k].train_idx).unique()
        train = rows.filter((pl.col("fold") < k) & pl.col("race_date").is_in(train_days))
        test = rows.filter(pl.col("fold") == k)
        if test.height == 0:
            continue
        for variant in VARIANTS:
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
                for v in VARIANTS
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
        "day_sd": [p[0] for p in priors],
        "noise_var": [p[1] for p in priors],
    }
    for v in VARIANTS:
        ll = per_race.get_column(f"ll_{v}").to_numpy()
        out[f"ll_{v}"] = float(ll.mean())
        out[f"pseudo_r2_{v}"] = float(1 - ll.sum() / eq.sum())
        out[f"gain_{v}_vs_equal"] = _bootstrap(eq - ll, meeting, n_boot, seed)
    gain = (per_race.get_column("ll_prior") - per_race.get_column("ll_update")).to_numpy()
    entered = per_race.get_column("bias_races_n").to_numpy()
    out["gain_update_vs_prior"] = _bootstrap(gain, meeting, n_boot, seed)
    for name, mask in (("0", entered == 0), ("1-3", (entered >= 1) & (entered <= 3)),
                       ("4+", entered >= 4)):  # fmt: skip
        if mask.any():
            out[f"gain_update_vs_prior_entered_{name}"] = _bootstrap(
                gain[mask], meeting[mask], n_boot, seed
            )
    return out
