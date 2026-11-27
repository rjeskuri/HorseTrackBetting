"""E32: the same-day draw and position bias as of t, at the final Win price (#348, epic #73).

The owner's answer on #97 (2026-11-26, item 4): E32 is "one shrunk bias column from #138's
within_day_posterior". Per declared runner, `bias` is its draw third's posterior mean (`draw_adj`)
plus the posterior mean of its own #129 running style (`style_adj_<style>`, 0 without a style; a
runner without a draw has no draw term), from `track_bias.track_bias_features` at the race's
decision time t = post - DECISION_LEAD (D2's primary, T-2), reading only the same day's earlier
races whose results were published strictly before t (`track_bias_report.published_times`: the
logged time, else post + 30 min).

The test is a conditional logit with the market as the offset: u = ln q + coef * bias, q the final
Win price normalised per race, so coef = 0 is the market exactly. `fit_coefficient` is the
unpenalised maximum likelihood with the meeting-clustered sandwich se (D7's unit;
`models.condlogit`) and a 95% two-sided interval. `score_folds` refits the coefficient for each
#70 fold k >= 1 on the rows of the earlier folds within fold k's training days (outside its
embargo) and scores fold k's races: winner log loss against the market's (#69: a dead heat's
winners averaged), ΔLL = ll_market - ll_model per race (positive: the bias helps), with D7's
one-sided meeting bootstrap. Each fold's bias is built from a prior fitted on the days up to the
fold's train_end, so every scored row is out of sample; the pooled coefficient is fitted on the
scored rows (folds >= 1).

Beside the posterior, the same tests run on the history prior alone (nothing published: the
bias the day's results have not moved), and the posterior's ΔLL over the prior's is #138's
same-day update measured at the market rather than against the prior in a form without it.

`run_study` freezes the family's spec in the run registry before any store row is read, reads the
JRA store column- and date-scoped (holdout days dropped by #114's reader; the holdout is never
opened), appends one registry record per variant (failures included) and writes the report
through `write_report`. CLI: `python -m horsetrack.countries.japan.studies.same_day_bias --run
[--from DAY] [--to DAY] [--out DIR]`. Measurement only: nothing is adopted (D16).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import Fold, FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.pricing import clean_odds
from horsetrack.countries.japan.handicap.factors import _read
from horsetrack.countries.japan.handicap.track_bias import (
    BiasPrior,
    fit_bias_prior,
    track_bias_features,
)
from horsetrack.countries.japan.handicap.track_bias_report import (
    PUBLISH_LAG,
    RUNNER_COLUMNS,
    _pick_style,
    _post,
    _read_races,
    _runner_style,
    published_times,
)
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.studies import calibration
from horsetrack.countries.japan.studies.calibration import Study, Variant, _markdown_table, _metric

COUNTRY = "japan"
TICKET = 348
SOURCE = calibration.SOURCE  # the store's JRA results, from netkeiba; no JRDB, no JRA-VAN
FAMILY_ID = "japan-same-day-bias-final-win"
DECISION_LEAD = timedelta(minutes=load_country_yaml(COUNTRY, "decision")["primary_min"])
Z95 = calibration.BETA_CI_Z
BOOTSTRAP_SEED = 348
COLUMNS = {"posterior": "bias", "history_prior": "bias_prior"}  # variant -> its bias column
STYLE_SCHEMA = {"race_id": pl.String, "horse_no": pl.Int64, "style": pl.String}


# --------------------------------------------------------------------------- the feature


def runner_bias(
    prior: BiasPrior,
    races: pl.DataFrame,
    runners: pl.DataFrame,
    *,
    race_id: str,
    decision_time: datetime,
    published: Mapping[str, datetime],
    styles: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per declared runner of `race_id`: `race_id`, `horse_no` and `bias`, the draw
    posterior plus the posterior of the runner's own style in `styles` (0 without one)."""
    parts = track_bias_features(
        prior, races, runners, race_id=race_id, decision_time=decision_time, published=published
    )
    own = pl.DataFrame(schema=STYLE_SCHEMA)
    if styles is not None:
        own = (
            styles.select("race_id", pl.col("horse_no").cast(pl.Int64), "style")
            .filter(pl.col("race_id") == race_id)
            .unique(["race_id", "horse_no"], keep="first", maintain_order=True)
        )
    return parts.join(own, on=["race_id", "horse_no"], how="left", maintain_order="left").select(
        "race_id",
        "horse_no",
        (pl.col("draw_adj").fill_null(0.0) + _pick_style("own").fill_null(0.0)).alias("bias"),
    )


def _day_rows(
    prior: BiasPrior,
    races: pl.DataFrame,
    runners: pl.DataFrame,
    styles: pl.DataFrame | None,
    *,
    with_prior: bool,
) -> pl.DataFrame:
    """`day_rows`, and with `with_prior` also `bias_prior`, the column with nothing published."""
    published = published_times(races)
    builds = {"bias": published, **({"bias_prior": {}} if with_prior else {})}
    outcome = runners.filter(~pl.col("is_scratched").fill_null(False)).select(
        "race_id",
        pl.col("horse_no").cast(pl.Int64),
        (pl.col("finish_pos") == 1).fill_null(False).alias("won"),
        pl.col("win_odds").cast(pl.Float64),
    )
    out = []
    order = races.select("race_id", "meeting_id", "race_date", "post_time_local").sort("race_id")
    for rid, meeting, day, hhmm in order.rows():
        post = _post(day, hhmm)
        if post is None:
            continue
        kw = dict(race_id=rid, decision_time=post - DECISION_LEAD, styles=styles)
        frame = None
        for name, pub in builds.items():
            got = runner_bias(prior, races, runners, **kw, published=pub).rename({"bias": name})
            frame = got if frame is None else frame.join(got, on=["race_id", "horse_no"])
        out.append(
            frame.join(outcome, on=["race_id", "horse_no"], how="left", maintain_order="left")
            .with_columns(pl.lit(meeting).alias("meeting_id"), pl.lit(day).alias("race_date"))
            .select("race_id", "meeting_id", "race_date", "horse_no", "won", "win_odds", *builds)
        )
    if not out:
        schema = {
            "race_id": pl.String,
            "meeting_id": pl.String,
            "race_date": pl.Date,
            "horse_no": pl.Int64,
            "won": pl.Boolean,
            "win_odds": pl.Float64,
        }
        return pl.DataFrame(schema={**schema, **dict.fromkeys(builds, pl.Float64)})
    return pl.concat(out)


def day_rows(
    prior: BiasPrior,
    races: pl.DataFrame,
    runners: pl.DataFrame,
    styles: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Every race of `races` with a post time, its declared runners with `race_id`,
    `meeting_id`, `race_date`, `horse_no`, `won`, `win_odds` (final) and `bias` at the race's
    own t = post - DECISION_LEAD."""
    return _day_rows(prior, races, runners, styles, with_prior=False)


# --------------------------------------------------------------------------- the fit


@dataclass(frozen=True)
class CoefFit:
    """The E32 coefficient: its plain and meeting-clustered se, the 95% two-sided interval on the
    clustered se, and the races (with a winner) and meetings the likelihood holds."""

    coef: float
    se: float
    se_cluster: float
    lower: float
    upper: float
    races: int
    meetings: int


def _ln_q(rows: pd.DataFrame) -> np.ndarray:
    inv = 1.0 / rows["win_odds"].astype(float)
    return np.log((inv / inv.groupby(rows["race_id"].to_numpy()).transform("sum")).to_numpy())


def fit_coefficient(rows: pd.DataFrame, *, ridge: float = 0.0) -> CoefFit:
    """The conditional logit u = ln q + coef * bias on `rows` (see the module doc)."""
    fit = fit_conditional_logit(
        rows[["bias"]].to_numpy(dtype=float),
        rows["race_id"].to_numpy(),
        rows["won"].to_numpy(dtype=bool),
        offset=_ln_q(rows),
        ridge=ridge,
        cluster=rows["meeting_id"].to_numpy(),
    )
    coef, se_cluster = float(fit.coef[0]), float(fit.se_cluster[0])
    has_winner = rows["won"].astype(bool).groupby(rows["race_id"].to_numpy()).transform("any")
    return CoefFit(
        coef=coef,
        se=float(fit.se[0]),
        se_cluster=se_cluster,
        lower=coef - Z95 * se_cluster,
        upper=coef + Z95 * se_cluster,
        races=int(fit.n_races),
        meetings=int(rows.loc[has_winner.to_numpy(), "meeting_id"].nunique()),
    )


def _winner_ll(test: pd.DataFrame, ln_q: np.ndarray, coef: float) -> pd.DataFrame:
    """Per race with a winner: -ln q and -ln p of its winners, averaged (a dead heat's 1/k)."""
    u = ln_q + coef * test["bias"].to_numpy(dtype=float)
    race = test["race_id"].to_numpy()
    top = pd.Series(u).groupby(race).transform("max").to_numpy()
    lse = top + np.log(pd.Series(np.exp(u - top)).groupby(race).transform("sum").to_numpy())
    won = test["won"].to_numpy(dtype=bool)
    frame = pd.DataFrame(
        {
            "race_id": race[won],
            "meeting_id": test["meeting_id"].to_numpy()[won],
            "ll_market": -ln_q[won],
            "ll_model": -(u - lse)[won],
        }
    )
    return frame.groupby(["race_id", "meeting_id"], as_index=False, sort=True).mean()


def _score_folds(rows: pd.DataFrame, folds: Sequence[Fold]) -> tuple[pd.DataFrame, dict]:
    dates = pd.to_datetime(rows["race_date"])
    ln_q = _ln_q(rows)
    parts, fits = [], {}
    for k in range(1, len(folds)):
        mine = (rows["fold"] == k).to_numpy()
        if not mine.any():
            continue
        train = rows[(rows["fold"] < k) & (dates <= pd.Timestamp(folds[k].train_end))]
        fits[k] = fit = fit_coefficient(train)
        scored = _winner_ll(rows[mine], ln_q[mine], fit.coef)
        parts.append(scored.assign(fold=k, coef=fit.coef))
    columns = ["race_id", "meeting_id", "fold", "coef", "ll_market", "ll_model"]
    if not parts:
        return pd.DataFrame(columns=columns), fits
    return pd.concat(parts, ignore_index=True)[columns], fits


def score_folds(rows: pd.DataFrame, folds: Sequence[Fold]) -> pd.DataFrame:
    """One row per scored race of folds k >= 1: `race_id`, `meeting_id`, `fold`, the fold's
    `coef`, `ll_market` and `ll_model` (see the module doc)."""
    return _score_folds(rows, folds)[0]


# --------------------------------------------------------------------------- the pure study


def _coef_row(label: str, fit: CoefFit) -> dict:
    return {
        "bias": label,
        "coef": fit.coef,
        "se": fit.se,
        "se cluster": fit.se_cluster,
        "lower 95%": fit.lower,
        "upper 95%": fit.upper,
        "races": fit.races,
        "meetings": fit.meetings,
    }


def _interval(diff: pd.DataFrame, boot: dict):
    return paired_bootstrap(
        diff.set_index("race_id")["gain"],
        diff.set_index("race_id")["meeting_id"],
        n_boot=boot["n_boot"],
        level=boot["level"],
        seed=boot["seed"],
    )


def _gain_row(label: str, diff: pd.DataFrame, interval) -> dict:
    return {
        "comparison": label,
        "ΔLL mean": interval.mean,
        "lower 95% one-sided": interval.lo,
        "upper 95% one-sided": interval.hi,
        "races": len(diff),
        "meetings": interval.n_clusters,
    }


def compute_study(rows: pd.DataFrame, folds: Sequence[Fold], spec: dict) -> tuple[Study, dict]:
    """Every variant of the frozen spec on `rows` (bias, bias_prior, fold); pure. Returns the
    study and the headline figures (`coefficient`, `delta_ll`; None where a variant failed)."""
    boot = spec["bootstrap"]
    scheme = "walk_forward:" + json.dumps(spec["fold_scheme"], sort_keys=True)
    scored_rows = rows[rows["fold"] >= 1].reset_index(drop=True)
    study = Study(counts={"rows": len(rows), "scored_rows": len(scored_rows)})
    headline: dict = {"coefficient": None, "delta_ll": None}
    per_race: dict[str, pd.DataFrame] = {}

    for variant, column in COLUMNS.items():
        components = {"fit": spec["fit"], "feature": spec["feature"][variant]}

        def pooled(column=column, variant=variant):
            fit = fit_coefficient(scored_rows.assign(bias=scored_rows[column]))
            if variant == "posterior":
                headline["coefficient"] = {
                    "estimate": fit.coef,
                    "se_cluster": fit.se_cluster,
                    "lower": fit.lower,
                    "upper": fit.upper,
                    "races": fit.races,
                    "meetings": fit.meetings,
                }
            metric = _metric("development", "coef", fit.coef, fit.races, fit.lower, fit.upper)
            return [metric], {"coefficient": [_coef_row(variant, fit)]}

        study.attempt(
            Variant(
                "E32",
                f"{variant}:pooled_coefficient",
                scheme,
                components,
                {"rows": "scored test-fold rows (folds >= 1)"},
            ),
            pooled,
        )

        def walk(column=column, variant=variant):
            got, fits = _score_folds(rows.assign(bias=rows[column]), folds)
            diff = got.assign(gain=got["ll_market"] - got["ll_model"])
            per_race[variant] = diff
            interval = _interval(diff, boot)
            if variant == "posterior":
                headline["delta_ll"] = {
                    "mean": interval.mean,
                    "lo": interval.lo,
                    "hi": interval.hi,
                    "races": len(diff),
                    "meetings": interval.n_clusters,
                }
            metrics = [
                _metric(
                    "development",
                    "delta_ll_vs_market",
                    interval.mean,
                    len(diff),
                    interval.lo,
                    interval.hi,
                )
            ]
            table = []
            for k, fit in fits.items():
                label = folds[k].test_start.isoformat()
                mine = diff[diff["fold"] == k]
                metrics.append(_metric(label, "coef", fit.coef, fit.races, fit.lower, fit.upper))
                metrics.append(_metric(label, "delta_ll_vs_market", mine["gain"].mean(), len(mine)))
                table.append(
                    {
                        "bias": variant,
                        "test start": label,
                        "train end": folds[k].train_end.isoformat(),
                        **_coef_row(variant, fit),
                        "scored races": len(mine),
                        "ΔLL mean": mine["gain"].mean(),
                    }
                )
            return metrics, {
                "gain": [_gain_row(f"{variant} vs market", diff, interval)],
                "folds": table,
            }

        study.attempt(
            Variant(
                "E32",
                f"{variant}:walk_forward_delta_ll",
                scheme,
                components,
                {"baseline": "market q (the offset alone)"},
            ),
            walk,
        )

    def update_over_prior():
        post, prior = per_race["posterior"], per_race["history_prior"]
        both = post.merge(prior[["race_id", "ll_model"]], on="race_id", suffixes=("", "_prior"))
        diff = both.assign(gain=both["ll_model_prior"] - both["ll_model"])
        interval = _interval(diff, boot)
        metric = _metric(
            "development",
            "delta_ll_posterior_vs_history_prior",
            interval.mean,
            len(diff),
            interval.lo,
            interval.hi,
        )
        return [metric], {"gain": [_gain_row("posterior vs history prior", diff, interval)]}

    components = {"fit": spec["fit"], "feature": "posterior against history_prior, paired"}
    study.attempt(
        Variant(
            "E32",
            "same_day_update:walk_forward_delta_ll",
            scheme,
            components,
            {"baseline": "history_prior's walk-forward model"},
        ),
        update_over_prior,
    )
    return study, headline


# --------------------------------------------------------------------------- spec and runner


def build_spec(date_from: date, date_to: date, scheme: FoldScheme, bootstrap: dict) -> dict:
    """The family's pre-registered spec, frozen before any store row is read."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypothesis": {
            "E32": "the same-day draw-third and running-style bias as of t adds to the final Win "
            "price: coef != 0 in u = ln q + coef * bias, and walk-forward ΔLL > 0 against q",
        },
        "null": "coef = 0, ΔLL = 0: the market already prices the same-day bias",
        "target": "the race's winner (Win pool) at the final Win odds",
        "decision_time": "post_time_local (Asia/Tokyo) - "
        f"{int(DECISION_LEAD.total_seconds() // 60)} min (D2's primary, config/decision.yaml)",
        "published": "races.result_published_utc where logged, else post + "
        f"{int(PUBLISH_LAG.total_seconds() // 60)} min; a race counts strictly before t",
        "feature": {
            "posterior": "draw_adj + style_adj of the runner's own #129 style (0 without one) "
            "from track_bias.track_bias_features, the day's earlier races published before t",
            "history_prior": "the same column with nothing published (the fold's history prior)",
        },
        "prior": "track_bias.fit_bias_prior per fold on the days up to train_end, day_sd the "
        "pooled predictive estimate (the module default, #258)",
        "fit": "conditional logit u = ln q + coef * bias, q = (1/odds) normalised per race, "
        "unpenalised MLE (models.condlogit); se_cluster the meeting sandwich, no small-sample "
        "factor; 95% two-sided normal interval",
        "pooled_coefficient": "fitted on the scored test-fold rows (folds >= 1)",
        "walk_forward": "fold k >= 1: coef fitted on rows of folds < k with race_date <= "
        "folds[k].train_end, scored on fold k's races; fold 0 only trains",
        "delta_ll": "per race ll_market - ll_model, winner log loss, a dead heat's winners "
        "averaged (D6); D7 meeting bootstrap, one-sided",
        "rows": "declared runners with a final Win price of races with a post time; development "
        f"days {date_from.isoformat()} to {date_to.isoformat()}; holdout days never read",
        "fold_scheme": dataclasses.asdict(scheme),
        "bootstrap": bootstrap,
        "decisions": {
            **{k: calibration.DECISIONS[k] for k in ("D6", "D7", "D9", "D10", "D14", "D15")},
            "#97 (4)": "E32 uses one shrunk bias column from #138's within_day_posterior "
            "(owner, 2026-11-26)",
        },
        "adoption": "none (D16): measurement only",
        "sources": [SOURCE],
    }


def _record(variant: Variant, git_sha: str, builds: dict[str, str]) -> RunRecord:
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=variant.params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def _read_rows(date_from: date, date_to: date, scheme: FoldScheme) -> tuple[pd.DataFrame, list]:
    """Both bias columns per declared runner of every fold's test races, tagged with its fold,
    unpriced starters dropped."""
    races = _read_races(date_from, date_to)
    runners = _read(
        "runners", (*RUNNER_COLUMNS, "win_odds"), date_from, date_to, touches_price=True
    )
    dates = races.get_column("race_date")
    folds = walk_forward(dates.to_list(), scheme=scheme)
    parts = []
    for k, fold in enumerate(folds):
        prior = fit_bias_prior(races, runners, train_end=fold.train_end)
        for day in sorted(dates.gather(fold.test_idx).unique().to_list()):
            day_races = races.filter(pl.col("race_date") == day)
            day_runners = runners.filter(pl.col("race_date") == day)
            got = _day_rows(prior, day_races, day_runners, _runner_style(day), with_prior=True)
            parts.append(got.with_columns(pl.lit(k).alias("fold")))
    del races, runners
    rows = pl.concat(parts).to_pandas()
    rows["win_odds"] = clean_odds(rows["win_odds"].to_numpy())
    return rows[rows["win_odds"].notna()].reset_index(drop=True), folds


SECTIONS = (
    (
        "coefficient",
        "E32 coefficient, pooled over the scored rows (meeting-clustered 95% CI)",
        "u = ln q + coef * bias; coef = 0 is the market. lower/upper: coef -/+ 1.96 se cluster.",
    ),
    (
        "gain",
        "Walk-forward ΔLL (nats per race, winner log loss; D7 meeting bootstrap)",
        "Positive: the model beats its baseline. The bounds are D7's one-sided 95% bounds.",
    ),
    ("folds", "Per fold: the coefficient from its training rows and its test races' ΔLL", ""),
)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """Aggregate tables only, never a runner or race row."""
    lines = [
        f"# E32: same-day draw and style bias at the final price ({COUNTRY}, #{TICKET})",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        "- development rows only: holdout days are never read and the holdout is not opened",
        f"- adoption: {spec['adoption']}",
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title, note in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", "", *([note, ""] if note else [])]
        lines.append(_markdown_table(rows) if rows else "_Not computed: see the variants below._")
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def run_study(
    date_from: date = PRICE_ERA_START,
    date_to: date | None = None,
    *,
    scheme: FoldScheme | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Run the pre-registered family on Japan's development data and write its report.

    Raises `paths.DataDirInsideRepo` for an `out_dir` inside the repository and
    `FamilyAlreadyFrozen` when the family is frozen with another spec."""
    date_to = date.today() if date_to is None else date_to
    scheme = load_country_scheme(COUNTRY) if scheme is None else scheme
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    boot = {**calibration.bootstrap_settings(), "seed": BOOTSTRAP_SEED}
    spec = build_spec(date_from, date_to, scheme, boot)
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before any store row is read

    builds = {
        **calibration.store_builds(),
        "jra_handicap_style": "features/jra/handicap/<day>.parquet (#129 style)",
    }
    rows, folds = _read_rows(date_from, date_to, scheme)
    study, headline = compute_study(rows, folds, spec)
    study.counts["folds"] = len(folds)
    del rows

    existing = registry.records(FAMILY_ID)
    records = [_record(v, git_sha, builds) for v in study.variants]
    appended = [registry.append(r) for r in records if r not in existing]
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(builds.values()),
        "run ids": ", ".join(run_id[:16] for run_id in appended) or "none appended",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"same-day-bias-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(study, spec, about),
        sources=(SOURCE,),
    )
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "registry_path": str(registry.path),
        "counts": study.counts,
        "records_appended": len(appended),
        "run_ids": [run_id[:16] for run_id in appended],
        "failed_variants": [v.name for v in study.variants if v.error],
        **headline,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.same_day_bias"
    )
    parser.add_argument("--run", action="store_true", required=True, help="run the study")
    parser.add_argument(
        "--from",
        dest="date_from",
        type=date.fromisoformat,
        default=PRICE_ERA_START,
        help="first day read (default 2014-06-07)",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        type=date.fromisoformat,
        default=None,
        help="last day read (default today)",
    )
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    result = run_study(args.date_from, args.date_to, out_dir=args.out)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
