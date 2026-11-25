"""The race-mean diagnostic behind the day-variance ledger row, with #138's choices as switches
(#257; the contract is pinned in tests/countries/japan/handicap/test_track_bias_null.py).

`null_diagnostic` reads the history `track_bias._history` reads (race days on or before
`train_end`; a holdout day there is a ValueError, D14) and works on its deviations e, a finisher's
score minus its race's cell prior mean at the level. Per level and (day, venue, surface) group, a
race j with c_j of its nf_j finishers at the level has race mean m_j of their e, and under a null
a mean mu_j and a variance v_j. With race weights a_j (1 / r, or c_j / sum c with `obs_weights`):

    d = sum a m,  q = sum (m - d)^2,  V0 = sum a^2 v,
    E0q = sum v (1 - 2 a) + r V0 + sum (mu - sum a mu)^2    (the null expectation of q)

and per level `observed_var` and `null_var` are the equal-group-weight means of q / (r - 1) and
E0q / (r - 1) over the groups of two or more races, `ratio` their quotient, `lam` the pooled
sum q / sum E0q (#138's fitted scale), and `tau2` the mean over the groups used (r >= 2, or every
group with `one_race_groups`) of (d - sum a mu)^2 - k V0, k = lam with `fitted_lam`, else 1.
All switches on with the shuffled null is #138's removed `_day_spread` (839ee72) per level; all
off is #224's diagnostic (PR #259).

Nulls:

- "shuffled": the race's places shuffled among its finishers: mu = 0 and v = w0 =
  s2 (nf - c) / ((nf - 1) c), s2 the population variance of the race's finisher scores.
- "corner": the level's race mean given its runners' first-corner positions, through the level
  itself (a style level is a band of corner positions), the finishers nf and the count c. Its
  moments are the history's own: per (factor, level, nf, c) cell of n races, mean M and sum of
  squares SS of the race means m, shrunk by PRIOR_RACES pseudo-races toward the shuffled null,
  mu = n M / (n + PRIOR_RACES) and v = (SS + PRIOR_RACES w0) / (n - 1 + PRIOR_RACES). A cell's
  variance is taken over every race in it, so it carries the day variance as well as the race
  noise: the corner null's v is high by about the level's day variance (#224: under 2% of v at
  every level), and its tau2 low by about that over r.

`run_report` is the development-data run (CLI `python -m
horsetrack.countries.japan.handicap.track_bias_null [FIRST_DAY]`): the scoped JRA store (holdout
days dropped, #114) from HISTORY_FROM (#224's first day) or FIRST_DAY, each variant at TRAIN_END,
one registry record per variant; it prints aggregates only and writes nothing else. Measurement
only: nothing here is adopted (D16).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import date

import polars as pl

from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import FoldMetric, RunRecord, default_registry
from horsetrack.countries.japan.handicap.pp import _read
from horsetrack.countries.japan.handicap.track_bias import (
    CELL,
    GROUP,
    LEVELS,
    PRIOR_RACES,
    _history,
)
from horsetrack.countries.japan.studies.calibration import store_builds

NULLS = ("shuffled", "corner")
SWITCHES = ("obs_weights", "one_race_groups", "fitted_lam")
CORNER_CELL = ("factor", "level", "nf", "c")
SCHEMA = {
    "factor": pl.String,
    "level": pl.String,
    "groups": pl.Int64,
    "races": pl.Int64,
    "observed_var": pl.Float64,
    "null_var": pl.Float64,
    "ratio": pl.Float64,
    "lam": pl.Float64,
    "tau2": pl.Float64,
}


def null_diagnostic(
    races: pl.DataFrame,
    runners: pl.DataFrame,
    *,
    train_end: date,
    obs_weights: bool = False,
    one_race_groups: bool = False,
    fitted_lam: bool = False,
    null: str = "shuffled",
) -> pl.DataFrame:
    """Seven rows, one per level in `track_bias.LEVELS` order (see the module doc)."""
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}, got {null!r}")
    obs, _ = _history(races, runners, train_end)
    return _diagnostic(
        _race_terms(obs),
        obs_weights=obs_weights,
        one_race_groups=one_race_groups,
        fitted_lam=fitted_lam,
        null=null,
    )


def _race_terms(obs: pl.DataFrame) -> pl.DataFrame:
    """One row per (group, level, race): c, the race mean m, nf and the shuffled null's w0."""
    c, nf = pl.col("c"), pl.col("nf")
    return (
        obs.group_by(*GROUP, "race_id")
        .agg(pl.len().alias("c"), pl.col("e").mean().alias("m"), nf.first(), pl.col("s2").first())
        .with_columns((pl.col("s2") * (nf - c) / ((nf - 1) * c)).alias("w0"))
        .sort(*GROUP, "race_id")
    )


def _null_moments(terms: pl.DataFrame, null: str) -> pl.DataFrame:
    """`terms` with each race mean's null mean `mu` and variance `v`."""
    if null == "shuffled":
        return terms.with_columns(pl.lit(0.0).alias("mu"), pl.col("w0").alias("v"))
    k = float(PRIOR_RACES)
    cells = terms.group_by(CORNER_CELL).agg(
        pl.len().alias("n"),
        pl.col("m").mean().alias("cell_m"),
        ((pl.col("m") - pl.col("m").mean()) ** 2).sum().alias("cell_ss"),
    )
    n = pl.col("n")
    return (
        terms.join(cells, on=list(CORNER_CELL), how="left", maintain_order="left")
        .with_columns(
            (n * pl.col("cell_m") / (n + k)).alias("mu"),
            ((pl.col("cell_ss") + k * pl.col("w0")) / (n - 1 + k)).alias("v"),
        )
        .drop("n", "cell_m", "cell_ss")
    )


def _diagnostic(
    terms: pl.DataFrame, *, obs_weights: bool, one_race_groups: bool, fitted_lam: bool, null: str
) -> pl.DataFrame:
    races = _null_moments(terms, null)
    a, m, mu, v = pl.col("a"), pl.col("m"), pl.col("mu"), pl.col("v")
    weight = (
        pl.col("c") / pl.col("c").sum().over(GROUP) if obs_weights else 1 / pl.len().over(GROUP)
    )
    groups = (
        races.with_columns(weight.alias("a"))
        .with_columns((a * m).sum().over(GROUP).alias("d"), (a * mu).sum().over(GROUP).alias("dmu"))
        .group_by(GROUP)
        .agg(
            pl.len().alias("r"),
            pl.col("d").first(),
            pl.col("dmu").first(),
            ((m - pl.col("d")) ** 2).sum().alias("q"),
            (a**2 * v).sum().alias("v0"),
            (v * (1 - 2 * a)).sum().alias("s"),
            ((mu - pl.col("dmu")) ** 2).sum().alias("sm"),
        )
        .with_columns((pl.col("s") + pl.col("r") * pl.col("v0") + pl.col("sm")).alias("e0q"))
        .sort(GROUP)
    )
    rows = []
    for factor, level in LEVELS:
        g = groups.filter((pl.col("factor") == factor) & (pl.col("level") == level))
        rows.append((factor, level, *_level_row(g, one_race_groups, fitted_lam)))
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _level_row(groups: pl.DataFrame, one_race_groups: bool, fitted_lam: bool) -> tuple:
    terms = list(groups.select("r", "d", "dmu", "q", "v0", "e0q").iter_rows())
    multi = [t for t in terms if t[0] >= 2]
    used = terms if one_race_groups else multi
    observed = _mean([q / (r - 1) for r, _, _, q, _, _ in multi])
    null_var = _mean([e0q / (r - 1) for r, _, _, _, _, e0q in multi])
    ratio = observed / null_var if observed is not None and null_var else None
    sum_e0q = math.fsum(t[5] for t in multi)
    lam = math.fsum(t[3] for t in multi) / sum_e0q if sum_e0q > 0 else None
    k = lam if fitted_lam and lam is not None else 1.0  # #138 took lam = 1 with no spread to fit
    tau2 = _mean([(d - dmu) ** 2 - k * v0 for _, d, dmu, _, v0, _ in used])
    return len(used), sum(t[0] for t in used), observed, null_var, ratio, lam, tau2


# --------------------------------------------------------------------------- development run

COUNTRY = "japan"
TICKET = 257
FAMILY_ID = "japan-track-bias-null-diagnostic"
HYPOTHESIS = "E32"  # 08 §4 E32: the same-day draw and position bias
SOURCE = "jra"  # the store's JRA results and corner positions; no JRDB, no JRA-VAN
TRAIN_END = date(2026, 9, 26)  # the last #70 fold's train_end, #224's diagnostic
HISTORY_FROM = date(2014, 6, 7)  # #224's first day; the store now starts earlier
RACE_COLUMNS = ("race_id", "race_date", *CELL)
RUNNER_COLUMNS = ("race_id", "race_date", "draw", "finish_pos", "running_positions", "is_scratched")
METRICS = ("observed_var", "null_var", "ratio", "lam", "tau2")


@dataclass(frozen=True)
class Variant:
    name: str
    obs_weights: bool = False
    one_race_groups: bool = False
    fitted_lam: bool = False
    null: str = "shuffled"


VARIANTS = (
    Variant("baseline"),
    Variant("obs_weights", obs_weights=True),
    Variant("one_race_groups", one_race_groups=True),
    Variant("fitted_lam", fitted_lam=True),
    Variant("obs_weights+one_race_groups", obs_weights=True, one_race_groups=True),
    Variant("obs_weights+fitted_lam", obs_weights=True, fitted_lam=True),
    Variant("one_race_groups+fitted_lam", one_race_groups=True, fitted_lam=True),
    Variant("all_138", obs_weights=True, one_race_groups=True, fitted_lam=True),
    Variant("corner", null="corner"),
    Variant(
        "corner+all_138", obs_weights=True, one_race_groups=True, fitted_lam=True, null="corner"
    ),  # fmt: skip
)


def run_report(train_end: date = TRAIN_END, date_from: date = HISTORY_FROM) -> dict:
    """Every variant on the JRA store's development days from `date_from` to `train_end`, one
    registry record each (failures included); returns the aggregates."""
    git_sha, builds = _git_sha(), store_builds()
    registry = default_registry(COUNTRY)
    races = _read("races", RACE_COLUMNS, date_from, train_end)
    runners = _read("runners", RUNNER_COLUMNS, date_from, train_end)
    obs, _ = _history(races, runners, train_end)
    del races, runners
    terms = _race_terms(obs)
    out: dict = {
        "history": [date_from.isoformat(), train_end.isoformat()],
        "races": obs.get_column("race_id").n_unique(),
    }
    del obs
    fold = f"history {date_from.isoformat()}..{train_end.isoformat()}"
    for variant in VARIANTS:
        switches = {s: getattr(variant, s) for s in SWITCHES}
        metrics, error = [], None
        try:
            table = _diagnostic(terms, null=variant.null, **switches)
            out[variant.name] = table.to_dicts()
            for row in table.iter_rows(named=True):
                for name in METRICS:
                    if row[name] is not None and math.isfinite(row[name]):
                        metric = f"{name}:{row['factor']}:{row['level']}"
                        metrics.append(
                            FoldMetric(fold, metric, row[name], None, None, row["groups"])
                        )
        except Exception as exc:  # every failure is a registry record, never dropped
            error = f"{type(exc).__name__}: {exc}"
            out[variant.name] = {"error": error}
        record = RunRecord(
            hypothesis_id=HYPOTHESIS,
            family_id=FAMILY_ID,
            variant=variant.name,
            ticket=TICKET,
            git_sha=git_sha,
            data_builds=builds,
            sources=(SOURCE,),
            fold_scheme=f"in_sample:{fold}",
            components={"estimator": "track_bias_null.null_diagnostic"},
            params={
                **switches,
                "null": variant.null,
                "date_from": date_from.isoformat(),
                "train_end": train_end.isoformat(),
            },  # fmt: skip
            metrics=tuple(metrics),
            status="failed" if error else "scored",
        )
        if record not in registry.records(FAMILY_ID):
            registry.append(record)
    return out


if __name__ == "__main__":
    first = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else HISTORY_FROM
    print(json.dumps(run_report(date_from=first), indent=1, default=str))
