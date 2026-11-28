"""#129's handicap score with #96's speed figure, re-measured beside the interim class par (#219).

#129 built the handicap factors with an interim speed figure (the final time against the in-fold
class-par median) and measured the out-of-fold `handicap_score` alone (V1) and melded into the
calibrated-market Win baseline v0 (V2). #96's expected top-3 time has landed, so the speed family
now reads it (`handicap/factors.py`). This study rebuilds the per-day table with it, each past
start's figure as of its own race day with its day variant (`speed_at_start_day=True`), and reruns
#129's two comparisons with `core.eval.scoring` and `core.eval.bootstrap`, next to the same
comparisons on #129's kept table (the interim par, pinned to `features/jra/handicap`), and scores
the swap itself: the new score against the old, paired on the same races. It adopts nothing (D16);
the holdout is never opened and no JRDB or JRA-VAN table is read (D21).

This is the v2 family. v1 (`japan-handicap-speed-96-final-win`) fitted every past start in the
model (no day variant, so it tested the new par alone); its records stay in the registry.

- V1: the out-of-fold score alone against equal chances (winner log loss, a dead heat's k co-winners
  at 1/k, pseudo-R2 against ln n, gain in nats per race), and against v0.
- V2: exp(a ln p_cal + b ln h) over v0, (a, b) the conditional logit fitted on the v0 rows of the
  folds before the one scored and no later than that fold's training cutoff (D9's embargo, the
  v0 manifest's `train_end`; walk-forward, `fit_conditional_logit`); the first fold only trains.
- The bounds are D7's: a paired bootstrap over meetings (venue x day), one-sided 95% (the lower
  bound a claim rests on), plus the two-sided 95% interval.

Order of `run_study` (CLI `python -m horsetrack.countries.japan.studies.handicap_speed --run
[--out DIR] [--rebuild]`): build the spec, freeze the family in the run registry (a different spec
under the same id stops the run), only then build the table (`build_handicap_table(...,
out_dir=TABLE_DIR, speed_at_start_day=True)`, a directory beside #129's, which is kept; an existing
one is reused only when its completion marker names this range and mode, else rebuilt), read the
outcomes, score, append one registry record per variant (failures included) and write the report
through `write_report`. `compute_study` is pure.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import signal
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import RunRecord, default_registry, summary_dir, write_summary
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.scoring import gain, per_race_log_loss, pseudo_r2
from horsetrack.countries.japan.handicap import factors, pp
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.studies import speed_run
from horsetrack.countries.japan.studies.calibration import Study, Variant, _markdown_table, _metric

COUNTRY = "japan"
LOG = logging.getLogger(__name__)
MAX_MINUTES = 90.0  # the deadline of a whole run (build, score, variants); past it the run fails
RUN_FAILED = "run:failed"  # the one record of a run that raised or passed its deadline
TICKET = 219
FAMILY_ID = "japan-handicap-speed-96-final-win-v2"
V1_FAMILY_ID = "japan-handicap-speed-96-final-win"  # superseded, kept in the registry
V1_SPEC_SHA256 = "16b385befd691c9e957de14b56f465ed6cfda04bcaffcec42b199b23ebabcad8"
HYPOTHESIS_ID = "E39"  # 08 §4: C3's figure; this is its use inside the handicap score (#129)
SOURCE = "jra"  # the store's JRA results (netkeiba); no JRDB, no JRA-VAN
BOOTSTRAP_SEED = 219  # reproducibility only; scoring.yaml signs no seed
FIRST_DAY = PRICE_ERA_START  # 2014-06-07, #129's first development day
LAST_DAY = date(2026, 11, 15)  # #129's last development day (every day is before the holdout)
V0_FIT_ID = "aac501f927b0"  # #129's baseline: v0's out-of-fold p_cal, `win_calibration_oof-<id>`
TABLE_DIR = "handicap-96-v2"  # features/jra/<TABLE_DIR>; v1's handicap-96 and #129's handicap stay
INTERIM_DIR = "handicap"  # features/jra/<INTERIM_DIR>: #129's table, the interim par, read-only
INTERIM = "interim_par"
V96 = "v96"
KEYS = ["race_id", "horse_no"]
DECISIONS = {
    "D6": "winner log loss; dead heat k co-winners at weight 1/k; pseudo-R2 against equal chances",
    "D7": "meeting clusters (venue x day), 95% one-sided; the two-sided 95% interval beside it",
    "D9": "#70's quarterly expanding folds (config/folds.yaml), for the score and the meld alike",
    "D10": "no price is read before 2014-06-07",
    "D11-D15": "Japan holdout A2 from 2026-11-28, fully locked: no holdout day is read",
    "D16": "nothing is adopted",
    "D21": "no licensed source (JRDB, JRA-VAN) is read",
}


def table_dir() -> Path:
    return paths.data_dir() / "features" / "jra" / TABLE_DIR


def interim_dir() -> Path:
    """#129's table, pinned here and not through `factors.handicap_dir()`, so a rebuild of the #96
    table can never be compared with itself."""
    return paths.data_dir() / "features" / "jra" / INTERIM_DIR


def v0_oof_path() -> Path:
    return paths.data_dir() / "models" / COUNTRY / "v0" / f"win_calibration_oof-{V0_FIT_ID}.parquet"


def v0_manifest_path() -> Path:
    return v0_oof_path().with_name(f"manifest-win_calibration-{V0_FIT_ID}.json")


def bootstrap_settings() -> dict:
    """D7's bootstrap from config/scoring.yaml, with this study's fixed seed."""
    boot = load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["cluster_unit"] != "meeting" or boot["sided"] != "one-sided":
        raise ValueError(f"this study implements meeting clusters, one-sided; scoring.yaml: {boot}")
    return {"cluster_unit": "meeting", "sided": "one-sided", "level": float(boot["level"])} | {
        "n_boot": int(boot["n_boot"]),
        "seed": BOOTSTRAP_SEED,
    }


def build_spec(boot: dict, scheme: FoldScheme) -> dict:
    """The family's pre-registered spec, frozen before the table is built or any outcome is read."""
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "null": "swapping the speed par for #96's expected time does not improve the handicap "
            "score: gain = ll(interim-par score) - ll(#96 score) <= 0 on the common races",
            "H1": "V1 holds: the #96 score alone still beats equal chances. Its pseudo-R2 is "
            "within 0.02 of #129's 0.165 (the figure feeds the speed, slope, fit and class-fit "
            "features, and #96's six-key par is a finer cell than the class par); a negative "
            "gain over equal chances would be a bug",
            "H2": "V2 stays small: the meld gains at most about 0.002 nats per race over v0, as "
            "#129's 0.0005 did, and its one-sided lower bound may not exclude 0; a gain above 0.01 "
            "would point to leakage before an edge",
            "H3": "the swap is worth little: |gain of the #96 score over the interim-par score| is "
            "under 0.01 nats per race",
            "note": "v2 supersedes the v1 family (spec sha256 "
            f"{V1_SPEC_SHA256}), whose per-day table put every past start in the model's fit "
            "(train_end = the cut), so its figure had the day variant at its prior and tested "
            "the new par alone. v1's results were known when this spec was frozen: H1-H3 are "
            "carried from v1 unchanged, not tuned to them. v2's figure is as of the start's own "
            "race day, with that day's earlier results observed (the day variant); two changes "
            "besides: the V2 meld admits only v0 rows no later than the scored fold's training "
            "cutoff (D9's embargo, which v1 skipped), and the #96 table is written to "
            f"features/jra/{TABLE_DIR}",
        },
        "supersedes": {"family_id": V1_FAMILY_ID, "spec_sha256": V1_SPEC_SHA256},
        "figure": "speed = SpeedModel(SpeedSpec(PAR_KEYS, VARIANT_KEYS, same_day_variant=True, "
        "t_df=5.0)).expected_top3_time at post - 2 min (D2) minus finish_time_s, for each past "
        "start as of its own race day E: the model fitted on speed_run.build_race_table("
        "estimand='mean_placed_1_3') of the races before E, having observed E's results (one fit "
        f"per race day); no figure for a start in the first {factors.SPEED_WARMUP_DAYS} days of "
        "the store, or in a cell with no earlier race (handicap/factors.py "
        "`_own_day_expected`, #219)",
        "table": f"features/jra/{TABLE_DIR}: build_handicap_table({FIRST_DAY}, {LAST_DAY}, "
        "speed_at_start_day=True), the default ridge conditional logit (alpha 1, SCORE_FEATURES), "
        "out of fold over #70's scheme",
        "interim_table": f"features/jra/{INTERIM_DIR}, #129's table (the in-fold class-par median),"
        " rescored here with the same code",
        "development_days": [FIRST_DAY.isoformat(), LAST_DAY.isoformat()],
        "baseline": f"v0's calibrated Win model out of fold, win_calibration_oof-{V0_FIT_ID} (#114,"
        " #123); its folds are the meld's",
        "variants": {
            "V1": "the score alone vs equal chances (ln n), pseudo-R2 and gain in nats per race",
            "handicap_vs_market": "gain = ll(v0) - ll(score) on the races both score",
            "V2": "exp(a ln p_cal + b ln h) normalised within the race; (a, b) the conditional "
            "logit on the v0 rows of the folds before the scored one and dated no later than "
            "its training cutoff (the v0 manifest's train_end); gain = ll(v0) - ll(meld) over "
            "the scored folds on the same rows",
            "swap": "the #96 score against the interim-par score on their common races, alone "
            "and melded; gain = ll(interim) - ll(#96)",
        },
        "scoring": "core.eval.scoring: per_race_log_loss, pseudo_r2, gain",
        "bootstrap": boot | {"two_sided": "the same draws' 2.5% and 97.5% quantiles"},
        "fold_scheme": dataclasses.asdict(scheme),
        "decisions": DECISIONS,
        "sources": [SOURCE],
    }


# --------------------------------------------------------------------------- the pure study


@dataclass
class Inputs:
    """`scores`: one table per source (race_id, race_date, horse_no, h, a null where no fold scores
    the row); `outcomes`: race_id, horse_no, won of the declared runners; `meetings`: race_id ->
    meeting_id; `v0`: race_id, horse_no, fold, p_cal; `v0_train_end`: v0 fold -> the last day that
    fold's fit trained on (D9's embargo)."""

    scores: dict[str, pd.DataFrame]
    outcomes: pd.DataFrame
    meetings: pd.Series
    v0: pd.DataFrame
    v0_train_end: dict[int, date]


def _rows(scores: pd.DataFrame, outcomes: pd.DataFrame, col: str) -> pd.DataFrame:
    """The score rows with `won`, over the races whose every row has a score."""
    rows = scores.merge(outcomes, on=KEYS, how="left")
    rows["won"] = rows["won"].fillna(False).astype(bool)
    whole = rows[col].notna().groupby(rows["race_id"]).transform("all")
    return rows[whole].reset_index(drop=True)


def _loss(rows: pd.DataFrame, col: str) -> pd.Series:
    """Winner log loss per race of the probabilities in `col` (renormalised within the race)."""
    none = pd.Series(False, index=rows.index)
    return per_race_log_loss(rows[col], rows["won"], rows["race_id"], none).per_race


def _equal_chances(rows: pd.DataFrame) -> pd.Series:
    return np.log(rows.groupby("race_id").size().astype(float))


def _bounds(per_race: pd.Series, clusters: pd.Series, boot: dict) -> dict:
    """The mean and the one-sided 95% lower bound, and the two-sided 95% interval, on one draw."""
    kw = {k: boot[k] for k in ("n_boot", "seed")}
    one = paired_bootstrap(per_race, clusters, level=boot["level"], **kw)
    two = paired_bootstrap(per_race, clusters, level=1 - (1 - boot["level"]) / 2, **kw)
    return {"mean": one.mean, "lb": one.lo, "lo": two.lo, "hi": two.hi, "meetings": one.n_clusters}


def _gain_row(label: str, per_race: pd.Series, inputs: Inputs, boot: dict) -> tuple[list, dict]:
    """Registry metrics and the report row of a per-race gain in nats."""
    b = _bounds(per_race, inputs.meetings, boot)
    n = len(per_race)
    metrics = [
        _metric("development", "gain_nats", b["mean"], n, b["lo"], b["hi"]),
        _metric("development", "gain_nats_lower_1s95", b["lb"], n),
    ]
    row = {"comparison": label, "races": n, "meetings": b["meetings"], "gain nats/race": b["mean"]}
    return metrics, row | {
        "1-sided 95% lower": b["lb"],
        "2-sided 95% lo": b["lo"],
        "2-sided 95% hi": b["hi"],
    }


def _alone(source: str, inputs: Inputs, boot: dict) -> tuple[list, dict]:
    """V1: the source's score alone vs equal chances, and against v0."""
    rows = _rows(inputs.scores[source], inputs.outcomes, "h")
    ll_h, ref = _loss(rows, "h"), _equal_chances(rows)
    r2 = pseudo_r2(ll_h, ref)
    metrics, row = _gain_row("V1: score vs equal chances", gain(ll_h, ref).per_race, inputs, boot)
    metrics.append(_metric("development", "pseudo_r2", r2.value, r2.n_common))
    mkt_rows = _rows(inputs.v0.rename(columns={"p_cal": "h"})[[*KEYS, "h"]], inputs.outcomes, "h")
    ll_m = _loss(mkt_rows, "h")
    r2_m = pseudo_r2(ll_m, ref.loc[ref.index.isin(ll_m.index)])
    vs, vs_row = _gain_row("score vs market v0 (ll_v0 - ll_score)", gain(ll_h, ll_m).per_race,
                           inputs, boot)  # fmt: skip
    metrics += [
        _metric("development", f"vs_market_{m.metric}", m.value, m.n, m.lo, m.hi) for m in vs
    ]
    metrics.append(_metric("development", "market_pseudo_r2", r2_m.value, r2_m.n_common))
    rows_out = [{"source": source, "pseudo-R2": r2.value, **row}, {"source": source, **vs_row}]
    rows_out.append({"source": source, "comparison": "market v0 vs equal chances, paired races",
                     "races": r2_m.n_common, "pseudo-R2": r2_m.value})  # fmt: skip
    return metrics, {"alone": rows_out}


def _meld(source: str, inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The walk-forward meld's rows (fold, race_date, ll_v0, ll_meld per race) and its (a, b) per
    fold. Fold k's meld is fitted on the v0 folds before k, on the days up to fold k's training
    cutoff (D9's one-week embargo: the days between it and the fold's first test day are not
    trained on); the first fold only trains."""
    rows = _rows(inputs.scores[source], inputs.outcomes, "h")
    rows = rows.merge(inputs.v0, on=KEYS, how="inner")
    rows["x_m"], rows["x_h"] = np.log(rows["p_cal"]), np.log(rows["h"])
    day = pd.to_datetime(rows["race_date"])
    coefs, parts = [], []
    for k in sorted(rows["fold"].unique())[1:]:
        cutoff = pd.Timestamp(inputs.v0_train_end[int(k)])
        train, test = rows[(rows["fold"] < k) & (day <= cutoff)], rows[rows["fold"] == k].copy()
        fit = fit_conditional_logit(
            train[["x_m", "x_h"]].to_numpy(), train["race_id"], train["won"].to_numpy()
        )
        a, b = fit.coef
        coefs.append({"fold": int(k), "a": a, "b": b, "races fitted": fit.n_races})
        test["u"] = a * test["x_m"] + b * test["x_h"]
        e = np.exp(test["u"] - test.groupby("race_id")["u"].transform("max"))
        test["p_meld"] = e / e.groupby(test["race_id"]).transform("sum")
        ll = pd.DataFrame({"v0": _loss(test, "p_cal"), "meld": _loss(test, "p_meld")})
        ll["fold"] = k
        ll["race_date"] = test.groupby("race_id")["race_date"].first()
        parts.append(ll)
    return pd.concat(parts), pd.DataFrame(coefs)


def _melded(source: str, inputs: Inputs, boot: dict) -> tuple[list, dict]:
    """V2: the meld over v0, with its coefficients and the gain by year."""
    ll, coefs = _meld(source, inputs)
    metrics, row = _gain_row("V2: meld over market v0", ll["v0"] - ll["meld"], inputs, boot)
    by_year = (ll["v0"] - ll["meld"]).groupby(pd.to_datetime(ll["race_date"]).dt.year)
    years = [{"source": source, "year": int(y), "races": len(g), "gain nats/race": g.mean()}
             for y, g in by_year]  # fmt: skip
    last = coefs.iloc[-1]
    coef = {"source": source, "a (last fold)": last["a"], "b (last fold)": last["b"],
            "b min": coefs["b"].min(), "b max": coefs["b"].max(), "folds": len(coefs)}  # fmt: skip
    return metrics, {"meld": [{"source": source, **row}], "meld_years": years, "meld_coef": [coef]}


def _swap(inputs: Inputs, boot: dict, *, melded: bool) -> tuple[list, dict]:
    """The #96 score against the interim-par score, paired on their common races."""
    if melded:
        new, old = (_meld(s, inputs)[0] for s in (V96, INTERIM))
        new, old = new["meld"], old["meld"]
        label = "swap, melded over v0: ll(interim meld) - ll(#96 meld)"
    else:
        new, old = (
            _loss(_rows(inputs.scores[s], inputs.outcomes, "h"), "h") for s in (V96, INTERIM)
        )
        label = "swap, alone: ll(interim) - ll(#96)"
    metrics, row = _gain_row(label, gain(new, old).per_race, inputs, boot)
    return metrics, {"swap": [row]}


def compute_study(inputs: Inputs, spec: dict) -> Study:
    """Every variant of the frozen `spec` on `inputs`. A failed variant is recorded."""
    boot = spec["bootstrap"]
    study = Study(counts={"races with an outcome": int(inputs.outcomes["race_id"].nunique())})
    scheme = "walk_forward:" + json.dumps(
        spec["fold_scheme"], sort_keys=True
    )  # speed_run.scheme_label
    for source in (V96, INTERIM):
        components = {"speed": source, "score": "ridge conditional logit, alpha 1",
                      "market": f"v0 win calibration {V0_FIT_ID}"}  # fmt: skip
        for name, compute in (
            (f"{source}:V1:handicap_alone", lambda s=source: _alone(s, inputs, boot)),
            (f"{source}:V2:meld_over_market_v0", lambda s=source: _melded(s, inputs, boot)),
        ):
            study.attempt(Variant(HYPOTHESIS_ID, name, scheme, components, {"source": source}),
                          compute)  # fmt: skip
    components = {"comparison": "#96 score against the interim-par score, paired"}
    for name, melded in (("swap:handicap_alone", False), ("swap:meld_over_market_v0", True)):
        study.attempt(
            Variant(HYPOTHESIS_ID, name, scheme, components, {"melded": melded}),
            lambda m=melded: _swap(inputs, boot, melded=m),
        )
    return study


# --------------------------------------------------------------------------- inputs and runner


def read_scores(directory: Path) -> pd.DataFrame:
    """race_id, race_date, horse_no and `h` (the out-of-fold score) of a per-day handicap table."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no handicap table under {directory}")
    cols = ["race_id", "race_date", pl.col("horse_no").cast(pl.Int64), "handicap_score"]
    frame = pl.scan_parquet([str(f) for f in files]).select(cols).collect().to_pandas()
    return frame.rename(columns={"handicap_score": "h"})


def read_outcomes() -> tuple[pd.DataFrame, pd.Series]:
    """Declared runners' win flags and each race's meeting, through the scoped reader (a holdout
    day is never returned)."""
    runners = pp._read("runners", [*KEYS, "finish_pos", "is_scratched"], FIRST_DAY, LAST_DAY)
    runners = runners.filter(~pl.col("is_scratched").fill_null(False)).select(
        "race_id", pl.col("horse_no").cast(pl.Int64), (pl.col("finish_pos") == 1).alias("won")
    )
    races = pp._read("races", ["race_id", "meeting_id"], FIRST_DAY, LAST_DAY).to_pandas()
    return runners.to_pandas(), races.set_index("race_id")["meeting_id"]


def read_v0() -> tuple[pd.DataFrame, dict[int, date]]:
    """v0's out-of-fold rows and each fold's training cutoff from its manifest."""
    frame = pd.read_parquet(v0_oof_path(), columns=[*KEYS, "fold", "p_cal"])
    frame["horse_no"] = frame["horse_no"].astype("int64")
    folds = json.loads(v0_manifest_path().read_text())["folds"]
    return frame, {int(f["fold"]): date.fromisoformat(f["train_end"]) for f in folds}


def store_builds(tables: dict[str, Path]) -> dict[str, str]:
    builds = speed_run.store_builds()
    builds[f"v0_oof_{V0_FIT_ID}"] = f"models/{COUNTRY}/v0/{v0_oof_path().name}"
    builds[f"v0_manifest_{V0_FIT_ID}"] = f"models/{COUNTRY}/v0/{v0_manifest_path().name}"
    for name, directory in tables.items():
        builds[f"handicap_table_{name}"] = f"{directory.relative_to(paths.data_dir())}: " + (
            f"{len(list(directory.glob('*.parquet')))} race days"
        )
    return builds


def render_report(study: Study, spec: dict, about: dict) -> str:
    """The full report: aggregate tables only, never a runner or race row."""
    titles = {
        "alone": "V1: the score alone, against equal chances and against market v0",
        "meld": "V2: exp(a ln p_cal + b ln h) over market v0, walk-forward",
        "meld_years": "V2 gain by year (nats per race)",
        "meld_coef": "V2 coefficients: last fold and the range of b over the folds",
        "swap": "The swap: the #96 score against the interim-par score, same races",
    }
    lines = [f"# Handicap speed figure from #96, re-measured ({COUNTRY}, #{TICKET})", ""]
    lines += [f"- {key}: {value}" for key, value in about.items()]
    lines += ["- development days only; the holdout is never opened; no JRDB or JRA-VAN input", ""]
    lines += ["## Hypotheses (frozen before anything was scored)", ""]
    lines += [f"- {key}: {value}" for key, value in spec["hypotheses"].items()]
    lines += ["", "## Owner decisions used", ""] + [
        f"- {k}: {v}" for k, v in spec["decisions"].items()
    ]
    lines += [
        "",
        "## Counts",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title in titles.items():
        rows = study.tables.get(key)
        lines += ["", f"## {title}", "", _markdown_table(rows) if rows else "_Not computed._"]
    variants = [
        {"variant": v.name, "status": "failed" if v.error else "scored"}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    return "\n".join([*lines, "", "## Variants (one registry record each)", "",
                      _markdown_table(variants), ""])  # fmt: skip


class StudyTimeout(BaseException):
    """The run's deadline passed. A BaseException so that no per-variant `attempt` records it as
    that variant's failure and carries on."""


@contextlib.contextmanager
def _deadline(minutes: float):
    """Raise `StudyTimeout` in the main thread after `minutes`, wherever the run is (a fit looping
    in Python is interrupted too). Unix only."""

    def fire(signum, frame):
        raise StudyTimeout(f"no result after {minutes:g} minutes")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, minutes * 60)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _inputs_and_study(spec: dict, rebuild: bool, new_dir: Path, old_dir: Path) -> Study:
    if rebuild or not factors.table_complete(new_dir, FIRST_DAY, LAST_DAY, speed_at_start_day=True):
        LOG.info("building %s", new_dir)
        factors.build_handicap_table(FIRST_DAY, LAST_DAY, out_dir=new_dir, speed_at_start_day=True)
    LOG.info("reading outcomes, v0 and the two tables")
    outcomes, meetings = read_outcomes()
    v0, v0_train_end = read_v0()
    inputs = Inputs(
        {V96: read_scores(new_dir), INTERIM: read_scores(old_dir)},
        outcomes, meetings, v0, v0_train_end,
    )  # fmt: skip
    LOG.info("scoring the variants")
    return compute_study(inputs, spec)


def run_study(
    out_dir: Path | None = None, *, rebuild: bool = False, max_minutes: float = MAX_MINUTES
) -> dict:
    """Freeze the spec, build the #96 table when absent (or `rebuild`), score, record, report.
    A run that fails or passes `max_minutes` appends one failed record (`RUN_FAILED`) and raises."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    scheme, boot = load_country_scheme(COUNTRY), bootstrap_settings()
    spec = build_spec(boot, scheme)
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    spec_hash = registry.freeze_family(FAMILY_ID, spec)  # before the table or any outcome is read

    new_dir, old_dir = table_dir(), interim_dir()
    if new_dir.resolve() == old_dir.resolve():
        raise ValueError("the #96 table and #129's interim table are one directory")
    fold_scheme = speed_run.scheme_label(scheme)
    try:
        with _deadline(max_minutes):
            study = _inputs_and_study(spec, rebuild, new_dir, old_dir)
    except (Exception, StudyTimeout) as exc:
        failed = RunRecord(
            hypothesis_id=HYPOTHESIS_ID, family_id=FAMILY_ID, variant=RUN_FAILED, ticket=TICKET,
            git_sha=git_sha, data_builds=store_builds({"96": new_dir, "129": old_dir}),
            sources=(SOURCE,), fold_scheme=fold_scheme,
            components={"run": "table build, out-of-fold scoring and the six variants"},
            params={"error": f"{type(exc).__name__}: {exc}", "bootstrap": boot,
                    "max_minutes": max_minutes},
            metrics=[], status="failed",
        )  # fmt: skip
        if failed not in registry.records(FAMILY_ID):
            registry.append(failed)
        raise
    builds = store_builds({"96": new_dir, "129": old_dir})
    run_ids = {}
    for v in study.variants:
        params = v.params | ({"error": v.error} if v.error else {}) | {"bootstrap": boot}
        record = RunRecord(
            hypothesis_id=v.hypothesis_id, family_id=FAMILY_ID, variant=v.name, ticket=TICKET,
            git_sha=git_sha, data_builds=builds, sources=(SOURCE,), fold_scheme=fold_scheme,
            components=v.components, params=params, metrics=v.metrics,
            status="failed" if v.error else "scored",
        )  # fmt: skip
        if record not in registry.records(FAMILY_ID):
            run_ids[v.name] = registry.append(record)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "store": ", ".join(f"{k}: {v}" for k, v in builds.items()),
        "run ids": ", ".join(f"{k} `{v}`" for k, v in run_ids.items()) or "already recorded",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"handicap-speed-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(study, spec, about),
        sources=(SOURCE,),
    )
    write_summary(registry, summary_dir(COUNTRY))
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "report_path": str(report),
        "run_ids": run_ids,
        "tables": study.tables,
        "failed_variants": {v.name: v.error for v in study.variants if v.error},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.handicap_speed"
    )
    parser.add_argument("--run", action="store_true", required=True, help="run the study")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the #96 table")
    parser.add_argument("--max-minutes", type=float, default=MAX_MINUTES, help="deadline of a run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = run_study(args.out, rebuild=args.rebuild, max_minutes=args.max_minutes)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
