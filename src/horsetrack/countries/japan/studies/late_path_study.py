"""The late-money path's real-data run (#102, epic #73): `horsetrack japan late-path-study`.

The pure pieces are `studies.late_path`. The design is pre-registered in `config/late_path.yaml`,
with D2's grid (`config/decision.yaml`), D9's folds (`config/folds.yaml`) and D7's bootstrap
(`config/scoring.yaml`), all in the family's spec and its hash. Two of its fields are pending the
owner's approval (E24's features, E22's p_model): until they are approved only a `--trial` runs.

Data: the path source's day files (`late_path.path_files`: "jravan", the JRA-VAN 0B41 series, or
"jra-logs", our compacted logs), WIN rows only and by column, one day at a time. D10: days from
2014-06-07. D14: a holdout day is left out by its file name before any read; the holdout is never
opened. A starter is a WIN runner priced on its final capture; a winner is a starter whose final
row carries a WIN dividend that is not a refund (a dead heat has two); a race without one is void
and dropped, counted. Grades are the source's races table's race_class (C1's effects).

Pass 1, every development day: each race's starters, final odds and winners, the late change at
each E23 window (`late_change`, the close at the scheduled post time), and C1's WIN cases and
pool totals (`models.nowcast.extract`). D9's walk-forward over the development days
(`models.nowcast.fold_days`); D4's calibrated market is fitted on each fold's training days and
predicts its test days: q_final, so only test days are scored. Pass 2, the test days only: each
fold's C1 is fitted on its training days, then `ex_ante_features` is read at every D2 tau for every
race of the fold's test days (each test day's file is read a second time).

Comparisons, one registry record each, failures included:
- E23 `paper_regression` on every development day's starters at each window, with race-, day-
  and meeting-clustered SEs, beside the paper's versions (the spec's `paper`);
- E23 `offset:late_change_vs_final` and E24 `offset:ex_ante_vs_final` (per tau): `offset_test`
  on the scored races, the path coefficients walk-forward over the scored quarters, one-sided
  bounds resampling race days (the ticket) and meetings (D7);
- E22 `alpha_by_tau` on the scored races' (race, tau) cases, p_model from the spec's source.

The report and a JSON dump go through `write_report` under the data directory (for the jravan
source every figure is JRA-VAN-derived and stays out of the repository, D21); every record's
`sources` is the path source alone, then `write_summary`. A `--trial` run writes its records to
the family's `-trial` family with params["trial"] = True and never freezes the family's spec.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Annotated

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from horsetrack.core import config, paths
from horsetrack.core.eval.folds import load_country_scheme
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.registry import RunRecord, default_registry, summary_dir, write_summary
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.board.snapshot import load_decision_grid
from horsetrack.countries.japan.models import nowcast as nc
from horsetrack.countries.japan.models.v0.calibration import fit_win_calibration
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.studies import late_path as lp
from horsetrack.countries.japan.studies.calibration import Study, Variant, _markdown_table, _metric
from horsetrack.countries.japan.studies.nowcast import _progress

COUNTRY = "japan"
TICKET = 102
FAMILY = "japan-late-path"
PENDING = "pending owner approval"
COLUMNS = [
    "race_id",
    "venue",
    "race_no",
    "pool",
    "combination",
    "snapshot_utc",
    "post_utc",
    "odds",
    "odds_hi",
    "investment",
    "is_final_capture",
    "dividend",
    "is_refund",
]
REGRESSION = "e23:paper_regression"
E23_OFFSET = "e23:offset:late_change_vs_final"
E24_OFFSET = "e24:offset:ex_ante_vs_final"
E22 = "e22:alpha_by_tau_vs_fixed_blend"
VARIANTS = {REGRESSION: "E23", E23_OFFSET: "E23", E24_OFFSET: "E24", E22: "E22"}
Z95 = NormalDist().inv_cdf(0.975)
COMPONENTS = {
    "path": "countries/japan/studies/late_path.py",
    "baseline": "countries/japan/models/v0/calibration.py",
    "c1": "countries/japan/models/nowcast.py",
}


def family_id(source: str) -> str:
    return f"{FAMILY}-{source}"


def pending_fields(plan: dict, prefix: str = "") -> list[str]:
    """The spec's fields marked pending owner approval, as dotted paths."""
    out = [prefix] if plan.get("status") == PENDING else []
    for key, value in plan.items():
        if isinstance(value, dict):
            out += pending_fields(value, f"{prefix}.{key}" if prefix else key)
    return out


def build_spec(source: str, plan: dict, grid, scheme, bootstrap: dict) -> dict:
    """The family's design, frozen before any development row is read."""
    return {
        "family_id": family_id(source),
        "ticket": TICKET,
        "source": source,
        "data": {
            "days": f"processed/{lp.PATH_SOURCES[source]}/odds_timeseries (late_path.path_files)",
            "first_day": PRICE_ERA_START.isoformat(),
            "pool": nc.WIN,
            "holdout": "D14: holdout days left out by file name, never read; never opened",
            "row_time": "snapshot_utc (owner 2026-11-26: no lag)",
            "starters": "WIN runners priced on their final capture",
            "winners": "starters whose final row carries a WIN dividend, not a refund; a race "
            "without one is void and dropped; a dead heat counts each co-winner at 1/k (#69)",
            "grades": "the source's races table's race_class",
        },
        "plan": plan,
        "grid": {"minutes": list(grid.minutes), "primary": grid.primary},
        "fold_scheme": dataclasses.asdict(scheme),
        "bootstrap": bootstrap,
        "comparisons": VARIANTS,
        "adoption": "none (D16): nothing is adopted and the holdout is never opened",
    }


# ---------------------------------------------------------------- inputs


def read_day(path: Path) -> pd.DataFrame:
    """One day file's WIN rows by column; Arrow's pool is released after."""
    table = pq.read_table(path, columns=COLUMNS, filters=[("pool", "==", nc.WIN)])
    frame = table.to_pandas()
    del table
    pa.default_memory_pool().release_unused()
    return frame


def read_grades(folder: str) -> dict[str, str | None]:
    races = pq.read_table(
        paths.processed_dir(folder) / "races.parquet", columns=["race_id", "race_class"]
    ).to_pandas()
    grade = races["race_class"].where(races["race_class"].notna(), None)
    return dict(zip(races["race_id"], grade, strict=True))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def data_builds(source: str, files: list[tuple[str, Path]], plan: dict) -> dict[str, str]:
    folder = lp.PATH_SOURCES[source]
    days = hashlib.sha256()
    for day, path in files:
        days.update(f"{day}:{path.stat().st_size}\n".encode())
    builds = {
        f"{folder}_odds_timeseries": f"processed/{folder}/odds_timeseries@{days.hexdigest()[:16]}"
    }
    for table in ("races", "runners"):
        path = paths.processed_dir(folder) / f"{table}.parquet"
        builds[f"{folder}_{table}"] = f"processed/{folder}/{table}.parquet@{_file_digest(path)}"
    p_model = paths.data_dir() / plan["e22"]["p_model"]["file"]
    found = _file_digest(p_model) if p_model.is_file() else "missing"
    builds["p_model"] = f"{plan['e22']['p_model']['file']}@{found}"
    return builds


def starters(win: pd.DataFrame) -> pd.DataFrame:
    """Each race's starters: the final odds, and whether the runner won."""
    final = win[win["is_final_capture"] & (win["combination"] != nc.POOL_ROW) & (win["odds"] > 0)]
    refund = final["is_refund"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return pd.DataFrame(
        {
            "race_id": final["race_id"].to_numpy(),
            "horse_id": final["combination"].to_numpy(),
            "odds_final": final["odds"].to_numpy(dtype=float),
            "won": final["dividend"].notna().to_numpy() & ~refund,
        }
    )


def late_changes(win: pd.DataFrame, windows: list[float]) -> pd.DataFrame:
    """`late_change` per runner at each window, the close at the race's scheduled post time."""
    prices = win.loc[win["combination"] != nc.POOL_ROW, ["race_id", "combination", "snapshot_utc",
                                                          "post_utc", "odds"]]  # fmt: skip
    parts = []
    for race_id, g in prices.groupby("race_id", sort=False):
        post = g["post_utc"].dropna()
        if post.empty:
            continue
        captures = g.rename(columns={"combination": "horse_id", "snapshot_utc": "fetched_utc"})
        changes = {
            f"late_change_{w:g}": lp.late_change(captures, close_utc=post.iloc[0], window_min=w)
            for w in windows
        }
        parts.append(pd.DataFrame(changes).reset_index().assign(race_id=race_id))
    if not parts:
        return pd.DataFrame(
            columns=["race_id", "horse_id", *(f"late_change_{w:g}" for w in windows)]
        )
    return pd.concat(parts, ignore_index=True)


def read_paths(files, windows, grades, log, started):
    """Pass 1: starters with their late changes, and C1's WIN cases and pool totals."""
    runner_parts, case_parts, growth_parts, case_chunks, growth_chunks = [], [], [], [], []
    counts: dict[str, int] = defaultdict(int)
    for i, (day, path) in enumerate(files):
        win = read_day(path)
        day_cases, day_growth, c = nc.extract(win, grades=grades)
        case_parts.append(day_cases)
        growth_parts.append(day_growth)
        for key, value in c.items():
            counts[f"c1_{key}"] += value
        rows = starters(win).merge(
            late_changes(win, windows), on=["race_id", "horse_id"], how="left"
        )
        del win
        winners = rows.groupby("race_id")["won"].transform("sum")
        counts["races_read"] += rows["race_id"].nunique()
        counts["void_races_dropped"] += rows.loc[winners == 0, "race_id"].nunique()
        rows = rows[winners > 0].assign(dead_heat=(winners >= 2)[winners > 0], day=day)
        runner_parts.append(rows.assign(meeting=rows["race_id"].str.rsplit(":", n=1).str[0]))
        if len(case_parts) == 100 or i == len(files) - 1:
            case_chunks.append(nc.Cases.concat(case_parts))
            growth_chunks.append(nc.Growth.concat(growth_parts))
            case_parts, growth_parts = [], []
            log(f"  {i + 1}/{len(files)} days read ({_progress(started)})")
    if not runner_parts:
        raise ValueError("no development day file to read")
    runners = pd.concat(runner_parts, ignore_index=True)
    return runners, nc.Cases.concat(case_chunks), nc.Growth.concat(growth_chunks), dict(counts)


def calibrated_market(runners: pd.DataFrame, folds, n_knots: int) -> pd.Series:
    """D4's calibrated market on each fold's test days, fitted on its training days."""
    q = pd.Series(np.nan, index=runners.index)
    for train, test, _ in folds:
        fit_rows = runners[runners["day"].isin(train) & ~runners["dead_heat"]]
        model = fit_win_calibration(
            pd.DataFrame(
                {
                    "race_id": fit_rows["race_id"].to_numpy(),
                    "race_date": pd.to_datetime(fit_rows["day"]).to_numpy(),
                    "win_odds": fit_rows["odds_final"].to_numpy(),
                    "won": fit_rows["won"].to_numpy(),
                }
            ),
            n_knots=n_knots,
        )
        rows = runners["day"].isin(test)
        inv = 1.0 / runners.loc[rows, "odds_final"]
        market = inv / inv.groupby(runners.loc[rows, "race_id"]).transform("sum")
        q[rows] = model.predict_many(market.to_numpy(), runners.loc[rows, "race_id"].to_numpy())
    return q


def _label(fold) -> str:
    return f"{fold.test_start.year}Q{(fold.test_start.month - 1) // 3 + 1}"


# ---------------------------------------------------------------- the comparisons


def e23_regression(runners: pd.DataFrame, plan: dict, clusters: dict) -> tuple[list, dict]:
    low, high = sorted(plan["paper"]["range"])
    metrics, rows, coverage = [], [], []
    for w in plan["e23"]["windows_min"]:
        change = runners[f"late_change_{w:g}"]
        frame = runners[["race_id", "won", "odds_final"]].assign(late_change=change)
        for unit, cl in (
            ("race", None),
            ("day", clusters["day"]),
            ("meeting", clusters["meeting"]),
        ):
            r = lp.paper_regression(frame, clusters=cl)
            lo, hi = r.delta - Z95 * r.se, r.delta + Z95 * r.se
            metrics.append(_metric(f"pooled:{unit}", f"delta:w{w:g}", r.delta, r.n_runners, lo, hi))
            rows.append(
                {"window_min": w, "se_unit": unit, "delta": r.delta, "se": r.se}
                | {"ci95_lo": lo, "ci95_hi": hi, "overlaps_paper_range": hi >= low and lo <= high}
                | {"n_runners": r.n_runners, "n_races": r.n_races, "n_clusters": r.n_clusters}
            )
        year = runners["day"].str[:4]
        for y, g in change.groupby(year):
            coverage.append(
                {"window_min": w, "year": y, "starters": len(g)}
                | {
                    "with_late_change": int(g.notna().sum()),
                    "nonzero": int((g.fillna(0) != 0).sum()),
                }
            )
    return metrics, {"e23_regression": rows, "e23_coverage": coverage}


def _offset(frame, cols, clusters, bootstrap, label, name) -> tuple[list, list]:
    """offset_test on `frame` by every bootstrap unit, with the in-sample coefficients."""
    first = min(label.unique())
    n_races = frame.loc[frame["race_id"].map(label) != first, "race_id"].nunique()
    coef = lp.fit_offset(frame, path_cols=cols)
    metrics, rows = [], []
    for unit in bootstrap["units"]:
        iv = lp.offset_test(
            frame,
            path_cols=cols,
            clusters=clusters[unit],
            n_boot=bootstrap["n_boot"],
            level=bootstrap["level"],
            seed=bootstrap["seed"],
            folds=label,
        )
        metrics.append(
            _metric(f"pooled:{unit}", f"gain_nats:{name}", iv.mean, n_races, iv.lo, iv.hi)
        )
        rows.append(
            {"comparison": name, "unit": unit, "gain_nats": iv.mean, "lo": iv.lo, "hi": iv.hi}
            | {"n_clusters": iv.n_clusters, "n_races": n_races}
            | {f"coef_in_sample:{c}": v for c, v in coef.items()}
        )
    return metrics, rows


def e23_offset(scored, plan, clusters, bootstrap, label) -> tuple[list, dict]:
    metrics, rows = [], []
    for w in plan["e23"]["windows_min"]:
        change = scored[f"late_change_{w:g}"]
        frame = scored[["race_id", "won", "q_final"]].assign(late_change=change.fillna(0.0))
        m, r = _offset(frame, ["late_change"], clusters, bootstrap, label, f"w{w:g}")
        metrics += m
        rows += [row | {"starters_at_0": int(change.isna().sum())} for row in r]
    return metrics, {"e23_offset": rows}


def ex_ante_rows(files, folds, cases, growth, grid, plan, grades, log, started) -> pd.DataFrame:
    """Pass 2: `ex_ante_features` at every tau for every race of each fold's test days, with
    that fold's C1 fitted on its training days."""
    by_day, e24 = dict(files), plan["e24"]
    parts, counts = [], defaultdict(int)
    for train, test, fold in folds:
        c1 = nc.Nowcast.fit_cases(cases, growth, features=True, days=train)
        for day in test:
            day_parts = []
            for race_id, g in read_day(by_day[day]).groupby("race_id", sort=False):
                post = g["post_utc"].dropna()
                if post.empty:
                    counts["no_post_time"] += len(grid.minutes)
                    continue
                for tau in grid.minutes:
                    at = post.iloc[0] - pd.Timedelta(minutes=tau)
                    try:
                        f = lp.ex_ante_features(
                            g, at_utc=at, nowcast=c1, grades=grades,
                            n_draws=int(e24["n_draws"]), seed=int(e24["draw_seed"]),
                        )  # fmt: skip
                    except KeyError:  # no WIN board at t, or no C1 cell for the tau
                        counts["no_board_at_t"] += 1
                        continue
                    day_parts.append(f.reset_index().assign(race_id=race_id, tau_min=float(tau)))
            if day_parts:
                parts.append(pd.concat(day_parts, ignore_index=True))
        log(f"  E24 fold {_label(fold)}: C1 on {len(train)} days ({_progress(started)})")
        del c1
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.attrs["counts"] = dict(counts)
    return out


def e24_offset(scored, features, grid, plan, clusters, bootstrap, label) -> tuple[list, dict]:
    cols = list(plan["e24"]["features"]["columns"])
    unknown = sorted(set(cols) - set(lp.FEATURES))
    if unknown:
        raise ValueError(f"E24 features {unknown} are not ex_ante_features columns")
    metrics, rows = [], []
    for tau in grid.minutes:
        at_tau = features.loc[features["tau_min"] == float(tau), ["race_id", "horse_id", *cols]]
        frame = scored[["race_id", "horse_id", "won", "q_final"]].merge(
            at_tau, on=["race_id", "horse_id"], how="left", validate="one_to_one"
        )
        missing = {f"starters_at_0:{c}": int(frame[c].isna().sum()) for c in cols}
        frame[cols] = frame[cols].fillna(0.0)
        m, r = _offset(frame, cols, clusters, bootstrap, label, f"T-{tau:g}")
        metrics += m
        rows += [row | {"tau_min": tau, "primary": tau == grid.primary} | missing for row in r]
    return metrics, {"e24_offset": rows, "e24_counts": [features.attrs.get("counts", {})]}


def read_p_model(plan: dict, folder: str, races: np.ndarray) -> pd.DataFrame:
    """The spec's p_model per (race_id, horse_id = horse number)."""
    spec = plan["e22"]["p_model"]
    if spec["source"] != "c2_rating_oof":
        raise ValueError(f"no p_model source {spec['source']!r} is implemented")
    rating = pq.read_table(
        paths.data_dir() / spec["file"], columns=["race_id", "horse_id", "oof_fold", "logp"]
    ).to_pandas()
    rating = rating[rating["oof_fold"].notna() & rating["race_id"].isin(races)]
    if rating.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("the p_model file has two out-of-fold rows for one runner")
    numbers = pq.read_table(
        paths.processed_dir(folder) / "runners.parquet",
        columns=["race_id", "horse_id", "horse_no"],
        filters=[("race_id", "in", list(races))],
    ).to_pandas()
    joined = rating.merge(numbers, on=["race_id", "horse_id"], how="inner", validate="one_to_one")
    return pd.DataFrame(
        {
            "race_id": joined["race_id"].to_numpy(),
            "horse_id": joined["horse_no"].astype(int).astype(str).to_numpy(),
            "p_model": np.exp(joined["logp"].to_numpy(dtype=float)),
        }
    )


def e22(cases, scored, grid, plan, folder) -> tuple[list, dict]:
    test = cases.select_days(sorted(scored["day"].unique()))
    rows = pd.DataFrame(
        {
            "race_id": np.repeat(test.race_id, test.k),
            "tau_min": np.repeat(test.tau, test.k),
            "horse_id": test.combination,
            "q_tau": test.s_t,
        }
    ).merge(scored[["race_id", "horse_id", "won"]], on=["race_id", "horse_id"], how="inner")
    p_model = read_p_model(plan, folder, scored["race_id"].unique())
    rows = rows.merge(p_model, on=["race_id", "horse_id"], how="left")
    whole = rows["p_model"].notna().groupby(rows["race_id"]).transform("all")
    counts = {"races_without_a_p_model_runner": int(rows.loc[~whole, "race_id"].nunique())}
    rows = rows[whole].copy()
    case = [rows["race_id"], rows["tau_min"]]
    for col in ("q_tau", "p_model"):
        rows[col] = rows[col] / rows.groupby(case)[col].transform("sum")
    table = lp.alpha_by_tau(rows, buckets=list(grid.minutes))
    metrics = []
    for row in table.to_dict("records"):
        if not row["n_races"]:
            continue
        b, n = f"T-{row['tau_min']:g}", int(row["n_races"])
        lo, hi = row["alpha"] - Z95 * row["alpha_se"], row["alpha"] + Z95 * row["alpha_se"]
        metrics += [
            _metric(b, "alpha", row["alpha"], n, lo, hi),
            _metric(b, "gw_pvalue", row["gw_pvalue"], n),
            _metric(b, "gw_mean_loss_diff_nats", row["gw_mean"], n),
        ]
    counts["e22_cases"] = int(table["n_races"].sum())
    counts["e22_races"] = int(rows["race_id"].nunique())
    return metrics, {"e22": table.to_dict("records"), "e22_counts": [counts]}


# ---------------------------------------------------------------- records and the report


def _record(variant: Variant, *, family, source, git_sha, builds, spec_hash, trial) -> RunRecord:
    """The variant's registry record; `sources` is the path source alone."""
    params = dict(variant.params) | {"spec_sha256": spec_hash}
    if trial:
        params["trial"] = True
    if variant.error is not None:
        params["error"] = variant.error
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=family,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(source,),
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


SECTIONS = (
    ("e23_regression", "E23: the paper's regression (ex post, every development day)"),
    ("e23_coverage", "E23: starters with a late change, by year and window"),
    ("e23_offset", "E23: ln q_final + late_change against ln q_final (ex post, scored quarters)"),
    ("e24_offset", "E24: ln q_final + the features at t against ln q_final (ex ante, per tau)"),
    ("e24_counts", "E24: cases without features"),
    ("e22", "E22: the blend weight alpha by tau bucket, Giacomini-White against one blend"),
    ("e22_counts", "E22: counts"),
)


def render_report(study: Study, plan: dict, about: dict, folds: list[dict]) -> str:
    lines = [f"# The late-money path: E23, E24 and E22 ({COUNTRY}, #{TICKET})", ""]
    lines += [*(f"- {key}: {value}" for key, value in about.items()), ""]
    lines += [
        "The two windows are distinct. E23's (ex post) runs from each runner's last board at or "
        "before close - w to its last board strictly before the close (the scheduled post time), "
        "after every decision time. E24's (ex ante) runs from t - 5 min to the decision time t "
        "= post - tau, and reads nothing after t; C1 is fitted on earlier days only.",
        "",
        "Gains are per race in nats (the final's winner log loss minus the final + path's): "
        "positive means the path helps. `lo` / `hi` are one-sided 95% bounds; the path "
        "coefficients are fitted walk-forward over the scored quarters (the first only trains).",
        "",
        f"## The paper (arXiv {plan['paper']['arxiv']})",
        "",
        _markdown_table(plan["paper"]["versions"]),
        "",
        "## Counts",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
        "",
        _markdown_table(folds) if folds else "(no fold)",
    ]
    for key, title in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", "", _markdown_table(rows) if rows else "_Not computed._"]
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- the run


def run_study(
    *,
    source: str = "jravan",
    end: date | None = None,
    trial: bool = False,
    out_dir: Path | None = None,
    log=print,
) -> dict:
    if source not in lp.PATH_SOURCES:
        raise ValueError(f"no path source {source!r}; one of {sorted(lp.PATH_SOURCES)}")
    plan = config.load_country_yaml(COUNTRY, "late_path")
    waiting = pending_fields(plan)
    if waiting and not trial:
        raise RuntimeError(
            f"spec fields pending owner approval ({', '.join(waiting)}): --trial only"
        )
    grid = load_decision_grid()
    scheme = load_country_scheme(COUNTRY)
    scoring = config.load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    bootstrap = {
        "n_boot": int(scoring["n_boot"]),
        "level": float(scoring["level"]),
        "seed": int(plan["bootstrap"]["seed"]),
        "units": list(plan["bootstrap"]["units"]),
    }
    spec = build_spec(source, plan, grid, scheme, bootstrap)
    # The registry's spec hash (sorted, compact JSON), so a trial's names the spec it would freeze.
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    registry = default_registry(COUNTRY)
    family = family_id(source) + ("-trial" if trial else "")
    if not trial:  # a trial's spec may still change, so it is not frozen
        spec_hash = registry.freeze_family(family, spec)  # before any development row is read
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    git_sha = _git_sha()
    folder = lp.PATH_SOURCES[source]
    files = [(d, p) for d, p in lp.path_files(source) if end is None or d <= end.isoformat()]
    builds = data_builds(source, files, plan)
    walk = "walk_forward:" + json.dumps(dataclasses.asdict(scheme), sort_keys=True)
    windows = [float(w) for w in plan["e23"]["windows_min"]]
    params = {"windows_min": windows, "taus": list(grid.minutes), "primary_tau": grid.primary}
    variants = {
        name: Variant(hyp, name, "in_sample:development_days" if name == REGRESSION else walk,
                      dict(COMPONENTS), params)
        for name, hyp in VARIANTS.items()
    }  # fmt: skip
    record = {"family": family, "source": source, "git_sha": git_sha, "builds": builds,
              "spec_hash": spec_hash, "trial": trial}  # fmt: skip
    started = time.monotonic()
    study = Study(counts={})
    log(f"reading {len(files)} development days of {source}")
    try:
        grades = read_grades(folder)
        runners, cases, growth, counts = read_paths(files, windows, grades, log, started)
        days = sorted(runners["day"].unique())
        folds = nc.fold_days(days, scheme)
        if not folds:
            raise ValueError(f"no D9 fold over {len(days)} development days")
        runners["q_final"] = calibrated_market(runners, folds, int(plan["baseline"]["n_knots"]))
    except Exception as exc:  # every comparison is a failed record, then the error is raised
        error = f"{type(exc).__name__}: {exc}"
        for v in variants.values():
            registry.append(_record(dataclasses.replace(v, error=error), **record))
        write_summary(registry, summary_dir(COUNTRY))
        raise
    label_of_day = {d: _label(fold) for _, test, fold in folds for d in test}
    by_race = runners.groupby("race_id")[["day", "meeting"]].first()
    clusters = {"day": by_race["day"], "meeting": by_race["meeting"]}
    scored = runners[runners["q_final"].notna()].reset_index(drop=True)
    label = scored.groupby("race_id")["day"].first().map(label_of_day)
    study.counts = counts | {
        "development_days": len(days),
        "starters": len(runners),
        "races": runners["race_id"].nunique(),
        "dead_heat_races": runners.loc[runners["dead_heat"], "race_id"].nunique(),
        "scored_races": scored["race_id"].nunique(),
        "folds": len(folds),
    }
    log(f"{len(runners)} starters on {len(days)} days, {len(folds)} folds ({_progress(started)})")

    study.attempt(variants[REGRESSION], lambda: e23_regression(runners, plan, clusters))
    study.attempt(
        variants[E23_OFFSET], lambda: e23_offset(scored, plan, clusters, bootstrap, label)
    )
    log(f"E23 done ({_progress(started)})")

    def e24() -> tuple[list, dict]:
        features = ex_ante_rows(files, folds, cases, growth, grid, plan, grades, log, started)
        return e24_offset(scored, features, grid, plan, clusters, bootstrap, label)

    study.attempt(variants[E24_OFFSET], e24)
    study.attempt(variants[E22], lambda: e22(cases, scored, grid, plan, folder))
    log(f"E24 and E22 done ({_progress(started)})")

    records = [_record(v, **record) for v in study.variants]
    fold_rows = [
        {
            "fold": _label(f),
            "train_days": len(tr),
            "test_days": len(te),
            "train_end": str(f.train_end),
        }
        for tr, te, f in folds
    ]
    about = {
        "family": f"`{family}`, spec sha256 `{spec_hash}`",
        "source": source,
        "git": git_sha,
        "data": ", ".join(builds.values()),
        "pending owner approval": ", ".join(waiting) or "none",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trial (records marked params.trial, spec not frozen)": trial,
    }
    stem = f"late-path-{COUNTRY}-{source}-{spec_hash[:12]}" + ("-trial" if trial else "")
    report = write_report(
        out_dir / f"{stem}.md", render_report(study, plan, about, fold_rows), sources=(source,)
    )
    dump = {"about": about, "counts": study.counts, "folds": fold_rows, "tables": study.tables}
    write_report(
        out_dir / f"{stem}.json", json.dumps(dump, indent=1, default=str), sources=(source,)
    )
    existing = registry.records(family)
    appended = 0
    for record in records:
        if record not in existing:
            registry.append(record)
            appended += 1
    write_summary(registry, summary_dir(COUNTRY))
    return {
        "report_path": str(report),
        "family_id": family,
        "spec_hash": spec_hash,
        "records_appended": appended,
        "failed_variants": [v.name for v in study.variants if v.error],
        "registry_path": str(registry.path),
    }


def late_path_study_command(
    source: Annotated[
        str, typer.Option("--source", help="Path source: jravan or jra-logs.")
    ] = "jravan",
    end: Annotated[
        datetime | None,
        typer.Option("--end", formats=["%Y-%m-%d"], help="Last race day read (default: all)."),
    ] = None,
    trial: Annotated[
        bool,
        typer.Option(
            "--trial",
            help=f"Records go to {FAMILY}-<source>-trial marked trial; the spec is not frozen; "
            "the report is suffixed -trial. The only run while spec fields await the owner.",
        ),
    ] = False,
) -> None:
    """E23, E24 and E22 of the late-money path on D9's folds; writes the private report."""
    result = run_study(
        source=source, end=None if end is None else end.date(), trial=trial, log=typer.echo
    )
    typer.echo(json.dumps(result, indent=2))
