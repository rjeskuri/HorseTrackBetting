"""The #131 exacta study's run: freeze the families, count the inputs, score E1 on development days.

`python -m horsetrack.countries.japan.research.exacta_run {freeze|count|score}`

- freeze: ``freeze_family`` for every spec of ``exacta_study.family_specs`` in the Japan registry,
  then rewrite the committed registry summary.
- count: row counts only (no outcome is scored): development races with a FORECAST dividend,
  out-of-fold Win probabilities, the ordering fit, and #130-logged meetings with exacta boards.
- score: E1 (B2) with its variants B1, B3 and the log-score total C; every variant is a registry
  record, SPA and StepM run over E1's members, and the tables go through ``write_report``.

D14: race days are filtered with ``is_holdout_day`` from the races table's dates before any
outcome, odds or dividend is read, and the readers take only the remaining race ids.
LockedHoldout is never opened. Public JRA data only (sources: jra).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from horsetrack.core import env, paths
from horsetrack.core.eval import snooping
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import (
    FoldMetric,
    RunRecord,
    default_registry,
    summary_path,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan import odds_log, takeout
from horsetrack.countries.japan.research import exacta_study as study
from horsetrack.countries.japan.research.exacta_consistency import (
    CATEGORIES,
    harville_exacta,
    implied_share,
)

COUNTRY = "japan"
FIRST_DAY = date(2014, 6, 7)
SOURCES = ("jra",)
MAX_TEST_ID = f"{study.E1_ID}/max-test"
RACE_COLUMNS = ["race_id", "meeting_id", "race_date", "race_no", "race_class", "race_class_raw"]


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        cwd=env.repo_root(),
    )  # fmt: skip
    return out.stdout.strip()


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _fits() -> tuple[str, Path, str, float]:
    """The current v0 fits: win fit id, its out-of-fold file, ordering fit id and lambda2."""
    model_dir = paths.data_dir() / "models" / "japan" / "v0"
    names = json.loads((model_dir / "current.json").read_text(encoding="utf-8"))
    win_id = json.loads((model_dir / names["win"]).read_text(encoding="utf-8"))["fit_id"]
    oof = model_dir / f"win_calibration_oof-{win_id}.parquet"
    ordering = json.loads((model_dir / names["ordering"]).read_text(encoding="utf-8"))
    manifest = json.loads(
        (model_dir / f"manifest-ordering-{ordering['fit_id']}.json").read_text(encoding="utf-8")
    )
    if Path(manifest["oof"]["path"]).name != oof.name:
        raise ValueError(f"the ordering fit was not fitted on {oof.name}")
    return win_id, oof, ordering["fit_id"], float(ordering["lambdas"][0])


def development_races() -> pd.DataFrame:
    """Races from 2014-06-07 on development days. Reads race keys and dates only."""
    path = paths.processed_dir("jra") / "races.parquet"
    races = pq.read_table(path, columns=RACE_COLUMNS).to_pandas()
    races = races[races["race_date"] >= FIRST_DAY]
    holdout = {d: is_holdout_day(COUNTRY, d) for d in races["race_date"].unique()}
    races = races[~races["race_date"].map(holdout)]
    if any(holdout[d] for d in races["race_date"].unique()):  # pragma: no cover - a guard
        raise AssertionError("a holdout day survived the filter")
    return races.reset_index(drop=True)


def _read(path: Path, columns: list[str], race_ids: list[str], extra=None) -> pd.DataFrame:
    flt = pc.field("race_id").isin(race_ids)
    if extra is not None:
        flt = flt & extra
    return pq.read_table(path, columns=columns, filters=flt).to_pandas()


def logged_exacta_meetings() -> list[str]:
    """#130-logged development meetings whose log holds FORECAST rows (holdout days unopened)."""
    root = paths.data_dir() / "odds_snapshots" / "jra"
    found = []
    for day_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        day = date.fromisoformat(day_dir.name)
        if is_holdout_day(COUNTRY, day):
            continue
        for log in sorted(day_dir.glob("*.jsonl.gz")):
            if any(r.get("pool") == "FORECAST" for r in odds_log.read_rows(log)):
                found.append(f"{day_dir.name}:{log.name.split('.')[0]}")
    return found


def count() -> dict:
    """Row counts only: nothing is scored."""
    races = development_races()
    ids = races["race_id"].tolist()
    win_id, oof, ordering_id, lambda2 = _fits()
    dividends = paths.processed_dir("jra") / "dividends.parquet"
    forecast = pq.read_table(
        dividends, columns=["race_id"], filters=pc.field("race_id").isin(ids)
        & (pc.field("pool") == "FORECAST"),
    )  # fmt: skip
    oof_ids = pq.read_table(oof, columns=["race_id"], filters=pc.field("race_id").isin(ids))
    logged = logged_exacta_meetings()
    return {
        "development_races_from_2014_06_07": len(races),
        "development_meetings": int(races["meeting_id"].nunique()),
        "forecast_dividend_rows": forecast.num_rows,
        "races_with_a_forecast_row": len(pc.unique(forecast["race_id"])),
        "win_fit": win_id,
        "oof_rows": oof_ids.num_rows,
        "oof_races": len(pc.unique(oof_ids["race_id"])),
        "ordering_fit": ordering_id,
        "lambda2_in_fold": lambda2,
        "logged_exacta_meetings": len(logged),
        "logged_exacta_meeting_ids": logged,
    }


def load_races() -> tuple[list[dict], dict]:
    """The scoring inputs of every development race with out-of-fold Win probabilities."""
    races = development_races()
    win_id, oof_path, ordering_id, lambda2 = _fits()
    oof = _read(oof_path, ["race_id", "horse_no", "p_cal"], races["race_id"].tolist())
    races = races[races["race_id"].isin(set(oof["race_id"]))]
    ids = races["race_id"].tolist()
    runners = _read(
        paths.processed_dir("jra") / "runners.parquet",
        ["race_id", "horse_no", "win_odds", "is_scratched"],
        ids,
    )
    runners = runners[~runners["is_scratched"].astype(bool)]
    dividends = _read(
        paths.processed_dir("jra") / "dividends.parquet",
        ["race_id", "combination", "dividend", "is_refund"],
        ids,
        pc.field("pool") == "FORECAST",
    )
    odds = {k: dict(zip(g["horse_no"], g["win_odds"], strict=True)) for k, g in
            runners.groupby("race_id")}  # fmt: skip
    p = {k: dict(zip(g["horse_no"], g["p_cal"], strict=True)) for k, g in oof.groupby("race_id")}
    rows: dict[str, list] = {}
    for r in dividends.itertuples(index=False):
        combo = tuple(int(h) for h in str(r.combination).split(","))
        rows.setdefault(r.race_id, []).append((combo, float(r.dividend), bool(r.is_refund)))

    out, dropped = [], {}
    for race in races.itertuples(index=False):
        field, probs = odds.get(race.race_id, {}), p.get(race.race_id, {})
        if set(field) != set(probs) or not all(math.isfinite(v) and v > 0 for v in field.values()):
            dropped["p_or_odds_mismatch"] = dropped.get("p_or_odds_mismatch", 0) + 1
            continue
        t = takeout.race_takeout(race.race_id, "FORECAST", race_class_raw=race.race_class_raw)
        out.append(
            {
                "race_id": race.race_id,
                "race_date": race.race_date,
                "meeting_id": race.meeting_id,
                "race_no": int(race.race_no),
                "race_class": race.race_class,
                "win_odds": {int(h): float(v) for h, v in field.items()},
                "p": {int(h): float(v) for h, v in probs.items()},
                "rate": t.payout_rate,
                "boost_may_be_withheld": t.boost_may_be_withheld,
                "exacta": rows.get(race.race_id, []),
            }
        )
    base = paths.processed_dir("jra")
    meta = {
        "win_fit": win_id,
        "ordering_fit": ordering_id,
        "lambda2_in_fold": lambda2,
        "dropped_before_scoring": dropped,
        "data_builds": {
            **{
                f"jra_{name}": f"processed/jra/{name}.parquet@{_digest(base / f'{name}.parquet')}"
                for name in ("races", "runners", "dividends")
            },
            "win_oof": f"{oof_path.name}@{_digest(oof_path)}",
        },
        "first_day": min(r["race_date"] for r in out).isoformat(),
        "last_day": max(r["race_date"] for r in out).isoformat(),
    }
    return out, meta


def _log_score(races: list[dict], lambda2: float) -> pd.DataFrame:
    """Per scored race: log q_w - log p_w (C); races paying at or below 110 yen are left out."""
    rows = []
    for race in races:
        if is_holdout_day(COUNTRY, race["race_date"]):
            continue
        reason = study.exclusion_reason(
            n_starters=len(race["win_odds"]),
            boost_may_be_withheld=race["boost_may_be_withheld"],
            exacta=race["exacta"],
        )
        if reason is not None:
            continue
        (first, second), dividend, _ = race["exacta"][0]
        q = implied_share(dividend, rate=race["rate"])
        if math.isnan(q):
            continue
        horses = sorted(race["p"])
        model = harville_exacta(np.array([race["p"][h] for h in horses]), lambda2=lambda2)
        p_w = model[horses.index(first), horses.index(second)]
        rows.append((race["race_id"], race["meeting_id"], math.log(q) - math.log(p_w)))
    return pd.DataFrame(rows, columns=["race_id", "meeting_id", "gain"]).set_index("race_id")


def _record(family, variant, params, components, metrics, meta, sha, status="scored"):
    return RunRecord(
        hypothesis_id="E43",
        family_id=family,
        variant=variant,
        ticket=131,
        git_sha=sha,
        data_builds=meta["data_builds"],
        sources=SOURCES,
        fold_scheme="oof:#114 walk-forward test quarters (folds.yaml)",
        components=components,
        params=params,
        metrics=tuple(metrics),
        status=status,
    )


def _member_metrics(sub: pd.DataFrame, column: str, ratio: bool, seed: int, n_boot: int) -> dict:
    rates = sub["rate"].to_numpy() if ratio else np.ones(len(sub))
    upper = study.rho_interval(sub[column], rates, sub["meeting_id"], alternative="less",
                               n_boot=n_boot, seed=seed)  # fmt: skip
    lower = study.rho_interval(sub[column], rates, sub["meeting_id"], alternative="greater",
                               n_boot=n_boot, seed=seed)  # fmt: skip
    return {"rho": upper.rho, "se": upper.se, "lo": lower.bound, "hi": upper.bound,
            "n": len(sub), "meetings": upper.n_meetings}  # fmt: skip


def score(registry, races: list[dict], meta: dict, sha: str, *, n_boot: int = 10000) -> dict:
    """Score E1, B1, B3 and C on ``races``; append every variant to ``registry``; SPA and StepM
    over E1's members. Refuses to run unless every family is frozen in ``registry``."""
    for family_id in study.family_specs():
        if registry.frozen_spec_hash(family_id) is None:
            raise RuntimeError(f"{family_id} is not frozen; run `freeze` first")
    lambdas = {"in_fold": meta["lambda2_in_fold"], 1.0: 1.0}
    frames = {
        key: study.score_races(races, holdout_day=is_holdout_day, lambda2=value)
        for key, value in lambdas.items()
    }
    meta["excluded"] = frames["in_fold"].attrs["excluded"]
    meta["scored_races"] = int(frames["in_fold"]["race_id"].nunique())
    meta["scored_meetings"] = int(frames["in_fold"]["meeting_id"].nunique())
    results, seed = [], 131
    plan = [("B2", study.E1_ID, "b2", True, list(lambdas)),
            ("B1", study.E1_B1_ID, "b1", True, ["in_fold"]),
            ("B3", study.E1_B3_ID, "b3", False, list(lambdas))]  # fmt: skip
    for estimator, family, column, ratio, keys in plan:
        for key in keys:
            frame = frames[key]
            for category in CATEGORIES:
                for cell in study.SEGMENT_CELLS:
                    sub = frame[(frame["category"] == category) & frame[cell]]
                    m = _member_metrics(sub, column, ratio, seed, n_boot)
                    results.append({"estimator": estimator, "family": family, "lambda2": key,
                                    "category": category, "cell": cell, **m})  # fmt: skip
    table = pd.DataFrame(results)
    table["rho_eb"] = np.nan
    for key in lambdas:
        for category in CATEGORIES:
            idx = table.index[
                (table["estimator"] == "B2")
                & (table["lambda2"].astype(str) == str(key))
                & (table["category"] == category)
            ]
            table.loc[idx, "rho_eb"] = study.eb_shrink(table.loc[idx, "rho"], table.loc[idx, "se"])

    components = {"win": meta["win_fit"]}
    for row in table.itertuples(index=False):
        variant = f"{row.estimator.lower()}|lambda2={row.lambda2}|{row.category}|{row.cell}"
        comps = {**components, "ordering": meta["ordering_fit"] if row.lambda2 == "in_fold"
                 else "plain_harville"}  # fmt: skip
        if row.estimator == "B1":
            comps = dict(components)
        metric = "rho" if row.estimator != "B3" else "mean_return"
        n = int(row.n)
        metrics = [
            FoldMetric("development", metric, float(row.rho), float(row.lo), float(row.hi), n),
            FoldMetric("development", "bootstrap_se", float(row.se), None, None, n),
        ]
        if row.estimator == "B2":
            metrics.append(FoldMetric("development", "rho_eb", float(row.rho_eb), None, None, n))
        params = {"estimator": row.estimator, "lambda2": row.lambda2, "category": row.category,
                  "cell": row.cell, "direction": study.DIRECTIONS[row.category]}  # fmt: skip
        registry.append(_record(row.family, variant, params, comps, metrics, meta, sha))

    # C: the log-score total, one paired comparison per lambda2.
    c_rows = []
    for key, value in lambdas.items():
        gain = _log_score(races, value)
        ci = paired_bootstrap(gain["gain"], gain["meeting_id"], n_boot=n_boot, level=0.95,
                              seed=seed)  # fmt: skip
        c_rows.append({"lambda2": key, "mean": ci.mean, "lo": ci.lo, "hi": ci.hi,
                       "n": len(gain), "meetings": ci.n_clusters})  # fmt: skip
        comps = {**components, "ordering": meta["ordering_fit"] if key == "in_fold"
                 else "plain_harville"}  # fmt: skip
        registry.append(_record(
            study.C_ID, f"c|lambda2={key}", {"estimator": "C", "lambda2": key}, comps,
            [FoldMetric("development", "log_score_gain", ci.mean, ci.lo, ci.hi, len(gain))],
            meta, sha,
        ))  # fmt: skip

    # SPA and StepM over E1's members (lambda2 x category x cell).
    first = frames["in_fold"].drop_duplicates("race_id").set_index("race_id")
    losses = {}
    for key in lambdas:
        frame = frames[key]
        for category in CATEGORIES:
            sub = frame[frame["category"] == category].set_index("race_id")
            for cell in study.SEGMENT_CELLS:
                name = f"b2|lambda2={key}|{category}|{cell}"
                losses[name] = sub["e1_loss"].where(sub[cell], 0.0).reindex(first.index)
    losses = pd.DataFrame(losses)
    bench = pd.Series(0.0, index=losses.index)
    kw = {"clusters": first["meeting_id"], "order": pd.to_datetime(first["race_date"]),
          "block_size": 5, "n_boot": n_boot, "seed": seed, "registry": registry,
          "family_id": study.E1_ID}  # fmt: skip
    spa = snooping.spa(bench, losses, **kw)
    stepm = snooping.stepm(bench, losses, level=0.95, **kw)
    registry.append(_record(
        MAX_TEST_ID, "spa+stepm",
        {"family": study.E1_ID, "superior": sorted(stepm.superior), "block_size": 5},
        {**components, "ordering": meta["ordering_fit"]},
        [FoldMetric("development", "spa_pvalue", spa.pvalue, spa.pvalue_lower,
                    spa.pvalue_upper, spa.n_clusters),
         FoldMetric("development", "stepm_superior", float(len(stepm.superior)), None, None,
                    stepm.n_clusters),
         FoldMetric("development", "family_size", float(spa.family_size), None, None,
                    spa.n_rules)],
        meta, sha,
    ))  # fmt: skip
    summary = {
        "meta": meta,
        "spa": {"pvalue": spa.pvalue, "lower": spa.pvalue_lower, "upper": spa.pvalue_upper,
                "n_rules": spa.n_rules, "family_size": spa.family_size,
                "n_clusters": spa.n_clusters},
        "stepm": {"superior": stepm.superior, "family_size": stepm.family_size},
        "members": table.to_dict(orient="records"),
        "log_score": c_rows,
    }  # fmt: skip
    return summary


def write_outputs(summary: dict, sha: str, out: Path) -> None:
    """The full member table and summary, through write_report (aggregates only)."""
    text = json.dumps(summary, indent=1, default=str, allow_nan=True)
    write_report(out / f"e1-{sha[:12]}.json", text, sources=SOURCES)
    table = pd.DataFrame(summary["members"])
    lines = [f"# #131 exacta study: E1 results ({sha[:12]})", "",
             "| estimator | lambda2 | category | cell | n | estimate | 95% lo | 95% hi | EB |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for r in table.itertuples(index=False):
        eb = "" if pd.isna(r.rho_eb) else f"{r.rho_eb:.3f}"
        lines.append(f"| {r.estimator} | {r.lambda2} | {r.category} | {r.cell} | {r.n} | "
                     f"{r.rho:.3f} | {r.lo:.3f} | {r.hi:.3f} | {eb} |")  # fmt: skip
    write_report(out / f"e1-{sha[:12]}.md", "\n".join(lines) + "\n", sources=SOURCES)


def freeze() -> dict[str, str]:
    registry = default_registry(COUNTRY)
    hashes = {fid: registry.freeze_family(fid, spec) for fid, spec in study.family_specs().items()}
    write_summary(registry, summary_path(COUNTRY))
    return hashes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.research.exacta_run"
    )
    parser.add_argument("action", choices=["freeze", "count", "score"])
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze()
    elif args.action == "count":
        result = count()
    else:
        sha, registry = _git_sha(), default_registry(COUNTRY)
        races, meta = load_races()
        try:
            result = score(registry, races, meta, sha)
        except Exception as exc:  # the failed run is a registry record too
            registry.append(_record(
                MAX_TEST_ID, "run", {"error": f"{type(exc).__name__}: {exc}"[:500]}, {}, [],
                meta, sha, status="failed",
            ))  # fmt: skip
            raise
        write_outputs(result, sha, paths.reports_dir() / "japan" / "131-exacta")
        write_summary(registry, summary_path(COUNTRY))
        result = {k: result[k] for k in ("meta", "spa", "stepm", "log_score")}
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
