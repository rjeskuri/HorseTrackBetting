"""The #131 exacta study's run: freeze the families, count the inputs, score E1 on development days.

`python -m horsetrack.countries.japan.research.exacta_run {freeze|count|score|max-test|breakage}`

- freeze: ``freeze_family`` for every spec of ``exacta_study.family_specs`` in the Japan registry.
  It records the specs only: the committed summary lists runs, and a freeze is not one.
- count: row counts only (no outcome is scored): development races with a FORECAST dividend,
  out-of-fold Win probabilities, the ordering fit, and #130-logged meetings with exacta boards.
- score: E1 (B2) with its variants B1, B3 and the log-score total C; every variant is a registry
  record, SPA and StepM run over E1's members, and the tables go through ``write_report``.
- breakage: descriptive, not a test: the share of B2's paid return that rounding the dividend
  down to 10 yen takes, per lambda2, category and cell, and rho with half a step added back.

Before writing anything, score and max-test refuse (``RunRefused``) a family frozen with a spec
other than the one in use, and a Win or ordering fit other than the one the family's scored
records hold: a refreshed fit needs its own frozen family. A rerun reuses identical records, so
an interrupted run completes and a repeated one appends nothing. The max-test command runs both
checks before it builds any frame.

A member whose cell is empty, or a statistic that cannot be computed (a 0 / 0 breakage share, a
shrinkage without a positive SE), keeps its row with its count; the statistic is NaN in the
returned rows, ``n/a`` in a markdown table, ``null`` in a JSON report, and in the registry it is no
FoldMetric but an entry of ``params["unavailable"]`` with the reason. Report names hold the commit
and a digest of ``data_builds``, so two data builds at one commit keep separate files.

D14: race days are filtered with ``is_holdout_day`` from the races table's dates (the only column
read for every row) before any outcome, odds, class or dividend is read, and the readers take only
the remaining race ids. ``data_builds`` fingerprints those development rows, never a whole file.
LockedHoldout is never opened. Public JRA data only (sources: jra).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
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
    summary_dir,
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
SCORED_FAMILIES = (study.E1_ID, study.E1_B1_ID, study.E1_B3_ID, study.C_ID)
BREAKAGE_ID = "japan/131-exacta-breakage"
BREAKAGE_STEP = 10.0  # JRA pays per 100 yen rounded down to 10 yen
STEPM_FWER = 0.05  # the spec's 95% level as a family-wise error rate
RACE_COLUMNS = ["race_id", "meeting_id", "race_date", "race_no", "race_class", "race_class_raw"]


class RunRefused(RuntimeError):
    """A run refused before it computes or records anything: not a failed run."""


def _git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        cwd=env.repo_root(),
    )  # fmt: skip
    return out.stdout.strip()


def _fingerprint(frame: pd.DataFrame) -> str:
    """16 hex digits of sha256 over the rows the run read (values as text, rows sorted): it
    depends on those rows alone, not on the file's other rows, row order or layout."""
    text = frame.astype(str)
    text = text.sort_values(list(text.columns), kind="mergesort").reset_index(drop=True)
    h = hashlib.sha256("\x1f".join(text.columns).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(text, index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def _spec_hash(spec: dict) -> str:
    """The hash ``Registry.freeze_family`` gives ``spec``."""
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_tag(meta: dict) -> str:
    """The data part of a report's name: 8 hex digits of sha256 over ``data_builds``."""
    text = json.dumps(meta["data_builds"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _stem(prefix: str, sha: str, meta: dict) -> str:
    return f"{prefix}-{sha[:12]}-{_build_tag(meta)}"


def _available(value) -> bool:
    return value is not None and math.isfinite(value)


def _json_safe(value):
    """``value`` with every non-finite float as None, so a JSON report holds null, not NaN."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def _fmt(value, digits: int) -> str:
    return f"{value:.{digits}f}" if _available(value) else "n/a"


def _metrics(fold: str, entries, n: int) -> tuple[list[FoldMetric], list[dict]]:
    """FoldMetrics of the available ``(metric, value, lo, hi, reason)`` entries; the others as
    ``params["unavailable"]`` entries carrying their reason."""
    metrics, unavailable = [], []
    for metric, value, lo, hi, reason in entries:
        if _available(value):
            metrics.append(FoldMetric(fold, metric, float(value), lo, hi, n))
        else:
            unavailable.append({"fold": fold, "metric": metric, "reason": reason})
    return metrics, unavailable


def _record_key(record: RunRecord) -> str:
    return json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))


def _appender(registry) -> Callable[[RunRecord], None]:
    """``registry.append`` that skips a record already in the registry, so a rerun of the same
    commit and data reuses its records and an interrupted run completes."""
    seen = {_record_key(r) for r in registry.records()}

    def append(record: RunRecord) -> None:
        key = _record_key(record)
        if key not in seen:
            registry.append(record)
            seen.add(key)

    return append


def _check_frozen(registry, families: Iterable[str]) -> None:
    """Refuse unless each family is frozen with the hash of the spec in use."""
    specs = study.family_specs()
    for family_id in families:
        frozen = registry.frozen_spec_hash(family_id)
        if frozen is None:
            raise RunRefused(f"{family_id} is not frozen; run `freeze` first")
        in_use = _spec_hash(specs[family_id])
        if frozen != in_use:
            raise RunRefused(
                f"{family_id} is frozen with spec {frozen[:12]}, not the spec in use "
                f"({in_use[:12]}): a changed design is a new family"
            )


def _check_fits(registry, meta: dict, families: Iterable[str]) -> None:
    """Refuse a Win or ordering fit other than the one the families' scored records hold, so a
    family's records (and its max test) never mix fits."""
    families = set(families)
    want = {"win": meta["win_fit"], "ordering": meta["ordering_fit"]}
    for r in registry.records():
        if r.family_id not in families or r.status != "scored":
            continue
        for key, fit in want.items():
            held = r.components.get(key)
            if held not in (None, fit, "plain_harville"):
                raise RunRefused(
                    f"{r.family_id} holds scored records under {key} fit {held}, not {fit}: "
                    "a refreshed fit needs its own frozen family"
                )


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
    """Races from 2014-06-07 on development days. Every row's date is read to classify the day;
    the other columns are read for development days only."""
    path = paths.processed_dir("jra") / "races.parquet"
    days = pc.unique(pq.read_table(path, columns=["race_date"])["race_date"]).to_pylist()
    development = [d for d in days if d >= FIRST_DAY and not is_holdout_day(COUNTRY, d)]
    races = pq.read_table(
        path, columns=RACE_COLUMNS, filters=pc.field("race_date").isin(development)
    ).to_pandas()
    if any(is_holdout_day(COUNTRY, d) for d in races["race_date"].unique()):  # pragma: no cover
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
        "lambda2_full_development": lambda2,
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
    runners_read = _read(
        paths.processed_dir("jra") / "runners.parquet",
        ["race_id", "horse_no", "win_odds", "is_scratched"],
        ids,
    )
    runners = runners_read[~runners_read["is_scratched"].astype(bool)]
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
    # "<identity>@<digest>": the identity is the table, the digest its development rows read
    # here (holdout rows are never read), so a data refresh is a rerun, not a new variant.
    read = {"races": races, "runners": runners_read, "dividends": dividends}
    meta = {
        "win_fit": win_id,
        "ordering_fit": ordering_id,
        "lambda2_full_development": lambda2,
        "dropped_before_scoring": dropped,
        "data_builds": {
            **{
                f"jra_{name}": f"processed/jra/{name}.parquet@{_fingerprint(frame)}"
                for name, frame in read.items()
            },
            "win_oof": f"{oof_path.name}@{_fingerprint(oof)}",
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


EMPTY_CELL = "no scored race in the cell"


def _member_metrics(sub: pd.DataFrame, column: str, ratio: bool, seed: int, n_boot: int) -> dict:
    """rho, its meeting-bootstrap SE and one-sided bounds; all NaN for an empty cell. A cell of
    one meeting redraws that meeting in every replicate, so its SE is 0 (not float residue)."""
    if sub.empty:
        return {"rho": math.nan, "se": math.nan, "lo": math.nan, "hi": math.nan, "n": 0,
                "meetings": 0}  # fmt: skip
    rates = sub["rate"].to_numpy() if ratio else np.ones(len(sub))
    upper = study.rho_interval(sub[column], rates, sub["meeting_id"], alternative="less",
                               n_boot=n_boot, seed=seed)  # fmt: skip
    lower = study.rho_interval(sub[column], rates, sub["meeting_id"], alternative="greater",
                               n_boot=n_boot, seed=seed)  # fmt: skip
    se = 0.0 if upper.n_meetings == 1 else upper.se
    return {"rho": upper.rho, "se": se, "lo": lower.bound, "hi": upper.bound,
            "n": len(sub), "meetings": upper.n_meetings}  # fmt: skip


def score(registry, races: list[dict], meta: dict, sha: str, *, n_boot: int = 10000) -> dict:
    """Score E1, B1, B3 and C on ``races``; append every variant to ``registry``; SPA and StepM
    over E1's members. ``RunRefused`` before anything is computed or written unless every scored
    family is frozen with the spec in use and its scored records hold ``meta``'s fits. Records
    already in ``registry`` are reused, not appended again."""
    _check_frozen(registry, SCORED_FAMILIES)
    _check_fits(registry, meta, (*SCORED_FAMILIES, MAX_TEST_ID))
    append = _appender(registry)
    lambdas = {study.FITTED: meta["lambda2_full_development"], 1.0: 1.0}
    frames = score_frames(races, meta)
    meta["excluded"] = frames[study.FITTED].attrs["excluded"]
    meta["scored_races"] = int(frames[study.FITTED]["race_id"].nunique())
    meta["scored_meetings"] = int(frames[study.FITTED]["meeting_id"].nunique())
    results, seed = [], 131
    plan = [("B2", study.E1_ID, "b2", True, list(lambdas)),
            ("B1", study.E1_B1_ID, "b1", True, [study.FITTED]),
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
            # Shrunk among the members with a finite estimate and a positive SE; the others
            # keep rho_eb NaN (no precision is invented for a zero-SE or empty cell).
            rho, se = table.loc[idx, "rho"], table.loc[idx, "se"]
            usable = idx[(np.isfinite(rho) & np.isfinite(se) & (se > 0)).to_numpy()]
            if len(usable):
                table.loc[usable, "rho_eb"] = study.eb_shrink(
                    table.loc[usable, "rho"], table.loc[usable, "se"]
                )

    components = {"win": meta["win_fit"]}
    for row in table.itertuples(index=False):
        variant = f"{row.estimator.lower()}|lambda2={row.lambda2}|{row.category}|{row.cell}"
        comps = {**components, "ordering": meta["ordering_fit"] if row.lambda2 == study.FITTED
                 else "plain_harville"}  # fmt: skip
        if row.estimator == "B1":
            comps = dict(components)
        metric = "rho" if row.estimator != "B3" else "mean_return"
        entries = [
            (metric, row.rho, float(row.lo), float(row.hi), EMPTY_CELL),
            ("bootstrap_se", row.se, None, None, EMPTY_CELL),
        ]
        if row.estimator == "B2":
            why = EMPTY_CELL if row.n == 0 else "bootstrap SE is 0: no precision to shrink with"
            entries.append(("rho_eb", row.rho_eb, None, None, why))
        metrics, unavailable = _metrics("development", entries, int(row.n))
        params = {"estimator": row.estimator, "lambda2": row.lambda2, "category": row.category,
                  "cell": row.cell, "direction": study.DIRECTIONS[row.category]}  # fmt: skip
        if unavailable:
            params["unavailable"] = unavailable
        append(_record(row.family, variant, params, comps, metrics, meta, sha))

    # C: the log-score total, one paired comparison per lambda2.
    c_rows = []
    for key, value in lambdas.items():
        gain = _log_score(races, value)
        ci = paired_bootstrap(gain["gain"], gain["meeting_id"], n_boot=n_boot, level=0.95,
                              seed=seed)  # fmt: skip
        c_rows.append({"lambda2": key, "mean": ci.mean, "lo": ci.lo, "hi": ci.hi,
                       "n": len(gain), "meetings": ci.n_clusters})  # fmt: skip
        comps = {**components, "ordering": meta["ordering_fit"] if key == study.FITTED
                 else "plain_harville"}  # fmt: skip
        append(_record(
            study.C_ID, f"c|lambda2={key}", {"estimator": "C", "lambda2": key}, comps,
            [FoldMetric("development", "log_score_gain", ci.mean, ci.lo, ci.hi, len(gain))],
            meta, sha,
        ))  # fmt: skip

    return {
        "meta": meta,
        **max_test(registry, frames, meta, sha, n_boot=n_boot),
        "members": table.to_dict(orient="records"),
        "log_score": c_rows,
    }


def score_frames(races: list[dict], meta: dict) -> dict[object, pd.DataFrame]:
    lambdas = {study.FITTED: meta["lambda2_full_development"], 1.0: 1.0}
    return {
        key: study.score_races(races, holdout_day=is_holdout_day, lambda2=value)
        for key, value in lambdas.items()
    }


def max_test(registry, frames, meta, sha, *, n_boot: int = 10000, note: str | None = None):
    """SPA and StepM over E1's members (lambda2 x category x cell), one registry record.

    The spec's level 0.95 is the confidence level: StepM's family-wise error (snooping.stepm's
    ``level``, arch's ``size``) is 0.05. ``RunRefused`` before anything is computed unless E1 is
    frozen with the spec in use and its scored records hold ``meta``'s fits; an identical
    record already in ``registry`` is reused."""
    _check_frozen(registry, (study.E1_ID,))
    _check_fits(registry, meta, (study.E1_ID, MAX_TEST_ID))
    first = frames[study.FITTED].drop_duplicates("race_id").set_index("race_id")
    losses = {}
    for key, frame in frames.items():
        for category in CATEGORIES:
            sub = frame[frame["category"] == category].set_index("race_id")
            for cell in study.SEGMENT_CELLS:
                name = f"b2|lambda2={key}|{category}|{cell}"
                losses[name] = sub["e1_loss"].where(sub[cell], 0.0).reindex(first.index)
    losses = pd.DataFrame(losses)
    bench = pd.Series(0.0, index=losses.index)
    kw = {"clusters": first["meeting_id"], "order": pd.to_datetime(first["race_date"]),
          "block_size": 5, "n_boot": n_boot, "seed": 131, "registry": registry,
          "family_id": study.E1_ID}  # fmt: skip
    spa = snooping.spa(bench, losses, **kw)
    stepm = snooping.stepm(bench, losses, level=STEPM_FWER, **kw)
    params = {"family": study.E1_ID, "superior": sorted(stepm.superior), "block_size": 5,
              "stepm_fwer": STEPM_FWER}  # fmt: skip
    if note:
        params["note"] = note
    _appender(registry)(_record(
        MAX_TEST_ID, "spa+stepm", params,
        {"win": meta["win_fit"], "ordering": meta["ordering_fit"]},
        [FoldMetric("development", "spa_pvalue", spa.pvalue, spa.pvalue_lower,
                    spa.pvalue_upper, spa.n_clusters),
         FoldMetric("development", "stepm_superior", float(len(stepm.superior)), None, None,
                    stepm.n_clusters),
         FoldMetric("development", "family_size", float(spa.family_size), None, None,
                    spa.n_rules)],
        meta, sha,
    ))  # fmt: skip
    return {
        "spa": {"pvalue": spa.pvalue, "lower": spa.pvalue_lower, "upper": spa.pvalue_upper,
                "n_rules": spa.n_rules, "family_size": spa.family_size,
                "n_clusters": spa.n_clusters},
        "stepm": {"superior": stepm.superior, "family_size": stepm.family_size,
                  "fwer": STEPM_FWER},
    }  # fmt: skip


def write_outputs(summary: dict, sha: str, out: Path) -> None:
    """The full member table and summary, through write_report (aggregates only)."""
    stem = _stem("e1", sha, summary["meta"])
    text = json.dumps(_json_safe(summary), indent=1, default=str, allow_nan=False)
    write_report(out / f"{stem}.json", text, sources=SOURCES)
    table = pd.DataFrame(summary["members"])
    lines = [f"# #131 exacta study: E1 results ({sha[:12]})", "",
             "| estimator | lambda2 | category | cell | n | estimate | 95% lo | 95% hi | EB |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for r in table.itertuples(index=False):
        eb = "" if r.estimator != "B2" else _fmt(r.rho_eb, 3)
        lines.append(f"| {r.estimator} | {r.lambda2} | {r.category} | {r.cell} | {r.n} | "
                     f"{_fmt(r.rho, 3)} | {_fmt(r.lo, 3)} | {_fmt(r.hi, 3)} | {eb} |")  # fmt: skip
    write_report(out / f"{stem}.md", "\n".join(lines) + "\n", sources=SOURCES)


def breakage(registry, races: list[dict], meta: dict, sha: str, *, n_boot: int = 10000):
    """Descriptive, not a test (review of 0c39a8d): how much of B2's return rounding takes.

    JRA pays per 100 yen rounded down to 10 yen, so the unrounded dividend lies in [D, D + 10).
    Taken at D + 5 (uniform within the step, as ``implied_share`` does), the unrounded return is
    R' = R (D + 5) / D. Per lambda2, category and cell, ``breakage_share`` = 1 - sum R / sum R'
    (a pool fair before rounding has a paid rho of about 1 - breakage_share) and
    ``rho_unrounded`` = sum R' / sum rate, null 1, with E1's one-sided meeting-bootstrap bounds.
    E1's races and frozen loss are unchanged; one registry record per lambda2. ``RunRefused``
    unless E1 is frozen with the spec in use and ``meta``'s fits are E1's. Holdout races are
    dropped (D14) before any dividend is read. An empty cell keeps its row with every statistic
    NaN; a cell without a winning return has rho 0 and a NaN share (0 / 0)."""
    _check_frozen(registry, (study.E1_ID,))
    _check_fits(registry, meta, (study.E1_ID,))
    races = [r for r in races if not is_holdout_day(COUNTRY, r["race_date"])]
    dividend = {r["race_id"]: r["exacta"][0][1] for r in races if r["exacta"]}
    append = _appender(registry)
    rows = []
    for key, frame in score_frames(races, meta).items():
        d = frame["race_id"].map(dividend)
        frame = frame.assign(b2_unrounded=frame["b2"] * (d + BREAKAGE_STEP / 2) / d)
        metrics, unavailable = [], []
        for category in CATEGORIES:
            for cell in study.SEGMENT_CELLS:
                sub = frame[(frame["category"] == category) & frame[cell]]
                m = _member_metrics(sub, "b2_unrounded", True, 131, n_boot)
                unrounded = sub["b2_unrounded"].sum()
                if sub.empty:
                    paid, share, why = math.nan, math.nan, EMPTY_CELL
                else:
                    paid = study.rho(sub["b2"], sub["rate"])
                    share = float(1 - sub["b2"].sum() / unrounded) if unrounded > 0 else math.nan
                    why = "no winning return in the cell: the share is 0 / 0"
                rows.append({"lambda2": key, "category": category, "cell": cell, "n": m["n"],
                             "rho": paid, "breakage_share": share,
                             "rho_unrounded": m["rho"], "lo": m["lo"], "hi": m["hi"]})  # fmt: skip
                got, missing = _metrics(
                    f"development:{category}:{cell}",
                    [("breakage_share", share, None, None, why),
                     ("rho_unrounded", m["rho"], m["lo"], m["hi"], EMPTY_CELL)],
                    m["n"],
                )  # fmt: skip
                metrics += got
                unavailable += missing
        comps = {"win": meta["win_fit"], "ordering": meta["ordering_fit"] if key == study.FITTED
                 else "plain_harville"}  # fmt: skip
        params = {"estimator": "B2 breakage", "lambda2": key, "step_yen": BREAKAGE_STEP,
                  "unrounded": "D + step / 2", "family": study.E1_ID,
                  "role": "post-hoc descriptive (review of 0c39a8d), not a test"}  # fmt: skip
        if unavailable:
            params["unavailable"] = unavailable
        append(_record(BREAKAGE_ID, f"breakage|lambda2={key}", params, comps, metrics, meta, sha))
    return rows


def _breakage_markdown(rows: list[dict], sha: str) -> str:
    lines = [f"# #131 exacta study: breakage of B2's paid return ({sha[:12]})", "",
             "| lambda2 | category | cell | n | rho (paid) | breakage share | 1 - share | "
             "rho unrounded | 95% lo | 95% hi |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
    for r in rows:
        lines.append(f"| {r['lambda2']} | {r['category']} | {r['cell']} | {r['n']} | "
                     f"{_fmt(r['rho'], 4)} | {_fmt(r['breakage_share'], 4)} | "
                     f"{_fmt(1 - r['breakage_share'], 4)} | {_fmt(r['rho_unrounded'], 4)} | "
                     f"{_fmt(r['lo'], 4)} | {_fmt(r['hi'], 4)} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def freeze() -> dict[str, str]:
    """Freeze every family spec; the committed summary (runs only) is left alone."""
    registry = default_registry(COUNTRY)
    return {fid: registry.freeze_family(fid, spec) for fid, spec in study.family_specs().items()}


def _recorded(registry, family: str, meta: dict, sha: str, run: Callable[[], object]):
    """``run()``; a failure is a registry record of ``family`` too, a refusal is not (nothing
    ran). An identical failure already recorded is not appended again."""
    try:
        return run()
    except RunRefused:
        raise
    except Exception as exc:
        _appender(registry)(_record(
            family, "run", {"error": f"{type(exc).__name__}: {exc}"[:500]}, {}, [], meta, sha,
            status="failed",
        ))  # fmt: skip
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.research.exacta_run"
    )
    parser.add_argument("action", choices=["freeze", "count", "score", "max-test", "breakage"])
    parser.add_argument("--note", help="max-test: why the max test is re-run (recorded)")
    args = parser.parse_args(argv)
    out = paths.reports_dir() / "japan" / "131-exacta"
    if args.action == "freeze":
        result = freeze()
    elif args.action == "count":
        result = count()
    elif args.action == "max-test":  # SPA and StepM alone, on E1's recorded members
        sha, registry = _git_sha(), default_registry(COUNTRY)
        races, meta = load_races()
        _check_frozen(registry, (study.E1_ID,))  # before any frame is built
        _check_fits(registry, meta, (study.E1_ID, MAX_TEST_ID))
        result = _recorded(registry, MAX_TEST_ID, meta, sha, lambda: max_test(
            registry, score_frames(races, meta), meta, sha, note=args.note))  # fmt: skip
        write_report(out / f"{_stem('max-test', sha, meta)}.json",
                     json.dumps(_json_safe(result), indent=1), sources=SOURCES)  # fmt: skip
        write_summary(registry, summary_dir(COUNTRY))
    elif args.action == "breakage":  # the committed summary is regenerated separately
        sha, registry = _git_sha(), default_registry(COUNTRY)
        races, meta = load_races()
        rows = _recorded(registry, BREAKAGE_ID, meta, sha,
                         lambda: breakage(registry, races, meta, sha))  # fmt: skip
        stem = _stem("breakage", sha, meta)
        write_report(out / f"{stem}.json", json.dumps(_json_safe(rows), indent=1, default=str),
                     sources=SOURCES)  # fmt: skip
        write_report(out / f"{stem}.md", _breakage_markdown(rows, sha), sources=SOURCES)
        result = [r for r in rows if r["cell"] == "all"]
    else:
        sha, registry = _git_sha(), default_registry(COUNTRY)
        races, meta = load_races()
        result = _recorded(registry, MAX_TEST_ID, meta, sha,
                           lambda: score(registry, races, meta, sha))  # fmt: skip
        write_outputs(result, sha, out)
        write_summary(registry, summary_dir(COUNTRY))
        result = {k: result[k] for k in ("meta", "spa", "stepm", "log_score")}
    print(json.dumps(_json_safe(result), indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
