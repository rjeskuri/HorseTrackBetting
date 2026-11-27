"""The shadow-calibration study on the logged development days (#101).

Pre-registration: docs/specs/shadow-calibration-prereg.md; the frozen spec is ``build_spec()``.
``python -m horsetrack.countries.japan.studies.shadow_run --freeze`` freezes it in the japan
registry (``Registry.freeze_family``; again with the same spec is a no-op), and ``--run`` refuses
to read a record, a dividend or a board unless the family is frozen with exactly that spec. Every
variant is one registry record, failures and empty results included. The full output goes
through ``write_report`` under the data dir: the report (strict JSON, a missing value as null)
and the settlement file, one JSON line per shadow record with its log key, decision time and
outcome. Both are named by the spec hash, the commit and the run's artifact id (``_artifact_id``:
its records' run ids and its settlement text), so a new commit, a rebuilt input or a changed
result writes new files beside the old ones, and a file that exists is never replaced: a rerun
that reproduces an earlier run keeps that run's report and settlement file as written.

Inputs: the spec's days' v0 shadow logs (``models.shadow.read_shadow``: development days only), the
jra store's dividends with the scratch refunds (``bets.settle.day_dividends``), each race's final
WIN board from the raw odds log (``board.snapshot.final_board``), every bet log on disk
(``models.shadow.read_bets``) and the leaderboard of the japan registry
(``core.eval.leaderboard.rows_from_registry``). Nothing here places a bet or opens the holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.holdout import _git_sha
from horsetrack.core.eval.leaderboard import rank, rows_from_registry, shrink
from horsetrack.core.eval.registry import (
    FoldMetric,
    RunRecord,
    _canonical,
    _sha256,
    _spec_hash,
    default_registry,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.bets.log import load_limits
from horsetrack.countries.japan.bets.settle import SETTLED, _day_of, day_dividends, settle_bets
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.models.shadow import (
    NotEstimable,
    calibration_slope,
    flat_stakes,
    log_growth_by_day,
    played_by_day,
    read_bets,
    read_shadow,
    selection_drift,
    settle_shadow,
)
from horsetrack.countries.japan.shadow.log import shadow_log_path

COUNTRY = "japan"
SOURCE = "jra"
TICKET = 101
FAMILY_ID = "japan-shadow-calibration-v1"
FOLD = "development"
FOLD_SCHEME = "development-days"
DAYS = ("2026-11-21", "2026-11-22")  # every logged shadow day before the holdout start
POOL = "WIN"  # the only pool v0 logs a point claim (value_mean) for
TAUS = (2, 1, 5, 16, 30, 60)  # the shadow grid; T-2 is D2's decision time and the primary
# E20 lags: 3 pairs T-2 with T-5, 11 pairs T-5 with T-16, 14 pairs T-16 with T-30 and T-2 with T-16
DRIFT_LAGS = (3, 11, 14)
PICK_OPTION = "pick-rule-q05-t2"
BOOTSTRAP_SEED = TICKET
# the settlement file's columns: a record's key in its log, its decision time, how it settled
SETTLEMENT_COLUMNS = (
    "race_id",
    "pool",
    "combination",
    "tau_min",
    "decision_utc",
    "outcome",
    "return_per_100",
    "claimed_ev",
    "realised",
)


def bootstrap_settings() -> dict:
    """D7's bootstrap from config/scoring.yaml, with this study's fixed seed."""
    boot = load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["cluster_unit"] != "meeting" or boot["sided"] != "one-sided":
        raise ValueError(f"this study implements meeting clusters, one-sided; scoring.yaml: {boot}")
    return {"cluster_unit": "meeting", "sided": "one-sided", "level": float(boot["level"])} | {
        "n_boot": int(boot["n_boot"]),
        "seed": BOOTSTRAP_SEED,
    }


def build_spec() -> dict:
    """The family's pre-registered spec (plain JSON), frozen before any scored read."""
    limits = load_limits()
    win_rate = 1.0 - float(load_source(COUNTRY, SOURCE).pools[POOL].takeout)
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "P5": "calibration: the realised net return per yen at the final dividend, regressed "
            "(OLS with an intercept) on the claimed net EV per yen of the v0 WIN shadow records, "
            "has slope 1; the claim is value_mean - 1",
            "E20": "selection drift: the fall ln board_odds(t) - ln final_odds is regressed on "
            "the claimed EV of the record at t + lag; E20 expects a positive slope (an overlay "
            "is shortened by the final), the null 0",
        },
        "days": list(DAYS),
        "pool": POOL,
        "calibration": {
            "taus": list(TAUS),
            "primary_tau": TAUS[0],
            "clusters": "meeting",
            "sensitivity": {"tau": TAUS[0], "clusters": "race"},
        },
        "pick_rule": {
            "option": PICK_OPTION,
            "tau": TAUS[0],
            "value_q05_above": 1.0,
            "flags": "none on the record",
            "rank": "value_q05 descending within a decision time",
            "stake": "flat_stakes with the committed limits",
            "bankroll_yen": limits.day_cap_yen,
            "payout_rate": win_rate,
        },
        "leaderboard_options": {
            "k": 0,
            "why": "no leaderboard row is measured at the freeze, so no option has a rank; a "
            "ranked option enters through a new family version that names it",
        },
        "drift": {
            "lags_min": list(DRIFT_LAGS),
            "primary_lag_min": DRIFT_LAGS[0],
            "final_odds": "the WIN figure of board.snapshot.final_board",
        },
        "limits": {"day_cap_yen": limits.day_cap_yen, "flat_stake_yen": limits.flat_stake_yen},
        "bets": "every bet-log day on disk, settled at the final dividends, beside each "
        "option's development rank; holdout days stay 'holdout'",
        "bootstrap": bootstrap_settings(),
        "adoption": "none: reported, never a gate (owner, 2026-11-27)",
        "sources": [SOURCE],
    }


# ---------------------------------------------------------------- reading


def _sha16(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "absent"


def _builds(days: Sequence[date]) -> dict[str, str]:
    """Each input's identity: name@sha16 of the shadow and odds logs, and the jra store build."""
    builds = {f"shadow_log_{d}": f"{d}.jsonl@{_sha16(shadow_log_path(d))}" for d in days}
    for day in days:
        for log in sorted(snapshot.jra_day_dir(day).glob("*.jsonl.gz")):
            builds[f"odds_log_{day}_{log.name}"] = f"{log.name}@{_sha16(log)}"
    build = json.loads((paths.processed_dir(SOURCE) / "_build.json").read_text(encoding="utf-8"))
    builds["jra_store"] = f"jra@{build['snapshot_id']}"
    return builds


def final_win_odds(days: Sequence[date]) -> dict[tuple[str, str], float]:
    """{(race_id, horse number): odds} of each race's final WIN board, from the raw odds logs."""
    out: dict[tuple[str, str], float] = {}
    for day in days:
        for log in sorted(snapshot.jra_day_dir(day).glob("*.jsonl.gz")):
            by_race: dict[str, list[dict]] = {}
            for member in odds_log.iter_members(log):
                for row in member:
                    if row.get("pool") == snapshot.WIN:
                        by_race.setdefault(row.get("race_id"), []).append(row)
            for race_id, rows in by_race.items():
                board = snapshot.final_board(rows, race_id)
                if board is not None:
                    out.update({(race_id, str(h)): float(o) for h, o in board.win.items()})
    return out


# ---------------------------------------------------------------- variants


Outcome = tuple[list[FoldMetric], str | None, bool]  # (metrics, note, failed)


def _variant(name: str, hypothesis: str, params: dict, run: Callable[[], Outcome]) -> dict:
    """One variant's record fields; an error in ``run`` fails it with no metric and its reason."""
    try:
        metrics, note, failed = run()
    except Exception as exc:  # every failure after the freeze is a record
        metrics, note, failed = [], f"{type(exc).__name__}: {exc}", True
    return {
        "variant": name,
        "hypothesis_id": hypothesis,
        "params": params | ({"note": note} if note else {}),
        "metrics": metrics,
        "status": "failed" if failed else "scored",
    }


def _metric(metric: str, value: float, n: int, lo: float | None = None, hi=None) -> FoldMetric:
    return FoldMetric(fold=FOLD, metric=metric, value=float(value), lo=lo, hi=hi, n=int(n))


def _meetings(race_ids: pd.Series) -> pd.Series:
    return race_ids.str.rsplit(":", n=1).str[0]


def _slope(
    name: str, unit: str, fit: Callable[[], tuple[float, float, float, int, int]]
) -> Outcome:
    """The slope with its interval and ``n_clusters[<unit>]``, the clusters the bootstrap resamples;
    when only the interval is not estimable, the point slope and that count (one meeting reads as
    1, not as an absence) and a failed variant: the pre-registered estimate has its interval."""
    count = f"n_clusters[{unit}]"
    try:
        slope, lo, hi, n, k = fit()
    except NotEstimable as exc:
        point = [] if exc.slope is None else [_metric(name, exc.slope, exc.n)]
        if exc.n_clusters is not None:
            point.append(_metric(count, exc.n_clusters, exc.n))
        return point, f"not estimable: {exc}", True
    return [_metric(name, slope, n, lo, hi), _metric(count, k, n)], None, False


def _calibration(settled: pd.DataFrame, tau: int, clusters: str, boot: dict) -> Callable:
    def run() -> Outcome:
        rows = settled[(settled["pool"] == POOL) & (settled["tau_min"] == tau)]
        used = rows[rows["claimed_ev"].notna() & rows["realised"].notna()]
        unit = _meetings(rows["race_id"]) if clusters == "meeting" else rows["race_id"]

        def fit() -> tuple[float, float, float, int, int]:
            got = calibration_slope(rows["claimed_ev"], rows["realised"], unit, **boot)
            return got.mean, got.lo, got.hi, len(used), got.n_clusters

        means = [
            _metric("mean_claimed_ev", used["claimed_ev"].mean(), len(used)),
            _metric("mean_realised", used["realised"].mean(), len(used)),
        ]
        slope, note, failed = _slope("calibration_slope", clusters, fit)
        return (means if len(used) else []) + slope, note, failed

    return run


def _picks(settled: pd.DataFrame, tau: int, floor: float) -> pd.DataFrame:
    """The pick rule's records: WIN at ``tau`` with value_q05 above ``floor`` and no avoid flag,
    ranked by value_q05 (highest 1) within each decision time."""
    rows = settled[(settled["pool"] == POOL) & (settled["tau_min"] == tau)]
    unflagged = rows["flags"].map(lambda f: isinstance(f, list) and not f).astype(bool)
    picks = rows[(pd.to_numeric(rows["value_q05"]) > floor) & unflagged]
    order = pd.to_numeric(picks["value_q05"]).groupby(picks["decision_utc"])
    return picks.assign(rank=order.rank(ascending=False, method="first"))


def _day_counts(frame: pd.DataFrame) -> Counter:
    return Counter(_day_of(race_id) for race_id in frame["race_id"])


def _growth_rows(
    settled: pd.DataFrame, staked: pd.DataFrame, growth: pd.DataFrame, rule: dict, days: list
) -> list[dict]:
    """One row per study day of the pick rule's paper stakes, by ``status``: 'settled' (the log
    growth over its settled picks), 'no picks' (the rule read the day's claims and picked nothing:
    nothing staked, the day's money kept, growth 0), 'pending' (picks, none settled yet) or 'no
    claims' (no record at the rule's decision time carries ``value_q05``, so the rule had no input).
    The last two have a null growth and say why; so does a day that ends at 0 yen (ln 0)."""
    at_tau = settled[(settled["pool"] == POOL) & (settled["tau_min"] == rule["tau"])]
    read, picks = _day_counts(settled), _day_counts(staked)
    claims = _day_counts(at_tau[pd.to_numeric(at_tau["value_q05"]).notna()])
    figures = {g["day"]: g for g in growth.to_dict("records")}
    at = f"T-{rule['tau']}"
    rows = []
    for day in days:
        got = figures.get(day)
        money = dict.fromkeys(("n_settled", "staked_yen", "returned_yen", "ending_yen"))
        value, reason = None, None
        if got is not None and not math.isnan(got["log_growth"]):
            status, money = "settled", {name: got[name] for name in money}
            if math.isinf(got["log_growth"]):
                reason = "the day ended at 0 yen: ln 0 has no value"
            else:
                value = got["log_growth"]
        elif picks[day]:
            status, reason = "pending", f"{picks[day]} pick(s), none staked and settled yet"
        elif claims[day]:
            status, value, reason = "no picks", 0.0, f"no record at {at} met the rule"
            money = {"n_settled": 0, "staked_yen": 0, "returned_yen": 0}
            money["ending_yen"] = rule["bankroll_yen"]  # nothing staked: the day's money kept
        else:
            status = "no claims"
            reason = (
                f"no record at {at} carries value_q05: the rule had no input"
                if read[day]
                else "no shadow record was read for the day"
            )
        rows.append(
            {"day": day, "status": status}
            | {"n_records": read[day], "n_claims": claims[day], "n_picks": picks[day]}
            | money
            | {"log_growth": value, "reason": reason}
        )
    return rows


def _pick_rule(settled: pd.DataFrame, spec: dict, limits, boot: dict, out: dict) -> Callable:
    rule = spec["pick_rule"]
    days = [date.fromisoformat(d) for d in spec["days"]]

    def run() -> Outcome:
        staked = flat_stakes(_picks(settled, rule["tau"], rule["value_q05_above"]), limits=limits)
        growth = log_growth_by_day(staked, bankroll_yen=rule["bankroll_yen"])
        out["pick_growth"] = _growth_rows(settled, staked, growth, rule, days)
        metrics = [_metric("n_picks", len(staked), len(staked))]
        metrics += [
            _metric(f"n_claims[{g['day']}]", g["n_claims"], g["n_claims"])
            for g in out["pick_growth"]
        ]
        metrics += [
            _metric(f"log_growth[{g['day']}]", g["log_growth"], g["n_settled"])
            for g in out["pick_growth"]
            if g["log_growth"] is not None
        ]
        bets = staked[staked["outcome"].isin(SETTLED) & (staked["stake_yen"] > 0)]
        if not len(bets):
            return metrics, f"{len(staked)} pick(s) staked and settled: no return", False
        per_100 = pd.Series(bets["return_per_100"].to_numpy(), index=range(len(bets)))
        got = paired_bootstrap(per_100, pd.Series(_meetings(bets["race_id"]).to_numpy()), **boot)
        metrics.append(_metric("return_per_100", got.mean, len(bets), got.lo, got.hi))
        return metrics, None, False

    return run


def _drift(shadow: pd.DataFrame, finals: dict, lag: int, boot: dict) -> Callable:
    def run() -> Outcome:
        rows = shadow[shadow["pool"] == POOL]
        keys = zip(rows["race_id"], rows["combination"], strict=True)
        frame = rows.assign(final_odds=[finals.get(key, np.nan) for key in keys])

        def fit() -> tuple[float, float, float, int, int]:
            got = selection_drift(frame, lag_minutes=float(lag), **boot)
            return got.slope, got.lo, got.hi, got.n, got.n_clusters

        return _slope("drift_slope", "meeting", fit)

    return run


def _leaderboard_block(spec: dict, boot: dict) -> dict:
    rule = spec["pick_rule"]
    return {
        "kind": "rule",
        "pool": POOL,
        "rule": f"WIN at T-{rule['tau']}: v0 value_q05 above {rule['value_q05_above']}, no avoid "
        "flag; flat stake under the race-day cap",
        "selection": "decision_time",
        "payout_rate": rule["payout_rate"],
        "ci_level": boot["level"],
        "n_unit": "bets",
    }


def _bets(ranked: list, limits, out: dict) -> Callable:
    """Every bet log on disk, settled, by day and option beside the option's rank, with each
    day's staked total (every bet of the day) against the cap: a day over it is flagged, never
    refused (#453 records an over-cap bet as a fact)."""

    def run() -> Outcome:
        folder = paths.data_dir() / "bets" / COUNTRY
        days = sorted(date.fromisoformat(p.stem) for p in folder.glob("*.jsonl"))
        bets = read_bets(days)
        if not len(bets):
            return [_metric("n_bets", 0, 0)], "no bet log under the data dir's bets/japan", False
        dividends = pd.concat([day_dividends(d) for d in days], ignore_index=True)
        by_day = played_by_day(settle_bets(bets, dividends), ranked)
        day_of = bets["race_id"].str.split(":").str[1]
        staked = pd.to_numeric(bets["stake_yen"]).groupby(day_of).sum()
        out["bets_by_day"] = by_day.to_dict("records")
        out["day_stakes"] = [
            {"day": d, "staked_yen": int(s), "over_cap": int(s) > limits.day_cap_yen}
            for d, s in staked.items()
        ]
        metrics = [_metric("n_bets", len(bets), len(bets))]
        metrics += [_metric(f"staked_yen[{d}]", s, 1) for d, s in staked.items()]
        return metrics, None, False

    return run


# ---------------------------------------------------------------- the run


def freeze() -> str:
    """Freeze ``build_spec()`` in the japan registry and return its hash."""
    return default_registry(COUNTRY).freeze_family(FAMILY_ID, build_spec())


def _plain(value: Any) -> Any:
    """``value`` for strict JSON: a missing value (None, NaN, NA, NaT) as None, a numpy scalar as
    its Python value, a date as ISO text, containers item by item. An infinity is kept, so
    ``json.dumps(..., allow_nan=False)`` refuses it rather than write a token JSON does not have."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value.isoformat() if isinstance(value, date) else value


def _json(value: Any, **kwargs: Any) -> str:
    return json.dumps(_plain(value), allow_nan=False, **kwargs)


def _settlement(settled: pd.DataFrame) -> str:
    """One JSON line per shadow record, in log order: its key in the log (race_id, pool,
    combination, tau_min), its decision_utc and how it settled (outcome, return_per_100,
    claimed_ev, realised; null where not measured), so each claim joins back to its log line."""
    rows = settled.reindex(columns=list(SETTLEMENT_COLUMNS)).to_dict("records")
    return "".join(_json(row, ensure_ascii=False) + "\n" for row in rows)


def _artifact_id(run_ids: Sequence[str], settlement: str | None) -> str:
    """The run's artifact id: 16 hex digits of the sha256 of its records' run ids, in plan order,
    and of its settlement text. A run id is the sha256 of the whole record: its commit, its
    ``data_builds`` (each shadow and odds log's bytes and the jra store build), its params and
    results. The settlement text holds each record's outcome at the dividend it used. So the id
    changes with any of them, and a run that reproduces another has that run's id."""
    return _sha256(_canonical({"run_ids": list(run_ids), "settlement": settlement}))[:16]


def _write_new(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` through ``write_report`` unless the file exists, and say whether
    it was written: an existing report or settlement file is never replaced."""
    if path.exists():
        return False
    write_report(path, text, sources=(SOURCE,))
    return True


def _primary(variants: list[dict], spec: dict) -> dict:
    """The primary estimate stated whole: P5 at the primary tau with D7's meeting clusters, its
    slope and either its interval or why there is none, and the meetings behind it."""
    name = f"calibration-win-t{spec['calibration']['primary_tau']}-meeting"
    variant = next(v for v in variants if v["variant"] == name)
    got = {m.metric: m for m in variant["metrics"]}
    slope, meetings = got.get("calibration_slope"), got.get("n_clusters[meeting]")
    k = None if meetings is None else int(meetings.value)
    if slope is not None and slope.lo is not None:
        interval = "D7: a bootstrap of whole meetings"
    elif k is not None:
        interval = f"not estimable: {k} meeting(s), and D7's bootstrap resamples whole meetings"
    else:
        interval = variant["params"].get("note", "not estimable")
    return {
        "variant": name,
        "slope": None if slope is None else slope.value,
        "lo": None if slope is None else slope.lo,
        "hi": None if slope is None else slope.hi,
        "n": None if slope is None else slope.n,
        "n_meetings": k,
        "interval": interval,
    }


def _record(v: dict, git_sha: str, builds: dict, components: dict) -> RunRecord:
    return RunRecord(
        hypothesis_id=v["hypothesis_id"],
        family_id=FAMILY_ID,
        variant=v["variant"],
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme=FOLD_SCHEME,
        components=components,
        params=v["params"],
        metrics=tuple(v["metrics"]),
        status=v["status"],
    )


def run_study(out_dir: Path | None = None) -> dict:
    """Score the frozen family on its days, append one record per variant and write the report."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    spec = build_spec()
    registry = default_registry(COUNTRY)
    spec_hash = _spec_hash(spec)
    if registry.frozen_spec_hash(FAMILY_ID) != spec_hash:
        raise RuntimeError(f"{FAMILY_ID} is not frozen with this spec: run --freeze first")
    boot = {k: spec["bootstrap"][k] for k in ("n_boot", "level", "seed")}
    limits = load_limits()
    days = [date.fromisoformat(d) for d in spec["days"]]
    try:
        shadow = read_shadow(days)
        settled = settle_shadow(shadow, pd.concat([day_dividends(d) for d in days]))
        finals = final_win_odds(days)
        rows = rows_from_registry(registry)
        load_error = None
    except Exception as exc:  # the run is still a record: every variant fails with the reason
        shadow, settled, finals, rows, load_error = None, None, {}, [], exc
    out: dict[str, Any] = {}
    base = {"pool": POOL, "days": spec["days"]}
    plan = [
        (
            f"calibration-win-t{tau}-meeting",
            "P5",
            base | {"tau": tau, "clusters": "meeting"},
            lambda tau=tau: _calibration(settled, tau, "meeting", boot),
        )
        for tau in spec["calibration"]["taus"]
    ]
    plan += [
        (
            f"calibration-win-t{TAUS[0]}-race",
            "P5",
            base | {"tau": TAUS[0], "clusters": "race"},
            lambda: _calibration(settled, TAUS[0], "race", boot),
        ),
        (
            PICK_OPTION,
            "P5",
            base | {"rule": spec["pick_rule"], "leaderboard": _leaderboard_block(spec, boot)},
            lambda: _pick_rule(settled, spec, limits, boot, out),
        ),
    ]
    plan += [
        (
            f"drift-win-lag{lag}",
            "E20",
            base | {"lag_min": lag},
            lambda lag=lag: _drift(shadow, finals, lag, boot),
        )
        for lag in spec["drift"]["lags_min"]
    ]
    plan.append(("hand-played-bets", "P5", {}, lambda: _bets(rank(shrink(rows)), limits, out)))

    def failed() -> Outcome:
        raise load_error

    variants = [
        _variant(name, hypothesis, params, failed if load_error else make())
        for name, hypothesis, params, make in plan
    ]
    git_sha = _git_sha()
    builds = _builds(days)
    models = set() if shadow is None else set(shadow["models"].dropna().map(json.dumps))
    components = {"shadow_models": "; ".join(sorted(models)) or "none"}
    existing = registry.records(FAMILY_ID)
    records = [_record(v, git_sha, builds, components) for v in variants]
    run_ids = [_sha256(_canonical(asdict(r))) for r in records]  # as Registry.append names a run
    appended = [registry.append(r) for r in records if r not in existing]
    counts = {
        "load_error": None if load_error is None else f"{type(load_error).__name__}: {load_error}",
        "leaderboard_rows": len(rows),
        "leaderboard_measured": sum(row.measured for row in rows),
    }
    if settled is not None:
        counts |= {
            "records": len(shadow),
            "with_claim": int(settled["claimed_ev"].notna().sum()),
            "outcomes": {
                f"{pool} {outcome}": int(n)
                for (pool, outcome), n in settled.groupby(["pool", "outcome"]).size().items()
            },
            "races_with_final_win_board": len({race for race, _ in finals}),
        }
    lines = None if settled is None else _settlement(settled)
    artifact_id = _artifact_id(run_ids, lines)
    stem = f"shadow-calibration-{COUNTRY}-{spec_hash[:12]}-{git_sha[:7]}-{artifact_id}"
    settlement, written = None, []
    if lines is not None:
        path = out_dir / f"{stem}-settlement.jsonl"
        written += [str(path)] if _write_new(path, lines) else []
        settlement = {
            "path": str(path),
            "n_records": len(settled),
            "key": ["race_id", "pool", "combination", "tau_min"],
            "columns": list(SETTLEMENT_COLUMNS),
        }
    payload = _plain(
        {
            "family_id": FAMILY_ID,
            "spec_hash": spec_hash,
            "git_sha": git_sha,
            "artifact_id": artifact_id,
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": counts,
            "primary": _primary(variants, spec),
            "variants": [v | {"metrics": [m.__dict__ for m in v["metrics"]]} for v in variants],
            **out,
            "settlement": settlement,
            "run_ids": run_ids,
            "records_appended": appended,
        }
    )
    report = out_dir / f"{stem}.json"
    written += [str(report)] if _write_new(report, _json(payload, indent=2)) else []
    return payload | {"report_path": str(report), "written": written}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.studies.shadow_run")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true", help="freeze the spec in the registry")
    action.add_argument("--run", action="store_true", help="run the frozen study")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    if args.freeze:
        print(f"frozen family={FAMILY_ID} spec_hash={freeze()}")
        return 0
    print(_json(run_study(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
