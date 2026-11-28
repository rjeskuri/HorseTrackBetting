"""The shadow page's live nowcast projections scored against the observed odds (#629).

The sealed ladder log of a race day (#628, `shadow.ladder_log`) keeps, at each checkpoint k in
(20, 10, 5, 2) of each race, the observed odds O_k of every key with the page's projection P of
every later checkpoint j (10, 5, 2 or 0, the final), and the final odds. Each `ok` record at k and
each non-null projection whose key has an `ok` record at j is one pair, scored twice: the nowcast,
log error ln(P / O_j), and the null, a random walk (projection = O_k), ln(O_k / O_j). A direction
hit is sign(ln(P / O_k)) == sign(ln(O_j / O_k)), three-valued, so the random walk hits exactly the
keys whose odds did not move.

Rows per (pool, odds band of O_k, from k, to j, predictor), plus an all-bands row per (pool, k, j,
predictor): n, the median log error with its meeting-bootstrap interval, the mean absolute log
error (MAE), the hit rate and, on the nowcast's rows, the MAE gain over the random walk on the same
pairs with its one-sided lower bound. The page's rule, the previous rung to the next
(`shadow_feed.PREVIOUS_RUNG`), is the primary pair; every other (k, j) is a secondary row.

`run` scores development days only (a holdout day's file is never opened), refuses unless the
registry holds FAMILY_ID frozen with `family_spec()`, writes its private report through
`write_report` (the projections come from the nowcast fitted on JRA-VAN history: D21) and appends
one registry record per run, failures included. The spec: docs/specs/ladder-nowcast-prereg.md.
"""

from __future__ import annotations

import hashlib
import math
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import typer

from horsetrack.core import config, env, paths
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import (
    FoldMetric,
    Registry,
    RunRecord,
    _canonical,
    _spec_hash,
    default_registry,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_inside, write_report
from horsetrack.countries.japan.board.shadow_feed import PREVIOUS_RUNG
from horsetrack.countries.japan.models.v0.nowcast import band_of
from horsetrack.countries.japan.shadow.ladder_log import (
    CHECKPOINTS,
    FINAL,
    LadderLog,
    ladder_log_path,
)
from horsetrack.countries.japan.shadow.log import HoldoutSealed, ShadowLogTampered

COUNTRY, STUDY, TICKET, HYPOTHESIS_ID = "japan", "ladder-nowcast", 629, "P629"
FAMILY_ID = "japan-ladder-nowcast"
VARIANT = "nowcast-v0-vs-random-walk"
FOLD_SCHEME = "in-sample:development-days"
# our own JRA captures (the odds) and the nowcast fitted on JRA-VAN history (the projections)
SOURCES = ("jra", "jravan")
MIN_DAYS = 4
BANDS = tuple(float(b) for b in config.load_country_yaml(COUNTRY, "nowcast_v0")["bands"])
PREDICTORS = ("nowcast", "random_walk")
TARGETS = (*CHECKPOINTS, FINAL)
ALL_BANDS = "all"
CI_LEVEL = 0.95  # two-sided interval of the median log error
DEFAULT_SEED, DEFAULT_N_BOOT = 20261129, 2000
NO_REPORT = "not-written"


def _gain_level() -> float:
    """D7's one-sided level, the lower bound of the MAE gain (config/scoring.yaml)."""
    boot = config.load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["cluster_unit"] != "meeting" or boot["sided"] != "one-sided":
        raise ValueError(f"the study implements meeting clusters, one-sided; scoring.yaml: {boot}")
    return float(boot["level"])


def family_spec() -> dict[str, Any]:
    """The plain-JSON spec the family is frozen with (docs/specs/ladder-nowcast-prereg.md)."""
    return {
        "study": STUDY,
        "ticket": TICKET,
        "hypothesis_id": HYPOTHESIS_ID,
        "sources": list(SOURCES),
        "hypothesis": "the page's nowcast projection of a later checkpoint's odds is closer to the "
        "observed odds than the random walk (projection = the odds now): MAE gain above 0 on the "
        "primary pairs; the null, the starting point: it is not",
        "input": "the sealed ladder logs (shadow/japan/<day>.ladder.jsonl, #628) of development "
        "days; an unsealed, tampered or holdout-logged file is skipped and counted",
        "min_days": MIN_DAYS,
        "bands": list(BANDS),
        "checkpoints": list(CHECKPOINTS),
        "final": FINAL,
        "pair": "an ok record at k, a non-null projection for j < k and an ok record of the key "
        "at j; a null projection is counted before the target is looked up",
        "predictors": {
            "nowcast": "log error ln(P / O_j)",
            "random_walk": "the null, P = O_k: log error ln(O_k / O_j)",
        },
        "hit": "sign(ln(P / O_k)) == sign(ln(O_j / O_k)), sign in (-1, 0, +1)",
        "cells": "pool x band of O_k (cut points at or below it) x (k, j) x predictor, and the "
        "same over all bands",
        "primary": "PREVIOUS_RUNG[j] == k: 20->10, 10->5, 5->2, 2->final; every other pair is "
        "secondary",
        "metrics": ["n", "median_log_error", "mae", "hit_rate", "ci_lo", "ci_hi", "mae_gain"],
        "bootstrap": {
            "cluster_unit": "meeting",
            "median_interval": f"two-sided {CI_LEVEL} percentile",
            "mae_gain": "random-walk MAE minus nowcast MAE on the same pairs; one-sided lower "
            "bound at scoring.yaml's level",
        },
        "holdout": "development days only: is_holdout_day days are never opened (D14)",
        "d21": "the report and the registry stay under $HORSETRACK_DATA_DIR",
    }


def freeze(registry: Registry) -> str:
    """Freeze the family's spec in `registry` and return its hash; again is a no-op."""
    return registry.freeze_family(FAMILY_ID, family_spec())


# ---------------------------------------------------------------- scoring


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _meeting(race_id: str) -> str:
    """The meeting (venue x day, D7) of a race: its race_id without the race number."""
    return race_id.rsplit(":", 1)[0]


@dataclass
class Scored:
    rows: list[dict[str, Any]]
    excluded: dict[str, int]


Pair = tuple[str, float, float, bool, bool]  # (meeting, nowcast error, walk error, hits)


def _pairs(records: Iterable[dict]) -> tuple[dict[tuple, list[Pair]], Counter]:
    records = list(records)
    observed = {
        (r["race_id"], r["pool"], r["combination"], r["checkpoint_min"]): float(r["odds"])
        for r in records
        if r["status"] == "ok"
    }
    excluded: Counter = Counter({"null_projection": 0, "no_board": 0})
    cells: dict[tuple, list[Pair]] = defaultdict(list)
    for r in records:
        if r["status"] != "ok":  # no_board, the log's only other status
            excluded["no_board"] += 1
            continue
        k = r["checkpoint_min"]
        if k == FINAL:
            continue
        o = float(r["odds"])
        band = band_of(o, BANDS)
        for text, p in r["projected"].items():
            if p is None:
                excluded["null_projection"] += 1
                continue
            try:
                j = int(text)
            except ValueError:
                j = None
            if j not in TARGETS or j >= k:
                excluded["invalid_target"] += 1
                continue
            target = observed.get((r["race_id"], r["pool"], r["combination"], j))
            if target is None:
                excluded["no_target"] += 1
                continue
            if not p > 0:
                excluded["invalid_projection"] += 1
                continue
            move = _sign(math.log(target / o))
            cells[(r["pool"], band, k, j)].append(
                (
                    _meeting(r["race_id"]),
                    math.log(p / target),
                    math.log(o / target),
                    _sign(math.log(p / o)) == move,
                    move == 0,
                )
            )
    return cells, excluded


def _seed(seed: int, key: tuple) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(repr(key).encode())])


def _cell_rows(key: tuple, pairs: list[Pair], *, seed: int, n_boot: int, level: float) -> list:
    pool, band, k, j = key
    meetings = sorted({p[0] for p in pairs})
    index = {m: i for i, m in enumerate(meetings)}
    of = np.array([index[p[0]] for p in pairs])
    errors = {
        "nowcast": np.array([p[1] for p in pairs]),
        "random_walk": np.array([p[2] for p in pairs]),
    }
    hits = {"nowcast": [p[3] for p in pairs], "random_walk": [p[4] for p in pairs]}
    rng = _seed(seed, key)
    medians = {name: np.empty(n_boot) for name in PREDICTORS}
    gains = np.empty(n_boot)
    for b in range(n_boot):
        weights = np.bincount(
            rng.integers(0, len(meetings), len(meetings)), minlength=len(meetings)
        )
        w = weights[of]
        for name in PREDICTORS:
            medians[name][b] = np.median(np.repeat(errors[name], w))
        gains[b] = (
            np.average(np.abs(errors["random_walk"]), weights=w)
            - np.average(np.abs(errors["nowcast"]), weights=w)
        )  # fmt: skip
    tail = (1 - CI_LEVEL) / 2 * 100
    mae = {name: float(np.mean(np.abs(errors[name]))) for name in PREDICTORS}
    rows = []
    for name in PREDICTORS:
        lo, hi = np.percentile(medians[name], [tail, 100 - tail])
        is_nowcast = name == "nowcast"
        rows.append(
            {
                "pool": pool,
                "band": band,
                "from_min": k,
                "to_min": j,
                "predictor": name,
                "primary": PREVIOUS_RUNG[j] == k,
                "n": len(pairs),
                "n_meetings": len(meetings),
                "median_log_error": float(np.median(errors[name])),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "mae": mae[name],
                "hit_rate": sum(hits[name]) / len(pairs),
                "mae_gain": mae["random_walk"] - mae["nowcast"] if is_nowcast else None,
                "gain_lo": (float(np.percentile(gains, (1 - level) * 100)) if is_nowcast else None),
            }
        )
    return rows


def _order(key: tuple) -> tuple:
    pool, band, k, j = key
    return (pool, -k, -j, (1, 0) if band == ALL_BANDS else (0, band))


def score(records: Iterable[dict], *, seed: int, n_boot: int) -> Scored:
    """The rows and exclusions of `records`, ladder-log records of any number of days (module
    docstring); the same records and seed give the same rows."""
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 2:
        raise ValueError(f"n_boot must be an integer of at least 2, got {n_boot!r}")
    cells, excluded = _pairs(records)
    merged: dict[tuple, list[Pair]] = dict(cells)
    for (pool, _, k, j), pairs in cells.items():
        merged.setdefault((pool, ALL_BANDS, k, j), []).extend(pairs)
    level = _gain_level()
    rows = [
        row
        for key in sorted(merged, key=_order)
        for row in _cell_rows(key, merged[key], seed=seed, n_boot=n_boot, level=level)
    ]
    return Scored(rows, dict(excluded))


# ---------------------------------------------------------------- the run


class FamilyNotFrozen(ValueError):
    """The registry does not hold FAMILY_ID frozen with `family_spec()`: nothing was read."""


@dataclass
class RunResult:
    status: str  # "scored" or "failed"
    rows: list[dict[str, Any]]
    days: list[date]
    report_path: Path | None
    run_ids: dict[str, str]
    skipped: dict[str, int]
    excluded: dict[str, int]


def _commit() -> str:
    try:
        return _git_sha()
    except RuntimeError:
        return "unknown"


def _read_days(start: date, end: date, skipped: Counter) -> tuple[list[date], list[dict], str]:
    """The development days with a sealed, verified ladder log, their records and a digest of the
    files read. A holdout day is skipped before its path is looked at."""
    days, records, digest = [], [], hashlib.sha256()
    day = start
    while day <= end:
        current, day = day, day + timedelta(days=1)
        if is_holdout_day(COUNTRY, current):
            skipped["holdout_day"] += 1
            continue
        path = ladder_log_path(current)
        if not path.is_file():
            continue
        log = LadderLog(path, day=current)
        try:
            if not log.verify().sealed:
                skipped["unsealed"] += 1
                continue
            day_records = log.read()
        except ShadowLogTampered:
            skipped["tampered"] += 1
            continue
        except HoldoutSealed:
            skipped["logged_as_holdout"] += 1
            continue
        digest.update(f"{current.isoformat()}:".encode() + path.read_bytes())
        days.append(current)
        records += day_records
    return days, records, digest.hexdigest()[:16]


def _metrics(rows: list[dict[str, Any]]) -> list[FoldMetric]:
    out = []
    for r in rows:
        fold = f"{r['pool']}/band:{r['band']}/{r['from_min']}->{r['to_min']}"
        name, n = r["predictor"], r["n"]
        out.append(
            FoldMetric(
                fold, f"median_log_error[{name}]", r["median_log_error"], r["ci_lo"], r["ci_hi"], n
            )  # fmt: skip
        )
        out.append(FoldMetric(fold, f"mae[{name}]", r["mae"], None, None, n))
        out.append(FoldMetric(fold, f"hit_rate[{name}]", r["hit_rate"], None, None, n))
        if r["mae_gain"] is not None:
            out.append(FoldMetric(fold, "mae_gain[nowcast]", r["mae_gain"], r["gain_lo"], None, n))
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def _table(rows: list[dict], cols: list[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    return lines


def _report(
    status: str,
    rows: list[dict],
    summary: dict[str, Any],
    skipped: Counter,
    excluded: dict[str, int],
) -> str:
    """The private report, in docs/experiment-report.md's headings; the projector against the
    random walk on the primary pairs, all bands, is its first figure."""
    nowcast = {
        (r["pool"], r["band"], r["from_min"], r["to_min"]): r
        for r in rows
        if r["predictor"] == "nowcast"
    }
    walk = {
        (r["pool"], r["band"], r["from_min"], r["to_min"]): r
        for r in rows
        if r["predictor"] == "random_walk"
    }
    head = [
        {
            "pool": key[0],
            "pair": f"{key[2]}->{key[3]}",
            "n": r["n"],
            "n_meetings": r["n_meetings"],
            "mae_nowcast": r["mae"],
            "mae_random_walk": walk[key]["mae"],
            "mae_gain": r["mae_gain"],
            "gain_lo": r["gain_lo"],
            "hit_nowcast": r["hit_rate"],
            "hit_random_walk": walk[key]["hit_rate"],
            "median_nowcast": r["median_log_error"],
            "median_random_walk": walk[key]["median_log_error"],
        }
        for key, r in nowcast.items()
        if key[1] == ALL_BANDS and r["primary"]
    ]
    ranked = sorted(
        (r for key, r in sorted(nowcast.items(), key=lambda item: _order(item[0]))
         if key[1] != ALL_BANDS),
        key=lambda r: (-r["gain_lo"], -r["mae_gain"]),
    )  # fmt: skip
    cols = ["pool", "band", "from_min", "to_min", "predictor", "primary", "n", "n_meetings"]
    cols += ["median_log_error", "ci_lo", "ci_hi", "mae", "hit_rate", "mae_gain", "gain_lo"]
    lines = [
        "# Live ladder nowcast vs the random walk (Japan, #629): private report",
        "",
        "Built on the nowcast fitted on JRA-VAN history (licensed, D21): never copy a figure into "
        "the repository.",
        "",
        *(f"- {key}: {value}" for key, value in summary.items()),
        "",
        "## Figure 1: the projector against the random walk, primary pairs, all bands",
        "",
        "mae_gain is the random walk's MAE minus the nowcast's on the same pairs (above 0: the "
        "nowcast is closer); gain_lo its one-sided lower bound (meeting bootstrap).",
        "",
        *(_table(head, list(head[0])) if head else ["No primary pair was scored."]),
        "",
        "## Hypothesis",
        "",
        "The hypothesis: " + family_spec()["hypothesis"] + ".",
        "",
        "## Variants tried",
        "",
        f"One variant, `{VARIANT}` (family `{FAMILY_ID}`), status {status}; every cell below, "
        "primary and secondary, nowcast and random walk.",
        "",
        "## Alternatives rejected and why",
        "",
        "- Scoring the nowcast on JRA-VAN history again: #85's study did that offline; this one "
        "scores the live projector on our own captures.",
        "- Unsealed or tampered ladder logs: not scored, counted below.",
        "",
        "## What this rules out",
        "",
        "A measurement, not a verdict: cells where gain_lo is above 0 rule out the random walk "
        "being as close as the nowcast; elsewhere it is not ruled out.",
        "",
        "## Leaderboard rows",
        "",
        "No betting option: the projector is display data. The nowcast cells, ranked by gain_lo:",
        "",
        *(_table(ranked, cols) if ranked else ["None scored."]),
        "",
        "## Every row",
        "",
        *(_table(rows, cols) if rows else ["None scored."]),
        "",
        "## Skipped days and excluded pairs",
        "",
        *(f"- {key}: {n}" for key, n in sorted({**skipped, **excluded}.items())),
        "",
        "## Follow-ups filed",
        "",
        "None by this run.",
    ]
    return "\n".join(lines) + "\n"


def run(
    *,
    date_from: date,
    date_to: date,
    out_dir: Path | str,
    registry: Registry,
    seed: int = 0,
    n_boot: int = 200,
) -> RunResult:
    """Score the development days from `date_from` to `date_to` and record the run (module
    docstring). Refuses, with nothing read or appended, when the family is not frozen with
    `family_spec()` or `out_dir` is inside the repository."""
    if date_to < date_from:
        raise ValueError(f"date_to {date_to} is before date_from {date_from}")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 2:
        raise ValueError(f"n_boot must be an integer of at least 2, got {n_boot!r}")
    spec_hash = _spec_hash(family_spec())
    held = registry.frozen_spec_hash(FAMILY_ID)
    if held != spec_hash:
        raise FamilyNotFrozen(f"family {FAMILY_ID} is frozen with {held}, not {spec_hash}")
    out_dir = Path(out_dir)
    if is_inside(out_dir, env.repo_root()):
        raise LicensedOutputInRepo(f"{out_dir} is inside the repository: the study reads jravan")
    params = {
        "spec_hash": spec_hash,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "seed": seed,
        "n_boot": n_boot,
    }
    run_utc = datetime.now(timezone.utc).isoformat()
    git_sha = _commit()
    # one report per run: the run time is in its name, so a rerun never overwrites one
    report_name = f"{STUDY}-japan-{_sha(_canonical({**params, 'git': git_sha, 'at': run_utc}))}.md"
    skipped: Counter = Counter()
    state: dict[str, Any] = {"days": [], "digest": "none", "report": NO_REPORT, "nowcast_ids": ""}

    def record(status: str, metrics: list[FoldMetric]) -> RunRecord:
        return RunRecord(
            hypothesis_id=HYPOTHESIS_ID,
            family_id=FAMILY_ID,
            variant=VARIANT,
            ticket=TICKET,
            git_sha=git_sha,
            data_builds={
                "ladder_log": f"shadow/{COUNTRY}/ladder@{state['digest']}",
                "report": f"reports/{STUDY}@{state['report']}",
            },
            sources=SOURCES,
            fold_scheme=FOLD_SCHEME,
            components={},
            params=params,
            metrics=tuple(metrics),
            status=status,
            notes={
                "run_utc": run_utc,
                "days": ",".join(d.isoformat() for d in state["days"]),
                "nowcast_ids": state["nowcast_ids"],
            },
        )

    try:
        days, records, state["digest"] = _read_days(date_from, date_to, skipped)
        state["days"] = days
        state["nowcast_ids"] = ",".join(sorted({str(r.get("nowcast_id")) for r in records}))
        status = "scored" if len(days) >= MIN_DAYS else "failed"
        scored = score(records, seed=seed, n_boot=n_boot) if status == "scored" else None
        rows = [] if scored is None else scored.rows
        excluded = {} if scored is None else scored.excluded
        summary = {
            **params,
            "status": status,
            "git_sha": git_sha,
            "family": f"{FAMILY_ID} {spec_hash}",
            "development days with a sealed ladder log": len(days),
            "min_days": MIN_DAYS,
            "nowcast ids": state["nowcast_ids"] or "none",
            "replay records": sum(r.get("mode") == "replay" for r in records),
        }
        text = _report(status, rows, summary, skipped, excluded)
        report_path = write_report(out_dir / report_name, text, sources=SOURCES)
        state["report"] = report_name
    except Exception:
        registry.append(record("failed", []))
        raise
    if status == "failed":
        metrics = [FoldMetric("all", "development_days", float(len(days)), None, None, len(days))]
    else:
        metrics = _metrics(rows)
    run_id = registry.append(record(status, metrics))
    return RunResult(status, rows, days, report_path, {VARIANT: run_id}, dict(skipped), excluded)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- the command

ladder_nowcast_app = typer.Typer(
    no_args_is_help=True, help="#629: the live ladder nowcast against the random walk."
)


@ladder_nowcast_app.command("freeze")
def freeze_command() -> None:
    """Freeze the family's spec in the japan registry (irreversible; again is a no-op)."""
    typer.echo(f"frozen {FAMILY_ID} {freeze(default_registry(COUNTRY))}")


@ladder_nowcast_app.command("run")
def run_command(
    date_from: str = typer.Option(..., "--from", help="First day (YYYY-MM-DD)."),
    date_to: str = typer.Option(..., "--to", help="Last day (YYYY-MM-DD)."),
    seed: int = typer.Option(DEFAULT_SEED, help="Bootstrap seed."),
    n_boot: int = typer.Option(DEFAULT_N_BOOT, help="Bootstrap replicates."),
) -> None:
    """Score the sealed ladder logs of development days and record the run; prints no figure."""
    result = run(
        date_from=date.fromisoformat(date_from),
        date_to=date.fromisoformat(date_to),
        out_dir=paths.data_dir() / "reports",
        registry=default_registry(COUNTRY),
        seed=seed,
        n_boot=n_boot,
    )
    typer.echo(f"status={result.status} days={len(result.days)}\nreport={result.report_path}")
    for variant, run_id in result.run_ids.items():
        typer.echo(f"run {variant} {run_id}")
    if result.status != "scored":
        raise typer.Exit(1)
