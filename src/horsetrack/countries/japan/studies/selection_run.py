"""The development-fold run of the Family P selection study on the JRA-VAN boards at t (#462).

#88 landed the machinery (`studies.selection`) and the owner's signed family
(`config/selection_family.yaml`, `japan-family-p-selection-v1`); this module feeds it years of
the Win board at t from the jravan store (#369's `odds_timeseries`) and records the run.

`build_frame(date_from=, date_to=)`: one row per starter (a runner not scratched, with a final
Win price) of every development race in the span, read from `processed/jravan/`:
- the span may not start before 2014-06-07 (D10, `PreReformPriceError`, raised before any file
  is opened); a holdout day's file is dropped by its name and never opened, nor is a day file
  outside the span, and the store tables are read through `read_store_table` (holdout days
  dropped) or by the development race ids;
- `post_utc` from `races.post_time_local` on the Asia/Tokyo clock of `race_date`, and t =
  `post_utc` minus the family's `decision_time` (T-2);
- the board at t (docs/specs/board-at-t.md, as-of.md): per (race, WIN, horse) the latest row
  available by t (`fetched_utc`, else `snapshot_utc`, the 発表 minute JRA-VAN stamps; exactly t
  counts); a horse whose latest row is unpriced or scratched ("SCR") is off the board. `odds_t`
  is its price, `rank_t` its min-rank on the board, `entropy_t` the board's normalised entropy
  and `entropy_t_bin` its bin at config/calibration.yaml's `entropy_cut_points`;
- the final price, the result and the pay (`odds_final`, `won`, `win_dividend` per unit stake),
  the Win `takeout` (config/jra.yaml), `weight_type`, `R12`, `meeting_id`;
- each column a rule reads declared as `<col>__source` and `<col>__available_utc`: the board
  columns from `board_at_t`, stamped by their board rows (a horse off the board is unstamped,
  genuinely missing); the card columns from `races`, stamped 00:00 JST of the race day (the
  availability config's card rule).
A race without a post time (no t), with fewer than two starters or with no winner among its
starters cannot be scored and is left out, counted in the private report.

`run_selection(...)`: before any store file is opened, freezes `asdict(load_family("japan"))` in
the registry and appends each `dropped` variant as a failed record of the family (so the family
size counts them before anything is scored); builds the frame; calls `selection.run_study` with
the frozen spec, the holdout (held, never opened) and `sources=("jra", "jravan")`; writes the
private report through `write_report` under `out_dir`. A crash after the freeze is a failed record
of every rule. Nothing is written inside the repository; `horsetrack japan selection-run` then
writes the family's committed registry summary (values withheld, D21).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import typer

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.folds import load_country_scheme
from horsetrack.core.eval.holdout import LockedHoldout, _git_sha, split_off
from horsetrack.core.eval.registry import (
    Registry,
    RunRecord,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.countries.japan.models.v0.store import (
    PRICE_ERA_START,
    PreReformPriceError,
    read_store_table,
)
from horsetrack.countries.japan.studies import selection
from horsetrack.countries.japan.studies.cross_pool_run import (
    _day_files,
    _Families,
    _final_odds,
    _index_family,
    _peak_gb,
)
from horsetrack.countries.japan.studies.selection import (
    FamilySpec,
    StudyReport,
    load_family,
    run_study,
)

COUNTRY = "japan"
TICKET = 462
SOURCE = "jravan"
SOURCES = ("jra", "jravan")
# The bootstrap seed: the ticket number, for reproducibility only (it chooses nothing). Not in
# the signed family spec: any spec edit is a new family id.
SEED = 462
SPAN_START = PRICE_ERA_START  # D10
WIN, POOL_ROW, SCRATCHED = "WIN", "__pool__", "SCR"
BOARD, CARD = "board_at_t", "races"  # selection.ADMITTED_SOURCES
BOARD_COLS = ("odds_t", "rank_t", "entropy_t_bin")
CARD_COLS = ("weight_type", "R12")
DAY_COLUMNS = ["race_id", "combination", "snapshot_utc", "fetched_utc", "odds", "status_raw"]
RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "race_no", "post_time_local", "weight_type"]
JST = ZoneInfo("Asia/Tokyo")
UTC_NS = "datetime64[ns, UTC]"
REPORT_NAME = "selection-family-p-japan.md"
NOT_BUILT = "selection_frame@not-built"  # run_study's build name, before the frame exists


@dataclass(frozen=True)
class SelectionRun:
    study: StudyReport
    report_path: Path
    counts: dict[str, int]


def _check_span(date_from: date, date_to: date) -> None:
    if date_from < SPAN_START:
        raise PreReformPriceError(
            f"the span may not start before {SPAN_START.isoformat()} (D10), got {date_from}"
        )
    if date_to < date_from:
        raise ValueError(f"the span ends ({date_to}) before it starts ({date_from})")


# --------------------------------------------------------------------------- the frame


def _post_utc(race_date: pd.Series, hhmm: pd.Series) -> pd.Series:
    """post_time_local ("H:MM" or "HH:MM") on the Asia/Tokyo clock of the race day, as UTC; NaT
    when missing or malformed."""
    text = race_date.dt.strftime("%Y-%m-%d") + " "
    text = text + hhmm.astype("string").str.strip().str.zfill(5)
    local = pd.to_datetime(text, format="%Y-%m-%d %H:%M", errors="coerce")
    return local.dt.tz_localize(JST).dt.tz_convert("UTC").astype(UTC_NS)


def _utc(column: pd.Series) -> pd.Series:
    return pd.to_datetime(column, utc=True).astype(UTC_NS)


def _horse_no(combination: pd.Series) -> pd.Series:
    return pd.to_numeric(combination.astype("string").str.strip(), errors="coerce")


def _read_day(path: Path) -> pd.DataFrame:
    """One day file's WIN rows (no pool totals), only the board's columns."""
    table = pq.read_table(
        path,
        columns=DAY_COLUMNS,
        filters=[("pool", "==", WIN), ("combination", "!=", POOL_ROW)],
    )
    frame = table.to_pandas()
    del table
    pa.default_memory_pool().release_unused()
    return frame


def _day_board(day: pd.DataFrame, t_of: pd.Series) -> pd.DataFrame:
    """The day's Win boards at t: `race_id, horse_no, odds_t, odds_t_utc, rank_t, entropy_t,
    board_utc` for every horse on its race's board at t (`t_of`: race_id -> t)."""
    rows = pd.DataFrame(
        {
            "race_id": day["race_id"].astype(object),
            "combination": day["combination"].astype(object),
            "odds": day["odds"].to_numpy(dtype=float, na_value=np.nan),
            "status": day["status_raw"].astype("string").str.strip(),
            "available": _utc(day["fetched_utc"]).fillna(_utc(day["snapshot_utc"])),
        }
    )
    t = rows["race_id"].map(t_of)
    rows = rows[(rows["available"] <= t).fillna(False).to_numpy(dtype=bool)]
    rows = rows.sort_values("available", kind="mergesort")
    rows = rows.drop_duplicates(["race_id", "combination"], keep="last")
    priced = np.isfinite(rows["odds"].to_numpy()) & (rows["odds"].to_numpy() > 0)
    priced &= ~(rows["status"] == SCRATCHED).fillna(False).to_numpy(dtype=bool)
    rows = rows[priced].assign(horse_no=_horse_no(rows.loc[priced, "combination"]))
    rows = rows.dropna(subset=["horse_no"])
    by_race = rows.groupby("race_id", sort=False)
    inv = 1.0 / rows["odds"]
    q = inv / inv.groupby(rows["race_id"]).transform("sum")
    size = by_race["odds"].transform("count").to_numpy(dtype=float)
    plogp = (q * np.log(q)).groupby(rows["race_id"]).transform("sum").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = np.where(size >= 2, -plogp / np.log(size), np.nan)
    return pd.DataFrame(
        {
            "race_id": rows["race_id"].to_numpy(),
            "horse_no": rows["horse_no"].to_numpy(dtype=np.int64),
            "odds_t": rows["odds"].to_numpy(),
            "odds_t_utc": rows["available"].values,  # naive UTC
            "rank_t": by_race["odds"].rank(method="min").to_numpy(),
            "entropy_t": entropy,
            "board_utc": by_race["available"].transform("max").values,
        }
    )


def _boards(files, t_of: pd.Series, log) -> pd.DataFrame:
    parts, started = [], time.monotonic()
    for i, (_, path) in enumerate(files):
        board = _day_board(_read_day(path), t_of)
        if len(board):
            parts.append(board)
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            log(
                f"  {i + 1}/{len(files)} day files read ({time.monotonic() - started:.0f} s, "
                f"peak {_peak_gb():.2f} GB)"
            )
    if parts:
        return pd.concat(parts, ignore_index=True)
    columns = ["race_id", "horse_no", "odds_t", "odds_t_utc", "rank_t", "entropy_t", "board_utc"]
    return pd.DataFrame({c: [] for c in columns}).astype({"horse_no": np.int64})


def _starters(races: pd.DataFrame, date_from: date, date_to: date) -> pd.DataFrame:
    """`race_id, horse_no, odds_final, won, win_dividend` of every runner not scratched with a
    final Win price, of `races` (development races)."""
    parts = []
    years = races["race_date"].dt.year
    for year in sorted(set(years)):
        ids = races.loc[years == year, "race_id"].tolist()
        start, end = max(date_from, date(year, 1, 1)), min(date_to, date(year, 12, 31))
        table = _final_odds(WIN, start, end, ids)
        parts.append(
            pd.DataFrame(
                {
                    "race_id": table["race_id"].to_numpy(zero_copy_only=False),
                    "horse_no": _horse_no(pd.Series(table["combination"].to_pylist())),
                    "odds_final": table["odds"].to_numpy(),
                }
            )
        )
        del table
    final = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame({"race_id": [], "horse_no": [], "odds_final": []})
    )
    final = final.dropna(subset=["horse_no"]).astype({"horse_no": np.int64})
    final = final.drop_duplicates(["race_id", "horse_no"])
    ids = pa.array(races["race_id"].tolist(), pa.string())
    runners = read_store_table(
        "runners",
        columns=["race_id", "horse_no", "finish_pos", "is_scratched"],
        start=date_from,
        end=date_to,
        touches_price=False,
        where=ds.field("race_id").isin(ids),
        source=SOURCE,
    ).to_pandas()
    runners = runners.dropna(subset=["horse_no"]).astype({"horse_no": np.int64})
    runners = runners.drop_duplicates(["race_id", "horse_no"])
    out = final.merge(runners, on=["race_id", "horse_no"], how="left")
    out = out[~out["is_scratched"].fillna(False).to_numpy(dtype=bool)]
    out["won"] = (out["finish_pos"] == 1).fillna(False).to_numpy(dtype=int)
    paid = pq.read_table(
        paths.processed_dir(SOURCE) / "dividends.parquet",
        columns=["race_id", "pool", "combination", "dividend", "is_refund"],
        filters=[("pool", "==", WIN), ("race_id", "in", races["race_id"].tolist())],
    ).to_pandas()
    paid = paid[~paid["is_refund"].fillna(False).to_numpy(dtype=bool)]
    paid = paid.assign(horse_no=_horse_no(paid["combination"])).dropna(subset=["horse_no"])
    paid = paid.astype({"horse_no": np.int64}).drop_duplicates(["race_id", "horse_no"])
    paid = paid.assign(win_dividend=paid["dividend"].astype(float) / 100.0)
    out = out.merge(paid[["race_id", "horse_no", "win_dividend"]], how="left")
    out["win_dividend"] = out["win_dividend"].where(out["won"] == 1)
    return out[["race_id", "horse_no", "odds_final", "won", "win_dividend"]]


def _build(date_from: date, date_to: date, log=None) -> tuple[pd.DataFrame, dict[str, int]]:
    log = log or (lambda message: None)
    _check_span(date_from, date_to)
    minutes = selection._decision_minutes(load_family(COUNTRY).decision_time)
    cuts = [float(c) for c in load_country_yaml(COUNTRY, "calibration")["entropy_cut_points"]]
    files, held = _day_files(date_from, date_to)
    counts = {"development_day_files": len(files), "holdout_day_files_skipped": held}
    races = read_store_table(
        "races",
        columns=RACE_COLUMNS,
        start=date_from,
        end=date_to,
        touches_price=False,
        source=SOURCE,
    ).to_pandas()
    races = races.drop_duplicates("race_id").reset_index(drop=True)
    races["race_date"] = pd.to_datetime(races["race_date"]).astype("datetime64[ns]")
    races["post_utc"] = _post_utc(races["race_date"], races["post_time_local"])
    counts["races_in_span"] = len(races)
    counts["excluded_no_post_time"] = int(races["post_utc"].isna().sum())
    races = races[races["post_utc"].notna()].reset_index(drop=True)
    file_days = {d for d, _ in files}
    counts["races_on_days_without_a_day_file"] = int(
        (~races["race_date"].dt.date.isin(file_days)).sum()
    )
    t_of = (races["post_utc"] - pd.Timedelta(minutes=minutes)).set_axis(races["race_id"])
    log(f"reading {len(files)} development day files")
    boards = _boards(files, t_of, log)
    starters = _starters(races, date_from, date_to)
    frame = starters.merge(races, on="race_id").merge(
        boards, on=["race_id", "horse_no"], how="left"
    )
    del boards, starters
    by_race = frame.groupby("race_id")
    few = by_race["horse_no"].transform("count") < 2
    no_winner = by_race["won"].transform("max") < 1
    counts["excluded_fewer_than_2_starters"] = int(frame.loc[few, "race_id"].nunique())
    counts["excluded_no_winner"] = int(frame.loc[~few & no_winner, "race_id"].nunique())
    frame = frame[~(few | no_winner)].copy()
    # Every starter of a race with a board at t carries the board's entropy and stamp, also one
    # quoted only after t.
    race_board = frame.groupby("race_id")[["entropy_t", "board_utc"]].transform("first")
    frame[["entropy_t", "board_utc"]] = race_board
    entropy = frame["entropy_t"].to_numpy(dtype=float)
    bins = np.searchsorted(cuts, entropy, side="right") + 1.0
    frame["entropy_t_bin"] = np.where(np.isnan(entropy), np.nan, bins)
    frame["R12"] = np.where(frame["race_no"] == 12, "in", "out")
    takeout = load_source(COUNTRY, "jra").pools[WIN]
    days = frame["race_date"].dt.date
    frame["takeout"] = days.map({d: float(takeout.takeout_on(d)) for d in set(days)})
    card_utc = frame["race_date"].dt.tz_localize(JST).dt.tz_convert("UTC").astype(UTC_NS)
    stamps = {
        "odds_t": _utc(frame["odds_t_utc"]),
        "rank_t": _utc(frame["odds_t_utc"]),
        "entropy_t_bin": _utc(frame["board_utc"]),
    }
    for col in BOARD_COLS:
        frame[col + selection.SOURCE_SUFFIX] = BOARD
        frame[col + selection.AVAILABLE_SUFFIX] = stamps[col]
    for col in CARD_COLS:
        frame[col + selection.SOURCE_SUFFIX] = CARD
        frame[col + selection.AVAILABLE_SUFFIX] = card_utc
    frame = frame.drop(columns=["odds_t_utc", "board_utc", "post_time_local"])
    frame = frame.sort_values(["race_date", "race_id", "horse_no"], kind="mergesort")
    frame = frame.reset_index(drop=True)
    selection._check_development(frame)  # no holdout row, whatever the reads above let through
    counts |= {
        "races": int(frame["race_id"].nunique()),
        "starters": len(frame),
        "races_with_a_board_at_t": int(frame.loc[frame["entropy_t"].notna(), "race_id"].nunique()),
        "starters_off_the_board_at_t": int(frame["odds_t"].isna().sum()),
        "race_days": int(frame["race_date"].nunique()),
        "meetings": int(frame["meeting_id"].nunique()),
    }
    return frame, counts


def build_frame(*, date_from: date, date_to: date) -> pd.DataFrame:
    """The selection frame of the development races in the span (module docstring)."""
    return _build(date_from, date_to)[0]


# --------------------------------------------------------------------------- the run


def _identity() -> dict:
    """The identity run_study gives its records, before the frame exists: its build name with no
    fingerprint yet (the same feature table, `Registry.family_size`), the fold scheme, sources."""
    scheme = load_country_scheme(COUNTRY)
    return {
        "git_sha": _git_sha(),
        "data_builds": {"selection_frame": NOT_BUILT},
        "sources": SOURCES,
        "fold_scheme": "walk_forward:" + json.dumps(dataclasses.asdict(scheme), sort_keys=True),
    }


def _dropped_records(spec: FamilySpec, frozen: str, identity: dict) -> list[RunRecord]:
    """The record run_study gives each dropped variant (`selection.run_study`'s `dropped`)."""
    return [
        RunRecord(
            hypothesis_id=selection._hypothesis_id(variant),
            family_id=spec.family_id,
            variant=variant,
            ticket=selection.TICKET,
            components={"claim": "dropped", "benchmark": spec.benchmark},
            params={"spec_sha256": frozen, "dropped_before_freeze": True},
            metrics=(),
            status="failed",
            **identity,
        )
        for variant in spec.dropped_variants
    ]


def _failed_records(spec: FamilySpec, frozen: str, identity: dict, why: str) -> list[RunRecord]:
    """A failed record of every rule, with run_study's variant identity (its `record`'s
    components and params), so a crash adds no variant to the family size."""
    return [
        RunRecord(
            hypothesis_id=selection._hypothesis_id(rule.rule_id),
            family_id=spec.family_id,
            variant=rule.rule_id,
            ticket=TICKET,
            components={
                "claim": rule.claim,
                "benchmark": spec.benchmark,
                "score": "winner_log_loss",
            },
            params={
                "spec_sha256": frozen,
                "segment": dict(rule.segment),
                "selection_cols": list(rule.selection_cols),
                "thresholds": {col: list(b) for col, b in rule.thresholds.items()},
                "direction": rule.direction,
                "decision_time": spec.decision_time,
                "cluster_unit": spec.cluster_unit,
                "max_stat": dict(spec.max_stat),
                "odds_band_cut_points": list(spec.odds_band_cut_points),
            },
            metrics=(),
            status="failed",
            notes={"run": f"selection_run (#{TICKET}) failed: {why}"},
            **identity,
        )
        for rule in spec.rules
    ]


def _append_dropped(registry: Registry, records: list[RunRecord]) -> None:
    """Each dropped variant the family does not hold yet, so the family size counts it once."""
    held = {r.variant for r in registry.records(records[0].family_id)} if records else set()
    for record in records:
        if record.variant not in held:
            registry.append(record)


def _append_failed(registry: Registry, records: list[RunRecord]) -> None:
    """Each failure record the registry does not hold already (the same crash rerun)."""
    held = registry.records(records[0].family_id) if records else []
    for record in records:
        if record not in held:
            registry.append(record)


def run_selection(
    *,
    date_from: date,
    date_to: date,
    out_dir: Path,
    seed: int,
    registry: Registry | None = None,
    holdout: LockedHoldout | None = None,
    log: Callable[[str], None] | None = None,
) -> SelectionRun:
    """Freeze the family and record its dropped variants, build the frame, run the study and
    write the private report (module docstring). Raises `PreReformPriceError` for a span before
    2014-06-07 and `paths.DataDirInsideRepo` for an `out_dir` inside the repository, before any
    file is opened or record appended."""
    log = log or (lambda message: None)
    started = time.monotonic()
    _check_span(date_from, date_to)
    out_dir = paths.outside_repo(Path(out_dir), what="out")
    registry = default_registry(COUNTRY) if registry is None else registry
    spec = load_family(COUNTRY)
    frozen = registry.freeze_family(spec.family_id, dataclasses.asdict(spec))
    identity = _identity()
    _append_dropped(registry, _dropped_records(spec, frozen, identity))
    try:
        frame, counts = _build(date_from, date_to, log)
        if holdout is None:  # the country's block, held and never opened; the frame has no row
            _, holdout = split_off(frame[["race_date", "race_id"]], "race_date", COUNTRY)
        log(f"running the study (peak {_peak_gb():.2f} GB)")
        study = run_study(
            spec, frame, registry=registry, holdout=holdout, seed=seed, sources=SOURCES
        )
    except Exception as exc:  # the run is a record too: every rule failed
        _append_failed(registry, _failed_records(spec, frozen, identity, f"{exc!r}"))
        raise
    about = {
        "family": f"`{spec.family_id}`, spec sha256 `{frozen}`",
        "family size": f"{study.family_size} (rules and dropped variants, failures included)",
        "span": f"{date_from.isoformat()} to {date_to.isoformat()} (development days only)",
        "sources": ", ".join(SOURCES),
        "decision time": spec.decision_time,
        "seed": str(seed),
        "git": identity["git_sha"],
        "holdout": "held, never opened",
        "wall time": f"{time.monotonic() - started:.0f} s",
        "peak RSS": f"{_peak_gb():.2f} GB",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    text = render_report(spec, study, counts, about)
    path = write_report(Path(out_dir) / REPORT_NAME, text, sources=SOURCES)
    log(f"done: {path}")
    return SelectionRun(study=study, report_path=Path(path), counts=counts)


# --------------------------------------------------------------------------- the report


def _num(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def render_report(spec: FamilySpec, study: StudyReport, counts: dict, about: dict) -> str:
    """The private report: family size, each rule's stages, survivors, P(omega_s > 1/(1 - t))
    and rung, the frame's counts, and every stage's full results."""
    stages = study.stages
    lines = [
        "# Family P selection study: development folds, JRA-VAN boards at t (#462)",
        "",
        "PRIVATE: built on JRA-VAN (D21). No figure here enters the repository.",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        "",
        f"Family size: {study.family_size}. Rules: {len(spec.rules)}; dropped variants "
        f"recorded as failed: {', '.join(spec.dropped_variants) or 'none'}.",
        "",
        "## Rules",
        "",
        "| rule | claim | segment | selection | stage 1 passed | delta LL [lo, hi] | coef "
        "| StepM superior | survivor | bets | P(omega_s > 1/(1 - t)) | omega selected "
        "| expected return [lo, hi] | ROI | rung |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- |",
    ]
    superior = set(stages["max_stat"].get("superior", []))
    for rule in spec.rules:
        rid = rule.rule_id
        ll, om, roi = stages["log_loss"][rid], stages["omega"][rid], stages["roi"][rid]
        er = om["expected_return"]
        segment = ", ".join(f"{k} = {v}" for k, v in rule.segment.items())
        chosen = ", ".join(f"{c} in {rule.thresholds[c]}" for c in rule.selection_cols)
        delta = f"{_num(ll.get('delta_ll'), 6)} [{_num(ll.get('lo'), 6)}, {_num(ll.get('hi'), 6)}]"
        expected = f"{_num(er.get('mean'))} [{_num(er.get('low'))}, {_num(er.get('high'))}]"
        cells = [
            rid,
            rule.claim,
            segment,
            chosen,
            "yes" if ll.get("passed") else "no",
            delta,
            _num(ll.get("coef")),
            "yes" if rid in superior else "no",
            "yes" if rid in study.survivors else "no",
            str(om.get("n_bets")),
            _num(study.p_beats_takeout[rid]),
            _num(om.get("omega_selected")),
            expected,
            _num(roi.get("roi")),
            study.rungs[rid],
        ]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    lines += [
        "",
        f"Survivors (stage 1 passed and StepM superior; a diagnostic): "
        f"{', '.join(study.survivors) or 'none'}.",
        "",
        "Rungs: every rule sits at S0, D19's default; no rule is above S1. A rule moves only "
        "through the leaderboard and the owner.",
        "",
        "## Errors by stage",
        "",
    ]
    errors = [
        f"- {rid}, {stage}: {entry['error']}"
        for stage in ("log_loss", "omega", "roi")
        for rid, entry in stages[stage].items()
        if "error" in entry
    ]
    if "error" in stages["max_stat"]:
        errors.append(f"- max statistic: {stages['max_stat']['error']}")
    lines += errors or ["- none"]
    lines += ["", "## The frame", ""]
    lines += [f"- {key}: {value}" for key, value in counts.items()]
    lines += [
        "",
        "## Every stage in full",
        "",
        "```json",
        json.dumps(stages, indent=1, sort_keys=True, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- summary and CLI


def write_committed_summary(registry: Registry) -> None:
    """The family's registry summary file (`write_summary`, values withheld for a licensed run)
    and the committed index with its line added; no other family's file is written."""
    family = load_family(COUNTRY).family_id
    directory = summary_dir(COUNTRY)
    mine = _Families(registry, {family})
    if not mine.records():
        return
    with tempfile.TemporaryDirectory() as scratch:
        write_summary(mine, scratch)
        for path in sorted(Path(scratch).glob("*.md")):
            if path.stem != "README":
                write_report(directory / path.name, path.read_text("utf-8"), sources=())
        fresh = Path(scratch, "README.md").read_text("utf-8").splitlines()
    index = directory / "README.md"
    committed = index.read_text("utf-8").splitlines() if index.exists() else []
    entries = {name: line for line in committed + fresh if (name := _index_family(line))}
    head = [line for line in fresh if not _index_family(line)]
    lines = [*head, *(entries[name] for name in sorted(entries))]
    write_report(index, "\n".join(lines) + "\n", sources=())


def last_development_day() -> date | None:
    """The last development day the jravan odds_timeseries holds (file names only)."""
    files, _ = _day_files(SPAN_START, date.max)
    return files[-1][0] if files else None


def selection_run_command(
    date_from: Annotated[
        str, typer.Option("--from", help="First day (default 2014-06-07, D10).")
    ] = SPAN_START.isoformat(),
    date_to: Annotated[
        str | None,
        typer.Option(
            "--to", help="Last day (default: the last development day of the jravan boards)."
        ),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Report directory, outside the repository.")
    ] = None,
) -> None:
    """Run the frozen Family P selection study (#462) on the development days of the JRA-VAN
    boards at t; the private report goes under the reports dir and the registry summary is
    rewritten. Prints no figure."""
    end = date.fromisoformat(date_to) if date_to else last_development_day()
    if end is None:
        raise typer.BadParameter("no jravan odds_timeseries day file; pass --to")
    registry = default_registry(COUNTRY)
    result = run_selection(
        date_from=date.fromisoformat(date_from),
        date_to=end,
        out_dir=out or paths.reports_dir() / "selection",
        seed=SEED,
        registry=registry,
        log=lambda message: print(message, file=sys.stderr),
    )
    write_committed_summary(registry)
    typer.echo(
        json.dumps(
            {
                "family_id": load_family(COUNTRY).family_id,
                "report_path": str(result.report_path),
                "rungs_at_most_s1": set(result.study.rungs.values()) <= {"S0", "S1"},
            },
            indent=2,
        )
    )
