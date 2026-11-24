"""Reconciliation checks on the built tables; report: reports_dir()/store-check-<source>.md.

Ported from HKTrack's store/checks.py, tolerances unchanged. Hard failures (`fail`): duplicate race
ids, duplicate (race_id, horse_id), or more than 1% of a table's race ids missing from `races`.
Everything else is counted and reported as `warn` while it stays rare - a few odd pages in many
seasons are expected and must not block the pipeline - and becomes a `fail` once it is systematic
(more than SYSTEMATIC_MIN cases AND more than 1% of what was checked): a parser reading the wrong
column must not pass. `integrity` holds the checks on the build itself: empty tables, pages that
failed to parse, tables that do not match `_build.json` (a build killed half-way through replacing
the files).

The unit stake, the WIN pool's code and the odds display rules come from the source's
`SourceConfig` (`core.config.load_source(country, source)`), passed in by the caller.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from horsetrack.core.config import SourceConfig
from horsetrack.core.paths import processed_dir, reports_dir
from horsetrack.core.store import tables
from horsetrack.core.store.build import BUILD_FILE, write_text_atomic

OK, WARN, FAIL = "ok", "warn", "fail"
ORPHAN_FAIL_RATE = 0.01
SYSTEMATIC_RATE = 0.01  # countable problems: a fail above this share of what was checked ...
SYSTEMATIC_MIN = 5  # ... but never for a handful of cases
# Displayed odds are the dividend per unit, floored: to 1 decimal below the threshold, to an integer
# at or above it. So dividend / unit - odds must fall in [0, step), with a little slack for the 0.05
# dividend steps.
ODDS_SLACK_BELOW = 0.06
ODDS_SLACK_AT_OR_ABOVE = 1.0
_EPS = 1e-9
_MAX_EXAMPLES = 10
RACE_CHILD_TABLES = ("runners", "dividends", "sectionals", "incidents", "running_comments")
_LEVELS = [OK, WARN, FAIL]


@dataclass
class Check:
    name: str
    level: str
    violations: int
    checked: int
    detail: str = ""
    examples: list[str] = field(default_factory=list)

    def line(self) -> str:
        text = f"{self.level.upper():<4} {self.name}: {self.violations} of {self.checked}"
        return f"{text} ({self.detail})" if self.detail else text


@dataclass
class CheckResult:
    source: str
    checks: list[Check] = field(default_factory=list)
    # the build itself: emptiness, lost pages, mixed files
    integrity: list[Check] = field(default_factory=list)
    seasons: list[dict[str, Any]] = field(default_factory=list)  # season, races, starters, share
    date_min: str | None = None
    date_max: str | None = None
    report_path: str = ""

    @property
    def failures(self) -> list[Check]:
        return [check for check in [*self.checks, *self.integrity] if check.level == FAIL]

    @property
    def ok(self) -> bool:
        return not self.failures

    def error_line(self) -> str | None:
        if self.ok:
            return None
        parts = "; ".join(f"{c.name} {c.violations} of {c.checked}" for c in self.failures)
        return f"store check failed for {self.source}: {parts} (see {self.report_path})"

    def lines(self) -> list[str]:
        """<= 15 lines for the terminal; the per-season table is in the report."""
        orphans = [c for c in self.checks if c.name.startswith("orphan race ids")]
        out = [f"store check {self.source}: {'OK' if self.ok else 'FAILED'}"]
        out += [check.line() for check in self.checks if check not in orphans]
        if orphans:
            level = max((c.level for c in orphans), key=_LEVELS.index)
            cells = ", ".join(
                f"{c.name.rsplit(' ', 1)[-1]} {c.violations}/{c.checked}" for c in orphans
            )
            out.append(f"{level.upper():<4} orphan race ids (not in races): {cells}")
        out += [check.line() for check in self.integrity]
        out.append(f"race dates {self.date_min} .. {self.date_max}, {len(self.seasons)} seasons")
        if self.seasons:
            shares = [s["rating_share"] for s in self.seasons if s["rating_share"] is not None]
            races = sum(s["races"] for s in self.seasons)
            starters = sum(s["starters"] for s in self.seasons)
            spread = f"{100 * min(shares):.1f}%..{100 * max(shares):.1f}%" if shares else "n/a"
            out.append(f"run races {races}, starters {starters}, rating share by season {spread}")
        out.append(f"report {self.report_path}")
        return out


def _counted(violations: int, checked: int) -> str:
    """ok / warn / fail for a problem tolerable while rare and a parser defect once systematic."""
    if not violations:
        return OK
    return FAIL if violations > max(SYSTEMATIC_MIN, SYSTEMATIC_RATE * checked) else WARN


def _examples(values: pd.Series) -> list[str]:
    return [str(v) for v in values.drop_duplicates().head(_MAX_EXAMPLES)]


def _flag(values: pd.Series) -> pd.Series:
    """A nullable boolean column as plain bools, missing = False."""
    return values.fillna(False).astype(bool)


def check_unique_races(races: pd.DataFrame) -> Check:
    duplicated = races["race_id"].notna() & races.duplicated("race_id", keep="first")
    n = int(duplicated.sum())
    examples = _examples(races.loc[duplicated, "race_id"])
    return Check("unique race_id", FAIL if n else OK, n, len(races), examples=examples)


def check_unique_runners(runners: pd.DataFrame) -> Check:
    known = runners.loc[runners["horse_id"].notna(), ["race_id", "horse_id"]]
    duplicated = known.duplicated(["race_id", "horse_id"], keep="first")
    n = int(duplicated.sum())
    dup = known.loc[duplicated]
    examples = _examples(dup["race_id"].astype(str) + " " + dup["horse_id"].astype(str))
    detail = f"{len(runners) - len(known)} rows without a horse id are not checked"
    return Check("unique (race_id, horse_id)", FAIL if n else OK, n, len(known), detail, examples)


def check_field_size(races: pd.DataFrame, runners: pd.DataFrame) -> Check:
    starters = runners.loc[~_flag(runners["is_scratched"])].groupby("race_id").size()
    counted = races["race_id"].map(starters).fillna(0).astype("int64")
    declared = races["field_size"]
    wrong = declared.isna() | (declared.fillna(-1).astype("int64") != counted)
    n = int(wrong.sum())
    detail = "races.field_size vs runner rows not scratched"
    examples = _examples(races.loc[wrong, "race_id"])
    return Check("field_size", _counted(n, len(races)), n, len(races), detail, examples)


def check_win_dividend(
    races: pd.DataFrame, runners: pd.DataFrame, dividends: pd.DataFrame, config: SourceConfig
) -> Check:
    """Single-winner races: the winner's final odds x unit stake is the WIN dividend, floored for
    display."""
    win_code = next((code for code, pool in config.pools.items() if pool.kind == "win"), "WIN")
    threshold = float(config.odds_display.get("threshold", 10))
    cap = config.odds_display.get("cap")

    run = set(races.loc[races["status"].eq("run"), "race_id"])
    first = runners.loc[_flag(runners["finish_pos"].eq(1)) & runners["race_id"].isin(run)]
    per_race = first.groupby("race_id")["horse_no"].transform("size")
    winners = first.loc[(per_race == 1) & ~_flag(first["dead_heat"])]
    win = dividends.loc[dividends["pool"].eq(win_code) & ~_flag(dividends["is_refund"])]
    win = win.loc[
        win.groupby("race_id")["pool"].transform("size") == 1,
        ["race_id", "combination", "dividend"],
    ]
    both = winners[["race_id", "horse_no", "win_odds"]].merge(
        win, on="race_id", how="left", validate="one_to_one"
    )

    odds = pd.to_numeric(both["win_odds"]).to_numpy(dtype="float64", na_value=np.nan)
    dividend = pd.to_numeric(both["dividend"]).to_numpy(dtype="float64", na_value=np.nan)
    per_unit = dividend / float(config.unit_stake)
    missing = np.isnan(odds) | np.isnan(per_unit)
    gap = per_unit - odds
    slack = np.where(odds < threshold, ODDS_SLACK_BELOW, ODDS_SLACK_AT_OR_ABOVE)
    if cap is not None:  # the display stops at the cap: any larger dividend is consistent with it
        slack = np.where(odds >= float(cap), np.inf, slack)
    paid_on = both["combination"].astype("string")
    other_horse = _flag(paid_on.notna() & (paid_on != both["horse_no"].astype("string")))
    wrong = ~missing & ((gap < -_EPS) | (gap >= slack) | other_horse.to_numpy(dtype=bool))
    n, n_missing, n_checked = int(wrong.sum()), int(missing.sum()), int((~missing).sum())
    detail = f"single-winner run races; {n_missing} more without odds or a {win_code} dividend"
    shown = both.loc[wrong].head(_MAX_EXAMPLES)
    examples = [f"{r.race_id} odds {r.win_odds} dividend {r.dividend}" for r in shown.itertuples()]
    # races that drop out of the comparison count too: "0 of 0" must never read as a pass
    level = max(_counted(n, n_checked), _counted(n_missing, len(both)), key=_LEVELS.index)
    return Check(f"winner odds vs {win_code} dividend", level, n, n_checked, detail, examples)


def check_unique_horse_no(runners: pd.DataFrame) -> Check:
    """Two runners of one race under the same saddle number: dividends and sectionals key on it."""
    known = runners.loc[
        runners["horse_no"].notna() & runners["race_id"].notna(), ["race_id", "horse_no"]
    ]
    duplicated = known.duplicated(["race_id", "horse_no"], keep="first")
    n = int(duplicated.sum())
    dup = known.loc[duplicated]
    examples = _examples(dup["race_id"].astype(str) + " #" + dup["horse_no"].astype(str))
    return Check(
        "unique (race_id, horse_no)", _counted(n, len(known)), n, len(known), examples=examples
    )


def check_not_empty(races: pd.DataFrame, runners: pd.DataFrame) -> Check:
    """A build whose parsers all failed writes valid, empty tables: every other check then passes
    on 0 rows."""
    n_run = int(races["status"].eq("run").sum())
    counts = (("races", len(races)), ("run races", n_run), ("runners", len(runners)))
    empty = [name for name, n in counts if not n]
    detail = f"{len(races)} races ({n_run} run), {len(runners)} runners"
    return Check("tables not empty", FAIL if empty else OK, len(empty), 3, detail, empty)


def check_build_coverage(record: dict[str, Any] | None) -> Check:
    """Pages the last build could not put into the tables, from `_build.json`."""
    name = "pages missing from the build"
    if not record:
        return Check(name, WARN, 0, 0, f"no {BUILD_FILE}: the tables were not written by the build")
    failed = record.get("failed") or []
    lost = len(failed) + len(record.get("unreadable") or [])
    parts = {
        "parse failures": len(failed),
        "unreadable": len(record.get("unreadable") or []),
        "key mismatches": len(record.get("key_mismatch") or [])
        + len(record.get("excluded_with_mismatch") or []),
        "duplicates": len(record.get("duplicate_pages") or []),
        "not parsed yet": int(record.get("n_not_parsed") or 0),
    }
    selected = int(record.get("n_selected") or 0)
    detail = ", ".join(f"{n} {label}" for label, n in parts.items() if n)
    level = _counted(lost, selected)
    if level == OK and any(parts.values()):
        level = WARN
    examples = [f"{f.get('page_type')}/{f.get('key')}" for f in failed[:_MAX_EXAMPLES] if f]
    return Check(name, level, lost, selected, detail, examples)


def check_build_record(record: dict[str, Any] | None, row_counts: dict[str, int]) -> Check:
    """The files on disk are the ones the last build wrote (it replaces the files one after the
    other, so a build killed half-way leaves a mix). A warning: the next store build repairs it."""
    name = f"tables match {BUILD_FILE}"
    if not record:
        return Check(name, WARN, 0, len(row_counts), f"no {BUILD_FILE}")
    expected = record.get("rows") or {}
    wrong = [
        f"{table}: {row_counts.get(table)} rows on disk, {n} in {BUILD_FILE}"
        for table, n in expected.items()
        if row_counts.get(table) != n
    ]
    detail = "run the store build again" if wrong else ""
    level = WARN if wrong else OK
    return Check(name, level, len(wrong), len(expected), detail, wrong[:_MAX_EXAMPLES])


def check_orphans(table: str, frame: pd.DataFrame, races: pd.DataFrame) -> Check:
    race_ids = frame["race_id"].dropna().drop_duplicates()
    orphans = race_ids[~race_ids.isin(set(races["race_id"]))]
    n, total = len(orphans), len(race_ids)
    level = OK if not n else FAIL if n > ORPHAN_FAIL_RATE * total else WARN
    return Check(f"orphan race ids in {table}", level, n, total, examples=_examples(orphans))


def season_stats(races: pd.DataFrame, runners: pd.DataFrame) -> list[dict[str, Any]]:
    """Per season: run races, starters in them, and the share of those starters with a rating."""
    run = races.loc[races["status"].eq("run")]
    starters = runners.loc[
        runners["race_id"].isin(set(run["race_id"])) & ~_flag(runners["is_scratched"])
    ]
    n_races = run.groupby("season").size()
    grouped = (
        starters.assign(rated=starters["rating"].notna())
        .groupby("season")["rated"]
        .agg(["size", "sum"])
    )
    out = []
    for season in sorted(set(n_races.index) | set(grouped.index)):
        n = int(grouped["size"].get(season, 0))
        rated = int(grouped["sum"].get(season, 0))
        out.append(
            {
                "season": season,
                "races": int(n_races.get(season, 0)),
                "starters": n,
                "rating_share": rated / n if n else None,
            }
        )
    return out


def _report(result: CheckResult) -> str:
    lines = [
        f"# Store check: {result.source}",
        "",
        f"Result: **{'OK' if result.ok else 'FAILED'}**",
        "",
    ]
    lines += ["| check | level | violations | checked | note |", "|---|---|---:|---:|---|"]
    lines += [
        f"| {c.name} | {c.level} | {c.violations} | {c.checked} | {c.detail} |"
        for c in [*result.checks, *result.integrity]
    ]
    for check in [*result.checks, *result.integrity]:
        if check.examples:
            lines += ["", f"## {check.name}: first examples", *(f"- {e}" for e in check.examples)]
    lines += ["", f"## Coverage: {result.date_min} .. {result.date_max}", ""]
    lines += ["| season | run races | starters | with rating |", "|---|---:|---:|---:|"]
    for s in result.seasons:
        share = "n/a" if s["rating_share"] is None else f"{100 * s['rating_share']:.1f}%"
        lines.append(f"| {s['season']} | {s['races']} | {s['starters']} | {share} |")
    return "\n".join(lines) + "\n"


def _load(source: str, name: str, columns: Sequence[str]) -> pd.DataFrame:
    try:
        return tables.read_table(source, name, columns)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{error}: run the store build first") from error


def _build_record(source: str) -> dict[str, Any] | None:
    path = processed_dir(source) / BUILD_FILE
    try:
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def run_checks(source: str, *, config: SourceConfig, write_report: bool = True) -> CheckResult:
    """Every check on the source's built tables; the report goes to
    reports_dir()/store-check-<source>.md unless `write_report` is False."""
    races = _load(source, "races", ["race_id", "race_date", "season", "field_size", "status"])
    runner_columns = [
        "race_id", "season", "horse_id", "horse_no", "finish_pos", "dead_heat", "win_odds",
        "is_scratched", "rating",
    ]  # fmt: skip
    runners = _load(source, "runners", runner_columns)
    dividends = _load(
        source, "dividends", ["race_id", "pool", "combination", "dividend", "is_refund"]
    )
    children = {"runners": runners, "dividends": dividends}
    for name in RACE_CHILD_TABLES:
        if name not in children:
            children[name] = _load(source, name, ["race_id"])

    result = CheckResult(source)
    result.checks = [
        check_unique_races(races),
        check_unique_runners(runners),
        check_field_size(races, runners),
        check_win_dividend(races, runners, dividends, config),
        *(check_orphans(name, children[name], races) for name in RACE_CHILD_TABLES),
    ]
    record = _build_record(source)
    result.integrity = [
        check_not_empty(races, runners),
        check_unique_horse_no(runners),
        check_build_coverage(record),
        check_build_record(record, tables.row_counts(source)),
    ]
    result.seasons = season_stats(races, runners)
    run_dates = races.loc[races["status"].eq("run"), "race_date"].dropna()
    result.date_min = min(run_dates).isoformat() if len(run_dates) else None
    result.date_max = max(run_dates).isoformat() if len(run_dates) else None
    path = reports_dir() / f"store-check-{source}.md"
    result.report_path = str(path)
    if write_report:
        write_text_atomic(path, _report(result))
    return result


def as_dict(result: CheckResult) -> dict[str, Any]:
    return {**asdict(result), "ok": result.ok}
