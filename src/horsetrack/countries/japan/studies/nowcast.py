"""The C1 nowcast's real-data run (#85): `horsetrack japan nowcast-study`.

Data (owner, 2026-11-26): the JRA-VAN time-series paths in the store,
`processed/jravan/odds_timeseries/<YYYY-MM-DD>.parquet` (0B41 WIN, PLACE, bracket; 0B42 QUINELLA;
exactly `logger.compact.SCHEMA`), read one day at a time by column and with the model's pools
only. D10: race days from 2014-06-07 (`PRICE_ERA_START`) only, since the nowcast is the price
half. Our own compacted logged days, `processed/jra/odds_timeseries/<YYYY-MM-DD>.parquet` (owner,
#85: added to the training as they accrue; #411), are read beside them, by the same columns and
pools plus status_raw, fetched_utc and `is_final_state`, with `--end` applied to both
(`read_logged_day`):
- only their rows with a post time;
- per (race, pool), the JRA-VAN path wins where it has a post time; otherwise our logged path
  supplies it (in practice our logs add Wide, D3). Each path is timed by its own source's post;
- a logged path counts only with a confirmed final capture (a capture whose priced rows of that
  pool all show the final state, docs/specs/board-at-t.md); one whose capture stopped before the
  close gives no case and no growth observation and is counted (`no_final_capture`);
- a runner scratched in our logged WIN or PLACE rows takes its Wide pairs off the Wide board from
  the reporting page's fetched_utc on (its snapshot_utc where it has none;
  `models.nowcast.extract`, `scratches=`), JRA-VAN's WIN rows or not; our logged prices and pool
  totals count from their page's fetched_utc too, so a capture round that straddles a grid
  cutoff never mixes pages fetched after it into that board.
D14: holdout days of either source are dropped by `is_holdout_day` on the file name, before any
read; the holdout is not opened. Grades are the jravan races table's `race_class` (None for a
race it does not hold). Wide is fitted and scored where the folds' training days hold Wide
cases; the report states plainly why when they do not (naming each missing input: a final
capture, a board before the post), and a (pool, tau) a fold's training gives no fit for is not
scored there (counted per fold, `unscored_no_fit`; a fold of such cases only is reported with no
PIT, and a run with no scored case still renders its report). Wide coverage reads the logged
days the folds read.

Folds: D9's walk-forward (`config/folds.yaml`) over the development race days. Each fold fits on
its training days only: the structural nowcast, the same with beta = 0 (`no_features`), and #116's
drift (`NowcastV0.fit` on the same boards' (low figure at t, final low figure) pairs with
nowcast_v0.yaml's bands and prior_n; its `max_board_age_s` limit is not applied, since it was set
for a logger capturing every ~30 s and on 5-6 minute paths it would drop most examples at T-2 and
T-1, and all four forecasters are fitted and scored on the same boards). The random walk is the
structural fit's. Each test day's cases are scored by every forecaster with the same view, n and
seed (`models.nowcast.score_cases`); with `--test-days-per-fold N`, a seeded sample of N test
days per fold is scored (training is never sampled).

A late scratch follows the owner's #69 rule: the case is scored with every forecaster's drawn
shares renormalised over the combinations priced at the close, and counted (`renormalised`). The
fits, the drift's included, train on the cases whose board at t is the final board.

Report (through `write_report`; every figure is withheld, D21): per comparison and (pool, tau),
the log-score, CRPS and KL gains with one-sided 95% bounds resampling race days (the ticket) and
meetings (D7); PIT deciles and 50% / 90% coverage per forecaster; sigma_t and beta per pool and
tau across folds; the phi_t profile against the observed V_t / V_final; the board's age at t; the
Place rule's check at t; the Place rule against the final votes (#410: the closing PLACE board
of every scored Place case, the folds' test cases after test-day sampling, read as scored, against
its final H1 votes in `pool_votes`, read through the store's scoped reader so holdout rows never
reach the study; after the folds); Wide coverage (development days only: a holdout day's file is
never read, D14); counts. A read's Place rows stand as a result only when it validated boards and
its shares were within the tolerance on at least `PLACE_MIN_WITHIN_TOLERANCE` of them
(`place_result_readers`); the gains and the PIT hold the Place cases of those reads only, and the
others (`undetermined` always) are reported apart (`gains_undetermined`, `gains_unvalidated`,
`pit_diagnostic`). One
registry record per comparison (E19) and one for the Place rule (#410), failures included (a
measurement that stands no read is a failed, unvalidated one, and the report then calls the Place
rows a diagnostic), then `write_summary` regenerates the committed summary. Family
`japan-c1-nowcast-final-shares-v6` supersedes v5, whose spec changed after it was frozen (Place's
P read from the field at the start of sales on the board reads and the drift baseline; v5's freeze
and records stay in the registry as history); v5 superseded v4, whose Place verdict took coverage
for agreement,
measured every extracted Place race rather than the scored ones and pooled the undetermined
boards' cases into the Place gains; v4 superseded v3, whose Place measurement validated the
`final_odds` boards rather than the scored closing boards, took non-final votes as final and
registered an empty measurement as scored; v3 superseded v2, whose Place shares inverted the low
figures by a rule that applied the takeout to the pool rather than the dividend; v2 superseded
`japan-c1-nowcast-final-shares`, whose spec dropped late-scratch cases against #69. v6 is the
family of a run on JRA-VAN days alone. A run that reads our logged days registers under
`japan-c1-nowcast-logged-wide-v1` (#411), whose spec (v6's design with our logged days) names
both sources and the rules above, and writes its reports as
`nowcast-japan-logged-wide-v1-<spec hash>`, so the JRA-VAN-only reports are never replaced. A
run with no fold (`--end` before D9's first test window) suffixes its reports `-no-fold`, so it
never replaces the full study's reports either. It
is a line of its own, not a successor of #410's `japan-c1-nowcast-final-shares-v3` to `-v6`. A
`--trial` run is a run too: its records go to the family's `-trial` family with
`params["trial"] = True`, and it never freezes or alters the family's spec.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import typer

from horsetrack.core import config, paths
from horsetrack.core.eval.folds import load_country_scheme
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import (
    FoldMetric,
    RunRecord,
    default_registry,
    summary_dir,
    write_summary,
)
from horsetrack.core.eval.reports import write_report
from horsetrack.core.pricing import stake_back_share_dividend
from horsetrack.countries.japan.board.snapshot import SCRATCHED, RaceBoard, load_decision_grid
from horsetrack.countries.japan.models import nowcast as nc
from horsetrack.countries.japan.models.v0.nowcast import NowcastV0
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table

COUNTRY = "japan"
TICKET = 85
SOURCE = "jravan"
HYPOTHESIS = "E19"
FAMILY_ID = "japan-c1-nowcast-final-shares-v6"  # JRA-VAN days alone (#85, #410)
TRIAL_FAMILY_ID = f"{FAMILY_ID}-trial"  # trial runs' records; a trial never freezes FAMILY_ID
SUPERSEDES = "japan-c1-nowcast-final-shares-v5"
LOGGED_FAMILY_ID = "japan-c1-nowcast-logged-wide-v1"  # with our own logged days (#411)
LOGGED_SOURCE = "jra"  # our own compacted logs, processed/jra/odds_timeseries
PLACE_TICKET = 410
PLACE_VARIANT = "place_rule_vs_votes"  # the Place rule measured against the final votes (#410)
POOL_VOTES = "pool_votes"
NULL_FIELD = "\\N"  # how `place_rule_builds` writes a null field; no count or data_kubun is it
# H1 データ区分 of a final record: 4 確定, 5 確定(月曜); 2 (前日売最終) is the day-before close
# and 9 a cancelled race, so their votes are not the money the closing board was printed from
PLACE_FINAL_KUBUN = ("4", "5")
# #410's tolerance for shares read from a board's figures against its money (per runner): a
# figure truncated by up to 0.1 moves the share it implies by up to 6.8% at a figure of 2.0
# (tests/countries/japan/test_nowcast.py states the derivation).
PLACE_REL_TOL = 0.075
PLACE_ABS_TOL = 0.01
# #410 re-check: the share of a read's validated boards whose shares are within that tolerance
# for the read's Place rows to stand as a result. By the derivation above a read that recovers
# the money is within the tolerance on every board whose figures the rule prints from the votes,
# so the bound is near 1; the slack leaves room for the few boards whose votes and figures
# disagree for reasons outside the read (which `share_reproduced` counts), while a read that
# fails structurally misses wherever its failure applies (the low figures on the tests' synthetic
# floored favourite). A data-validity check on the share read, not an adoption gate.
PLACE_MIN_WITHIN_TOLERANCE = 0.95
# `models.nowcast.place_shares`'s read whose figures do not determine the money: a diagnostic
UNDETERMINED = "undetermined"
PLACE_READS = ("joint", "low_only", UNDETERMINED)
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
]
# Our logs add the scratch markers (Wide boards), each page's fetch time (when its prices, totals
# and scratches are known) and the final-state flag (compact.py).
LOGGED_COLUMNS = [*COLUMNS, "status_raw", "fetched_utc"]
FINAL_STATE = "is_final_state"
# A fold's score table (`nc.score_cases` and the fold's label)
SCORE_COLUMNS = ["case", "forecaster", "log_score", "crps", "kl", "race_id", "day", "venue"] + [
    "pool", "tau_min", "reader", "fold"
]  # fmt: skip
FORECASTERS = ("structural", "no_features", "random_walk", "drift_v0")
COMPARISONS = (
    ("structural", "random_walk"),
    ("structural", "drift_v0"),
    ("no_features", "random_walk"),
    ("structural", "no_features"),
    ("drift_v0", "random_walk"),
)
N_DRAWS = 200
SEED = 85
PIT_BINS = 10
_AT = datetime(2000, 1, 1, tzinfo=timezone.utc)  # the drift's RaceBoard needs a time; unused


# ---------------------------------------------------------------- inputs


def day_files(start: date, end: date | None, source: str = SOURCE) -> list[tuple[str, Path]]:
    """The day files of `source` from `start` to `end`, by file name; nothing is opened."""
    folder = paths.processed_dir(source) / "odds_timeseries"
    out = []
    for path in sorted(folder.glob("*.parquet")) if folder.is_dir() else []:
        day = path.stem
        if day >= start.isoformat() and (end is None or day <= end.isoformat()):
            out.append((day, path))
    return out


@dataclasses.dataclass(frozen=True)
class LoggedDay:
    frame: pd.DataFrame  # the paths our log supplies
    scratches: pd.DataFrame  # every WIN / PLACE scratch row of the log (for the Wide boards)
    no_final_capture: dict[str, int]  # per pool: paths dropped for want of a final capture
    final_capture: dict[str, int]  # per pool: paths kept, each with a confirmed final capture


def _paths(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[["race_id", "pool"]])


def _final_paths(frame: pd.DataFrame) -> pd.MultiIndex:
    """The (race, pool) paths with a confirmed final capture: a capture (rows sharing a
    snapshot_utc) whose priced rows of that pool all show the final state (`is_final_state`;
    docs/specs/board-at-t.md, as `board.snapshot._pool_rows_final`)."""
    priced = frame[
        (frame["combination"] != nc.POOL_ROW)
        & (frame["odds"] > 0)
        & (frame["status_raw"] != SCRATCHED)
    ]
    final = priced[FINAL_STATE].eq(True) if FINAL_STATE in priced.columns else False
    captures = priced.assign(_final=final).groupby(["race_id", "pool", "snapshot_utc"])["_final"]
    done = captures.all()
    return pd.MultiIndex.from_tuples(
        sorted({(r, p) for (r, p, _), ok in done.items() if ok}), names=["race_id", "pool"]
    )


def read_logged_day(path: Path, jravan: pd.DataFrame | None) -> LoggedDay:
    """One of our own compacted logged days, as `read_day` (with status_raw and, where the file
    has them, fetched_utc and `is_final_state`). Kept: rows with a post time (a row without one
    cannot be placed on the tau grid); the (race, pool) paths the same day's JRA-VAN rows `jravan`
    do not hold with a post time (JRA-VAN wins a path only where it can be timed); and, of those,
    the paths with a confirmed final capture (counted per pool, `final_capture`): a logged path
    whose capture stopped before the close has no final board or V_final, so it is no target and
    no growth observation, and is counted (`no_final_capture`). A file compacted before #411 has
    no final-state flag, so none of its paths is kept (recompact). The rows keep their
    fetched_utc: a logged price, total or scratch is known from its page's fetch
    (`models.nowcast._available`)."""
    present = set(pq.read_schema(path).names)
    frame = read_day(path, [c for c in (*LOGGED_COLUMNS, FINAL_STATE) if c in present])
    scratches = frame[(frame["status_raw"] == SCRATCHED) & frame["pool"].isin((nc.WIN, nc.PLACE))]
    frame = frame[frame["post_utc"].notna()]
    if jravan is not None and len(jravan) and len(frame):
        timed = jravan[jravan["post_utc"].notna()][["race_id", "pool"]].drop_duplicates()
        frame = frame[~_paths(frame).isin(pd.MultiIndex.from_frame(timed))]
    final = _final_paths(frame)
    paths = frame[["race_id", "pool"]].drop_duplicates()
    stopped = paths[~_paths(paths).isin(final)]
    kept = paths[_paths(paths).isin(final)]
    frame = frame[_paths(frame).isin(final)].drop(columns=[FINAL_STATE], errors="ignore")
    return LoggedDay(
        frame.reset_index(drop=True),
        scratches.reset_index(drop=True),
        {str(k): int(v) for k, v in stopped["pool"].value_counts().items()},
        {str(k): int(v) for k, v in kept["pool"].value_counts().items()},
    )


def read_day(path: Path, columns: list[str] = COLUMNS) -> pd.DataFrame:
    """One day file, by column and with the model's pools only; Arrow's pool is released after."""
    table = pq.read_table(path, columns=columns, filters=[("pool", "in", list(nc.POOLS))])
    frame = table.to_pandas()
    del table
    pa.default_memory_pool().release_unused()
    return frame


def read_grades(start: date) -> dict[str, str | None]:
    races = pq.read_table(
        paths.processed_dir(SOURCE) / "races.parquet",
        columns=["race_id", "race_class"],
        filters=[("race_date", ">=", start)],
    ).to_pandas()
    return dict(
        zip(races["race_id"], races["race_class"].where(races["race_class"].notna()), strict=True)
    )


def _files_digest(files: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for day, path in files:
        digest.update(f"{day}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()[:16]


def data_builds(
    files: list[tuple[str, Path]], logged: list[tuple[str, Path]] | None = None
) -> dict[str, str]:
    """The inputs' builds: the JRA-VAN day files and races table, and our own logged development
    days when any were read (by name and size; a holdout day is not among them)."""
    races = paths.processed_dir(SOURCE) / "races.parquet"
    race_digest = hashlib.sha256(races.read_bytes()).hexdigest()[:16]
    builds = {
        "jravan_odds_timeseries": f"processed/{SOURCE}/odds_timeseries@{_files_digest(files)}",
        "jravan_races": f"processed/{SOURCE}/races.parquet@{race_digest}",
    }
    if logged:
        builds["jra_odds_timeseries"] = (
            f"processed/{LOGGED_SOURCE}/odds_timeseries@{_files_digest(logged)}"
        )
    return builds


def wide_coverage(logged: list[tuple[str, Path]] | None = None) -> dict:
    """Our own logged development days (processed/jra/odds_timeseries): Wide rows and post times.
    `logged`: the day files the run reads (its `--end` applied); by default every file in the
    folder. A holdout day is fully locked (D14): it is skipped by its file name, before any read."""
    if logged is None:
        folder = paths.processed_dir(LOGGED_SOURCE) / "odds_timeseries"
        logged = [(p.stem, p) for p in sorted(folder.glob("*.parquet"))] if folder.is_dir() else []
    days = []
    for day, path in logged:
        if is_holdout_day(COUNTRY, date.fromisoformat(day)):
            continue
        t = pq.read_table(path, columns=["race_id", "pool", "post_utc"]).to_pandas()
        wide = t[t["pool"] == nc.WIDE]
        days.append(
            {
                "day": day,
                "wide_races": int(wide["race_id"].nunique()),
                "wide_races_with_post_time": int(
                    wide.dropna(subset=["post_utc"])["race_id"].nunique()
                ),
            }
        )
    return {"own_logged_days": days}


def wide_status(
    coverage: dict,
    cases: nc.Cases,
    folds,
    cells: list[dict],
    no_final_capture: int = 0,
    final_capture: int = 0,
) -> dict:
    """Whether Wide (D3) was fitted, and when not, a sentence saying why: no Wide in our logs, no
    post time on the logged Wide races, no logged Wide path giving a case, no fold at all, Wide
    cases on no fold's training day, or too few of them for a cell. Confirmed finals are tracked
    apart from usable grid boards, and a path-level reason names each missing input:
    `no_final_capture` Wide paths dropped for want of a final capture, `final_capture` Wide paths
    kept with one that still give no case (no usable board before the post on the tau grid). The
    Wide cases and days are counted either way."""
    logged = coverage["own_logged_days"]
    wide_races = sum(e["wide_races"] for e in logged)
    with_post = sum(e["wide_races_with_post_time"] for e in logged)
    wide_days = sorted(set(cases.day[cases.pool == nc.WIDE].tolist()))
    trained = sorted({d for train, _, _ in folds for d in train} & set(wide_days))
    fitted = any(c["pool"] == nc.WIDE for c in cells)
    missing = []
    if no_final_capture > 0:
        missing.append(
            f"{no_final_capture} reached no confirmed final capture (a capture showing the final "
            "state), so they give no final board (a day compacted before #411 carries no "
            "final-state flag: recompact it)"
        )
    if final_capture > 0:
        missing.append(
            f"the {final_capture} with a confirmed final capture have no usable pre-post board "
            "on the tau grid (none of their captures before the post gives a grid case)"
        )
    reasons = (
        (
            not wide_races,
            f"Wide is not fitted: none of our {len(logged)} logged development days holds a Wide "
            "race (the JRA-VAN series has no Wide pool).",
        ),
        (
            not with_post,
            f"Wide is not fitted: none of the {wide_races} Wide races on our logged development "
            "days has a post time, so no Wide board can be placed on the tau grid (recompact the "
            "days: the race card now gives the post times, #411).",
        ),
        (
            not wide_days and (no_final_capture > 0 or final_capture > 0),
            "Wide is not fitted: no logged Wide path with a post time gives a case; "
            + "; ".join(missing)
            + ".",
        ),
        (
            not folds,
            "Wide is not fitted: the run has no fold (its development days end before D9's "
            "first test window), so no cell of any pool is fitted.",
        ),
        (
            not trained,
            f"Wide is not fitted: the Wide cases are on {len(wide_days)} logged day(s) "
            f"({', '.join(wide_days) or 'none'}), none of them a training day of any fold, so no "
            "fold can fit Wide yet; too few logged days hold Wide.",
        ),
        (
            True,
            f"Wide is not fitted: the folds' training days hold Wide cases on {len(trained)} "
            "day(s), too few for a pool-growth profile or any (pool, tau) cell.",
        ),
    )
    reason = None if fitted else next(text for applies, text in reasons if applies)
    return {
        "fitted": fitted,
        "reason": reason,
        "wide_cases": int((cases.pool == nc.WIDE).sum()),
        "wide_case_days": wide_days,
        "wide_training_days": trained,
    }


# ---------------------------------------------------------------- the Place rule (#410)


def final_place_votes(days: list[str]) -> pd.DataFrame:
    """The PLACE votes (H1 slots of `pool_votes`) of `days`, with each row's `data_kubun`, read one
    calendar year at a time through the store's scoped reader (`models.v0.store.read_store_table`,
    a price read): the pool, slot and day predicates run inside the scan and holdout days are
    dropped there, so only development rows reach the study (D14, #415)."""
    columns = ["race_id", "combination", "votes", "data_kubun"]
    wanted = sorted({date.fromisoformat(d) for d in days})
    parts = []
    for year in sorted({d.year for d in wanted}):
        in_year = pa.array([d for d in wanted if d.year == year], pa.date32())
        table = read_store_table(
            POOL_VOTES,
            columns=columns,
            start=max(date(year, 1, 1), PRICE_ERA_START),
            end=date(year, 12, 31),
            touches_price=True,
            where=(pc.field("pool") == nc.PLACE)
            & (pc.field("combination") != nc.POOL_ROW)
            & pc.field("race_date").isin(in_year),
            source=SOURCE,
        )
        parts.append(table.to_pandas())
        del table
        pa.default_memory_pool().release_unused()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def place_rule_builds(votes: pd.DataFrame) -> dict[str, str]:
    """The final votes' build: a digest of the scoped rows the measurement reads
    (`final_place_votes` of the development days, as the odds build covers them), not of the
    whole file, so votes outside them (holdout days among them) never enter it. The rows are
    serialised as sorted tab-separated lines (votes as integers), so the digest does not depend
    on the scan's row order. A null field (a blank vote count, a missing data_kubun) is written as
    `NULL_FIELD`, a value of its own, before the sort: it never raises, whatever string dtype
    pandas gives, and it differs from any count and from the row dropped. The build keeps the
    table identity the registered records carry, `processed/{SOURCE}/pool_votes.parquet`; only
    the digest after the `@` follows the rows."""
    fields = (
        votes["race_id"],
        votes["combination"],
        votes["votes"].astype("Int64"),
        votes["data_kubun"],
    )
    cells = [[NULL_FIELD if pd.isna(v) else str(v) for v in column] for column in fields]
    lines = sorted("\t".join(row) for row in zip(*cells, strict=True))
    digest = hashlib.sha256("".join(f"{line}\n" for line in lines).encode())
    table = f"processed/{SOURCE}/{POOL_VOTES}.parquet"
    return {"jravan_pool_votes": f"{table}@{digest.hexdigest()[:16]}"}


def jra_place_figures(share: np.ndarray, places: int) -> tuple[np.ndarray, np.ndarray]:
    """The low and high Place figures JRA prints for money shares `share`: each runner's
    `core.pricing.stake_back_share_dividend` with the P - 1 most-backed others (low) and the
    P - 1 least-backed (high), per 100 yen, over 100."""
    pool = config.load_source(COUNTRY, "jra").pools[nc.PLACE]
    unit = float(config.load_source(COUNTRY, "jra").unit_stake)
    kw = {
        "rate": 1.0 - float(pool.takeout),
        "unit": unit,
        "dividend_step": float(pool.dividend_step),
        "min_dividend": float(pool.min_dividend),
    }
    order = np.argsort(-share, kind="stable")
    lo, hi = np.empty(len(share)), np.empty(len(share))
    for i in range(len(share)):
        others = share[order[order != i]]
        lo[i] = stake_back_share_dividend(share[i], others[: places - 1].tolist(), **kw) / unit
        hi[i] = stake_back_share_dividend(share[i], others[-(places - 1) :].tolist(), **kw) / unit
    return lo, hi


def place_rule_measurement(boards: Iterable[tuple], votes: pd.DataFrame) -> dict:
    """#410: the Place rule against the money on the closing boards the study scores. `boards`
    are (race_id, combinations, low, high[, starters]) as `models.nowcast.closing_boards` gives
    them (every priced runner, a high figure NaN where the board has none, the field at the start
    of sales); `votes` the PLACE H1 rows (`race_id`, `combination`, `votes`, `data_kubun`). Each
    board is read as the study reads it (`models.nowcast.place_shares`: P from the field at sale,
    or from its priced runners on a board without one, `joint`, `undetermined` or `low_only`; an
    undetermined board is counted under its own read, a diagnostic). It is
    validated when its race's H1 record is final (data_kubun in `PLACE_FINAL_KUBUN`) with positive
    votes on every priced runner; otherwise it is unavailable, counted by reason. Per validated
    board: whether the rule reproduces every printed figure from the votes (P as scored, and
    again from the registered slots), and how far the shares as scored, and the low figures'
    read, are from the vote shares."""
    by_race = {race_id: g for race_id, g in votes.groupby("race_id", sort=False)}
    place_pool = config.load_source(COUNTRY, "jra").pools[nc.PLACE]
    counts: dict[str, int] = defaultdict(int)
    scored, rows = [], []
    for race_id, combinations, lo, hi, *field in boards:
        starters = field[0] if field else None
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        got, rule_sum, reader = nc.place_shares(lo, hi, starters=starters)
        if got is None:
            counts["boards_not_sold"] += 1
            continue
        places = place_pool.places_for(len(lo) if starters is None else int(starters))
        scored.append({"reader": reader, "places": places})
        g = by_race.get(race_id)
        if g is None:
            counts[f"unavailable_no_votes:{reader}"] += 1
            continue
        if not g["data_kubun"].isin(PLACE_FINAL_KUBUN).all():
            counts[f"unavailable_votes_not_final:{reader}"] += 1
            continue
        money = (
            g.drop_duplicates("combination")
            .set_index("combination")["votes"]
            .reindex(list(combinations))
            .to_numpy(dtype=float, na_value=np.nan)
        )
        if not (np.isfinite(money).all() and (money > 0).all()):
            counts[f"unavailable_votes_missing_on_a_priced_runner:{reader}"] += 1
            continue
        share = money / money.sum()
        alt = place_pool.places_for(int(g["combination"].nunique()))
        row = {
            "reader": reader,
            "places": places,
            "reproduced": _reproduces(share, lo, hi, places),
            "places_from_slots_differ": alt != places,
        }
        row["reproduced_with_slot_places"] = (
            row["reproduced"] if alt == places else alt >= 2 and _reproduces(share, lo, hi, alt)
        )
        low_only, _ = nc.pool_shares(nc.PLACE, lo, starters=starters)
        for name, read in (("as_scored", got), ("low_only", low_only)):
            err = np.abs(np.asarray(read) - share)
            row[f"{name}_max_rel_error"] = float((err / share).max())
            row[f"{name}_max_abs_error"] = float(err.max())
        row["rule_sum"] = rule_sum
        rows.append(row)
    counts["boards_scored"] = len(scored)
    counts["boards_validated"] = len(rows)
    return {
        "counts": dict(counts),
        "summary": _place_summary(pd.DataFrame(scored), pd.DataFrame(rows)),
    }


def _reproduces(share: np.ndarray, lo: np.ndarray, hi: np.ndarray, places: int) -> bool:
    """Whether the rule prints every low figure, and every high figure the board has, from
    `share`."""
    want_lo, want_hi = jra_place_figures(share, places)
    has = np.isfinite(hi)
    return bool(
        np.allclose(want_lo, lo, rtol=0, atol=1e-6)
        and np.allclose(want_hi[has], hi[has], rtol=0, atol=1e-6)
    )


def _place_summary(scored: pd.DataFrame, boards: pd.DataFrame) -> list[dict]:
    """Per read used in scoring (`joint`, `low_only`, `undetermined`) and number of places (and
    over all): the
    boards scored, those validated and their share (coverage); on the validated boards, the share
    whose printed figures the rule reproduces from the votes, the share read within #410's
    tolerance as scored (and by the low figures alone, the fallback, on the same boards), and
    quantiles of the worst per-runner errors and of the rule's money total."""
    if scored.empty:
        return []
    out = []
    for reader, s in scored.groupby("reader", sort=True):
        v_all = boards[boards["reader"] == reader] if not boards.empty else boards
        labels = [("all", None)] + [(f"P={p}", p) for p in sorted(s["places"].unique())]
        for label, p in labels:
            n_scored = int(len(s) if p is None else (s["places"] == p).sum())
            v = v_all if p is None or v_all.empty else v_all[v_all["places"] == p]
            row = {
                "reader": reader,
                "boards": label,
                "n_scored": n_scored,
                "n": len(v),
                "coverage": len(v) / n_scored,
            }
            if len(v):
                row |= _validated_stats(v)
            out.append(row)
    return out


def _validated_stats(v: pd.DataFrame) -> dict:
    rel, ab = v["as_scored_max_rel_error"].to_numpy(), v["as_scored_max_abs_error"].to_numpy()
    low_rel, low_ab = v["low_only_max_rel_error"].to_numpy(), v["low_only_max_abs_error"].to_numpy()
    total = v["rule_sum"].to_numpy()
    return {
        "share_reproduced": float(v["reproduced"].mean()),
        "share_slot_places_differ": float(v["places_from_slots_differ"].mean()),
        "share_reproduced_with_slot_places": float(v["reproduced_with_slot_places"].mean()),
        "share_within_tolerance": float(((rel <= PLACE_REL_TOL) & (ab <= PLACE_ABS_TOL)).mean()),
        "low_only_share_within_tolerance": float(
            ((low_rel <= PLACE_REL_TOL) & (low_ab <= PLACE_ABS_TOL)).mean()
        ),
        "max_rel_error_q50": float(np.quantile(rel, 0.5)),
        "max_rel_error_q95": float(np.quantile(rel, 0.95)),
        "max_rel_error_max": float(rel.max()),
        "max_abs_error_q95": float(np.quantile(ab, 0.95)),
        "rule_sum_q05": float(np.quantile(total, 0.05)),
        "rule_sum_q50": float(np.quantile(total, 0.5)),
        "rule_sum_q95": float(np.quantile(total, 0.95)),
    }


def place_result_readers(measured: dict) -> set[str]:
    """The determined reads (`joint`, `low_only`) whose Place rows stand as a result: the read
    validated boards (coverage) and its shares are within the tolerance on at least
    `PLACE_MIN_WITHIN_TOLERANCE` of them (agreement). An empty or failed measurement gives none,
    and `undetermined` is never one (ruling #410)."""
    return {
        str(row["reader"])
        for row in measured.get("summary", [])
        if row["boards"] == "all"
        and row["reader"] != UNDETERMINED
        and row["n"] > 0
        and row.get("share_within_tolerance", 0.0) >= PLACE_MIN_WITHIN_TOLERANCE
    }


def place_validated(measured: dict) -> bool:
    """Whether any read's Place rows stand as a result (`place_result_readers`)."""
    return bool(place_result_readers(measured))


def _place_metrics(measured: dict) -> tuple[FoldMetric, ...]:
    out = []
    for row in measured["summary"]:
        for key in ("coverage", "share_reproduced", "share_within_tolerance", "max_rel_error_q95"):
            value = row.get(key)
            if value is not None and np.isfinite(value):
                name = f"{key}:{row['reader']}:{row['boards']}"
                n = int(row["n_scored"] if key == "coverage" else row["n"])
                out.append(FoldMetric("pooled", name, float(value), None, None, n))
    return tuple(out)


def build_spec(grid, scheme, bootstrap: dict, v0_config: dict, *, logged: bool = False) -> dict:
    """The family's design, frozen before any development row is read: `FAMILY_ID`'s (#85, #410)
    on the JRA-VAN days alone, or with `logged` (our own logged development days are read, #411)
    `LOGGED_FAMILY_ID`'s."""
    spec = {
        "family_id": FAMILY_ID,
        "supersedes": f"{SUPERSEDES}: its spec changed after the family was frozen (#410 round "
        "7): Place's P is read from the field at the start of sales (the runners priced on the "
        "race's earliest PLACE board), not from the runners still priced after a scratch, on the "
        "board reads and on #116's drift baseline; v5 superseded v4, whose Place verdict "
        "counted a read as validated when its "
        "boards had final votes (coverage) whatever its agreement with them, measured every "
        "extracted Place race rather than the cases the folds scored, and pooled the "
        "undetermined boards' Place cases into the Place gains (#410's re-check); v4 superseded "
        "v3, whose Place rule measurement validated the final_odds boards rather than the "
        "closing boards the cases score (runners without a high figure dropped, P and the read "
        "changed), took non-final H1 votes as final and registered an empty measurement as "
        "scored (#410's review); v3 superseded v2, whose Place shares "
        "inverted the low figures alone by a rule that applied the takeout to the pool, not to "
        "the dividend; v2 superseded japan-c1-nowcast-final-shares, whose spec dropped "
        "late-scratch cases against the owner's #69 rule",
        "ticket": TICKET,
        "place_ticket": PLACE_TICKET,
        "hypotheses": {
            HYPOTHESIS: "the structural nowcast (phi_t x s_late, logistic-normal late composition "
            "with velocity and favourite-rank features) forecasts the final ln-shares of the Win "
            "and Place pools better than the random walk q_final = q_t and than #116's fitted "
            "drift, per decision horizon: log score (D8, primary) and CRPS gains with one-sided "
            "95% day-clustered bounds; PIT and 50%/90% coverage by horizon",
        },
        "null": "no gain over the random walk at any horizon (the starting point)",
        "data": {
            "source": f"processed/{SOURCE}/odds_timeseries (JRA-VAN 0B41/0B42)",
            "first_day": PRICE_ERA_START.isoformat(),
            "pools": [nc.WIN, nc.PLACE, nc.WIDE],
            "snapshot_rule": "a row counts from its snapshot_utc (owner 2026-11-26: no lag)",
            "final": "each combination's latest row (is_final_capture), V_final the latest total",
            "late_scratches": "the owner's #69 rule (core/eval/scoring.py): a combination priced "
            "on the board at t and not at the close is removed; every forecaster gets the board "
            "at t, its drawn shares are restricted to the combinations priced at the close and "
            "renormalised, and the case is scored and counted (renormalised); the fits train on "
            "the cases whose board at t is the final board",
            "dropped": "cases with no board at t or no final board, and cases whose final board "
            "prices a combination the board at t does not (added_by_close), each counted; a race "
            "without a post time gives no case",
            "place_shares": "no per-runner money in the series, so the figures are read by JRA's "
            "stake-back-share rule (core.pricing.stake_back_share_dividend: rate x (1 + D / "
            "(P x share)), truncated to 10 yen, floored at the minimum dividend): a board with a "
            "high figure on every runner by models.nowcast.place_money_fit (least squares over "
            "both figures at their truncation midpoints, floored figures left out; a fit whose "
            "used figures do not determine the money, by an SVD rank test of their design with "
            "relative tolerance models.nowcast.PLACE_RANK_RTOL, keeps its minimum-norm shares, "
            "is still scored and is flagged undetermined by models.nowcast.place_shares, and its "
            "Place rows are a diagnostic; the rank is computed only where the read is named: "
            "place_shares and the closing board of each case, never in the draws), else the "
            "low figures alone in closed form (owner, #410); the rule's money total at t is "
            "reported; #116's drift draws are printed as JRA prints (truncated, floored) and "
            "read by the same rule",
            "place_rule_measurement": "after the folds, on the closing PLACE board of every "
            "scored Place case (the folds' test cases after test-day sampling; no broader "
            "development measurement) "
            "(models.nowcast.closing_boards: the final board the cases score, every priced "
            "runner kept), read as scored (models.nowcast.place_shares: P from the field at the "
            "start of sales, the runners priced on the race's earliest PLACE board, the joint, "
            "undetermined or low-only read), against the PLACE votes (H1) of "
            "processed/"
            f"{SOURCE}/{POOL_VOTES}.parquet, read by the store's scoped reader (development "
            "days only); a board is validated when its H1 record is final (data_kubun "
            f"{'/'.join(PLACE_FINAL_KUBUN)}) with positive votes on every priced runner, else "
            "it is unavailable and counted by reason; per read used in scoring and P, coverage "
            "(validated over scored), whether the rule reproduces every printed figure from the "
            "votes (P as scored, and again from the registered slots), the share read within "
            f"{PLACE_REL_TOL} relative and {PLACE_ABS_TOL} absolute of the vote shares on every "
            "runner (as scored, and the low figures alone on the same boards), the worst errors "
            "and the rule's money total; one registry record: scored when a read stands as a "
            "result, failed (params.unvalidated) when none does or the measurement cannot run, "
            "and then the Place rows are a diagnostic",
            "place_result": "coverage and agreement are separate: a determined read (joint, "
            "low_only; never undetermined) stands as a result when it validated a board and its "
            "shares as scored are within the tolerance on at least "
            f"{PLACE_MIN_WITHIN_TOLERANCE} of its validated boards (a data-validity check on the "
            "share read, not an adoption gate: by the tolerance's derivation a read that "
            "recovers the money is within it on every board whose figures the rule prints from "
            "the votes, the slack leaving room for boards whose votes and figures disagree for "
            "reasons outside the read); each score row carries its closing board's read; the "
            "result gain tables (day, meeting, fold) hold the Place cases of the reads that "
            "stand only, the undetermined boards' Place cases go to gains_undetermined and, when "
            "some read stands, every other read's to gains_unvalidated (by read); when no read "
            "stands, the determined reads stay in the gain tables and the report calls their "
            "Place rows a diagnostic",
        },
        "grid": {"minutes": list(grid.minutes), "primary": grid.primary},
        "fold_scheme": dataclasses.asdict(scheme),
        "variants": {
            "structural": "models.nowcast.Nowcast.fit_cases(features=True)",
            "no_features": "the same with beta = 0",
            "random_walk": "the structural fit's random_walk (softmax(ln s_t + sigma_rw eps))",
            "drift_v0": "models.v0.nowcast.NowcastV0.fit on the same boards, bands and prior_n of "
            f"nowcast_v0.yaml ({v0_config['bands']}, {v0_config['prior_n']}), no board-age limit",
            PLACE_VARIANT: "data.place_rule_measurement",
        },
        "comparisons": [f"{a}_vs_{b}" for a, b in COMPARISONS],
        "scores": "per case (race, pool, tau): mean over combinations of the Gaussian log score "
        "(draws' mean and sd of ln share) and of the ensemble CRPS of the final ln-share; KL of "
        "the final shares from the mean draw; same view, n and seed for every forecaster",
        "bootstrap": {**bootstrap, "units": ["day (the ticket)", "meeting (D7)"]},
        "model_constants": {
            "knots": list(nc.KNOTS),
            "profile_penalty": nc.PROFILE_PENALTY,
            "effect_ridge": nc.EFFECT_RIDGE,
            "r_floor": nc.R_FLOOR,
            "n_cal": nc.N_CAL,
            "n_cal_draws": nc.N_CAL_DRAWS,
            "cal_seed": nc.CAL_SEED,
        },
    }
    return _logged_spec(spec) if logged else spec


def _logged_spec(base: dict) -> dict:
    """`LOGGED_FAMILY_ID`: `FAMILY_ID`'s design on the JRA-VAN days and our own logged
    development days (#411), a line of its own beside #410's final-shares-v3 to -v6."""
    spec = {k: v for k, v in base.items() if k != "supersedes"}
    spec["family_id"] = LOGGED_FAMILY_ID
    spec["extends"] = (
        f"{FAMILY_ID}: the same design, with our own logged development days read beside the "
        f"JRA-VAN days and Wide (D3) fitted from them (#411); {FAMILY_ID} stays the family of a "
        "run on JRA-VAN days alone"
    )
    spec["supersedes"] = (
        "none: a separate line from #410's japan-c1-nowcast-final-shares-v3 to -v6, not a "
        "successor of any"
    )
    spec["hypotheses"] = {
        HYPOTHESIS: base["hypotheses"][HYPOTHESIS].replace(
            "of the Win and Place pools", "of the Win, Place and Wide pools"
        )
    }
    spec["data"] = base["data"] | {
        "source": f"processed/{SOURCE}/odds_timeseries (JRA-VAN 0B41/0B42) and "
        f"processed/{LOGGED_SOURCE}/odds_timeseries (our own compacted logged days, #411)",
        "holdout": "holdout days of either source are skipped by file name before any read (D14)",
        "snapshot_rule": "a row counts from its page's fetched_utc, its snapshot_utc where it "
        "has none (JRA-VAN rows; docs/specs/board-at-t.md; owner 2026-11-26: no lag), prices, "
        "pool totals and scratches alike, so a capture round that straddles a grid cutoff puts "
        "only the pages fetched by the cutoff on that board",
        "merge": "per day and per (race, pool): the JRA-VAN path where it has a post time, else "
        "our logged path (its rows with a post time); each path is timed by its own source's "
        "post time",
        "final_capture": "a logged (race, pool) path counts only with a confirmed final capture, "
        "a capture whose priced rows of that pool all show the final state (最終オッズ, no page "
        "clock: docs/specs/board-at-t.md; compaction's is_final_state); otherwise it gives no "
        "case and no growth observation and is counted (no_final_capture)",
        "wide_shares": "the low figures inverted by JRA's pair rule (models.nowcast.wide_money): "
        "the pair {a,b} wins with {a,x} and {b,x} for one third runner x, its low figure with "
        "max over x of (m_ax + m_bx), each winning pair paid by the stake-back-share identity "
        "(core.pricing): (1 - takeout) x (1 + D / (3 m)), D the losers' money; the drift's Wide "
        "draws are read back by the same rule",
        "wide_scratches": "a runner scratched in our logged WIN or PLACE rows (status_raw SCR) "
        "takes its Wide pairs off the Wide board from the reporting row's fetched_utc on (its "
        "snapshot_utc where it has none; JRA's page stops listing them), so a case from before "
        "the scratch is renormalised by #69's rule",
        "unscored": "a (pool, tau) a fold's training days give no fit for (Wide before our logs "
        "accrue) is not scored in that fold and is counted (unscored_no_fit)",
    }
    return spec


# ---------------------------------------------------------------- the drift baseline


def drift_examples(cases: nc.Cases, days: list[str]) -> pd.DataFrame:
    """#116's training examples from the boards of `days`: one per combination and tau, with
    the columns `NowcastV0.fit` reads."""
    rep = cases.row_case()
    fitted = nc._in_days(cases.day, days) & (cases.scratched == 0)  # as the structural fits
    ok = np.repeat(fitted, cases.k) & (cases.odds_t > 0) & (cases.odds_f > 0)
    rep = rep[ok]
    return pd.DataFrame(
        {
            "day": pd.Categorical(cases.day[rep]),
            "pool": pd.Categorical(cases.pool[rep]),
            "tau_min": cases.tau[rep].astype(np.int16),
            "odds_t": cases.odds_t[ok],
            "odds_final": cases.odds_f[ok],
        }
    )


def drift_forecaster(model: NowcastV0):
    """#116's drift as a share forecaster: its odds draws, read back into shares by the pool's
    share rule (WIN 1/O; PLACE its low and high draws printed as JRA prints them, then read by
    the Place rule as a board is, P from the board's field at sale; Wide the pair rule's,
    `nc.wide_money`). A Wide pair's shock is v0's (eps_a + eps_b) / sqrt 2 of per-runner
    shocks."""

    def forecast(view: nc.RaceView, *, pool: str, n: int, seed: int) -> nc.Draws:
        board = view.pools[pool]
        keys = [nc._combination_key(c) for c in board.combinations]
        keys = [k if pool == nc.WIDE else k[0] for k in keys]
        lo = np.asarray(board.odds, dtype=float)
        hi = np.asarray(board.odds_hi, dtype=float)
        win = dict(zip(keys, lo.tolist(), strict=True)) if pool == nc.WIN else {}
        figures = {
            k: (o, None if not np.isfinite(x) else x)
            for k, o, x in zip(keys, lo.tolist(), hi.tolist(), strict=True)
        }
        place = figures if pool == nc.PLACE else {}
        wide = figures if pool == nc.WIDE else {}
        race = RaceBoard(view.race_id, None, _AT, _AT, False, win, place, wide, {}, frozenset())
        runners = sorted({h for k in keys for h in k}) if pool == nc.WIDE else keys
        eps = np.random.default_rng(seed).standard_normal((n, len(runners)))
        shocks = {h: eps[:, j] for j, h in enumerate(runners)}
        drawn = model.draws(race, pool=pool, tau_min=int(round(view.tau_min)), shocks=shocks)
        odds, odds_hi = drawn.odds, drawn.odds_hi
        if pool == nc.PLACE:  # read as the boards are: printed figures by the Place rule
            odds, odds_hi = nc.printed_figures(pool, odds), nc.printed_figures(pool, odds_hi)
        shares, _ = nc.pool_shares(
            pool,
            odds,
            odds_hi=odds_hi,
            starters=board.starters,
            combinations=drawn.combinations,
        )
        return nc.Draws(shares=shares, v_final=np.full(n, np.nan))

    return forecast


# ---------------------------------------------------------------- aggregation


def _pit_summary(pit: pd.DataFrame, cases: nc.Cases, offset: int, acc: dict) -> None:
    """Add PIT decile counts and 50% / 90% coverage counts per (forecaster, pool, tau, closing
    board's read; `none` off the Place rule), split by read once the reads that stand are known
    (`_split_pit`)."""
    local = pit["case"].to_numpy() - offset
    keys = pd.DataFrame(
        {
            "forecaster": pit["forecaster"].to_numpy(),
            "pool": cases.pool[local],
            "tau": cases.tau[local],
            "reader": pd.Series(cases.reader[local], dtype=object).fillna("none").astype(str),
            "u": pit["pit"].to_numpy(),
        }
    )
    for (name, pool, tau, reader), g in keys.groupby(["forecaster", "pool", "tau", "reader"]):
        u = g["u"].to_numpy()
        slot = acc.setdefault((name, pool, float(tau), reader), np.zeros(PIT_BINS + 3))
        slot[:PIT_BINS] += np.histogram(u, bins=PIT_BINS, range=(0.0, 1.0))[0]
        slot[PIT_BINS] += ((u >= 0.25) & (u <= 0.75)).sum()
        slot[PIT_BINS + 1] += ((u >= 0.05) & (u <= 0.95)).sum()
        slot[PIT_BINS + 2] += len(u)


def _quantiles(x: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean())} | {
        f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs
    }


def board_age(cases: nc.Cases) -> list[dict]:
    rows = []
    for pool in nc.POOLS:
        for tau in sorted(set(cases.tau.tolist()), reverse=True):
            m = (cases.pool == pool) & (cases.tau == tau)
            if not m.any():
                continue
            age = cases.board_min[m] - tau
            rows.append(
                {
                    "pool": pool,
                    "tau_min": tau,
                    **_quantiles(age),
                    "share_over_1min": float((age > 1).mean()),
                    "share_over_3min": float((age > 3).mean()),
                    "max": float(age.max()),
                }
            )
    return rows


def place_rule_check(cases: nc.Cases) -> list[dict]:
    rows = []
    for tau in sorted(set(cases.tau.tolist()), reverse=True):
        m = (cases.pool == nc.PLACE) & (cases.tau == tau)
        x = cases.rule_sum[m]
        if not m.any():
            continue
        rows.append(
            {
                "tau_min": tau,
                **_quantiles(x),
                "share_within_2pct": float((np.abs(x - 1) <= 0.02).mean()),
                "share_within_5pct": float((np.abs(x - 1) <= 0.05).mean()),
            }
        )
    return rows


def _table(rows: list[dict], digits: int = 4) -> str:
    if not rows:
        return "(none)"
    head = list(dict.fromkeys(key for row in rows for key in row))

    def cell(v):
        if isinstance(v, float):
            return f"{v:.{digits}g}"
        return str(v)

    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(cell(row.get(k, "")) for k in head) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- the run


def _progress(started: float) -> str:
    """Elapsed time and the process's peak resident memory so far."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 2**30 if sys.platform == "darwin" else peak / 2**20  # bytes on macOS, KiB else
    return f"{time.monotonic() - started:.0f} s, peak {peak_gb:.2f} GB"


def run_study(
    *,
    end: date | None = None,
    test_days_per_fold: int | None = None,
    n_draws: int = N_DRAWS,
    trial: bool = False,
    out_dir: Path | None = None,
    log=print,
) -> dict:
    grid = load_decision_grid()
    scheme = load_country_scheme(COUNTRY)
    scoring = config.load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    bootstrap = {"n_boot": int(scoring["n_boot"]), "level": float(scoring["level"]), "seed": SEED}
    v0_config = config.load_country_yaml(COUNTRY, "nowcast_v0")

    # D14: both sources' holdout days are dropped by file name, before any of their files is read.
    files = day_files(PRICE_ERA_START, end)
    logged = day_files(PRICE_ERA_START, end, source=LOGGED_SOURCE)
    held = sorted({d for d, _ in files + logged if is_holdout_day(COUNTRY, date.fromisoformat(d))})
    files = [(d, p) for d, p in files if d not in set(held)]
    logged = [(d, p) for d, p in logged if d not in set(held)]

    # A run that reads our logged days is LOGGED_FAMILY_ID's (#411); one on JRA-VAN days alone
    # stays FAMILY_ID's.
    base = LOGGED_FAMILY_ID if logged else FAMILY_ID
    sources = (SOURCE, LOGGED_SOURCE) if logged else (SOURCE,)
    spec = build_spec(grid, scheme, bootstrap, v0_config, logged=bool(logged))
    # The registry's spec hash (sorted, compact JSON), so a trial's names the spec it would freeze.
    spec_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    registry = default_registry(COUNTRY)
    family = f"{base}-trial" if trial else base
    if not trial:  # a trial's spec may still change, so it is not frozen
        spec_hash = registry.freeze_family(base, spec)  # before any development row is read
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    git_sha = _git_sha()
    builds = data_builds(files, logged)
    by_day: dict[str, list[Path | None]] = defaultdict(lambda: [None, None])
    for slot, source_files in enumerate((files, logged)):
        for d, p in source_files:
            by_day[d][slot] = p
    days = sorted(by_day.items())
    started = time.monotonic()
    grades = read_grades(PRICE_ERA_START)
    log(f"reading {len(days)} development days ({len(files)} JRA-VAN, {len(logged)} our logs)")
    case_chunks, growth_chunks, case_parts, growth_parts = [], [], [], []
    closing = []  # the closing PLACE boards, for the Place rule's measurement (#410)
    counts: dict[str, int] = defaultdict(int)
    if logged:
        counts["no_final_capture"] = 0
    no_final_wide = final_wide = 0
    for i, (_, (jravan_path, logged_path)) in enumerate(days):
        frame = None if jravan_path is None else read_day(jravan_path)
        scratches = None
        if logged_path is not None:
            own = read_logged_day(logged_path, frame)
            counts["no_final_capture"] += sum(own.no_final_capture.values())
            no_final_wide += own.no_final_capture.get(nc.WIDE, 0)
            final_wide += own.final_capture.get(nc.WIDE, 0)
            scratches = own.scratches
            if frame is None:
                frame = own.frame
            elif len(own.frame):  # a JRA-VAN path without a post time gives way to our log's
                frame = pd.concat([frame[frame["post_utc"].notna()], own.frame], ignore_index=True)
            del own
        day_cases, day_growth, c = nc.extract(frame, grades=grades, scratches=scratches)
        closing.extend(nc.closing_boards(frame, nc.PLACE))
        del frame, scratches
        case_parts.append(day_cases)
        growth_parts.append(day_growth)
        for key, value in c.items():
            counts[key] += value
        if len(case_parts) == 100 or i == len(days) - 1:
            case_chunks.append(nc.Cases.concat(case_parts))
            growth_chunks.append(nc.Growth.concat(growth_parts))
            case_parts, growth_parts = [], []
            log(f"  {i + 1}/{len(days)} days read ({_progress(started)})")
    cases = nc.Cases.concat(case_chunks)
    growth = nc.Growth.concat(growth_chunks)
    del case_chunks, growth_chunks
    on_place = cases.pool == nc.PLACE
    place_races = set(cases.race_id[on_place].tolist())
    closing = [b for b in closing if b[0] in place_races]  # kept for the scored races' check
    del on_place, place_races
    dev_days = sorted(set(cases.day.tolist()))
    folds = nc.fold_days(dev_days, scheme)
    log(f"{len(cases)} cases on {len(dev_days)} days, {len(folds)} folds ({_progress(started)})")

    rng = np.random.default_rng(SEED)
    scores, pit_acc, cells, phi_rows, fold_rows = [], {}, [], [], []
    scored_place = []  # (race_id, day) of every scored Place case, for the Place rule (#410)
    offset = wide_scored = 0
    try:
        for train, test, fold in folds:
            label = f"{fold.test_start.year}Q{(fold.test_start.month - 1) // 3 + 1}"
            scored_days = list(test)
            if test_days_per_fold is not None and len(test) > test_days_per_fold:
                scored_days = sorted(rng.choice(test, size=test_days_per_fold, replace=False))
            structural = nc.Nowcast.fit_cases(cases, growth, features=True, days=train)
            plain = nc.Nowcast.fit_cases(cases, growth, features=False, days=train)
            drift = NowcastV0.fit(
                drift_examples(cases, train),
                bands=v0_config["bands"],
                grid=grid,
                prior_n=float(v0_config["prior_n"]),
            )
            test_cases = cases.select_days(scored_days)
            # A (pool, tau) the training days give no fit for (Wide before our logs accrue) is
            # not scored: every forecaster needs its cell.
            fitted = [
                (pool, int(tau)) in structural.cells
                and (pool, int(tau)) in plain.cells
                and (pool, int(tau)) in drift.sigma0
                for pool, tau in zip(test_cases.pool, test_cases.tau, strict=True)
            ]
            unscored = len(test_cases) - sum(fitted)
            test_cases = test_cases.select(np.array(fitted, dtype=bool))
            on_place = test_cases.pool == nc.PLACE
            scored_place.extend(
                zip(
                    test_cases.race_id[on_place].tolist(),
                    test_cases.day[on_place].tolist(),
                    strict=True,
                )
            )
            forecasters = {
                "structural": structural.draws,
                "no_features": plain.draws,
                "random_walk": structural.random_walk,
                "drift_v0": drift_forecaster(drift),
            }
            table, pit = nc.score_cases(
                forecasters, test_cases, n=n_draws, seed=SEED, keep_pit=True
            )
            table["case"] += offset
            table["fold"] = label
            pit["case"] += offset
            if len(test_cases):  # a fold of unscorable cases only has no PIT to add
                _pit_summary(pit, test_cases, offset, pit_acc)
            scores.append(table)
            for cell in structural.cells.values():
                cells.append({"fold": label, **cell.__dict__, "beta": list(cell.beta)})
            for i in range(len(test_cases)):
                view, pool = test_cases.view(i), str(test_cases.pool[i])
                phi_rows.append(
                    (
                        pool,
                        test_cases.tau[i],
                        structural.phi(view, pool=pool),
                        test_cases.total_t[i] / test_cases.total_f[i],
                    )
                )
            fold_rows.append(
                {
                    "fold": label,
                    "train_days": len(train),
                    "test_days": len(test),
                    "scored_days": len(scored_days),
                    "cases": len(test_cases),
                    "unscored_no_fit": unscored,
                    "train_end": fold.train_end.isoformat(),
                }
            )
            offset += len(test_cases)
            wide_scored += int((test_cases.pool == nc.WIDE).sum())
            log(f"  fold {label}: {len(test_cases)} cases scored ({_progress(started)})")
    except Exception as exc:
        for a, b in COMPARISONS:
            record = _record(
                a,
                b,
                (),
                git_sha,
                builds,
                spec_hash,
                test_days_per_fold,
                n_draws,
                family=family,
                sources=sources,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            registry.append(record)
        error = f"the folds failed before the scored boards were known: {type(exc).__name__}"
        registry.append(
            _place_record(
                (),
                git_sha,
                builds,
                spec_hash,
                family=family,
                sources=sources,
                status="failed",
                error=error,
            )
        )
        write_summary(registry, summary_dir(COUNTRY))
        raise

    # From here `builds` names the final votes too: they choose the Place cases in the gains.
    place_measured, place_appended, builds = _measure_place_rule(
        closing,
        scored_place,
        dev_days,
        registry,
        family,
        git_sha,
        builds,
        spec_hash,
        sources=sources,
        log=log,
    )
    del closing
    log(f"  Place rule measured on the scored boards ({_progress(started)})")
    readers = place_result_readers(place_measured)
    # A run with no fold (`--end` before D9's first test window) has no score table: an empty one
    # with the scored columns lets every comparison and the report render with no scored case.
    empty = pd.DataFrame(columns=SCORE_COLUMNS)
    scores = pd.concat([t for t in scores if len(t)] or scores or [empty], ignore_index=True)
    in_result, diagnostic = _split_place(scores, readers)
    pit_result, pit_diagnostic = _split_pit(pit_acc, readers)
    gains, gains_undetermined, gains_unvalidated, records = {}, {}, {}, []
    for a, b in COMPARISONS:
        pair = in_result[in_result["forecaster"].isin([a, b])]
        name = f"{a}_vs_{b}"
        for reader, rows in diagnostic.items():
            tables = _diagnostic_gains(rows[rows["forecaster"].isin([a, b])], a, b, bootstrap)
            if reader == UNDETERMINED:
                gains_undetermined[name] = tables
            else:
                gains_unvalidated.setdefault(name, {})[reader] = tables
        by_day = nc.gain_table(
            pair,
            a,
            b,
            n_boot=bootstrap["n_boot"],
            level=bootstrap["level"],
            seed=SEED,
            cluster="day",
        )
        by_meeting = nc.gain_table(
            pair,
            a,
            b,
            n_boot=bootstrap["n_boot"],
            level=bootstrap["level"],
            seed=SEED,
            cluster="meeting",
        )
        per_fold = []
        primary = pair[pair["tau_min"] == grid.primary]
        for label, g in primary.groupby("fold", sort=True):
            per_fold.append(
                (
                    label,
                    nc.gain_table(
                        g, a, b, n_boot=200, level=bootstrap["level"], seed=SEED, cluster="day"
                    ),
                )
            )
        gains[f"{a}_vs_{b}"] = {"day": by_day, "meeting": by_meeting, "fold": per_fold}
        records.append(
            _record(
                a,
                b,
                _metrics(by_day, by_meeting, per_fold, grid.primary),
                git_sha,
                builds,
                spec_hash,
                test_days_per_fold,
                n_draws,
                family=family,
                sources=sources,
                place_readers=sorted(readers),
            )
        )

    phi = pd.DataFrame(phi_rows, columns=["pool", "tau", "phi_hat", "phi_obs"])
    report_data = {
        "counts": dict(counts)
        | {
            "development_days": len(dev_days),
            "jravan_day_files": len(files),
            "own_logged_day_files": len(logged),
            "holdout_days_dropped": len(held),
            "cases": len(cases),
            "folds": len(folds),
        },
        "folds": fold_rows,
        "board_age_min": board_age(cases),
        "place_rule_sum": place_rule_check(cases),
        "place_measurement": place_measured,
        "phi": _phi_diagnostics(phi),
        "cells": cells,
        "pit": _pit_rows(pit_result),
        "pit_diagnostic": _pit_rows(pit_diagnostic),
    }
    coverage = wide_coverage(logged)  # the logged days the folds read
    report_data["wide"] = (
        wide_status(coverage, cases, folds, cells, no_final_wide, final_wide)
        | {"wide_scored_cases": wide_scored}
        | coverage
    )
    about = {
        "family": f"`{family}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "data": ", ".join(builds.values()),
        "sampling": "every test day"
        if test_days_per_fold is None
        else f"{test_days_per_fold} seeded test days per fold (seed {SEED}); training unsampled",
        "draws per case": n_draws,
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trial (records marked params.trial, spec not frozen)": trial,
    }
    # A logged run's reports carry their family's name, so they never take a JRA-VAN-only
    # report's name; a run with no fold shares the full study's family and spec, so its reports
    # are suffixed -no-fold and never replace the full study's.
    version = "-logged-wide-v1" if logged else ""
    stem = (
        f"nowcast-{COUNTRY}{version}-{spec_hash[:12]}"
        + ("" if folds else "-no-fold")
        + ("-trial" if trial else "")
    )
    diagnostics = {"undetermined": gains_undetermined, "unvalidated": gains_unvalidated}
    report = write_report(
        out_dir / f"{stem}.md",
        render_report(gains, report_data, about, grid, diagnostics=diagnostics),
        sources=sources,
    )

    def records_of(tables: dict) -> dict:
        return {
            u: (
                t.to_dict("records") if u != "fold" else [(f, ft.to_dict("records")) for f, ft in t]
            )
            for u, t in tables.items()
        }

    dump = {k: v for k, v in report_data.items()} | {
        "place_result_readers": sorted(readers),
        "gains": {k: records_of(v) for k, v in gains.items()},
        "gains_undetermined": {k: records_of(v) for k, v in gains_undetermined.items()},
        "gains_unvalidated": {
            k: {reader: records_of(t) for reader, t in v.items()}
            for k, v in gains_unvalidated.items()
        },
        "about": about,
    }
    write_report(out_dir / f"{stem}.json", json.dumps(dump, indent=1, default=str), sources=sources)
    appended = place_appended
    existing = registry.records(family)
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
        "registry_path": str(registry.path),
    }


def _split_place(
    scores: pd.DataFrame, readers: set[str]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The score rows of the result gains, and the Place rows kept out of them by closing-board
    read (#410 re-check): `undetermined` always; when some read stands as a result
    (`readers`), every other read too, so the result tables hold only validated reads. With no
    read standing, the determined reads stay in the result tables, whose Place rows the report
    then calls a diagnostic as a whole."""
    on_place = (scores["pool"] == nc.PLACE).to_numpy()
    reader = scores["reader"].fillna("none").astype(str).to_numpy()
    out = on_place & np.isin(reader, sorted(_diagnostic_reads(set(reader[on_place]), readers)))
    diagnostic = {str(r): scores[out & (reader == r)] for r in sorted(set(reader[out].tolist()))}
    return scores[~out], diagnostic


def _diagnostic_reads(seen: set[str], readers: set[str]) -> set[str]:
    """The Place reads among `seen` kept out of the result (`_split_place`, `_split_pit`):
    `undetermined` always; when some read stands (`readers`), every other read too."""
    return {UNDETERMINED} | ({r for r in seen if r not in readers} if readers else set())


def _split_pit(acc: dict, readers: set[str]) -> tuple[dict, dict]:
    """The PIT counts by (forecaster, pool, tau, read) split as `_split_place` splits the gains:
    the result's, summed over the reads kept in it, by (forecaster, pool, tau); and the Place
    reads kept out, by (forecaster, pool, tau, read)."""
    seen = {key[3] for key in acc if key[1] == nc.PLACE}
    out = _diagnostic_reads(seen, readers)
    result, diagnostic = {}, {}
    for (name, pool, tau, reader), slot in acc.items():
        if pool == nc.PLACE and reader in out:
            diagnostic[(name, pool, tau, reader)] = slot.copy()
        else:
            result.setdefault((name, pool, tau), np.zeros(PIT_BINS + 3))
            result[(name, pool, tau)] += slot
    return result, diagnostic


def _diagnostic_gains(pair: pd.DataFrame, a: str, b: str, bootstrap: dict) -> dict:
    """The day- and meeting-clustered gain tables of diagnostic Place rows."""
    return {
        unit: nc.gain_table(
            pair,
            a,
            b,
            n_boot=bootstrap["n_boot"],
            level=bootstrap["level"],
            seed=SEED,
            cluster=unit,
        )
        for unit in ("day", "meeting")
    }


def _phi_diagnostics(phi: pd.DataFrame) -> list[dict]:
    rows = []
    for (pool, tau), g in phi.groupby(["pool", "tau"], sort=False):
        ok = np.isfinite(g["phi_obs"]) & (g["phi_obs"] > 0)
        err = np.log(g["phi_obs"][ok]) - np.log(g["phi_hat"][ok])
        rows.append(
            {
                "pool": pool,
                "tau_min": tau,
                "n": int(ok.sum()),
                "mean_phi_obs": float(g["phi_obs"][ok].mean()),
                "mean_phi_hat": float(g["phi_hat"].mean()),
                "mean_ln_obs_over_hat": float(err.mean()),
                "sd_ln_obs_over_hat": float(err.std()),
                "mean_abs_phi_error": float(np.abs(g["phi_obs"][ok] - g["phi_hat"][ok]).mean()),
            }
        )
    return sorted(rows, key=lambda r: (nc.POOLS.index(r["pool"]), -r["tau_min"]))


def _pit_rows(acc: dict) -> list[dict]:
    """PIT rows of counts keyed (forecaster, pool, tau) or (forecaster, pool, tau, read); the
    latter's rows lead with `reader`."""
    rows = []
    for key, slot in sorted(acc.items(), key=lambda kv: (kv[0][3:], kv[0][0], kv[0][1], -kv[0][2])):
        name, pool, tau = key[:3]
        n = slot[PIT_BINS + 2]
        rows.append(
            ({"reader": key[3]} if len(key) == 4 else {})
            | {
                "forecaster": name,
                "pool": pool,
                "tau_min": tau,
                "n": int(n),
                "coverage_50": slot[PIT_BINS] / n,
                "coverage_90": slot[PIT_BINS + 1] / n,
                "deciles": [round(float(c / n), 4) for c in slot[:PIT_BINS]],
            }
        )
    return rows


def _metrics(by_day, by_meeting, per_fold, primary) -> tuple[FoldMetric, ...]:
    out = []
    for unit, table in (("day", by_day), ("meeting", by_meeting)):
        for row in table.to_dict("records"):
            for gain in ("log_score_gain", "crps_gain", "kl_gain"):
                values = (row[gain], row[f"{gain}_lo"], row[f"{gain}_hi"])
                if all(np.isfinite(values)):
                    out.append(
                        FoldMetric(
                            f"pooled:{unit}",
                            f"{gain}:{row['pool']}:T-{row['tau_min']:g}",
                            float(values[0]),
                            float(values[1]),
                            float(values[2]),
                            int(row["n_cases"]),
                        )
                    )
    for label, table in per_fold:
        for row in table.to_dict("records"):
            for gain in ("log_score_gain", "crps_gain"):
                if np.isfinite(row[gain]):
                    out.append(
                        FoldMetric(
                            label,
                            f"{gain}:{row['pool']}:T-{primary}",
                            float(row[gain]),
                            None,
                            None,
                            int(row["n_cases"]),
                        )
                    )
    return tuple(out)


def _record(
    candidate,
    baseline,
    metrics,
    git_sha,
    builds,
    spec_hash,
    days_per_fold,
    n_draws,
    *,
    family: str = FAMILY_ID,
    sources: tuple[str, ...] = (SOURCE,),
    status: str = "scored",
    error: str | None = None,
    place_readers: list[str] | None = None,
) -> RunRecord:
    """One comparison's record in `family` (a `-trial` family marks the record a trial)."""
    trial = family.endswith("-trial")
    params = {
        "candidate": candidate,
        "baseline": baseline,
        "spec_sha256": spec_hash,
        "n_draws": n_draws,
        "seed": SEED,
        "test_days_per_fold": days_per_fold,
    }
    if place_readers is not None:  # the Place reads whose cases are in the gains as a result
        params["place_result_readers"] = place_readers
    if trial:
        params["trial"] = True
    if error is not None:
        params["error"] = error
    return RunRecord(
        hypothesis_id=HYPOTHESIS,
        family_id=family,
        variant=f"{candidate}_vs_{baseline}",
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=sources,
        fold_scheme="walk_forward:"
        + json.dumps(dataclasses.asdict(load_country_scheme(COUNTRY)), sort_keys=True),
        components={
            "model": "countries/japan/models/nowcast.py",
            "drift": "countries/japan/models/v0/nowcast.py",
        },
        params=params,
        metrics=tuple(metrics),
        status=status,
    )


def _measure_place_rule(
    closing,
    scored_place,
    dev_days,
    registry,
    family,
    git_sha,
    builds,
    spec_hash,
    *,
    sources: tuple[str, ...] = (SOURCE,),
    log,
) -> tuple[dict, int, dict[str, str]]:
    """#410's measurement on the closing boards of the scored Place cases (`scored_place`: the
    (race_id, day) of each, the folds' test cases after test-day sampling), against the final
    votes of the development days (`dev_days`, whose build `place_rule_builds` takes), and its
    registry record in the run's `family` (with the run's `sources`): scored when a read stands
    as a result (`place_result_readers`), failed (unvalidated) when none does or it could not
    run. Returns the measurement, the records
    appended (0 or 1) and the builds it read: `builds` plus the final votes' build when that could
    be taken, which the comparison records and the report then carry."""
    log("measuring the Place rule on the scored closing boards (#410)")
    scored = {race_id for race_id, _ in scored_place}
    boards = [b for b in closing if b[0] in scored]
    try:
        votes = final_place_votes(sorted({str(day)[:10] for day in dev_days}))
        place_builds = builds | place_rule_builds(votes)
        measured = place_rule_measurement(boards, votes)
        del votes
        if place_validated(measured):
            record = _place_record(
                _place_metrics(measured),
                git_sha,
                place_builds,
                spec_hash,
                family=family,
                sources=sources,
                readers=sorted(place_result_readers(measured)),
            )
        else:
            measured["error"] = (
                "unvalidated: no determined read of the scored Place boards has final votes on "
                f"a board and shares within the tolerance on at least {PLACE_MIN_WITHIN_TOLERANCE} "
                "of its validated boards"
            )
            record = _place_record(
                _place_metrics(measured),
                git_sha,
                place_builds,
                spec_hash,
                family=family,
                sources=sources,
                status="failed",
                error=measured["error"],
            )
    except Exception as exc:  # recorded as a failed run; the gains are still reported
        measured = {"error": f"{type(exc).__name__}: {exc}"}
        record = _place_record(
            (),
            git_sha,
            builds,
            spec_hash,
            family=family,
            sources=sources,
            status="failed",
            error=measured["error"],
        )
        place_builds = builds
    if record in registry.records(family):
        return measured, 0, place_builds
    registry.append(record)
    return measured, 1, place_builds


def _place_record(
    metrics,
    git_sha,
    builds,
    spec_hash,
    *,
    family: str = FAMILY_ID,
    sources: tuple[str, ...] = (SOURCE,),
    status: str = "scored",
    error=None,
    readers: list[str] | None = None,
) -> RunRecord:
    """The Place rule's record (#410) in `family` (a `-trial` family marks the record a trial)."""
    trial = family.endswith("-trial")
    params = {
        "spec_sha256": spec_hash,
        "rel_tol": PLACE_REL_TOL,
        "abs_tol": PLACE_ABS_TOL,
        "min_within_tolerance": PLACE_MIN_WITHIN_TOLERANCE,
        "result_readers": readers or [],
    }
    if trial:
        params["trial"] = True
    if status != "scored":
        params["unvalidated"] = True
    if error is not None:
        params["error"] = error
    return RunRecord(
        hypothesis_id=HYPOTHESIS,
        family_id=family,
        variant=PLACE_VARIANT,
        ticket=PLACE_TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=sources,
        fold_scheme="none: the closing boards of the scored Place cases",
        components={
            "rule": "core/pricing.py stake_back_share_dividend",
            "boards": "countries/japan/models/nowcast.py closing_boards",
            "reader": "countries/japan/models/nowcast.py place_shares",
        },
        params=params,
        metrics=tuple(metrics),
        status=status,
    )


# ---------------------------------------------------------------- the report


def _place_status(measured: dict) -> str:
    """How the Place rows are to be read, from #410's measurement."""
    readers = place_result_readers(measured)
    if not readers:
        return (
            "**Place rows: a diagnostic.** The Place rule's measurement against the final votes "
            "on the scored boards stands no read as a result "
            f"({measured.get('error', 'empty')}), so the Place shares are not shown to be money "
            "shares and the Place gains are not evidence; the `undetermined` boards' Place rows "
            "are in the diagnostic tables; the Win rows are unaffected."
        )
    alls = [row for row in measured.get("summary", []) if row["boards"] == "all"]
    others = sorted(
        row["reader"] for row in alls if row["reader"] not in readers and row["n_scored"] > 0
    )
    tail = (
        f" The {', '.join(f'`{r}`' for r in others)} read's Place rows are a diagnostic (the "
        f"`{UNDETERMINED}` read always: figures that hold some runners' money only as a sum; a "
        "determined read when it validated no board or its shares were within the tolerance on "
        f"fewer than {PLACE_MIN_WITHIN_TOLERANCE} of its validated boards): they are left out "
        "of the gains below and reported in the diagnostic tables."
        if others
        else ""
    )
    return (
        "**Place rows: a result for the "
        f"{' and '.join(f'`{r}`' for r in sorted(readers))} read.** Its shares were checked "
        "against the final votes on the scored closing boards and were within the tolerance on "
        f"at least {PLACE_MIN_WITHIN_TOLERANCE} of its validated boards (coverage, reproduction "
        f"and tolerance per read, in the section on the rule below); the Place rows of the gains "
        f"hold only its boards.{tail}"
    )


def render_report(
    gains: dict, data: dict, about: dict, grid, *, diagnostics: dict | None = None
) -> str:
    diagnostics = diagnostics or {}
    wide = data["wide"]
    lines = [
        f"# The C1 final-price nowcast against the random walk and #116's drift "
        f"({COUNTRY}, #{TICKET})",
        "",
        "Built on JRA-VAN data: every figure here is withheld from the repository (D21).",
        "",
        *(f"- {key}: {value}" for key, value in about.items()),
        "",
        "Gains are per case (race, pool, tau): the mean over combinations of the score on the "
        "final ln-share, candidate minus baseline for the log score (higher better), baseline "
        "minus candidate for CRPS and KL (lower better); positive means the candidate is better. "
        "`_lo` / `_hi` are the one-sided 95% bounds (the 5% and 95% quantiles of the bootstrap "
        "mean).",
        "",
        _place_status(data["place_measurement"]),
        "",
        "## Counts",
        "",
        _table([data["counts"]]),
        "",
        _table(data["folds"]),
        "",
    ]
    for name, by_unit in gains.items():
        lines += [f"## {name}", "", "### Race-day clusters (the ticket)", ""]
        lines += [_table(by_unit["day"].to_dict("records")), "", "### Meeting clusters (D7)", ""]
        cols = ["pool", "tau_min", "n_cases"] + [
            c for c in by_unit["meeting"].columns if "gain" in c
        ]
        lines += [_table(by_unit["meeting"][cols].to_dict("records")), ""]
        lines += [
            f"### By fold at the primary horizon T-{grid.primary} (day clusters, 200 replicates)",
            "",
        ]
        rows = []
        for label, table in by_unit["fold"]:
            for row in table.to_dict("records"):
                rows.append(
                    {
                        "fold": label,
                        "pool": row["pool"],
                        "n_cases": row["n_cases"],
                        **{
                            c: row[c]
                            for c in (
                                "log_score_gain",
                                "log_score_gain_lo",
                                "crps_gain",
                                "crps_gain_lo",
                            )
                        },
                    }
                )
        lines += [_table(rows), ""]
    lines += _diagnostic_sections(diagnostics)
    lines += [
        "## PIT deciles and 50% / 90% coverage (runner level)",
        "",
        "PIT of each combination's final ln-share under its draws, pooled over the scored folds. "
        "Combinations of one race are dependent, so no test statistic is given. The Place rows "
        "are split as the gains are: they hold the Place cases of the reads that stand as a "
        "result (with none standing, of every determined read, then a diagnostic as the Place "
        "status says); the others are in the PIT diagnostic below.",
        "",
        _table(data["pit"]),
        "",
        *_pit_diagnostic_section(data.get("pit_diagnostic", [])),
        "## sigma_t and beta by pool and tau (structural fit, every fold)",
        "",
        _table(_cell_summary(data["cells"])),
        "",
        "## phi_t: the pool-growth profile against the observed V_t / V_final (scored cases)",
        "",
        _table(data["phi"]),
        "",
        "## Board age at t (minutes from the board's oldest row to t)",
        "",
        _table(data["board_age_min"]),
        "",
        "## Place shares: the rule against the final votes (#410)",
        "",
        "The closing PLACE board of every scored Place case, read as the cases read it, "
        "against its final PLACE votes (H1, data_kubun "
        f"{'/'.join(PLACE_FINAL_KUBUN)}). `reader`: the read used in scoring (`joint`: both "
        "figures; `low_only`: the low figures, on a board without a high figure on every "
        "runner or whose joint fit left a runner without money; `undetermined`: the joint fit "
        "where the figures above the floor do not determine the money, read at its "
        "minimum-norm shares, a diagnostic). `n_scored`: boards scored "
        "with it; `n`: those validated (final votes on every priced runner); `coverage`: "
        "their share. `share_reproduced`: validated boards on which "
        "`core.pricing.stake_back_share_dividend` gives every printed figure from the votes (P "
        "as scored; `_with_slot_places` with P from the registered slots, which differ on "
        "`share_slot_places_differ`). `share_within_tolerance`: validated boards whose shares as "
        f"scored are within {PLACE_REL_TOL} relative and {PLACE_ABS_TOL} absolute of the vote "
        "shares on every runner; `low_only_share_within_tolerance` the same for the low figures' "
        "read on those boards. Errors are each board's worst runner; `rule_sum` the read money's "
        "total.",
        "",
        _place_status(data["place_measurement"]),
        "",
        _table(data["place_measurement"].get("summary", [])),
        "",
        _table([data["place_measurement"].get("counts", {})]),
        "",
        "## Place shares: the rule's money total at t (1 when the rule holds)",
        "",
        "The series carries no per-runner Place money, so Place shares are read from the figures "
        "at t by the rule measured above.",
        "",
        _table(data["place_rule_sum"]),
        "",
        "## Wide coverage",
        "",
        "The JRA-VAN series carries no Wide pool: Wide comes from our own logged days only (D3).",
        "",
        (
            f"Wide was fitted and {wide['wide_scored_cases']} Wide case(s) scored (its rows are "
            "in the tables above)."
            if wide["wide_scored_cases"]
            else "Wide was fitted (a fold's training days give a Wide cell), but no Wide test "
            "case was scored: no fold's scored test days hold a Wide case its fits cover."
        )
        if wide["fitted"]
        else wide["reason"],
        "",
        f"Wide cases: {wide['wide_cases']} on {len(wide['wide_case_days'])} day(s), of which "
        f"{len(wide['wide_training_days'])} are a fold's training day.",
        "",
        "Our own logged development days:",
        "",
        _table(wide["own_logged_days"]),
        "",
    ]
    return "\n".join(lines)


def _diagnostic_sections(diagnostics: dict) -> list[str]:
    """The Place gains kept out of the result tables (#410 re-check), by read."""
    by_read: dict[str, dict] = defaultdict(dict)
    for name, tables in diagnostics.get("undetermined", {}).items():
        by_read[UNDETERMINED][name] = tables
    for name, per_reader in diagnostics.get("unvalidated", {}).items():
        for reader, tables in per_reader.items():
            by_read[reader][name] = tables
    if not by_read:
        return []
    lines = [
        "## Place diagnostics: the gains of the boards whose read is not a result",
        "",
        "The Place cases whose closing board is read by a read that does not stand as a result "
        "(see the Place status above), kept out of the gains above. Not evidence.",
        "",
    ]
    for reader in sorted(by_read):
        for name, tables in by_read[reader].items():
            lines += [f"### `{reader}`: {name}", "", "Race-day clusters:", ""]
            lines += [_table(tables["day"].to_dict("records")), "", "Meeting clusters:", ""]
            lines += [_table(tables["meeting"].to_dict("records")), ""]
    return lines


def _pit_diagnostic_section(rows: list[dict]) -> list[str]:
    """The PIT of the Place cases kept out of the result PIT (#410 review), by read."""
    if not rows:
        return []
    return [
        "## PIT diagnostic: the Place boards whose read is not a result",
        "",
        "The PIT and coverage of the Place cases whose closing board's read does not stand as a "
        "result (`undetermined` always), by read, kept out of the PIT above. Not evidence.",
        "",
        _table(rows),
        "",
    ]


def _cell_summary(cells: list[dict]) -> list[dict]:
    frame = pd.DataFrame(cells)
    if frame.empty:
        return []
    rows = []
    for (pool, tau), g in frame.groupby(["pool", "tau_min"], sort=False):
        beta = np.array(g["beta"].tolist())
        last = g.iloc[-1]
        rows.append(
            {
                "pool": pool,
                "tau_min": tau,
                "folds": len(g),
                "sigma_last": last["sigma"],
                "sigma_min": g["sigma"].min(),
                "sigma_max": g["sigma"].max(),
                "sigma_rw_last": last["sigma_rw"],
                **{f"beta_{f}_last": float(b) for f, b in zip(nc.FEATURES, beta[-1], strict=True)},
                "share_floored_last": last["share_floored"],
                "cases_last": last["n_cases"],
            }
        )
    return sorted(rows, key=lambda r: (nc.POOLS.index(r["pool"]), -r["tau_min"]))


def nowcast_study_command(
    end: Annotated[
        datetime | None,
        typer.Option("--end", formats=["%Y-%m-%d"], help="Last race day read (default: all)."),
    ] = None,
    test_days_per_fold: Annotated[
        int | None,
        typer.Option("--test-days-per-fold", help="Score a seeded sample of N test days per fold."),
    ] = None,
    n_draws: Annotated[int, typer.Option("--n-draws", help="Draws per case.")] = N_DRAWS,
    trial: Annotated[
        bool,
        typer.Option(
            "--trial",
            help=f"Records go to {TRIAL_FAMILY_ID} marked trial; the spec is not frozen; the "
            "report is suffixed -trial.",
        ),
    ] = False,
) -> None:
    """Fit the C1 nowcast on D9's folds of the JRA-VAN paths and write the private report."""
    result = run_study(
        end=None if end is None else end.date(),
        test_days_per_fold=test_days_per_fold,
        n_draws=n_draws,
        trial=trial,
        log=typer.echo,
    )
    typer.echo(json.dumps(result, indent=2))
