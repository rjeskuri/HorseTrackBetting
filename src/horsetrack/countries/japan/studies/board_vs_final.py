"""Board-fitted against final-fitted terms on JRDB's fixed-time board (#103, epic #73; 08 §4 E2,
E45, R13).

E2: the Win price's own power and quadratic terms (#87's stage-2 market terms, beta on ln q and
gamma on (ln q)^2) fitted on the board are partly a price-forecast correction, |gamma_final| <
|gamma_board|. E45: #99's ordering coefficients (`models.exotic`, quadratic_context) depend on the
horizon of the Win probabilities fed to them. Each fits on one price and scores on the other
(`cross_apply`), on the races priced both ways only (so the same seasons on both sides), with
D7's meeting-clustered one-sided bounds.

The board is JRDB's TYB board: `tyb_win_odds` taken at `tyb_odds_time` (HHMM on the JST clock of
the race date), read through `core.store.asof.read_asof` under a column rule on its release time,
the odds time plus the configured lag (config/board_vs_final.yaml), at each race's off. It is
never filled from another price. Every output is JRDB-derived (D21): the report and the private
leaderboard go through `write_report` outside the repository, the registry records carry the
source `jrdb`, and the committed registry summary withholds their values.

E45's Place and Wide outcomes, and whether a race is scored in them, come from the race's paid
combinations in the dividends table (`payout_rules`; family v2): the store does not hold the field
when sales opened, which sets the places paid. v1 took them from the field of starters and stays
in the registry as recorded, superseded.

Order: freeze the family's spec (before any outcome is read), read the development store, fit
and score, append one registry record per fitted model (failures included), write the outputs and
the family's summary file. A run in which a table could not be produced writes its report to an
attempt-named file, leaves the last complete outputs as they are and exits non-zero. CLI:
`python -m horsetrack.countries.japan.studies.board_vs_final (--freeze | --run) [--out DIR]`,
DIR under $HORSETRACK_DATA_DIR (default its reports directory).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from horsetrack.core import env, paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, split_off
from horsetrack.core.eval.leaderboard import LeaderboardRow, to_json, write_leaderboard
from horsetrack.core.eval.registry import (
    FoldMetric,
    Registry,
    RunRecord,
    _canonical,
    _record_data,
    _sha256,
    default_registry,
    summary_dir,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_inside, write_report
from horsetrack.core.eval.scoring import gain, per_race_log_loss
from horsetrack.core.pricing import clean_odds
from horsetrack.core.store.asof import Availability, load_availability, read_asof
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.models.exotic import PLACE, TRIO, WIDE, WIN, Ordering, pool_array
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table
from horsetrack.countries.japan.research.pool_depth import update_summary
from horsetrack.countries.japan.studies.calibration import _markdown_table

COUNTRY = "japan"
TICKET = 103
CONFIG = load_country_yaml(COUNTRY, "board_vs_final")
FAMILY_ID = CONFIG["family_id"]
SOURCES = ("netkeiba", "jrdb")
FINAL, BOARD = "q_final", "q_board"
STATS = ("mean", "lo", "hi")
E45_POOLS = (PLACE, WIDE, TRIO)
CLUSTER_COLUMNS = {"meeting": "meeting_id"}  # D7's unit, venue x day
MODELS = {
    "E2": "conditional logit u = beta ln q + gamma (ln q)^2 on the fitted price (#87's stage-2 "
    "market terms), unbounded",
    "E45": "quadratic_context ordering (#99, models.exotic.Ordering)",
}
BASELINES = {"E2": "the scored price itself (q normalised)", "E45": "Harville on the scored price"}
_VIEW = "__board_vs_final_tyb"  # the release-time view jrdb_board reads through read_asof
_KEYS = "__board_vs_final_races"
JRDB_COLUMNS = ["race_id", "race_date", "horse_no", "tyb_win_odds", "tyb_odds_time"]
FETCHED = "tyb_fetched_utc"  # jrdb_pre_race's measured fetch time of the TYB file (#603)
BOARD_COLUMNS = ["race_id", "horse_no", "tyb_win_odds", "release_utc", "fetched_utc"]
RUNNER_COLUMNS = ["race_id", "race_date", "horse_no", "win_odds", "finish_pos", "dead_heat"]
RUNNER_COLUMNS += ["is_scratched"]
# the races view needs its rule's label and gated columns (availability.yaml) to derive post_utc
RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "status", "post_time_local", "going"]
RACE_COLUMNS += ["going_raw", "finish_time_s", "split_times_s", "result_published_utc"]
RACE_COLUMNS += ["result_unpublished_utc"]
DIVIDEND_COLUMNS = ["race_id", "pool", "combination", "is_refund"]
PAID_POOLS = (WIN, PLACE, WIDE)  # a paid Win marks a race whose paid record is held
SCORED, NO_RECORD, NOT_PAID = "scored", "no paid record", "not paid"
MISMATCH = "paid combinations do not match the result"


class OutsideDataDir(ValueError):
    """An output directory that does not resolve under $HORSETRACK_DATA_DIR."""


# --------------------------------------------------------------------------- the board


def _lag(rules: Mapping) -> timedelta:
    minutes = rules.get("release_lag_min") if isinstance(rules, Mapping) else None
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not minutes >= 0:
        raise ValueError(f"rules need release_lag_min, minutes >= 0, got {minutes!r}")
    if not math.isfinite(minutes):
        raise ValueError(f"release_lag_min must be finite, got {minutes!r}")
    return timedelta(minutes=minutes)


def _release(rules: Mapping) -> str:
    release = rules.get("release", "lag") if isinstance(rules, Mapping) else "lag"
    if not isinstance(release, str) or release not in ("lag", "fetched"):
        raise ValueError(f'rules["release"] must be "lag" or "fetched", got {release!r}')
    return release


def jrdb_board(
    con,
    race_ids: Sequence[str],
    *,
    rules: Mapping,
    decision_utc: pd.Series | None = None,
) -> pd.DataFrame:
    """JRDB's fixed-time board of `race_ids` from the store table `jrdb_pre_race` in `con`:
    race_id, horse_no, tyb_win_odds, release_utc and fetched_utc, at most one row per runner.

    `rules["release"]` picks the release time. "lag" (the default): a row is released at race_date,
    tyb_odds_time ("HHMM" on the wall clock of `rules["tz"]`, default Asia/Tokyo) +
    `rules["release_lag_min"]` (required); a missing or malformed odds time is never released.
    "fetched" (#603): a row is released when its TYB file was fetched (`jrdb_pre_race.
    tyb_fetched_utc`, measured, not assumed); `release_lag_min` is not needed and a row with no
    recorded fetch time is never released. Any other value raises ValueError. `fetched_utc` is the
    row's fetch time under either rule (null where unrecorded, or when the table has no such
    column). The rows go through `read_asof` under a column rule on the release time, at
    `decision_utc` (a Series of timezone-aware decision times indexed by race_id: a runner's board
    counts only if released by its own race's time; a race without one has no board) or, without
    it, now. Nothing is filled from another price: a runner without TYB odds reads null."""
    fetched_rule = _release(rules) == "fetched"
    lag = timedelta(0) if fetched_rule else _lag(rules)
    tz = rules.get("tz", "Asia/Tokyo")
    ZoneInfo(tz)  # an unknown zone fails here
    has_fetched = FETCHED in {r[0] for r in con.execute("DESCRIBE jrdb_pre_race").fetchall()}
    if fetched_rule and not has_fetched:
        raise ValueError(f"release 'fetched' needs jrdb_pre_race.{FETCHED}")
    fetched = f"j.{FETCHED}" if has_fetched else "CAST(NULL AS TIMESTAMPTZ)"
    lagged = (
        "CASE WHEN regexp_full_match(j.tyb_odds_time, '[0-9]{4}') THEN "
        "timezone(?, try_strptime(strftime(j.race_date, '%Y-%m-%d') || ' ' || j.tyb_odds_time, "
        "'%Y-%m-%d %H%M')) + to_microseconds(?) END"
    )
    release_sql, params = (
        (fetched, []) if fetched_rule else (lagged, [tz, lag // timedelta(microseconds=1)])
    )
    ids = pd.DataFrame(
        {"race_id": pd.Series(list(dict.fromkeys(map(str, race_ids))), dtype=object)}
    )
    decision, at = None, datetime.now(timezone.utc)
    if decision_utc is not None:
        decision = pd.to_datetime(pd.Series(decision_utc), utc=True)
        decision.index = decision.index.astype(object)
        decision = decision.loc[decision.index.isin(ids["race_id"]) & decision.notna()]
        at = decision.max().to_pydatetime() if len(decision) else None
    con.register(_KEYS, ids)
    try:
        con.execute(
            f'CREATE OR REPLACE TEMP TABLE "{_VIEW}" AS SELECT j.race_id, j.horse_no, '
            f"j.tyb_win_odds, {release_sql} AS release_utc, {fetched} AS fetched_utc "
            f'FROM jrdb_pre_race AS j WHERE j.race_id IN (SELECT race_id FROM "{_KEYS}")',
            params,
        )
        rule = Availability(_VIEW, "column", column="release_utc")
        board = pd.DataFrame(columns=BOARD_COLUMNS)
        if at is not None:
            board = read_asof(con, _VIEW, at, rules={_VIEW: rule})
    finally:
        con.execute(f'DROP TABLE IF EXISTS "{_VIEW}"')
        con.unregister(_KEYS)
    board = board[BOARD_COLUMNS].astype({"race_id": object})
    if decision is not None:
        off = pd.to_datetime(board["race_id"].map(decision), utc=True)
        board = board.loc[(board["release_utc"] <= off).to_numpy(dtype=bool, na_value=False)]
    board = board.sort_values("release_utc", kind="stable")  # the latest release by t counts
    return board.drop_duplicates(["race_id", "horse_no"], keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------- cross-application


def bootstrap_settings() -> dict:
    """D7's bootstrap: scoring.yaml's level and replicates, this study's cluster unit and seed."""
    boot = load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["sided"] != "one-sided" or boot["cluster_unit"] != CONFIG["cluster_unit"]:
        raise ValueError(f"board_vs_final.yaml's cluster unit against scoring.yaml's: {boot}")
    return {
        "cluster_unit": CONFIG["cluster_unit"],
        "sided": "one-sided",
        "level": float(boot["level"]),
        "n_boot": int(boot["n_boot"]),
        "seed": int(CONFIG["bootstrap_seed"]),
    }


def matched(frame: pd.DataFrame, prices: Sequence[str]) -> np.ndarray:
    """Each row's race has every price present on every one of its rows."""
    priced = pd.Series(frame[list(prices)].notna().all(axis=1).to_numpy(), index=frame["race_id"])
    return priced.groupby(level=0).transform("all").to_numpy(dtype=bool)


def cross_apply(
    frame: pd.DataFrame,
    *,
    fit_price: str,
    score_price: str,
    fit: Callable[[pd.DataFrame, str], object],
    score: Callable[[object, pd.DataFrame, str], pd.Series],
    cluster: str | None = None,
    bootstrap: Mapping | None = None,
) -> pd.DataFrame:
    """Fit on each price and score each fit on each price, on the races of `frame` (one row per
    runner) priced both ways on every row: `fit(rows, price)` once per price, `score(model, rows,
    price)` (a per-race Series on race_id) once per fit and price. Index: the price fitted on;
    columns: (the price scored on, mean | lo | hi), the mean over races and D7's one-sided
    bounds (a paired bootstrap over `cluster`, default the config's unit). `attrs["cells"]` holds
    each cell's races and clusters."""
    prices = (fit_price, score_price)
    if fit_price == score_price:
        raise ValueError(f"fit_price and score_price must differ, got {fit_price!r} twice")
    cluster = CLUSTER_COLUMNS[CONFIG["cluster_unit"]] if cluster is None else cluster
    boot = bootstrap_settings() if bootstrap is None else dict(bootstrap)
    rows = frame.loc[matched(frame, prices)]
    if rows.empty:
        raise ValueError(f"no race has both {fit_price} and {score_price} on every runner")
    clusters = rows[cluster].groupby(rows["race_id"].to_numpy(dtype=object)).first()
    models = {price: fit(rows, price) for price in prices}
    columns = pd.MultiIndex.from_product([prices, STATS], names=["score_price", "stat"])
    table = pd.DataFrame(np.nan, index=pd.Index(prices, name="fit_price"), columns=columns)
    cells = {}
    for f in prices:
        for s in prices:
            per_race = pd.Series(score(models[f], rows, s), dtype=float)
            if per_race.empty or not np.isfinite(per_race.to_numpy()).all():
                raise ValueError(f"the score of the {f} fit on {s} is empty or not finite")
            iv = paired_bootstrap(
                per_race, clusters, n_boot=boot["n_boot"], level=boot["level"], seed=boot["seed"]
            )
            for stat, value in zip(STATS, (iv.mean, iv.lo, iv.hi), strict=True):
                table.loc[f, (s, stat)] = value
            cells[(f, s)] = {"races": len(per_race), "clusters": iv.n_clusters}
    table.attrs["cells"] = cells
    return table


def _cells(tables: Mapping) -> list[tuple]:
    """(hypothesis, pool, fit price, score price, mean, lo, hi, races, clusters) per cell."""
    out = []
    for (hypothesis, pool), table in tables.items():
        for f in table.index:
            for s in table.index:
                n = table.attrs.get("cells", {}).get((f, s), {})
                stats = (float(table.loc[f, (s, stat)]) for stat in STATS)
                out.append((hypothesis, pool, f, s, *stats, n.get("races"), n.get("clusters")))
    return out


def _selection(price: str) -> str:
    return {FINAL: "look_ahead", BOARD: "decision_time"}.get(price, "unknown")


def leaderboard_rows(tables: Mapping) -> list[LeaderboardRow]:
    """One withheld member row per fit x score cell, in its table's pool. A cell is a ΔLL per
    race, not a return per ¥100: it carries no return and ranks unranked; its ΔLL is the note."""
    cells = _cells(tables)
    return [
        LeaderboardRow(
            option_id=f"{FAMILY_ID}:{h}:{pool}:fit={f}:score={s}",
            family_id=FAMILY_ID,
            ticket=TICKET,
            kind="member",
            pool=pool,
            segment="all",
            rule=f"{h} {MODELS.get(h, h)}: fitted on {f}, scored on {s}, ΔLL per race against "
            f"{BASELINES.get(h, 'the scored price')}",
            selection=_selection(s),
            sources=SOURCES,
            variants_searched=len(cells),
            status="scored",
            n=races,
            n_unit=None if races is None else "races",
            n_meetings=clusters,
            licensed=True,
            note=f"ΔLL {mean:.6g} nats per race, one-sided bounds [{lo:.6g}, {hi:.6g}]",
        )
        for h, pool, f, s, mean, lo, hi, races, clusters in cells
    ]


def _fold_scheme() -> str:
    return "walk_forward:" + json.dumps(
        dataclasses.asdict(load_country_scheme(COUNTRY)), sort_keys=True
    )


def run_records(
    tables: Mapping,
    *,
    git_sha: str,
    data_builds: Mapping[str, str],
    coefficients: Mapping[tuple[str, str], Sequence[FoldMetric]] | None = None,
) -> list[RunRecord]:
    """One record per fitted model (hypothesis, pool, price fitted on): its ΔLL on each price
    scored, mean with D7's bounds; then one per `coefficients` entry ((hypothesis, price) ->
    its per-fold coefficient metrics)."""
    scheme = _fold_scheme()

    def record(hypothesis: str, variant: str, components: dict, metrics) -> RunRecord:
        return RunRecord(
            hypothesis_id=hypothesis,
            family_id=FAMILY_ID,
            variant=variant,
            ticket=TICKET,
            git_sha=git_sha,
            data_builds=dict(data_builds),
            sources=SOURCES,
            fold_scheme=scheme,
            components={**components, "model": MODELS.get(hypothesis, hypothesis)},
            params={"baseline": BASELINES.get(hypothesis, "the scored price")},
            metrics=tuple(metrics),
            status="scored",
        )

    by_fit: dict[tuple, list[FoldMetric]] = {}
    for h, pool, f, s, mean, lo, hi, races, _ in _cells(tables):
        metric = FoldMetric("scored_folds", f"delta_ll[score={s}]", mean, lo, hi, races or 0)
        by_fit.setdefault((h, pool, f), []).append(metric)
    out = [
        record(h, f"{pool}:fit={f}", {"pool": pool, "fit_price": f}, metrics)
        for (h, pool, f), metrics in by_fit.items()
    ]
    for (h, f), metrics in (coefficients or {}).items():
        out.append(record(h, f"coefficients:fit={f}", {"fit_price": f}, metrics))
    return out


def _cell_text(table: pd.DataFrame, f: str, s: str) -> str:
    mean, lo, hi = (float(table.loc[f, (s, stat)]) for stat in STATS)
    return f"{mean:+.6g} [{lo:+.6g}, {hi:+.6g}]"


def render_report(
    tables: Mapping,
    *,
    coefficients: Mapping | None = None,
    about: Mapping | None = None,
    failures: Mapping | None = None,
    payout: Mapping | None = None,
    leads: Mapping | None = None,
) -> str:
    """`payout`: E45's races per pool by the status of their paid record (`payout_rules`);
    `leads`: per hypothesis, the board's lead to the off per fold (`board_leads`)."""
    lines = [f"# Board-fitted against final-fitted terms on the JRDB board ({COUNTRY}, #{TICKET})"]
    lines += ["", *(f"- {k}: {v}" for k, v in (about or {}).items())]
    lines += ["- built on JRDB (D21): private, never published", ""]
    lines += [
        "Each cell: the mean ΔLL per race in nats (the fitted model's log-likelihood gain over "
        "the baseline on the price scored on; positive: the fit helps), with D7's one-sided "
        "bounds [lo, hi] (meeting-clustered paired bootstrap). Rows: the price fitted on.",
    ]
    for (h, pool), table in tables.items():
        rows = [
            {"fit on": f, **{f"scored on {s}": _cell_text(table, f, s) for s in table.index}}
            | {"races": table.attrs.get("cells", {}).get((f, f), {}).get("races")}
            for f in table.index
        ]
        lines += ["", f"## {h} {pool}: {MODELS.get(h, h)}", ""]
        lines += [f"Baseline: {BASELINES.get(h, 'the scored price')}.", "", _markdown_table(rows)]
    for (h, f), metrics in (coefficients or {}).items():
        rows = [
            {"fold": m.fold, "term": m.metric, "value": m.value, "lo": m.lo, "hi": m.hi, "n": m.n}
            for m in metrics
        ]
        lines += ["", f"## {h} coefficients fitted on {f} (R13: with their snapshot)", ""]
        lines.append(_markdown_table(rows) if rows else "_None._")
    for h, rows in (leads or {}).items():
        lines += ["", f"## {h}: the board's lead to the off per fold (R13: the snapshot)", ""]
        lines += [
            "Minutes from each matched race's latest board release to its off, the median over "
            "the fold's training races (what the board fits are fitted at) and its test races.",
            "",
            _markdown_table(rows) if rows else "_None._",
        ]
    if payout:
        lines += ["", "## E45's races by pool and paid record (the dividends table)", ""]
        lines.append(_markdown_table([{"pool": p, **dict(n)} for p, n in payout.items()]))
    if failures:
        lines += ["", "## Failed (each a registry record)", ""]
        lines += [f"- {key}: {error}" for key, error in failures.items()]
    return "\n".join(lines) + "\n"


def write_outputs(tables: Mapping, *, out_dir: Path | str, **report) -> dict[str, Path]:
    """The report, the private leaderboard (Markdown and JSON) and the cross tables (JSON) under
    `out_dir`, through `write_report` with JRDB among the sources. A directory inside the
    repository raises LicensedOutputInRepo, and one not under $HORSETRACK_DATA_DIR
    OutsideDataDir, before any file is written. `report` goes to `render_report`
    (coefficients, about, failures)."""
    out_dir = _guarded(out_dir)
    stem = _stem()
    rows = leaderboard_rows(tables)
    cells = [
        dict(
            zip(("hypothesis", "pool", "fit", "score", *STATS, "races", "clusters"), c, strict=True)
        )
        for c in _cells(tables)
    ]
    text = render_report(tables, **report)
    return {
        "report": write_report(out_dir / f"{stem}.md", text, sources=SOURCES),
        "tables": write_report(
            out_dir / f"{stem}-tables.json", json.dumps(cells, indent=2), sources=SOURCES
        ),
        "leaderboard": write_leaderboard(
            out_dir / f"{stem}-leaderboard.md", rows, title=f"{FAMILY_ID} (private)", public=False
        ),
        "leaderboard_json": write_report(
            out_dir / f"{stem}-leaderboard.json", to_json(rows), sources=SOURCES
        ),
    }


def _stem() -> str:
    return f"board-vs-final-{COUNTRY}-{FAMILY_ID}"


def _guarded(out_dir: Path | str) -> Path:
    """`out_dir` resolved, once it is outside the repository (LicensedOutputInRepo) and under
    $HORSETRACK_DATA_DIR (`output_dir`'s OutsideDataDir)."""
    if is_inside(out_dir, env.repo_root()):
        raise LicensedOutputInRepo(f"{out_dir} is inside the repository (D21)")
    return output_dir(out_dir)


def write_attempt(tables: Mapping, *, out_dir: Path | str, attempt: str, **report) -> dict:
    """A run with a failed table: its report alone, under `out_dir` in a file named by `attempt`,
    through `write_report`, guarded as `write_outputs`. A name already taken gets the first free
    `-<n>` suffix, so no attempt report replaces another; `write_outputs`' files, the last
    complete run's, are not touched."""
    out_dir = _guarded(out_dir)
    text = render_report(tables, **report)
    out_dir.mkdir(parents=True, exist_ok=True)
    name, n = f"{_stem()}-attempt-{attempt}", 1
    while True:  # claim the name (O_EXCL) so a concurrent attempt cannot take it as well
        path = out_dir / (f"{name}.md" if n == 1 else f"{name}-{n}.md")
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
            break
        except FileExistsError:
            n += 1
    try:
        return {"attempt": write_report(path, text, sources=SOURCES)}
    except BaseException:
        path.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- the two fits


def _folds(rows: pd.DataFrame):
    """D9's walk-forward folds of `rows` (positions), the same for both prices."""
    return walk_forward(rows["race_date"], scheme=load_country_scheme(COUNTRY))


def _ln_q(rows: pd.DataFrame, price: str) -> np.ndarray:
    q = rows[price].to_numpy(dtype=float)
    total = pd.Series(q).groupby(rows["race_id"].to_numpy(dtype=object)).transform("sum")
    return np.log(q / total.to_numpy())


def _e2_probs(coef: np.ndarray, rows: pd.DataFrame, price: str) -> pd.Series:
    ln_q = _ln_q(rows, price)
    u = coef[0] * ln_q + coef[1] * ln_q * ln_q
    race = rows["race_id"].to_numpy(dtype=object)
    e = np.exp(u - pd.Series(u).groupby(race).transform("max").to_numpy())
    return pd.Series(e / pd.Series(e).groupby(race).transform("sum").to_numpy(), index=rows.index)


def e2_fit(rows: pd.DataFrame, price: str) -> list:
    """Per fold, the unbounded conditional-logit (beta, gamma) of u = beta ln q + gamma (ln q)^2
    on the fold's training rows, meeting-clustered se beside the inverse-information se."""
    fits = []
    for fold in _folds(rows):
        train = rows.iloc[fold.train_idx]
        ln_q = _ln_q(train, price)
        fits.append(
            fit_conditional_logit(
                np.column_stack([ln_q, ln_q * ln_q]),
                train["race_id"].to_numpy(dtype=object),
                train["won"].to_numpy(dtype=bool),
                cluster=train[CLUSTER_COLUMNS[CONFIG["cluster_unit"]]].to_numpy(dtype=object),
            )
        )
    return fits


def e2_score(model: list, rows: pd.DataFrame, price: str) -> pd.Series:
    """Per test race of each fold, the winner log loss of the scored price minus the fold's fit's
    on it (core.eval.scoring: a dead heat at 1/k): positive when the terms improve on the price."""
    parts = []
    for fit, fold in zip(model, _folds(rows), strict=True):
        test = rows.iloc[fold.test_idx]
        won, race = test["won"], test["race_id"].astype(object)
        none = pd.Series(False, index=test.index)
        ll_q = per_race_log_loss(test[price].astype(float), won, race, none).per_race
        ll_m = per_race_log_loss(_e2_probs(fit.coef, test, price), won, race, none).per_race
        parts.append(gain(ll_m, ll_q).per_race)
    return pd.concat(parts)


def e2_coefficients(model: list, rows: pd.DataFrame, level: float) -> list[FoldMetric]:
    z = NormalDist().inv_cdf(level)
    out = []
    for fit, fold in zip(model, _folds(rows), strict=True):
        for k, term in enumerate(("beta", "gamma")):
            c, s = float(fit.coef[k]), float(fit.se_cluster[k])
            out.append(
                FoldMetric(
                    fold.test_start.isoformat(),
                    f"coef[{term}]",
                    c,
                    c - z * s,
                    c + z * s,
                    fit.n_races,
                )
            )
    return out


def e45_fit(rows: pd.DataFrame, price: str) -> Ordering:
    """#99's quadratic_context ordering refitted on each fold's training rows of the price."""
    frame = rows[["race_id", "horse_no", "finish_pos", "dead_heat"]].assign(p_win=rows[price])
    return Ordering("quadratic_context").fit(frame, _folds(rows))


def e45_coefficients(model: Ordering, rows: pd.DataFrame) -> list[FoldMetric]:
    out = []
    for params, fold in zip(model.fold_params, _folds(rows), strict=True):
        n = int(rows["race_id"].iloc[fold.train_idx].nunique())
        out += [
            FoldMetric(fold.test_start.isoformat(), f"coef[{k}]", v, None, None, n)
            for k, v in params.items()
        ]
    return out


def _bernoulli(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float((y * np.log(p) + (1 - y) * np.log1p(-p)).sum())


def pool_loglik(t: np.ndarray, finish: np.ndarray, *, places: int, wide: bool) -> dict[str, float]:
    """One race's log-likelihood per pool from a trifecta array and the finishing positions (one
    runner each in first, second and third): PLACE, the Bernoulli sum over runners of placing in
    the first `places`, the places the race paid (NaN for 0: not scored in Place);
    QUINELLA_PLACE, the Bernoulli sum over pairs of both finishing in the first three (NaN unless
    `wide`, the race paid Wide); TRIO, ln P of the first three as a set."""
    n = len(finish)
    top = [int(np.flatnonzero(finish == k)[0]) for k in (1, 2, 3)]
    out = {PLACE: math.nan, WIDE: math.nan}
    if places:
        y = np.zeros(n)
        y[top[:places]] = 1.0
        positions = (t.sum(axis=(1, 2)), t.sum(axis=(0, 2)), t.sum(axis=(0, 1)))
        out[PLACE] = _bernoulli(sum(positions[:places]), y)
    trio = pool_array(t, TRIO)
    if wide:
        i, j = np.triu_indices(n, 1)
        y = np.isin(i, top) & np.isin(j, top)
        out[WIDE] = _bernoulli(trio.sum(axis=2)[i, j], y.astype(float))
    out[TRIO] = float(np.log(trio[top[0], top[1], top[2]]))
    return out


def _horses(combination: object) -> frozenset[int]:
    """The horse numbers of a paid combination ("4", "4,7"); empty for one that does not parse,
    which then matches no result."""
    parts = str(combination).split(",")
    return (
        frozenset(int(h) for h in parts) if all(h.strip().isdigit() for h in parts) else frozenset()
    )


def payout_rules(rows: pd.DataFrame, paid: pd.DataFrame) -> pd.DataFrame:
    """Per race of `rows` (one runner each in first, second and third), from its paid
    combinations in `paid` (dividends rows: race_id, pool, combination, is_refund; refunds are
    not paid): `places`, the Place places scored (0: not scored in Place), `wide`, whether Wide
    is scored, and each pool's status. Place scores the first `places` finishers when the paid
    PLACE horses are exactly those and `places` is a count jra.yaml pays; Wide scores when the
    paid pairs are exactly the pairs of the first three. A race with no paid Win has no paid
    record and scores in neither (NO_RECORD); one without a paid row of the pool was not sold it
    or lost its record (NOT_PAID); one whose paid combinations differ from the result, MISMATCH."""
    refund = paid["is_refund"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    combos: dict[tuple, set] = {}
    for race, pool, combination in paid.loc[~refund, ["race_id", "pool", "combination"]].itertuples(
        index=False
    ):
        combos.setdefault((str(race), str(pool)), set()).add(_horses(combination))
    counts = {rule.places for rule in load_source(COUNTRY, "jra").pools[PLACE].places_paid}
    top = rows.loc[rows["finish_pos"].isin([1.0, 2.0, 3.0])]
    top = top.pivot(index="race_id", columns="finish_pos", values="horse_no")
    top = top.reindex(columns=[1.0, 2.0, 3.0])
    out = {}
    for race_id, *first3 in top.astype("int64").itertuples():
        place, wide = combos.get((race_id, PLACE), set()), combos.get((race_id, WIDE), set())
        p, p_status, w_status = len(place), NOT_PAID, NOT_PAID
        if (race_id, WIN) not in combos:
            p, p_status, w_status = 0, NO_RECORD, NO_RECORD
        elif place:
            ok = p in counts and place == {frozenset([h]) for h in first3[:p]}
            p, p_status = (p, SCORED) if ok else (0, MISMATCH)
        if w_status != NO_RECORD and wide:
            ok = wide == {frozenset(pair) for pair in itertools.combinations(first3, 2)}
            w_status = SCORED if ok else MISMATCH
        out[race_id] = {"places": p, "wide": w_status == SCORED}
        out[race_id] |= {"place_status": p_status, "wide_status": w_status}
    frame = pd.DataFrame.from_dict(out, orient="index")
    columns = ["places", "wide", "place_status", "wide_status"]
    return frame.reindex(columns=columns).rename_axis("race_id")


def e45_deltas(model: Ordering, rows: pd.DataFrame, price: str) -> pd.DataFrame:
    """Per test race of each fold and per pool, the fold's fitted ordering's log-likelihood on
    the price minus Harville's on it; Place and Wide by the race's `places_paid` and
    `wide_paid` (`payout_rules`)."""
    harville = Ordering("harville")
    out = {}
    for k, fold in enumerate(_folds(rows)):
        fitted = model.at_fold(k)
        test = rows.iloc[fold.test_idx]
        for race_id, race in test.groupby(test["race_id"].to_numpy(dtype=object), sort=False):
            p, finish = race[price].to_numpy(dtype=float), race["finish_pos"].to_numpy(dtype=float)
            paid = {
                "places": int(race["places_paid"].iloc[0]),
                "wide": bool(race["wide_paid"].iloc[0]),
            }
            m, h = (
                pool_loglik(fitted.trifecta(p), finish, **paid),
                pool_loglik(harville.trifecta(p), finish, **paid),
            )
            out[race_id] = {pool: m[pool] - h[pool] for pool in E45_POOLS}
    frame = pd.DataFrame.from_dict(out, orient="index", columns=list(E45_POOLS), dtype=float)
    return frame.rename_axis("race_id")


def board_leads(rows: pd.DataFrame) -> list[dict]:
    """Per fold of `rows` (with `lead_min`, each runner's minutes from its board's release to
    its race's off): the median over the fold's training and test races of each race's lead,
    from its latest release (R13: the snapshot the board's terms are fitted and scored at)."""
    race = rows["race_id"].to_numpy(dtype=object)
    lead = rows["lead_min"].astype(float).groupby(race).min()
    out = []
    for fold in _folds(rows):
        row = {"fold": fold.test_start.isoformat()}
        for part, idx in (("train", fold.train_idx), ("test", fold.test_idx)):
            races = lead.loc[pd.unique(race[idx])]
            row |= {f"{part} median (min)": float(races.median()), f"{part} races": len(races)}
        out.append(row)
    return out


class _Memo:
    """A fit or score computed once per key: both E45 pools' tables reuse a price's fit."""

    def __init__(self, compute: Callable) -> None:
        self.compute, self.cache = compute, {}

    def __call__(self, key, *args):
        if key not in self.cache:
            self.cache[key] = self.compute(*args)
        return self.cache[key]


# --------------------------------------------------------------------------- the run


def build_spec(scheme, boot: dict) -> dict:
    """The family's pre-registered spec, frozen in the registry before any outcome is read."""
    board = CONFIG["board"]
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "E2": "the quadratic/power terms fitted on the board are partly a price-forecast "
            "correction: |gamma_final| < |gamma_board|; cross-applied, the board's terms gain "
            "less on the final than the final's own",
            "E45": "the quadratic_context ordering coefficients depend on the horizon of the Win "
            "probabilities fed to them (board vs final): cross-applied Place, Wide and Trio ΔLL",
        },
        "readings": "descriptive, no decision rule and no adoption (rank, not adopt; D16): each "
        "fit x score cell is a withheld member row of the private leaderboard; the E2 "
        "coefficients are read per fold with their meeting-clustered bounds",
        "board": {
            "table": "jrdb_pre_race (TYB): tyb_win_odds at tyb_odds_time, HHMM on the "
            f"{board['tz']} clock of race_date",
            "release": "odds time + release_lag_min (an assumption: no JRDB release log)",
            "release_lag_min": board["release_lag_min"],
            "read": "core.store.asof.read_asof under a column rule on release_utc, at each race's "
            "off (races.post_utc, availability.yaml's derived post); a board released after its "
            "race's off is not used",
            "never": "filled from runners.win_odds or kyi_base_odds; a missing or malformed odds "
            "time, or missing TYB odds, gives no board",
        },
        "final": "runners.win_odds (netkeiba), cleaned (core.pricing.clean_odds)",
        "prices": "q = 1/odds normalised over the race's starters, for each price",
        "races": "development days only (split_off; holdout days dropped in the scan) from "
        f"{PRICE_ERA_START.isoformat()} (D10); run races; starters = not scratched with a final "
        "price; a race needs a winner; cross_apply keeps the races with both prices on every "
        "starter (matching races and seasons for every fit and score)",
        "folds": {
            "scheme": dataclasses.asdict(scheme),
            "calendar": "walk_forward over each hypothesis's matched rows; the same folds for "
            "both prices",
            "scored": "every fold's test rows (D9: at least min_train quarters of training)",
        },
        "E2": {
            "fit": MODELS["E2"] + "; q normalised within race, ridge 0, a dead heat at 1/k "
            "(models.condlogit), refit per fold on its training rows",
            "score": "per test race, winner log loss of the scored q minus the fitted model's at "
            "the scored q (core.eval.scoring.gain)",
            "coefficients": "per fold beta and gamma, meeting-clustered sandwich se, bounds at "
            "D7's one-sided level",
        },
        "E45": {
            "fit": MODELS["E45"] + ", refit per fold on its training rows (its own rule: races "
            "with one runner each in first, second and third); its loose QC_BOUNDS stand",
            "races": "matched races with one runner each in first, second and third, no dead heat",
            "score": "per test race and pool, the fitted ordering's log-likelihood minus "
            "Harville's, both on the scored price",
            "paid": "Place's places and Wide's sale from the race's paid combinations "
            "(dividends: race_id, pool, combination, non-refund rows), not from the field of "
            "starters: the places paid follow the field when sales opened (jra.yaml), which the "
            "store does not hold, so a late scratch from eight to seven still pays three",
            "pools": {
                PLACE: "Bernoulli sum over starters of placing in the first p, p the count of "
                "paid PLACE horses; scored only when p is a count jra.yaml pays and the paid "
                "horses are exactly the first p finishers",
                WIDE: "Bernoulli sum over pairs of both finishing in the first three; scored only "
                "when the paid QUINELLA_PLACE pairs are exactly the pairs of the first three",
                TRIO: "ln P(the first three as a set)",
            },
            "not scored": "in Place and Wide, a race without a paid Win (no paid record); in a "
            "pool, a race without a paid row of it (not sold, or its record missing) or whose "
            "paid combinations differ from the result; each race counted by status per pool in "
            "the private report",
        },
        "supersedes": "japan-board-vs-final-e2-e45-v1, whose Place places and Wide sale came "
        "from the field of starters (a late scratch from eight to seven scored as two places); "
        "its records stay in the registry as recorded",
        "lead": "R13: per fold, the median minutes from each matched race's latest board "
        "release to its off, over the training and the test races, in the private report",
        "bootstrap": boot,
        "cluster_unit": CONFIG["cluster_unit"],
        "records": "one per (hypothesis, pool, price fitted on): its ΔLL on each price scored "
        "with D7's bounds; one per (hypothesis, price fitted on): its per-fold coefficients; a "
        "failed table gives failed records, never none; a run with a failed table writes its "
        "report to an attempt-named file, leaves the last complete outputs and exits non-zero",
        "leaderboard": "each fit x score cell a member row, withheld (D21): a ΔLL, not a return "
        "per ¥100, so it carries no return and ranks unranked",
        "sources": list(SOURCES),
        "holdout": "never read and never opened (D11-D15)",
        "d21": "every output is JRDB-derived: written through write_report outside the "
        "repository; registry values withheld in the committed summary",
    }


def read_development() -> dict[str, pd.DataFrame]:
    """Development rows of the price era, column- and date-scoped (holdout days dropped). The
    dividends table has no date column: its Win, Place and Wide rows are read column-scoped, of
    the development races only (filtered in the scan)."""
    out = {}
    for name, columns in (
        ("runners", RUNNER_COLUMNS),
        ("races", RACE_COLUMNS),
        ("jrdb_pre_race", JRDB_COLUMNS),
    ):
        table = read_store_table(
            name, columns=columns, start=PRICE_ERA_START, end=date.max, touches_price=True
        )
        out[name], _ = split_off(table.to_pandas(date_as_object=False), "race_date", COUNTRY)
    races = pa.array(out["races"]["race_id"].astype(str).unique())
    dataset = ds.dataset(paths.processed_dir("jra") / "dividends.parquet", format="parquet")
    out["dividends"] = dataset.to_table(
        columns=DIVIDEND_COLUMNS,
        filter=ds.field("pool").isin(pa.array(PAID_POOLS)) & ds.field("race_id").isin(races),
    ).to_pandas()
    return out


def _arrow(frame: pd.DataFrame) -> pa.Table:
    """The frame as an Arrow table for DuckDB, race_date a DATE again (a local date, not an
    instant: the as-of rules read it on the Tokyo clock)."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    where = table.schema.get_field_index("race_date")
    return table.set_column(where, "race_date", table["race_date"].cast(pa.date32()))


def post_times(races: pd.DataFrame) -> pd.Series:
    """Each race's off, races.post_utc of the as-of view (availability.yaml), by race_id."""
    con = duckdb.connect(":memory:")
    try:
        con.register("races", _arrow(races))
        view = read_asof(con, "races", datetime.now(timezone.utc), rules=load_availability(COUNTRY))
    finally:
        con.close()
    return pd.Series(view["post_utc"].to_numpy(), index=view["race_id"].astype(object).to_numpy())


def study_frame(data: Mapping[str, pd.DataFrame], board: pd.DataFrame) -> pd.DataFrame:
    """One row per priced starter of a run race with a winner: race_id, race_date, meeting_id,
    horse_no, finish_pos, dead_heat, won, q_final, q_board (NaN without a released board) and
    lead_min (the board's minutes before the off, `_board`)."""
    runners, races = data["runners"], data["races"].astype({"race_id": object})
    kept = runners.loc[runners["race_id"].isin(races.loc[races["status"].eq("run"), "race_id"])]
    scratched = kept["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)
    odds = clean_odds(kept["win_odds"].to_numpy(dtype=float, na_value=np.nan))
    keep = ~scratched & ~np.isnan(odds)
    frame = kept.loc[keep, RUNNER_COLUMNS[:6]].assign(final_odds=odds[keep])
    frame = frame.astype({"race_id": object, "horse_no": "int64"})
    frame["finish_pos"] = frame["finish_pos"].astype(float)
    frame["dead_heat"] = frame["dead_heat"].eq(True).to_numpy(dtype=bool, na_value=False)
    frame["won"] = frame["finish_pos"].eq(1)
    frame = frame.loc[frame.groupby("race_id")["won"].transform("any").to_numpy(dtype=bool)]
    frame = frame.merge(races[["race_id", "meeting_id"]], on="race_id", validate="many_to_one")
    tyb = board[["race_id", "horse_no", "tyb_win_odds", "lead_min"]].astype({"horse_no": "int64"})
    frame = frame.merge(tyb, on=["race_id", "horse_no"], how="left", validate="one_to_one")
    frame["board_odds"] = clean_odds(frame["tyb_win_odds"].to_numpy(dtype=float, na_value=np.nan))
    for odds_col, q in (("final_odds", FINAL), ("board_odds", BOARD)):
        inv = 1.0 / frame[odds_col]
        frame[q] = inv / inv.groupby(frame["race_id"]).transform("sum")
    frame = frame.drop(columns=["win_odds", "tyb_win_odds", "final_odds", "board_odds"])
    return frame.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)


def clean_top3(frame: pd.DataFrame) -> pd.DataFrame:
    """The races with one runner each in first, second and third and no dead heat among them."""
    race = frame["race_id"].to_numpy(dtype=object)
    ok = np.ones(len(frame), dtype=bool)
    for k in (1.0, 2.0, 3.0):
        ok &= frame["finish_pos"].eq(k).groupby(race).transform("sum").to_numpy() == 1
    top = frame["finish_pos"].isin([1.0, 2.0, 3.0]) & frame["dead_heat"]
    ok &= ~top.groupby(race).transform("any").to_numpy(dtype=bool)
    return frame.loc[ok].reset_index(drop=True)


def compute(frame: pd.DataFrame, paid: pd.DataFrame, boot: dict) -> tuple[dict, dict, dict, dict]:
    """The cross tables, the coefficients, the failures and the report's notes (`render_report`'s
    `payout` and `leads`) of both hypotheses; E45's Place and Wide by `paid` (`payout_rules`)."""
    tables, coefficients, failures, notes = {}, {}, {}, {"payout": {}, "leads": {}}
    prices = (FINAL, BOARD)
    rows = frame.loc[matched(frame, prices)].reset_index(drop=True)  # cross_apply's own rows
    notes["leads"]["E2"] = board_leads(rows) if len(rows) else []
    fits = _Memo(e2_fit)
    try:
        tables[("E2", WIN)] = cross_apply(
            rows,
            fit_price=FINAL,
            score_price=BOARD,
            fit=lambda r, p: fits(p, r, p),
            score=e2_score,
            bootstrap=boot,
        )
    except Exception as exc:  # a failed variant is a registry record, never dropped
        failures[("E2", WIN)] = f"{type(exc).__name__}: {exc}"
    for price in prices:
        if price in fits.cache:
            coefficients[("E2", price)] = e2_coefficients(fits.cache[price], rows, boot["level"])
    rows = clean_top3(rows)
    rules = payout_rules(rows, paid)
    by_race = rows["race_id"].map
    rows = rows.assign(places_paid=by_race(rules["places"]), wide_paid=by_race(rules["wide"]))
    for pool, status in ((PLACE, "place_status"), (WIDE, "wide_status")):
        notes["payout"][pool] = rules[status].value_counts().sort_index().to_dict()
    notes["leads"]["E45"] = board_leads(rows) if len(rows) else []
    fits, deltas = _Memo(e45_fit), _Memo(e45_deltas)
    for pool in E45_POOLS:
        try:
            tables[("E45", pool)] = cross_apply(
                rows,
                fit_price=FINAL,
                score_price=BOARD,
                fit=lambda r, p: fits(p, r, p),
                score=lambda m, r, p, pool=pool: deltas((id(m), p), m, r, p)[pool].dropna(),
                bootstrap=boot,
            )
        except Exception as exc:
            failures[("E45", pool)] = f"{type(exc).__name__}: {exc}"
    for price in prices:
        if price in fits.cache:
            coefficients[("E45", price)] = e45_coefficients(fits.cache[price], rows)
    return tables, coefficients, failures, notes


def _failed_records(failures: Mapping, git_sha: str, builds: dict) -> list[RunRecord]:
    """A failed record per price fitted on of each failed (hypothesis, pool) table."""
    out = []
    for h, pool in failures:
        for price in (FINAL, BOARD):
            out.append(
                RunRecord(
                    h,
                    FAMILY_ID,
                    f"{pool}:fit={price}",
                    TICKET,
                    git_sha,
                    dict(builds),
                    SOURCES,
                    _fold_scheme(),
                    {"pool": pool, "fit_price": price, "model": MODELS[h]},
                    {"baseline": BASELINES[h]},
                    (),
                    "failed",
                )
            )
    return out


def _append_new(registry: Registry, records: list[RunRecord]) -> tuple[dict, int]:
    """Append the records the family does not already hold (a record equal to one in the
    registry but for its git commit is the same run reproduced); the run ids by variant, a held
    record's own id for one not appended."""

    def same(r: RunRecord) -> str:
        return json.dumps(dataclasses.asdict(dataclasses.replace(r, git_sha="")), sort_keys=True)

    held = {same(r): _sha256(_canonical(_record_data(r))) for r in registry.records(FAMILY_ID)}
    ids, appended = {}, 0
    for record in records:
        key = f"{record.hypothesis_id}:{record.variant}"
        if same(record) in held:
            ids[key] = held[same(record)]
            continue
        ids[key] = registry.append(record)
        appended += 1
    return ids, appended


def freeze() -> tuple[Registry, dict, str]:
    boot = bootstrap_settings()
    spec = build_spec(load_country_scheme(COUNTRY), boot)
    registry = default_registry(COUNTRY)
    return registry, boot, registry.freeze_family(FAMILY_ID, spec)


def output_dir(out_dir: Path | str | None) -> Path:
    """The run's output directory, resolved: the data directory's reports directory by default.
    Either one, the default included (it may be a symlink), must lie under $HORSETRACK_DATA_DIR
    (OutsideDataDir otherwise) and so outside the repository (DataDirInsideRepo). Checked before
    anything is written."""
    if out_dir is None:
        out_dir = paths.reports_dir()
    resolved = paths.outside_repo(Path(out_dir), what="out")
    data = paths.data_dir()
    # A data dir not created yet holds no symlink: both resolved paths compare as written.
    inside = is_inside(resolved, data) if data.exists() else resolved.is_relative_to(data)
    if not inside:
        raise OutsideDataDir(f"out={resolved} is not under {paths.DATA_DIR_VAR}={data}")
    return resolved


def run_study(out_dir: Path | str | None = None) -> dict:
    """Freeze, then run the family on Japan's development data and write its private outputs.
    When a table could not be produced, the failed records are appended and only the attempt's
    report is written (`write_attempt`); the result's `failed` lists the tables."""
    out_dir = output_dir(out_dir)
    registry, boot, spec_hash = freeze()  # before any outcome is read
    builds = {
        **{f"jra_{name}": _build(name) for name in ("runners", "races")},  # store_builds' keys
        **{name: _build(name) for name in ("dividends", "jrdb_pre_race")},
    }
    notes: dict = {}
    try:
        data = read_development()
        frame, paid = study_frame(data, _board(data)), data["dividends"]
        del data
        tables, coefficients, failures, notes = compute(frame, paid, boot)
    except Exception as exc:  # a failed read fails every table: each a registry record
        error = f"{type(exc).__name__}: {exc}"
        tables, coefficients = {}, {}
        failures = {key: error for key in [("E2", WIN), *(("E45", p) for p in E45_POOLS)]}
    git_sha = _git_sha()
    records = run_records(tables, git_sha=git_sha, data_builds=builds, coefficients=coefficients)
    records += _failed_records(failures, git_sha, builds)
    ids, appended = _append_new(registry, records)
    now = datetime.now(timezone.utc)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "builds": ", ".join(builds.values()),
        "records": f"{appended} appended; ids by variant: {json.dumps(ids)}",
        "written (UTC)": now.isoformat(timespec="seconds"),
    }
    report = {"coefficients": coefficients, "about": about, "failures": failures, **notes}
    if failures:  # the last complete run's outputs stay; this attempt gets its own file
        written = write_attempt(
            tables, out_dir=out_dir, attempt=now.strftime("%Y%m%dT%H%M%SZ"), **report
        )
    else:
        written = write_outputs(tables, out_dir=out_dir, **report)
    update_summary(registry, [FAMILY_ID], summary_dir(COUNTRY))
    return {
        "family_id": FAMILY_ID,
        "spec_hash": spec_hash,
        "registry_path": str(registry.path),
        "record_ids": ids,
        "records_appended": appended,
        "outputs": {k: str(v) for k, v in written.items()},
        "failed": sorted(failures),
    }


def _build(name: str) -> str:
    """'processed/jra/<name>.parquet@<sha256 prefix>' of the store file; '@missing' (or
    '@unreadable') and no digest when it cannot be opened or read through, so the read then
    fails inside the run's failure handling, each table a failed record."""
    digest = hashlib.sha256()
    try:
        with open(paths.processed_dir("jra") / f"{name}.parquet", "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        state = "missing" if isinstance(exc, FileNotFoundError) else "unreadable"
        return f"processed/jra/{name}.parquet@{state}"
    return f"processed/jra/{name}.parquet@{digest.hexdigest()[:16]}"


def _board(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """`jrdb_board` at each race's off, with `lead_min`, the minutes from release to the off."""
    races = data["races"]
    offs = post_times(races)
    con = duckdb.connect(":memory:")
    try:
        con.register("jrdb_pre_race", _arrow(data["jrdb_pre_race"]))
        board = jrdb_board(
            con, races["race_id"].astype(str).tolist(), rules=CONFIG["board"], decision_utc=offs
        )
    finally:
        con.close()
    off = pd.to_datetime(board["race_id"].map(pd.to_datetime(offs, utc=True)), utc=True)
    lead = off - pd.to_datetime(board["release_utc"], utc=True)
    return board.assign(lead_min=lead.dt.total_seconds() / 60.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.board_vs_final"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="freeze the family's spec only")
    mode.add_argument("--run", action="store_true", help="freeze, then run the family")
    parser.add_argument(
        "--out", type=Path, default=None, help="output directory, under $HORSETRACK_DATA_DIR"
    )
    args = parser.parse_args(argv)
    if args.freeze:
        _, _, spec_hash = freeze()
        print(json.dumps({"family_id": FAMILY_ID, "spec_hash": spec_hash}, indent=2))
        return 0
    try:
        result = run_study(args.out)
    except (OutsideDataDir, paths.DataDirInsideRepo) as exc:  # refused, nothing written
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    if result["failed"]:
        print(f"error: tables not produced: {result['failed']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
