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

Order: freeze the family's spec (before any outcome is read), read the development store, fit
and score, append one registry record per fitted model (failures included), write the outputs and
the family's summary file. CLI:
`python -m horsetrack.countries.japan.studies.board_vs_final (--freeze | --run) [--out DIR]`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
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
from horsetrack.countries.japan.studies.calibration import _markdown_table, store_builds

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
RUNNER_COLUMNS = ["race_id", "race_date", "horse_no", "win_odds", "finish_pos", "dead_heat"]
RUNNER_COLUMNS += ["is_scratched"]
# the races view needs its rule's label and gated columns (availability.yaml) to derive post_utc
RACE_COLUMNS = ["race_id", "race_date", "meeting_id", "status", "post_time_local", "going"]
RACE_COLUMNS += ["going_raw", "finish_time_s", "split_times_s", "result_published_utc"]
RACE_COLUMNS += ["result_unpublished_utc"]


# --------------------------------------------------------------------------- the board


def _lag(rules: Mapping) -> timedelta:
    minutes = rules.get("release_lag_min") if isinstance(rules, Mapping) else None
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not minutes >= 0:
        raise ValueError(f"rules need release_lag_min, minutes >= 0, got {minutes!r}")
    if not math.isfinite(minutes):
        raise ValueError(f"release_lag_min must be finite, got {minutes!r}")
    return timedelta(minutes=minutes)


def jrdb_board(
    con,
    race_ids: Sequence[str],
    *,
    rules: Mapping,
    decision_utc: pd.Series | None = None,
) -> pd.DataFrame:
    """JRDB's fixed-time board of `race_ids` from the store table `jrdb_pre_race` in `con`:
    race_id, horse_no, tyb_win_odds and release_utc, at most one row per runner.

    A row is released at race_date, tyb_odds_time ("HHMM" on the wall clock of `rules["tz"]`,
    default Asia/Tokyo) + `rules["release_lag_min"]`; a missing or malformed odds time is never
    released. The rows go through `read_asof` under a column rule on that release time, at
    `decision_utc` (a Series of timezone-aware decision times indexed by race_id: a runner's board
    counts only if released by its own race's time; a race without one has no board) or, without
    it, now. Nothing is filled from another price: a runner without TYB odds reads null."""
    lag, tz = _lag(rules), rules.get("tz", "Asia/Tokyo")
    ZoneInfo(tz)  # an unknown zone fails here
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
            "j.tyb_win_odds, CASE WHEN regexp_full_match(j.tyb_odds_time, '[0-9]{4}') THEN "
            "timezone(?, try_strptime(strftime(j.race_date, '%Y-%m-%d') || ' ' || j.tyb_odds_time, "
            "'%Y-%m-%d %H%M')) + to_microseconds(?) END AS release_utc FROM jrdb_pre_race AS j "
            f'WHERE j.race_id IN (SELECT race_id FROM "{_KEYS}")',
            [tz, lag // timedelta(microseconds=1)],
        )
        rule = Availability(_VIEW, "column", column="release_utc")
        board = pd.DataFrame(columns=["race_id", "horse_no", "tyb_win_odds", "release_utc"])
        if at is not None:
            board = read_asof(con, _VIEW, at, rules={_VIEW: rule})
    finally:
        con.execute(f'DROP TABLE IF EXISTS "{_VIEW}"')
        con.unregister(_KEYS)
    board = board[["race_id", "horse_no", "tyb_win_odds", "release_utc"]].astype(
        {"race_id": object}
    )
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
) -> str:
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
    if failures:
        lines += ["", "## Failed (each a registry record)", ""]
        lines += [f"- {key}: {error}" for key, error in failures.items()]
    return "\n".join(lines) + "\n"


def write_outputs(tables: Mapping, *, out_dir: Path | str, **report) -> dict[str, Path]:
    """The report, the private leaderboard (Markdown and JSON) and the cross tables (JSON) under
    `out_dir`, through `write_report` with JRDB among the sources. A directory inside the
    repository raises LicensedOutputInRepo before any file is written. `report` goes to
    `render_report` (coefficients, about, failures)."""
    out_dir = Path(out_dir)
    if is_inside(out_dir, env.repo_root()):
        raise LicensedOutputInRepo(f"{out_dir} is inside the repository (D21)")
    stem = f"board-vs-final-{COUNTRY}-{FAMILY_ID}"
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


def pool_loglik(t: np.ndarray, finish: np.ndarray) -> dict[str, float]:
    """One race's log-likelihood per pool from a trifecta array and the finishing positions (one
    runner each in first, second and third): PLACE, the Bernoulli sum over runners of placing in
    the places the field pays (jra.yaml; NaN where Place is not sold); QUINELLA_PLACE, the
    Bernoulli sum over pairs of both finishing in the first three (NaN where Wide is not sold);
    TRIO, ln P of the first three as a set."""
    n = len(finish)
    top = [int(np.flatnonzero(finish == k)[0]) for k in (1, 2, 3)]
    pools = load_source(COUNTRY, "jra").pools
    out = {PLACE: math.nan, WIDE: math.nan}
    places = pools[PLACE].places_for(n)
    if places:
        y = np.zeros(n)
        y[top[:places]] = 1.0
        out[PLACE] = _bernoulli(pool_array(t, PLACE, field_at_sale=n), y)
    trio = pool_array(t, TRIO)
    if pools[WIDE].places_for(n):
        i, j = np.triu_indices(n, 1)
        y = np.isin(i, top) & np.isin(j, top)
        out[WIDE] = _bernoulli(trio.sum(axis=2)[i, j], y.astype(float))
    out[TRIO] = float(np.log(trio[top[0], top[1], top[2]]))
    return out


def e45_deltas(model: Ordering, rows: pd.DataFrame, price: str) -> pd.DataFrame:
    """Per test race of each fold and per pool, the fold's fitted ordering's log-likelihood on
    the price minus Harville's on it."""
    harville = Ordering("harville")
    out = {}
    for k, fold in enumerate(_folds(rows)):
        fitted = model.at_fold(k)
        test = rows.iloc[fold.test_idx]
        for race_id, race in test.groupby(test["race_id"].to_numpy(dtype=object), sort=False):
            p, finish = race[price].to_numpy(dtype=float), race["finish_pos"].to_numpy(dtype=float)
            m, h = (
                pool_loglik(fitted.trifecta(p), finish),
                pool_loglik(harville.trifecta(p), finish),
            )
            out[race_id] = {pool: m[pool] - h[pool] for pool in E45_POOLS}
    frame = pd.DataFrame.from_dict(out, orient="index", columns=list(E45_POOLS), dtype=float)
    return frame.rename_axis("race_id")


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
            "pools": {
                PLACE: "Bernoulli sum over starters of placing in the places paid to the field of "
                "starters (jra.yaml); races where Place is not sold are not scored",
                WIDE: "Bernoulli sum over pairs of both finishing in the first three; races where "
                "Wide is not sold are not scored",
                TRIO: "ln P(the first three as a set)",
            },
        },
        "bootstrap": boot,
        "cluster_unit": CONFIG["cluster_unit"],
        "records": "one per (hypothesis, pool, price fitted on): its ΔLL on each price scored "
        "with D7's bounds; one per (hypothesis, price fitted on): its per-fold coefficients; a "
        "failed table gives failed records, never none",
        "leaderboard": "each fit x score cell a member row, withheld (D21): a ΔLL, not a return "
        "per ¥100, so it carries no return and ranks unranked",
        "sources": list(SOURCES),
        "holdout": "never read and never opened (D11-D15)",
        "d21": "every output is JRDB-derived: written through write_report outside the "
        "repository; registry values withheld in the committed summary",
    }


def read_development() -> dict[str, pd.DataFrame]:
    """Development rows of the price era, column- and date-scoped (holdout days dropped)."""
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
    horse_no, finish_pos, dead_heat, won, q_final and q_board (NaN without a released board)."""
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
    tyb = board[["race_id", "horse_no", "tyb_win_odds"]].astype({"horse_no": "int64"})
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


def compute(frame: pd.DataFrame, boot: dict) -> tuple[dict, dict, dict]:
    """The cross tables, the coefficients and the failures of both hypotheses."""
    tables, coefficients, failures = {}, {}, {}
    prices = (FINAL, BOARD)
    rows = frame.loc[matched(frame, prices)].reset_index(drop=True)  # cross_apply's own rows
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
    return tables, coefficients, failures


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
    registry but for its git commit is the same run reproduced); their ids by variant."""

    def same(r: RunRecord) -> str:
        return json.dumps(dataclasses.asdict(dataclasses.replace(r, git_sha="")), sort_keys=True)

    held = {same(r) for r in registry.records(FAMILY_ID)}
    ids, appended = {}, 0
    for record in records:
        key = f"{record.hypothesis_id}:{record.variant}"
        if same(record) in held:
            ids[key] = "already in the registry"
            continue
        ids[key] = registry.append(record)
        appended += 1
    return ids, appended


def freeze() -> tuple[Registry, dict, str]:
    boot = bootstrap_settings()
    spec = build_spec(load_country_scheme(COUNTRY), boot)
    registry = default_registry(COUNTRY)
    return registry, boot, registry.freeze_family(FAMILY_ID, spec)


def run_study(out_dir: Path | None = None) -> dict:
    """Freeze, then run the family on Japan's development data and write its private outputs."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    registry, boot, spec_hash = freeze()  # before any outcome is read
    builds = {**store_builds(), "jrdb_pre_race": _build("jrdb_pre_race")}
    try:
        data = read_development()
        frame = study_frame(data, _board(data))
        del data
        tables, coefficients, failures = compute(frame, boot)
    except Exception as exc:  # a failed read fails every table: each a registry record
        error = f"{type(exc).__name__}: {exc}"
        tables, coefficients = {}, {}
        failures = {key: error for key in [("E2", WIN), *(("E45", p) for p in E45_POOLS)]}
    git_sha = _git_sha()
    records = run_records(tables, git_sha=git_sha, data_builds=builds, coefficients=coefficients)
    records += _failed_records(failures, git_sha, builds)
    ids, appended = _append_new(registry, records)
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "builds": ", ".join(builds.values()),
        "records": f"{appended} appended; ids by variant: {json.dumps(ids)}",
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    written = write_outputs(
        tables, out_dir=out_dir, coefficients=coefficients, about=about, failures=failures
    )
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
    digest = hashlib.sha256()
    with open(paths.processed_dir("jra") / f"{name}.parquet", "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"processed/jra/{name}.parquet@{digest.hexdigest()[:16]}"


def _board(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    races = data["races"]
    con = duckdb.connect(":memory:")
    try:
        con.register("jrdb_pre_race", _arrow(data["jrdb_pre_race"]))
        return jrdb_board(
            con,
            races["race_id"].astype(str).tolist(),
            rules=CONFIG["board"],
            decision_utc=post_times(races),
        )
    finally:
        con.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.board_vs_final"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="freeze the family's spec only")
    mode.add_argument("--run", action="store_true", help="freeze, then run the family")
    parser.add_argument("--out", type=Path, default=None, help="output directory, outside the repo")
    args = parser.parse_args(argv)
    if args.freeze:
        _, _, spec_hash = freeze()
        print(json.dumps({"family_id": FAMILY_ID, "spec_hash": spec_hash}, indent=2))
    else:
        print(json.dumps(run_study(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
