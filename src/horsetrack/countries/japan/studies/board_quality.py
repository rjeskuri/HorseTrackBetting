"""Board quality by horizon and the price-forecast EV bias (#86; 08 §4 E10, E18, E21; 08 R5).

Diagnostics, nothing adopted. Inputs are long frames, one row per (race, runner, capture):
race_id, horse_id, minutes_to_post (to the scheduled post at the row's availability; negative
after it), odds (decimal, stake included; cleaned by `core.pricing.clean_odds` inside the de-vigs
and here), is_final (the rows of the race's final board, docs/specs/board-at-t.md), plus won
(`board_vs_final`) or pool (the drift functions).

- The board at tau, per race (per race and pool for the drift), is the non-final capture with the
  smallest minutes_to_post at or above tau: the latest one available by post - tau. A race with no
  capture that early has no board at tau.
- E18, `board_vs_final`: per race and tau, the winner log loss of the de-vigged board minus that of
  the de-vigged final, one de-vig (#77's, or any `devig(odds, race_id)`) for both. A dead heat
  scores each of its k co-winners at 1/k; a race without a final, without a winner (void) or with
  an unpriced winner on the final is left out of every tau, and one whose winner is unpriced or
  missing on the board at tau is left out at that tau; each is counted in `attrs`.
  `board_quality_table` adds D7's meeting-clustered bounds (`core.eval.bootstrap`).
- E21: given the board, ln O_F is normal with sd sigma, and the final price is efficient
  (p = rate / O_F). Multiplying the two conditional expectations, p_hat * O_hat, overstates
  E[p * O_F] = rate by rate * (exp(sigma^2) - 1) (`spurious_ev_bias`), which is why a gate never
  multiplies them (08 R5). `simulate_spurious_ev_bias` checks the formula by Monte Carlo.
- E10: `drift_sd_by_tau` and `drift_sd_by_segment` give the sample SD of ln(O_final / O_tau) over
  the runners priced on both boards; `spurious_ev_table` turns that SD into the E21 bias per pool.

The study (`run_study`; CLI `python -m horsetrack.countries.japan.studies.board_quality --run
[--out DIR]`) is a thin I/O layer over the pure `compute_study`. Boards at tau come from #80's
as-of view (`core.store.asof.read_asof`, availability.yaml's rule, label columns dropped) of each
development day's compacted odds log (`logger/compact.py`); the final board from the raw log by the
final-state rule (`board.snapshot.final_board`: compaction drops `source_update_utc` and
`page_state`, which the rule reads); post times from `board.snapshot.post_times_for`; winners from
the store's runners (holdout days dropped by `read_store_table`; the holdout is never opened).
The power de-vig's beta is fitted out of sample (POWER_FIT, `fit_power_window`): on the store's
development races of the D9 fold whose test window holds the earliest logged day, never on the
boards it scores; its value, fit window and races go into the record's params and the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.registry import RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.pricing import clean_odds
from horsetrack.core.store.asof import load_availability, read_asof
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.logger.compact import day_file
from horsetrack.countries.japan.models.v0.store import read_store_table
from horsetrack.countries.japan.studies.calibration import (
    Study,
    Variant,
    _markdown_table,
    _metric,
    development_frame,
    devig_normalised,
    devig_power,
    devig_shin,
    fit_power_beta,
    read_development,
    store_builds,
)

COUNTRY = "japan"


def _grid() -> list[float]:
    """D2's decision grid (config/decision.yaml), minutes before post."""
    return [float(m) for m in load_country_yaml(COUNTRY, "decision")["grid_min"]]


# --------------------------------------------------------------------------- boards


def _board_at(frame: pd.DataFrame, tau: float, keys: list[str]) -> pd.DataFrame:
    """Per `keys` group, the non-final rows at the smallest minutes_to_post at or above tau."""
    mtp = frame["minutes_to_post"]
    early = frame.loc[~frame["is_final"].astype(bool) & (mtp >= tau)]
    latest = early.groupby(keys, sort=False)["minutes_to_post"].transform("min")
    return early.loc[early["minutes_to_post"] == latest]


def _one_board(rows: pd.DataFrame, keys: list[str], what: str) -> None:
    if rows.duplicated([*keys, "horse_id"]).any():
        raise ValueError(f"{what} lists a runner twice: one capture per board, one row per runner")


def _finals(frame: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, int]:
    """The final rows of the groups that have them, and the races that have none."""
    final = frame.loc[frame["is_final"].astype(bool)]
    _one_board(final, keys, "a final board")
    return final, frame["race_id"].nunique() - final["race_id"].nunique()


def _winner_loss(board: pd.DataFrame, devig, k: pd.Series) -> tuple[pd.Series, int]:
    """Per race, -(1/k) sum ln p over its k winners (k from the final), and the races whose winner
    is unpriced or missing on this board (left out)."""
    p = pd.Series(devig(board["odds"], board["race_id"]), index=board.index).to_numpy(dtype=float)
    winner = board["won"].to_numpy(dtype=bool)
    ok = winner & np.isfinite(p) & (p > 0)
    races = board["race_id"].to_numpy()[ok]
    loss = pd.Series(-np.log(p[ok])).groupby(races).agg(["mean", "size"])
    full = loss["size"] == k.reindex(loss.index)
    seen = board["race_id"].nunique()
    return loss.loc[full, "mean"], seen - int(full.sum())


def board_vs_final(frame: pd.DataFrame, *, taus: list[float], devig) -> pd.DataFrame:
    """One row per (race, tau): race_id, tau_min, logloss_board, logloss_final and gap (board
    minus final), the same `devig(odds, race_id)` on both boards. `attrs` counts the races left
    out: n_races_without_final, n_void_races, n_unpriced_winner_final, and per tau
    n_no_board_by_tau and n_unpriced_winner_by_tau."""
    final, without_final = _finals(frame, ["race_id"])
    k = final.groupby("race_id")["won"].sum()
    void = k.index[k == 0]
    k = k[k > 0]
    ll_final, unpriced_final = _winner_loss(final.loc[final["race_id"].isin(k.index)], devig, k)
    scored = frame.loc[frame["race_id"].isin(ll_final.index)]
    parts, no_board, unpriced = [], {}, {}
    for tau in taus:
        board = _board_at(scored, float(tau), ["race_id"])
        _one_board(board, ["race_id"], f"the board at T-{tau:g}")
        ll_board, unpriced[float(tau)] = _winner_loss(board, devig, k)
        no_board[float(tau)] = len(ll_final) - board["race_id"].nunique()
        parts.append(
            pd.DataFrame(
                {
                    "race_id": ll_board.index,
                    "tau_min": float(tau),
                    "logloss_board": ll_board.to_numpy(),
                    "logloss_final": ll_final.reindex(ll_board.index).to_numpy(),
                }
            )
        )
    columns = ["race_id", "tau_min", "logloss_board", "logloss_final"]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    out["gap"] = out["logloss_board"] - out["logloss_final"]
    out.attrs = {
        "n_races_without_final": int(without_final),
        "n_void_races": len(void),
        "n_unpriced_winner_final": int(unpriced_final),
        "n_no_board_by_tau": no_board,
        "n_unpriced_winner_by_tau": unpriced,
    }
    return out


def board_quality_table(
    gaps: pd.DataFrame, clusters: pd.Series, *, n_boot: int, level: float, seed: int
) -> pd.DataFrame:
    """Per tau in `gaps`: the mean gap, its one-sided bounds at `level` resampling whole clusters
    (`clusters`: race id -> meeting, D7), and the races and clusters behind it."""
    rows = []
    for tau, part in gaps.groupby("tau_min", sort=False):
        diff = part.set_index("race_id")["gap"]
        iv = paired_bootstrap(diff, clusters, n_boot=n_boot, level=level, seed=seed)
        rows.append(
            {"tau_min": tau, "mean": iv.mean, "lo": iv.lo, "hi": iv.hi}
            | {"n_races": len(diff), "n_clusters": iv.n_clusters}
        )
    return pd.DataFrame(rows, columns=["tau_min", "mean", "lo", "hi", "n_races", "n_clusters"])


# --------------------------------------------------------------------------- the EV bias


def spurious_ev_bias(sigma: float, rate: float) -> float:
    """rate * (exp(sigma^2) - 1): how far p_hat * O_hat overstates E[p * O_F] (E21)."""
    return float(rate) * math.expm1(float(sigma) ** 2)


def simulate_spurious_ev_bias(
    sigma: float, rate: float, *, n: int, seed: int
) -> tuple[float, float]:
    """Monte Carlo of E21: n draws of ln O_F ~ N(0, sigma^2), p = rate / O_F. Returns p_hat * O_hat
    minus the mean of p * O_F (the true expected return), and its delta-method standard error."""
    x = np.random.default_rng(seed).normal(0.0, float(sigma), size=int(n))
    o = np.exp(x)
    p = float(rate) / o
    p_hat, o_hat = p.mean(), o.mean()
    influence = o_hat * (p - p_hat) + p_hat * (o - o_hat)
    return float(p_hat * o_hat - (p * o).mean()), float(influence.std(ddof=1) / math.sqrt(n))


# --------------------------------------------------------------------------- drift


def _drift(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    keys = ["race_id", "pool"]
    final, _ = _finals(frame, keys)
    final = final[[*keys, "horse_id", "odds"]].rename(columns={"odds": "odds_final"})
    rows = []
    for tau in _grid():
        board = _board_at(frame, tau, keys)
        _one_board(board, keys, f"the board at T-{tau:g}")
        both = board.merge(final, on=[*keys, "horse_id"], how="inner")
        ratio = np.log(clean_odds(both["odds_final"].to_numpy()) / clean_odds(both["odds"]))
        both = both.assign(log_ratio=ratio).dropna(subset=["log_ratio", *by])
        stats = both.groupby([*by, "pool"], sort=True)["log_ratio"].agg(["std", "size"])
        rows.append(stats.reset_index().assign(tau_min=tau))
    table = pd.concat(rows, ignore_index=True).rename(columns={"std": "sd", "size": "n"})
    full = pd.MultiIndex.from_product(
        [*(sorted(frame[c].dropna().unique()) for c in [*by, "pool"]), _grid()],
        names=[*by, "pool", "tau_min"],
    )
    table = table.set_index([*by, "pool", "tau_min"]).reindex(full).reset_index()
    return table.assign(n=table["n"].fillna(0).astype(int))[[*by, "pool", "tau_min", "sd", "n"]]


def drift_sd_by_tau(frame: pd.DataFrame) -> pd.DataFrame:
    """Per pool and D2 grid tau: the sample SD of ln(O_final / O_tau) over the runners priced on
    both boards (sd; NaN under two runners) and their number (n)."""
    return _drift(frame, [])


def drift_sd_by_segment(frame: pd.DataFrame, segment: str) -> pd.DataFrame:
    """`drift_sd_by_tau` per value of `frame[segment]` (constant within a race), in a column named
    `segment`; rows with a missing value are left out."""
    varies = frame.groupby("race_id")[segment].nunique(dropna=False) > 1
    if varies.any():
        raise ValueError(f"{segment!r} varies within {int(varies.sum())} race(s)")
    return _drift(frame.rename(columns={segment: "segment"}), ["segment"])


def spurious_ev_table(drift: pd.DataFrame, *, rates: dict[str, float]) -> pd.DataFrame:
    """The drift table with each pool's payout `rate` and its `spurious_ev`, rate *
    (exp(sd^2) - 1)."""
    rate = drift["pool"].map(rates).astype(float)
    return drift.assign(rate=rate, spurious_ev=rate * np.expm1(drift["sd"].astype(float) ** 2))


# =========================================================================== the study

TICKET = 86
SOURCE = "jra"  # the Mac's JRA odds logs and the store's JRA results (netkeiba); no JRDB, JRA-VAN
FAMILY_ID = "japan-board-quality-by-horizon"
BOOTSTRAP_SEED = 86  # reproducibility only; scoring.yaml signs no seed
SIM_DRAWS = 200_000
IN_SAMPLE = "in_sample:logged_development_days"
POOLS = (snapshot.WIN, snapshot.PLACE)
VIEW_COLUMNS = ["race_id", "venue", "race_no", "pool", "combination", "snapshot_utc"]
VIEW_COLUMNS += ["fetched_utc", "odds", "investment", "status_raw"]
VIEW_COLUMNS += ["dividend", "is_refund", "is_final_capture"]  # label columns: the view drops them
RUNNER_COLUMNS = ["race_id", "race_date", "horse_no", "finish_pos", "is_scratched"]
SEGMENTS = {
    "late_card": "in: race_no 10, 11 or 12; out: every earlier race",
    "low_turnover": "in: the race's latest logged WIN pool total before the post is under the "
    "median of the run's races that have one; out: at or above it; neither without a total",
}
POWER_FIT = (
    "#77's fit_power_beta (conditional logit on ln(1/O), all races) on the training rows of the D9 "
    "fold (config/folds.yaml: expanding, 1-week embargo) whose test window holds the earliest "
    "logged day: the store's final Win prices of development race days from 2014-06-07 (D10), "
    "#77's development_frame (void races, scratched and unpriced runners out), all strictly "
    "before the earliest logged day; a diagnostic candidate until the owner picks the de-vig"
)
DEVIGS = {
    "normalised": "q_i = (1/O_i) / sum_j (1/O_j)",
    "power": "p_i proportional to (1/O_i)^beta, the same beta for every board, fitted out of "
    "sample by POWER_FIT",
    "shin": "Shin (1993), z solved per race (#77)",
}


def bootstrap_settings() -> dict:
    """D7's bootstrap from config/scoring.yaml, with the study's fixed seed."""
    boot = load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["cluster_unit"] != "meeting" or boot["sided"] != "one-sided":
        raise ValueError(f"this study implements meeting clusters, one-sided; scoring.yaml: {boot}")
    return {"cluster_unit": "meeting", "sided": "one-sided", "level": float(boot["level"])} | {
        "n_boot": int(boot["n_boot"]),
        "seed": BOOTSTRAP_SEED,
    }


def payout_rates() -> dict[str, float]:
    """1 - takeout per pool read (jra.yaml)."""
    pools = load_source(COUNTRY, SOURCE).pools
    return {pool: 1.0 - float(pools[pool].takeout) for pool in POOLS}


def build_spec() -> dict:
    return {
        "family_id": FAMILY_ID,
        "ticket": TICKET,
        "hypotheses": {
            "E18": "per race and tau, winner log loss of the de-vigged board at T-tau minus that "
            "of the de-vigged final board, meeting-clustered bounds",
            "E10": "SD of ln(O_final / O_tau) by pool, tau and segment",
            "E21": "spurious EV rate * (exp(sd^2) - 1) by pool and tau, with a Monte Carlo check",
        },
        "grid_min": _grid(),
        "devigs": DEVIGS,
        "power_fit": POWER_FIT,
        "segments": SEGMENTS,
        "rates": payout_rates(),
        "bootstrap": bootstrap_settings(),
        "simulation_draws": SIM_DRAWS,
        "adoption": "none: diagnostics; the owner picks the de-vig",
        "sources": [SOURCE],
    }


# --------------------------------------------------------------------------- reading


def _logged_days() -> list[date]:
    root = paths.data_dir() / "odds_snapshots" / SOURCE
    days = [p.name for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    return sorted(date.fromisoformat(d) for d in days if any((root / d).glob("*.jsonl.gz")))


def _quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _boards_at(day: date, posts: dict[str, dict[int, datetime]]) -> pd.DataFrame:
    """The day's WIN and PLACE rows on #80's as-of view at its last post, with minutes to post."""
    con = duckdb.connect(":memory:")
    con.execute(
        f"CREATE VIEW odds_timeseries AS SELECT {', '.join(VIEW_COLUMNS)} FROM read_parquet("
        f"{_quoted(day_file(day.isoformat(), SOURCE))}) WHERE pool IN ('WIN', 'PLACE')"
    )
    last = max(t for times in posts.values() for t in times.values())
    view = read_asof(con, "odds_timeseries", last, rules=load_availability(COUNTRY))
    con.close()
    post = pd.Series(
        [posts.get(v, {}).get(n) for v, n in zip(view["venue"], view["race_no"], strict=True)],
        index=view.index,
        dtype="datetime64[us, UTC]",
    )
    available = view["fetched_utc"].fillna(view["snapshot_utc"])
    return view.assign(minutes_to_post=(post - available) / timedelta(minutes=1))


def _final_rows(day: date, posts: dict[str, dict[int, datetime]]) -> pd.DataFrame:
    """Each race's final WIN and PLACE board by the final-state rule, from the raw logs."""
    out = []
    for log in sorted(snapshot.jra_day_dir(day).glob("*.jsonl.gz")):
        by_race: dict[str, list[dict]] = {}
        for member in odds_log.iter_members(log):
            for row in member:
                if row.get("pool") in POOLS:
                    by_race.setdefault(row.get("race_id"), []).append(row)
        for race_id, rows in by_race.items():
            board = snapshot.final_board(rows, race_id)
            if board is None:
                continue
            post = posts.get(board.context().venue, {}).get(board.context().race_no)
            mtp = (post - board.at_utc) / timedelta(minutes=1) if post else np.nan
            prices = [(snapshot.WIN, h, o) for h, o in board.win.items()]
            prices += [(snapshot.PLACE, h, lo) for h, (lo, _) in board.place.items()]
            out += [(race_id, pool, h, o, mtp) for pool, h, o in prices]
    columns = ["race_id", "pool", "horse_id", "odds", "minutes_to_post"]
    return pd.DataFrame(out, columns=columns).assign(is_final=True)


def read_day(day: date) -> tuple[pd.DataFrame, dict]:
    """The day's long frame (boards at every capture, then the final boards) and its counts."""
    venues = sorted(
        p.name[: -len(".jsonl.gz")] for p in snapshot.jra_day_dir(day).glob("*.jsonl.gz")
    )
    posts = {v: snapshot.post_times_for(day, v) for v in venues}
    if not any(posts.values()):
        return pd.DataFrame(), {"days_without_post_times": 1}
    view = _boards_at(day, posts)
    totals = view.loc[
        view["combination"].eq(odds_log.POOL_TOTAL)
        & view["pool"].eq(snapshot.WIN)
        & (view["minutes_to_post"] >= 0)
        & view["investment"].notna()
    ]
    total = totals.sort_values("minutes_to_post").groupby("race_id")["investment"].first()
    prices = view.loc[view["combination"].ne(odds_log.POOL_TOTAL)]
    horse = pd.to_numeric(prices["combination"], errors="coerce")
    scratched = prices["status_raw"].eq(snapshot.SCRATCHED)
    boards = pd.DataFrame(
        {
            "race_id": prices["race_id"],
            "pool": prices["pool"],
            "horse_id": horse,
            "odds": prices["odds"].where(~scratched),
            "minutes_to_post": prices["minutes_to_post"],
            "is_final": False,
        }
    )
    counts = {"view_rows_without_post_time": int(boards["minutes_to_post"].isna().sum())}
    boards = boards.dropna(subset=["minutes_to_post", "horse_id"])
    frame = pd.concat([boards, _final_rows(day, posts)], ignore_index=True)
    frame["horse_id"] = frame["horse_id"].astype(int)
    return frame.assign(win_pool_total=frame["race_id"].map(total)), counts


def read_results(days: Sequence[date]) -> pd.Series:
    """won (bool) by (race_id, horse_no) for the days' races in the store; holdout days dropped."""
    table = read_store_table(
        "runners", columns=RUNNER_COLUMNS, start=min(days), end=max(days), touches_price=False
    )
    runners = table.to_pandas()
    runners = runners.loc[~runners["is_scratched"].eq(True).to_numpy(dtype=bool, na_value=False)]
    won = runners["finish_pos"].eq(1).fillna(False).astype(bool)
    return pd.Series(
        won.to_numpy(), index=pd.MultiIndex.from_frame(runners[["race_id", "horse_no"]])
    )


def _sha16(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def fit_power_window(fit_rows: pd.DataFrame, first_day: date, scheme: FoldScheme) -> dict:
    """POWER_FIT on `fit_rows` (#77's development_frame: race_id, race_date, odds, won): beta,
    its SE and races, and the training window of the scheme's fold whose test window holds
    `first_day`; every row fitted is dated on or before that fold's train_end < `first_day`."""
    dates = pd.concat(
        [pd.Series(fit_rows["race_date"].to_numpy()), pd.Series([pd.Timestamp(first_day)])],
        ignore_index=True,
    )
    folds = walk_forward(dates, scheme=scheme)
    fold = next((f for f in folds if f.test_start <= first_day <= f.test_end), None)
    if fold is None:
        raise ValueError(f"no D9 fold's test window holds {first_day}")
    train = fit_rows.iloc[fold.train_idx[fold.train_idx < len(fit_rows)]]
    if train.empty:
        raise ValueError(f"no development race on or before {fold.train_end} to fit beta on")
    fit = fit_power_beta(train["odds"], train["race_id"], train["won"])
    return {"beta": fit.beta, "beta_se": fit.se, "fit_races": fit.n_races} | {
        "fit_from": str(pd.Timestamp(train["race_date"].min()).date()),
        "fit_to": str(pd.Timestamp(train["race_date"].max()).date()),
        "fit_train_end": str(fold.train_end),
        "fold_test_start": str(fold.test_start),
    }


def power_beta_out_of_sample(first_day: date) -> dict:
    """POWER_FIT on the store (#77's development reads: holdout days dropped, never opened)."""
    runners, races = read_development()
    frame, _ = development_frame(runners, races)
    del runners, races
    return fit_power_window(
        frame[["race_id", "race_date", "odds", "won"]], first_day, load_country_scheme(COUNTRY)
    )


# --------------------------------------------------------------------------- the pure study


def with_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame with the race's meeting (venue x day, D7) and SEGMENTS' values."""
    parts = frame["race_id"].str.split(":", expand=True)
    total = frame.drop_duplicates("race_id").set_index("race_id")["win_pool_total"]
    low = total.dropna() < total.dropna().median()
    return frame.assign(
        meeting=parts[1] + ":" + parts[2],
        late_card=np.where(parts[3].astype(int) >= 10, "in", "out"),
        low_turnover=frame["race_id"].map(low.map({True: "in", False: "out"})),
    )


def _board_quality(
    frame: pd.DataFrame, spec: dict, name: str, power: dict | None
) -> tuple[list, dict]:
    has_result = frame["won"].notna().groupby(frame["race_id"]).transform("any")
    win = frame.loc[frame["pool"].eq(snapshot.WIN) & has_result]
    if win.empty:
        raise ValueError("no logged race with a final board has a stored result")
    win = win.assign(won=win["won"].fillna(False).astype(int))  # a logged runner the store lacks
    params = {}
    if name == "power":
        if power is None:
            raise ValueError("no out-of-sample beta: POWER_FIT was not run")
        params = {"beta": power["beta"], "beta_se": power["beta_se"]}
        devig = lambda o, r: devig_power(o, r, power["beta"])  # noqa: E731
    else:
        devig = {"normalised": devig_normalised, "shin": devig_shin}[name]
    gaps = board_vs_final(win, taus=spec["grid_min"], devig=devig)
    clusters = frame.drop_duplicates("race_id").set_index("race_id")["meeting"]
    boot = spec["bootstrap"]
    table = board_quality_table(
        gaps, clusters, n_boot=boot["n_boot"], level=boot["level"], seed=boot["seed"]
    )
    metrics = [
        _metric(f"T-{r.tau_min:g}", "mean_gap_ll_board_minus_final", r.mean, r.n_races, r.lo, r.hi)
        for r in table.itertuples()
    ]
    rows = [{"de-vig": name, **params} | r for r in table.to_dict("records")]
    return metrics, {"board_quality": rows, "left_out": [{"de-vig": name, **gaps.attrs}]}


def _ev(frame: pd.DataFrame, spec: dict) -> tuple[list, dict]:
    table = spurious_ev_table(drift_sd_by_tau(frame), rates=spec["rates"])
    metrics, rows = [], []
    for i, r in enumerate(table.itertuples()):
        row = {"pool": r.pool, "tau_min": r.tau_min, "n": r.n, "sd": r.sd, "rate": r.rate}
        row["spurious_ev"] = r.spurious_ev
        if r.n >= 2:
            fold = f"{r.pool}:T-{r.tau_min:g}"
            metrics += [_metric(fold, "sd_ln_final_over_tau", r.sd, r.n)]
            metrics += [_metric(fold, "spurious_ev", r.spurious_ev, r.n)]
            est, se = simulate_spurious_ev_bias(r.sd, r.rate, n=SIM_DRAWS, seed=TICKET + i)
            row |= {"simulated": est, "sim se": se, "|z|": abs(est - r.spurious_ev) / se}
        rows.append(row)
    return metrics, {"ev": rows}


def _segment(frame: pd.DataFrame, name: str) -> tuple[list, dict]:
    if frame[name].isna().all():
        if name == "low_turnover":
            raise ValueError(
                "no race has a WIN pool total logged before the post: the compacted logs' WIN "
                "totals all sit after it (pre-post per-race Win sales: #278)"
            )
        raise ValueError(f"no race has a value for {name}")
    table = drift_sd_by_segment(frame, name)
    metrics = [
        _metric(f"{name}={r.segment}:{r.pool}:T-{r.tau_min:g}", "sd_ln_final_over_tau", r.sd, r.n)
        for r in table.itertuples()
        if r.n >= 2
    ]
    return metrics, {
        "segments": [r | {"segment": f"{name}={r['segment']}"} for r in table.to_dict("records")]
    }


def compute_study(frame: pd.DataFrame, spec: dict, power: dict | None = None) -> Study:
    """E18 under each de-vig, E10 by pool and tau and by segment, and E21 with its simulation
    check, on `read_day`'s frames with `won` joined and `with_segments` applied. `power` is
    `fit_power_window`'s out-of-sample beta with its fit window; the power variant fails
    without it. Pure: no I/O."""
    study = Study(counts={})
    for name in DEVIGS:
        params = ({"power_fit": POWER_FIT} | (power or {})) if name == "power" else {}
        variant = Variant("E18", f"board_vs_final:{name}", IN_SAMPLE, {"devig": name}, params)
        study.attempt(variant, lambda name=name: _board_quality(frame, spec, name, power))
    variant = Variant("E21", "spurious_ev:by_pool_and_tau", IN_SAMPLE, {"drift": "sd"}, {})
    study.attempt(variant, lambda: _ev(frame, spec))
    for name, definition in SEGMENTS.items():
        params = {"definition": definition}
        variant = Variant("E10", f"drift_sd:{name}", IN_SAMPLE, {"segment": name}, params)
        study.attempt(variant, lambda n=name: _segment(frame, n))
    return study


# --------------------------------------------------------------------------- report and runner

SECTIONS = (
    (
        "board_quality",
        "E18: winner log loss of the board at T-tau minus the final's (Win pool)",
        "mean, lo and hi in nats per race; lo and hi are the (1 - level) and level quantiles of the"
        " bootstrap over meetings (D7: one-sided). Positive: the board is worse than the final.",
    ),
    ("left_out", "E18: races left out", ""),
    (
        "ev",
        "E10 and E21: drift SD of ln(O_final / O_tau) and the spurious EV, by pool and tau",
        "spurious_ev = rate * (exp(sd^2) - 1); simulated is the Monte Carlo of p_hat * O_hat - "
        "E[p * O_F] at that sd, with its SE. PLACE uses the low end of the range.",
    ),
    ("segments", "E10: drift SD by segment", ""),
)


def render_report(study: Study, spec: dict, about: dict) -> str:
    """Aggregate tables only, never a runner or race row."""
    lines = [f"# Board quality by horizon and the EV bias ({COUNTRY}, #{TICKET})", ""]
    lines += [f"- {key}: {value}" for key, value in about.items()]
    lines += ["- development days only: holdout days are never read and the holdout is not opened"]
    lines += [
        "",
        "## Data",
        "",
        _markdown_table([{"count": k, "n": v} for k, v in study.counts.items()]),
    ]
    for key, title, note in SECTIONS:
        rows = study.tables.get(key)
        lines += ["", f"## {title}", ""] + ([note, ""] if note else [])
        lines.append(_markdown_table(rows) if rows else "_Not computed: see the variants below._")
    variants = [
        {"E": v.hypothesis_id, "variant": v.name, "status": "failed" if v.error else "scored"}
        | ({"error": v.error} if v.error else {})
        for v in study.variants
    ]
    lines += ["", "## Variants (one registry record each)", "", _markdown_table(variants), ""]
    return "\n".join(lines)


def run_study(out_dir: Path | None = None) -> dict:
    """Read every logged development day that is compacted, compute the study, append one
    registry record per variant (failures included) and write the report."""
    out_dir = paths.reports_dir() if out_dir is None else paths.outside_repo(out_dir, what="out")
    spec = build_spec()
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    counts = {"logged_days": 0, "holdout_days_skipped": 0, "days_not_compacted": 0}
    frames, builds = [], {}
    for day in _logged_days():
        counts["logged_days"] += 1
        if is_holdout_day(COUNTRY, day):
            counts["holdout_days_skipped"] += 1
            continue
        if not day_file(day.isoformat(), SOURCE).is_file():
            counts["days_not_compacted"] += 1
            continue
        frame, day_counts = read_day(day)
        for key, n in day_counts.items():
            counts[key] = counts.get(key, 0) + n
        if not frame.empty:
            frames.append(frame)
            builds[f"odds_timeseries_{day}"] = f"odds_timeseries/{day}@{_sha16(day_file(str(day)))}"
            for log in sorted(snapshot.jra_day_dir(day).glob("*.jsonl.gz")):
                builds[f"odds_log_{day}_{log.name}"] = f"{log.name}@{_sha16(log)}"
    if not frames:
        raise RuntimeError(f"no compacted development day with post times to read: {counts}")
    frame = pd.concat(frames, ignore_index=True)
    days = sorted({date.fromisoformat(r.split(":")[1]) for r in frame["race_id"].unique()})
    won = read_results(days)
    builds |= store_builds()
    key = pd.MultiIndex.from_arrays([frame["race_id"], frame["horse_id"]])
    frame = with_segments(frame.assign(won=won.reindex(key).to_numpy()))
    races = frame.drop_duplicates("race_id")
    with_final = frame.loc[frame["is_final"], "race_id"].unique()
    counts |= {"race_days": len(days), "races_logged": len(races)}
    counts |= {"races_with_a_final_board": len(with_final)}
    counts |= {"races_with_a_stored_result": int(frame.groupby("race_id")["won"].any().sum())}

    try:
        power, power_error = power_beta_out_of_sample(min(days)), None
    except ValueError as exc:  # the power variant then fails and is recorded
        power, power_error = None, str(exc)
    study = compute_study(frame, spec, power)
    study.counts = counts
    git_sha = _git_sha()
    registry = default_registry(COUNTRY)
    existing = registry.records(FAMILY_ID)
    records = [_record(v, git_sha, builds) for v in study.variants]
    appended = [registry.append(r) for r in records if r not in existing]
    about = {
        "family": f"`{FAMILY_ID}`, spec sha256 `{spec_hash}`",
        "git": git_sha,
        "days": ", ".join(str(d) for d in days),
        "power beta": (
            f"{power['beta']:.4f} (SE {power['beta_se']:.4f}) on {power['fit_races']} development "
            f"races, {power['fit_from']} to {power['fit_to']} (the D9 fold of "
            f"{power['fold_test_start']}, train_end {power['fit_train_end']}), out of sample; a "
            "diagnostic candidate pending the owner's de-vig pick"
            if power
            else f"not fitted: {power_error}"
        ),
        "written (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = write_report(
        out_dir / f"board-quality-{COUNTRY}-{spec_hash[:12]}.md",
        render_report(study, spec, about),
        sources=(SOURCE,),
    )
    return {
        "family_id": FAMILY_ID,
        "report_path": str(report),
        "counts": counts,
        "records_appended": len(appended),
        "failed_variants": {v.name: v.error for v in study.variants if v.error},
        "power_beta": power,
        "tables": {k: study.tables.get(k) for k in ("board_quality", "ev")},
    }


def _record(variant: Variant, git_sha: str, builds: dict[str, str]) -> RunRecord:
    return RunRecord(
        hypothesis_id=variant.hypothesis_id,
        family_id=FAMILY_ID,
        variant=variant.name,
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=builds,
        sources=(SOURCE,),
        fold_scheme=variant.fold_scheme,
        components=variant.components,
        params=variant.params,
        metrics=variant.metrics,
        status="failed" if variant.error else "scored",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.studies.board_quality"
    )
    parser.add_argument("--run", action="store_true", required=True, help="run the study")
    parser.add_argument("--out", type=Path, default=None, help="report directory, outside the repo")
    args = parser.parse_args(argv)
    print(json.dumps(run_study(args.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
