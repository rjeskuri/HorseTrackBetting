"""The v0 nowcast: a fitted drift of the final price by pool, odds band and horizon (#116, D8).

Target per (pool, combination): y = ln O_final - ln O_t on the low figure, for WIN, PLACE and
QUINELLA_PLACE, at t = post - tau for each tau of the decision grid. Cells are (pool, tau, band),
band = band_of(O_t): mu = n * ybar / (n + prior_n) and sigma^2 = (sum (y - ybar)^2 +
prior_n * sigma0^2) / (n + prior_n), sigma0 the population SD of every y of that (pool, tau); an
empty cell has mu = 0 and sigma = sigma0.

Draws: ln O = ln O_t + mu + sigma * eps with caller-supplied shocks per runner (a Wide pair uses
(eps_a + eps_b) / sqrt 2); high figures move by the low figure's factor; nothing goes below 1.0;
WIN draws keep the board's sum of 1/O (see `_keep_payout` for when the floor binds); tau <= 0
gives the board itself.

Memory: a log is read member by member, keeping per race only the latest WIN, PLACE and Wide row
per combination available by each grid time, and the rows of captures that show a final price.

CLI: `python -m horsetrack.countries.japan.models.v0.nowcast --fit --day YYYY-MM-DD:VEN ...
[--out DIR]`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from horsetrack.core import config, paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan import odds_log
from horsetrack.countries.japan.board import snapshot
from horsetrack.countries.japan.board.snapshot import (
    DecisionGrid,
    RaceBoard,
    board_at,
    bucket_for,
    final_board,
    load_decision_grid,
    post_times_for,
)
from horsetrack.countries.japan.models.v0.ordering import _git_sha

WIN, PLACE, WIDE = snapshot.WIN, snapshot.PLACE, snapshot.WIDE
POOLS = (WIN, PLACE, WIDE)
COUNTRY = "japan"
CONFIG_STEM = "nowcast_v0"
COLUMNS = [
    "race_id",
    "day",
    "pool",
    "combination",
    "tau_min",
    "odds_t",
    "odds_final",
    "board_age_s",
]
FLOOR = 1.0


class HoldoutDayError(RuntimeError):
    """A holdout day was offered for training or scoring (D14: holdout days are fully locked)."""


def band_of(odds: float, bands: Sequence[float]) -> int:
    """The number of cut points at or below `odds`."""
    return bisect_right([float(b) for b in bands], float(odds))


# ---------------------------------------------------------------- training examples


def _collect(path: Path, times: Mapping[str, Mapping[int, datetime]]):
    """Stream the log once. Returns, per race, {tau: {board key: (available, row)}} with the latest
    row available by post - tau, and {snapshot_utc: rows} of the captures holding a final row."""
    latest = {race_id: {tau: {} for tau in taus} for race_id, taus in times.items()}
    finals: dict[str, dict[datetime, list[dict]]] = {race_id: {} for race_id in times}
    for member in odds_log.iter_members(path):
        groups: dict[tuple[str, datetime], list[dict]] = {}
        for row in member:
            race_id = row.get("race_id")
            if race_id not in times or row.get("pool") not in POOLS:
                continue
            key = snapshot._key(row)
            available = snapshot._available(row)
            if key is None or key[1] == odds_log.POOL_TOTAL or available is None:
                continue
            for tau, at in times[race_id].items():
                if available <= at:
                    slot = latest[race_id][tau]
                    current = slot.get(key)
                    if current is None or available >= current[0]:  # later row wins a tie
                        slot[key] = (available, row)
            snap = odds_log._instant(row.get("snapshot_utc"))
            if snap is not None:
                groups.setdefault((race_id, snap), []).append(row)
        for (race_id, snap), rows in groups.items():
            if any(snapshot._is_final(r) for r in rows):
                finals[race_id].setdefault(snap, []).extend(rows)
    return latest, finals


def _pairs(board: RaceBoard, final: RaceBoard):
    """(pool, combination, odds_t, odds_final) for each combination priced on both boards."""
    for h in sorted(board.win.keys() & final.win.keys()):
        yield WIN, str(h), board.win[h], final.win[h]
    for h in sorted(board.place.keys() & final.place.keys()):
        yield PLACE, str(h), board.place[h][0], final.place[h][0]
    for a, b in sorted(board.wide.keys() & final.wide.keys()):
        yield WIDE, f"{a},{b}", board.wide[(a, b)][0], final.wide[(a, b)][0]


def training_examples(
    race_date: date, venue: str, *, grid: DecisionGrid, max_board_age_s: Mapping[int, float]
) -> pd.DataFrame:
    """One row per (race, pool, combination, tau) of a development meeting.

    Raises HoldoutDayError on a holdout day. A board older than max_board_age_s[tau] at
    post - tau drops its examples; `attrs["dropped"]` counts them per tau, and `attrs["skipped"]`
    counts races without a final board and (per tau) without any board.
    """
    if is_holdout_day(race_date, COUNTRY):
        raise HoldoutDayError(f"{race_date.isoformat()} is a {COUNTRY} holdout day (D14)")
    missing = [tau for tau in grid.minutes if tau not in max_board_age_s]
    if missing:
        raise ValueError(f"max_board_age_s has no limit for tau {missing}")
    day = race_date.isoformat()
    posts = {
        f"jra:{day}:{venue}:{no:02d}": post for no, post in post_times_for(race_date, venue).items()
    }
    times = {
        race_id: {tau: post - timedelta(minutes=tau) for tau in grid.minutes}
        for race_id, post in posts.items()
    }
    latest, finals = _collect(paths.odds_snapshot_path("jra", day, venue), times)

    records = []
    dropped = dict.fromkeys(grid.minutes, 0)
    no_board = dict.fromkeys(grid.minutes, 0)
    no_final = 0
    for race_id in sorted(posts):
        post = posts[race_id]
        final_rows = [row for snap in sorted(finals[race_id]) for row in finals[race_id][snap]]
        final = final_board(final_rows, race_id, post_utc=post)
        if final is None:
            no_final += 1
            continue
        for tau, at in times[race_id].items():
            rows = [row for _, row in latest[race_id][tau].values()]
            board = board_at(rows, race_id, at, post_utc=post)
            if board is None:
                no_board[tau] += 1
                continue
            age = (at - board.capture_utc).total_seconds()
            pairs = [p for p in _pairs(board, final) if p[2] > 0 and p[3] > 0]
            if age > max_board_age_s[tau]:
                dropped[tau] += len(pairs)
                continue
            for pool, combination, odds_t, odds_final in pairs:
                records.append((race_id, day, pool, combination, tau, odds_t, odds_final, age))
    examples = pd.DataFrame.from_records(records, columns=COLUMNS)
    examples.attrs["dropped"] = dropped
    examples.attrs["skipped"] = {"no_final_board": no_final, "no_board": no_board}
    return examples


# ---------------------------------------------------------------- the model


@dataclass(frozen=True)
class DriftCell:
    pool: str
    tau_min: int
    band: int
    n: int
    mu: float
    sigma: float


@dataclass(frozen=True)
class PoolDraws:
    pool: str
    combinations: tuple[str, ...]
    odds: np.ndarray  # n draws x k combinations, low figure
    odds_hi: np.ndarray | None  # None for WIN; NaN where the board shows no high figure


def _log_ratio(examples: pd.DataFrame) -> np.ndarray:
    return np.log(examples["odds_final"].to_numpy(float)) - np.log(
        examples["odds_t"].to_numpy(float)
    )


def _keep_payout(raw: np.ndarray, total: float) -> np.ndarray:
    """WIN draws rescaled so each row's sum of 1/O is `total`, no price below FLOOR. Runners the
    rescale pushes below the floor are held at it and the rest rescaled to fill what is left; both
    hold whenever they can (total <= runners), and the floor wins when they cannot."""
    share, cap = 1.0 / raw, 1.0 / FLOOR  # implied shares; the floor caps each share
    fixed = np.zeros(share.shape, dtype=bool)
    scaled = share
    for _ in range(share.shape[1] + 1):  # each pass fixes at least one more runner, or stops
        free = np.where(fixed, 0.0, share).sum(axis=1, keepdims=True)
        budget = total - cap * fixed.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(fixed, cap, share * budget / free)
        over = (scaled > cap) & ~fixed
        if not over.any():
            break
        fixed |= over
    return 1.0 / scaled


@dataclass(frozen=True)
class NowcastV0:
    bands: tuple[float, ...]
    grid: DecisionGrid
    prior_n: float
    days: tuple[str, ...]
    sigma0: dict[tuple[str, int], float]
    cells: dict[tuple[str, int, int], DriftCell]
    fit_id: str = ""

    @classmethod
    def fit(
        cls,
        examples: pd.DataFrame,
        *,
        bands: Sequence[float],
        grid: DecisionGrid,
        prior_n: float,
    ) -> NowcastV0:
        if examples.empty:
            raise ValueError("no examples to fit")
        if prior_n < 0:
            raise ValueError(f"prior_n must be >= 0, got {prior_n}")
        bands = tuple(float(b) for b in bands)
        taus = examples["tau_min"].to_numpy(int)
        off_grid = sorted(set(taus.tolist()) - set(grid.minutes))
        if off_grid:
            raise ValueError(f"examples at tau {off_grid} are not on the grid {grid.minutes}")
        frame = pd.DataFrame(
            {
                "pool": examples["pool"].to_numpy(),
                "tau": taus,
                "band": np.searchsorted(bands, examples["odds_t"].to_numpy(float), side="right"),
                "y": _log_ratio(examples),
            }
        )
        sigma0 = {
            (str(p), int(t)): float(np.std(g["y"].to_numpy(), ddof=0))
            for (p, t), g in frame.groupby(["pool", "tau"])
        }
        cells = {}
        for (p, t, b), g in frame.groupby(["pool", "tau", "band"]):
            y = g["y"].to_numpy()
            n, ybar = len(y), float(y.mean())
            s0 = sigma0[(str(p), int(t))]
            mu = n * ybar / (n + prior_n)
            sigma = math.sqrt((float(((y - ybar) ** 2).sum()) + prior_n * s0**2) / (n + prior_n))
            cells[(str(p), int(t), int(b))] = DriftCell(str(p), int(t), int(b), n, mu, sigma)
        days = tuple(sorted({str(d)[:10] for d in examples["day"]}))
        model = cls(bands, grid, float(prior_n), days, sigma0, cells)
        digest = hashlib.sha256(model.to_json().encode()).hexdigest()[:12]
        return dataclasses.replace(model, fit_id=digest)

    def cell(self, pool: str, tau_min: int, band: int) -> DriftCell:
        """The fitted cell; an empty one (n = 0) falls back to mu = 0, sigma = sigma0."""
        found = self.cells.get((pool, int(tau_min), int(band)))
        if found is not None:
            return found
        if (pool, int(tau_min)) not in self.sigma0:
            raise KeyError(f"no training data for {pool} at tau {tau_min}")
        return DriftCell(pool, int(tau_min), int(band), 0, 0.0, self.sigma0[(pool, int(tau_min))])

    def _params(self, examples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(mu, sigma) of each example's cell."""
        keys = pd.DataFrame(
            {
                "pool": examples["pool"].to_numpy(),
                "tau": examples["tau_min"].to_numpy(int),
                "band": np.searchsorted(
                    self.bands, examples["odds_t"].to_numpy(float), side="right"
                ),
            }
        )
        mu, sigma = np.empty(len(keys)), np.empty(len(keys))
        for (p, t, b), idx in keys.groupby(["pool", "tau", "band"]).indices.items():
            c = self.cell(str(p), int(t), int(b))
            mu[idx], sigma[idx] = c.mu, c.sigma
        return mu, sigma

    def draws(
        self, board: RaceBoard, *, pool: str, tau_min: int, shocks: Mapping[int, np.ndarray]
    ) -> PoolDraws:
        """n draws of the final price of every combination of `pool` on `board`, as seen tau_min
        minutes before the post (a tau off the grid uses `bucket_for`'s horizon)."""
        if pool == WIN:
            keys, lo, hi = sorted(board.win), [board.win[h] for h in sorted(board.win)], None
        elif pool in (PLACE, WIDE):
            prices = board.place if pool == PLACE else board.wide
            keys = sorted(prices)
            lo = [prices[k][0] for k in keys]
            hi = np.array([np.nan if prices[k][1] is None else prices[k][1] for k in keys], float)
        else:
            raise ValueError(f"no nowcast for pool {pool!r}")
        lo = np.array(lo, dtype=float)
        lengths = {len(np.asarray(s)) for s in shocks.values()}
        if len(lengths) != 1:
            raise ValueError("shocks must be non-empty and the same length for every runner")
        n = lengths.pop()
        combinations = tuple(f"{k[0]},{k[1]}" if pool == WIDE else str(k) for k in keys)
        if tau_min <= 0:  # after the post the board is final
            odds = np.tile(lo, (n, 1))
            return PoolDraws(pool, combinations, odds, None if hi is None else np.tile(hi, (n, 1)))

        tau = bucket_for(tau_min, self.grid)
        eps = np.empty((n, len(keys)))
        for j, k in enumerate(keys):
            if pool == WIDE:
                eps[:, j] = (np.asarray(shocks[k[0]]) + np.asarray(shocks[k[1]])) / math.sqrt(2.0)
            else:
                eps[:, j] = np.asarray(shocks[k])
        cells = [self.cell(pool, tau, band_of(o, self.bands)) for o in lo]
        mu = np.array([c.mu for c in cells])
        sigma = np.array([c.sigma for c in cells])
        factor = np.exp(mu + sigma * eps)
        raw = lo * factor
        if pool == WIN:
            return PoolDraws(pool, combinations, _keep_payout(raw, float((1.0 / lo).sum())), None)
        odds = np.maximum(raw, FLOOR)
        return PoolDraws(pool, combinations, odds, np.maximum(hi * factor, odds))

    def to_json(self) -> str:
        return json.dumps(
            {
                "model": "nowcast_v0",
                "fit_id": self.fit_id,
                "bands": list(self.bands),
                "grid": {"minutes": list(self.grid.minutes), "primary": self.grid.primary},
                "prior_n": self.prior_n,
                "days": list(self.days),
                "sigma0": [
                    {"pool": p, "tau_min": t, "sigma0": s} for (p, t), s in self.sigma0.items()
                ],
                "cells": [dataclasses.asdict(c) for c in self.cells.values()],
            }
        )

    @classmethod
    def from_json(cls, s: str) -> NowcastV0:
        data = json.loads(s)
        grid = DecisionGrid(tuple(data["grid"]["minutes"]), data["grid"]["primary"])
        sigma0 = {(r["pool"], int(r["tau_min"])): float(r["sigma0"]) for r in data["sigma0"]}
        cells = {(c["pool"], c["tau_min"], c["band"]): DriftCell(**c) for c in data["cells"]}
        return cls(
            tuple(data["bands"]),
            grid,
            float(data["prior_n"]),
            tuple(data["days"]),
            sigma0,
            cells,
            data["fit_id"],
        )


# ---------------------------------------------------------------- scores


def log_predictive_density(model: NowcastV0, examples: pd.DataFrame) -> pd.Series:
    """log N(y; mu, sigma) of each example's final log share change."""
    mu, sigma = model._params(examples)
    return pd.Series(stats.norm.logpdf(_log_ratio(examples), mu, sigma), index=examples.index)


def pit(model: NowcastV0, examples: pd.DataFrame) -> pd.Series:
    """Phi((y - mu) / sigma) of each example."""
    mu, sigma = model._params(examples)
    return pd.Series(stats.norm.cdf((_log_ratio(examples) - mu) / sigma), index=examples.index)


def _inside(u: np.ndarray, level: float) -> np.ndarray:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    return (u >= (1.0 - level) / 2.0) & (u <= (1.0 + level) / 2.0)


def coverage(model: NowcastV0, examples: pd.DataFrame, *, level: float) -> float:
    """The share of examples inside the central `level` predictive interval."""
    return float(_inside(np.asarray(pit(model, examples), dtype=float), level).mean())


# ---------------------------------------------------------------- the fit run


def _random_walk(model: NowcastV0) -> NowcastV0:
    """The same cells with mu = 0: the random walk the drift is scored against."""
    cells = {k: dataclasses.replace(c, mu=0.0) for k, c in model.cells.items()}
    return dataclasses.replace(model, cells=cells)


def _leave_one_day_out(examples: pd.DataFrame, *, bands, grid, prior_n) -> dict:
    """Per tau: LPD, PIT and 50%/90% coverage of each held-out day under the fit on the others."""
    days = sorted({str(d)[:10] for d in examples["day"]})
    if len(days) < 2:
        return {"note": "leave-one-day-out needs two development days", "by_tau": {}}
    day = examples["day"].astype(str).str[:10]
    parts, unscored = [], 0
    for held in days:
        model = NowcastV0.fit(examples[day != held], bands=bands, grid=grid, prior_n=prior_n)
        test = examples[day == held]
        keys = zip(test["pool"], test["tau_min"], strict=True)
        known = [(p, int(t)) in model.sigma0 for p, t in keys]
        unscored += len(test) - sum(known)
        test = test[known]
        rw = _random_walk(model)
        parts.append(
            pd.DataFrame(
                {
                    "tau_min": test["tau_min"].to_numpy(int),
                    "lpd": log_predictive_density(model, test).to_numpy(),
                    "lpd_rw": log_predictive_density(rw, test).to_numpy(),
                    "pit": pit(model, test).to_numpy(),
                    "pit_rw": pit(rw, test).to_numpy(),
                }
            )
        )
    scored = pd.concat(parts, ignore_index=True)
    by_tau = {}
    for tau, g in scored.groupby("tau_min"):
        u, u_rw = g["pit"].to_numpy(), g["pit_rw"].to_numpy()
        by_tau[str(int(tau))] = {
            "n": len(g),
            "lpd_mean": float(g["lpd"].mean()),
            "lpd_rw_mean": float(g["lpd_rw"].mean()),
            "lpd_minus_rw_mean": float((g["lpd"] - g["lpd_rw"]).mean()),
            "coverage_50": float(_inside(u, 0.5).mean()),
            "coverage_90": float(_inside(u, 0.9).mean()),
            "coverage_50_rw": float(_inside(u_rw, 0.5).mean()),
            "coverage_90_rw": float(_inside(u_rw, 0.9).mean()),
            "pit_ks_pvalue": float(stats.kstest(u, "uniform").pvalue),
            "pit_deciles": np.histogram(u, bins=10, range=(0.0, 1.0))[0].tolist(),
        }
    return {"held_out_days": days, "unscored_examples": unscored, "by_tau": by_tau}


def run_fit(meetings: Sequence[tuple[date, str]], out_dir: Path | None = None) -> dict:
    """Fit on the meetings' examples, score leave-one-day-out, write the fit and its manifest."""
    cfg = config.load_country_yaml(COUNTRY, CONFIG_STEM)
    bands = tuple(float(b) for b in cfg["bands"])
    prior_n = float(cfg["prior_n"])
    max_age = {int(t): float(s) for t, s in cfg["max_board_age_s"].items()}
    grid = load_decision_grid()
    frames, dropped, skipped = [], {}, {}
    for race_date, venue in meetings:
        ex = training_examples(race_date, venue, grid=grid, max_board_age_s=max_age)
        label = f"{race_date.isoformat()}:{venue}"
        dropped[label] = {str(t): n for t, n in ex.attrs["dropped"].items()}
        skipped[label] = ex.attrs["skipped"]
        frames.append(ex)
    examples = pd.concat(frames, ignore_index=True)
    model = NowcastV0.fit(examples, bands=bands, grid=grid, prior_n=prior_n)
    scores = _leave_one_day_out(examples, bands=bands, grid=grid, prior_n=prior_n)

    out_dir = Path(out_dir) if out_dir is not None else paths.data_dir() / "models" / "japan" / "v0"
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_path = out_dir / f"nowcast-{model.fit_id}.json"
    manifest_path = out_dir / f"manifest-nowcast-{model.fit_id}.json"
    cells = [dataclasses.asdict(c) for c in model.cells.values()]
    manifest = {
        "fit_id": model.fit_id,
        "days": list(model.days),
        "meetings": sorted(dropped),
        "config": {"bands": list(bands), "prior_n": prior_n, "max_board_age_s": max_age},
        "grid": list(grid.minutes),
        "n_examples": len(examples),
        "examples_per_cell": [{k: c[k] for k in ("pool", "tau_min", "band", "n")} for c in cells],
        "cells_with_n_below_prior_n": sum(c["n"] < prior_n for c in cells),
        "n_cells": len(cells),
        "dropped_stale_examples": dropped,
        "skipped_races": skipped,
        "scores_leave_one_day_out": scores,
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit_path": str(fit_path),
        "manifest_path": str(manifest_path),
    }
    fit_path.write_text(model.to_json())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _meeting(text: str) -> tuple[date, str]:
    day, sep, venue = text.partition(":")
    if not sep or not venue:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD:VEN, got {text!r}")
    return date.fromisoformat(day), venue.upper()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.models.v0.nowcast")
    parser.add_argument("--fit", action="store_true", required=True, help="fit the drift cells")
    parser.add_argument(
        "--day", type=_meeting, action="append", required=True, help="a meeting, YYYY-MM-DD:VEN"
    )
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)
    print(json.dumps(run_fit(args.day, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
