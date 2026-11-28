"""The v0 calibrated-market Win model P(win | q_final) (#114, D4).

q_i = (1/O_i) / sum_j (1/O_j) over a race's priced runners, from the final Win odds. The model is a
conditional logit, P(i wins) proportional to exp(f(ln q_i)), with f a natural cubic spline in ln q
(linear beyond the boundary knots; the intercept cancels within a race). The knots sit at the
5%..95% quantiles, evenly spaced, of the training rows' ln q, and the coefficients are the maximum
likelihood fit (scipy trust-exact with the analytic gradient and Hessian). A runner in a dead heat
for first counts as a winner of its race; `run_fit` leaves those races out.

Memory: the fit works on flat arrays sorted by race (np.add.reduceat over race segments), and the
out-of-fold run predicts in chunks of at most `chunk_races` whole races.

CLI: `python -m horsetrack.countries.japan.models.v0.calibration --fit [--out DIR]`; an
`--out` inside the repository is refused (`paths.DataDirInsideRepo`, non-zero exit).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize

from horsetrack.core import paths
from horsetrack.core.eval.folds import Fold, load_country_scheme, walk_forward
from horsetrack.countries.japan.models.v0.ordering import _git_sha
from horsetrack.countries.japan.models.v0.store import (
    PRICE_ERA_START,
    PreReformPriceError,
    read_store_table,
)

COUNTRY = "japan"
DEFAULT_SOURCES = ("jra",)  # the store copy a fit is trained on unless it says otherwise
STORE_COLUMNS = [
    "race_id",
    "race_date",
    "horse_no",
    "win_odds",
    "finish_pos",
    "is_scratched",
    "dead_heat",
]
QUANTILE_RANGE = (0.05, 0.95)
GRAD_TOL = 1e-6  # on the mean negative log-likelihood per race


def market_q(odds: np.ndarray) -> np.ndarray:
    """One race's market probabilities from its Win odds; NaN odds stay NaN."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return inv / np.nansum(inv)


def _q_by_race(odds: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """market_q applied race by race to flat arrays (codes: 0..n-1 race labels)."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    totals = np.bincount(codes, weights=np.nan_to_num(inv, nan=0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return inv / totals[codes]


def _basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """The natural cubic spline basis without its constant (ESL 5.4-5.5): x, then N_3..N_K."""
    last = knots[-1]

    def d(k: float) -> np.ndarray:
        return (np.maximum(x - k, 0.0) ** 3 - np.maximum(x - last, 0.0) ** 3) / (last - k)

    cols = [x]
    if len(knots) > 2:
        d_penult = d(knots[-2])
        cols.extend(d(k) - d_penult for k in knots[:-2])
    return np.column_stack(cols)


def _segments(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For codes sorted ascending: each segment's start and length."""
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    return starts, np.diff(np.r_[starts, len(codes)])


def _group_softmax(s: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """exp(s) normalised within each code, in the input order; codes need not be contiguous."""
    order = np.argsort(codes, kind="stable")
    s_sorted = s[order]
    starts, sizes = _segments(codes[order])
    e = np.exp(s_sorted - np.repeat(np.maximum.reduceat(s_sorted, starts), sizes))
    p = np.empty_like(s)
    p[order] = e / np.repeat(np.add.reduceat(e, starts), sizes)
    return p


@dataclass(frozen=True)
class WinCalibration:
    knots: tuple[float, ...]
    coef: tuple[float, ...]
    fit_id: str
    train_start: date
    train_end: date
    n_races: int
    # The store copies the fit was trained on (#404). Left out of `_content`, so out of the fit
    # id, and out of a 'jra' fit file, so those stay as before; a file without it is a 'jra' fit.
    sources: tuple[str, ...] = DEFAULT_SOURCES

    def _score(self, q: np.ndarray) -> np.ndarray:
        knots = np.asarray(self.knots, dtype=float)
        return _basis(np.log(q), knots) @ np.asarray(self.coef, dtype=float)

    def predict_many(self, q: np.ndarray, races: np.ndarray) -> np.ndarray:
        """P(win) for many races at once: `q` flat, `races` labelling each runner's race."""
        q = np.asarray(q, dtype=float)
        races = np.asarray(races)
        if races.shape != q.shape:
            raise ValueError(f"races has shape {races.shape}, q has {q.shape}")
        p = np.full(q.shape, np.nan)
        priced = ~np.isnan(q)
        if priced.any():
            codes = pd.factorize(races[priced])[0]
            p[priced] = _group_softmax(self._score(q[priced]), codes)
        return p

    def predict(self, q: np.ndarray, race: object | None = None) -> np.ndarray:
        """P(win) for one race; NaN entries stay NaN. `race` is ignored (WinModel protocol)."""
        q = np.asarray(q, dtype=float)
        p = np.full(q.shape, np.nan)
        priced = ~np.isnan(q)
        if priced.any():
            codes = np.zeros(int(priced.sum()), dtype=np.int64)
            p[priced] = _group_softmax(self._score(q[priced]), codes)
        return p

    def _content(self) -> dict:
        return {
            "model": "win_calibration_v0",
            "basis": "natural_cubic_spline_ln_q",
            "knots": list(self.knots),
            "coef": list(self.coef),
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "n_races": self.n_races,
        }

    def to_json(self) -> str:
        provenance = {} if self.sources == DEFAULT_SOURCES else {"sources": list(self.sources)}
        return json.dumps({"fit_id": self.fit_id, **self._content(), **provenance}, indent=2)

    @classmethod
    def from_json(cls, s: str) -> WinCalibration:
        data = json.loads(s)
        return cls(
            knots=tuple(float(k) for k in data["knots"]),
            coef=tuple(float(c) for c in data["coef"]),
            fit_id=str(data["fit_id"]),
            train_start=date.fromisoformat(data["train_start"]),
            train_end=date.fromisoformat(data["train_end"]),
            n_races=int(data["n_races"]),
            sources=tuple(str(s) for s in data.get("sources", DEFAULT_SOURCES)),
        )


def _check_price_era(race_dates: pd.Series) -> pd.Series:
    """The dates as Timestamps; a race before D10's price era is a PreReformPriceError."""
    stamps = pd.to_datetime(race_dates)
    if len(stamps) and stamps.min() < pd.Timestamp(PRICE_ERA_START):
        raise PreReformPriceError(
            f"a race on {stamps.min().date().isoformat()} is before the price era "
            f"{PRICE_ERA_START.isoformat()} (D10)"
        )
    return stamps


def _nll(beta, x, won, w, w_rows, starts, sizes, n_races):
    """Mean negative log-likelihood over races, its gradient, and p within races (w: winners
    per race, w_rows: the same repeated over the race's rows)."""
    s = x @ beta
    m = np.maximum.reduceat(s, starts)
    e = np.exp(s - np.repeat(m, sizes))
    z = np.add.reduceat(e, starts)
    p = e / np.repeat(z, sizes)
    value = -(s[won].sum() - (w * (np.log(z) + m)).sum()) / n_races
    wp = w_rows * p
    grad = -(x[won].sum(axis=0) - wp @ x) / n_races
    return value, grad, p


def fit_win_calibration(frame: pd.DataFrame, *, n_knots: int = 5) -> WinCalibration:
    """The conditional-logit spline in ln q fitted on `frame` (race_id, race_date, win_odds,
    won). Races without a priced winner are skipped; `n_races` counts the races fitted."""
    if n_knots < 2:
        raise ValueError(f"n_knots must be at least 2, got {n_knots}")
    stamps = _check_price_era(frame["race_date"])
    codes = pd.factorize(frame["race_id"])[0]
    q = _q_by_race(frame["win_odds"].to_numpy(dtype=float), codes)
    priced = ~np.isnan(q)
    won = frame["won"].to_numpy(dtype=bool) & priced
    # races without a priced winner carry no likelihood; a dead heat counts each winner
    winners = np.bincount(codes[priced], weights=won[priced], minlength=codes.max() + 1)
    keep = priced & (winners[codes] > 0)
    if not keep.any():
        raise ValueError("no race with a priced winner to fit")

    order = np.flatnonzero(keep)
    order = order[np.argsort(codes[order], kind="stable")]
    codes, q, won = codes[order], q[order], won[order]
    stamps = stamps.to_numpy()[order]
    starts, sizes = _segments(codes)
    n_races = len(starts)
    w = winners[codes[starts]]
    w_rows = np.repeat(w, sizes)

    ln_q = np.log(q)
    knots = np.quantile(ln_q, np.linspace(*QUANTILE_RANGE, n_knots))
    if np.any(np.diff(knots) <= 0):
        raise ValueError(f"the training ln q gives tied knots {knots.tolist()}; use fewer knots")
    x = _basis(ln_q, knots)
    del ln_q

    def fun(beta):
        value, grad, _ = _nll(beta, x, won, w, w_rows, starts, sizes, n_races)
        return value, grad

    def hess(beta):
        _, _, p = _nll(beta, x, won, w, w_rows, starts, sizes, n_races)
        xp = x * p[:, None]
        mean = np.add.reduceat(xp, starts, axis=0)
        return ((xp * w_rows[:, None]).T @ x - (mean * w[:, None]).T @ mean) / n_races

    beta0 = np.zeros(x.shape[1])
    beta0[0] = 1.0  # the identity, f(ln q) = ln q
    result = minimize(fun, beta0, jac=True, hess=hess, method="trust-exact", options={"gtol": 1e-9})
    # The loss is convex; trust-exact can stop on rounding just short of gtol, so convergence is
    # judged by the gradient it stopped at.
    converged = result.success or np.max(np.abs(result.jac)) < GRAD_TOL
    if not converged or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"the calibration fit did not converge: {result.message}")

    model = WinCalibration(
        knots=tuple(float(k) for k in knots),
        coef=tuple(float(c) for c in result.x),
        fit_id="",
        train_start=pd.Timestamp(stamps.min()).date(),
        train_end=pd.Timestamp(stamps.max()).date(),
        n_races=int(n_races),
    )
    digest = hashlib.sha256(json.dumps(model._content()).encode()).hexdigest()[:12]
    return dataclasses.replace(model, fit_id=digest)


def oof_predict(
    frame: pd.DataFrame,
    *,
    folds: list[Fold],
    n_knots: int = 5,
    chunk_races: int = 2000,
) -> pd.DataFrame:
    """Out-of-fold p_cal: fold k's test rows (positions into `frame`) through the fit on its
    training rows, predicted in chunks of at most `chunk_races` whole races."""
    if chunk_races < 1:
        raise ValueError(f"chunk_races must be at least 1, got {chunk_races}")
    _check_price_era(frame["race_date"])
    race_ids = frame["race_id"].to_numpy()
    codes = pd.factorize(race_ids)[0]
    q = _q_by_race(frame["win_odds"].to_numpy(dtype=float), codes)
    horse_no = frame["horse_no"].to_numpy()

    parts = []
    for k, fold in enumerate(folds):
        test = np.asarray(fold.test_idx)
        if len(test) == 0:
            continue
        model = fit_win_calibration(frame.iloc[fold.train_idx], n_knots=n_knots)
        order = np.argsort(codes[test], kind="stable")
        starts, _ = _segments(codes[test][order])
        bounds = np.r_[starts[::chunk_races], len(test)]
        p = np.empty(len(test))
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            rows = test[order[a:b]]
            p[order[a:b]] = model.predict_many(q[rows], codes[rows])
        parts.append(
            pd.DataFrame(
                {
                    "race_id": race_ids[test],
                    "horse_no": horse_no[test],
                    "fold": np.full(len(test), k, dtype=np.int32),
                    "q": q[test],
                    "p_cal": p,
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["race_id", "horse_no", "fold", "q", "p_cal"])
    return pd.concat(parts, ignore_index=True)


def _development_frame(table: pa.Table) -> tuple[pd.DataFrame, dict, dict]:
    """The fit frame from store runners, with the races and runners left out and why."""
    runners = table.to_pandas(date_as_object=False)
    runners["race_id"] = runners["race_id"].astype("category")
    codes = runners["race_id"].cat.codes.to_numpy()
    n_races = len(runners["race_id"].cat.categories)
    first = runners["finish_pos"].to_numpy(dtype=float) == 1
    dead_heat = runners["dead_heat"].fillna(False).to_numpy(dtype=bool)
    is_dh_first = np.bincount(codes, weights=first & dead_heat, minlength=n_races) >= 2

    scratched = runners["is_scratched"].fillna(False).to_numpy(dtype=bool)
    odds = runners["win_odds"].to_numpy(dtype=float)
    unpriced = ~scratched & ~(np.isfinite(odds) & (odds > 0))
    usable = ~scratched & ~unpriced
    # a race whose winner is not among its usable runners has nothing to fit
    no_winner = ~is_dh_first & (np.bincount(codes, weights=first & usable, minlength=n_races) == 0)
    keep = usable & ~is_dh_first[codes] & ~no_winner[codes]
    kept = runners.loc[keep]

    frame = pd.DataFrame(
        {
            "race_id": kept["race_id"].cat.remove_unused_categories(),
            "race_date": kept["race_date"],
            "horse_no": kept["horse_no"].astype(np.int32),
            "win_odds": kept["win_odds"].astype(np.float64),
            "won": kept["finish_pos"].to_numpy(dtype=float) == 1,
        }
    ).reset_index(drop=True)
    races_left_out = {
        "dead_heat_for_first": int(is_dh_first.sum()),
        "no_winner": int(no_winner.sum()),
    }
    runners_dropped = {
        "scratched": int(scratched.sum()),
        "unpriced": int(unpriced.sum()),
    }
    return frame, races_left_out, runners_dropped


def run_fit(
    out_dir: Path | None = None,
    *,
    n_knots: int = 5,
    chunk_races: int = 2000,
    source: str = "jra",
) -> dict:
    """Fit the live model on every development race from the price era, write out-of-fold
    predictions over Japan's fold scheme, and write the fit, the predictions and a manifest.

    `source` names the store copy read (#404): 'jra' by default, 'jravan' the JRA-VAN copy; a
    run on another copy than 'jra' names it in the manifest's and the fit file's `sources` and in
    its output file names (`win_calibration-<source>-<fit_id>.json` and the like).

    An `out_dir` that resolves inside the repository raises `paths.DataDirInsideRepo` before
    anything is read or written."""
    if out_dir is None:
        out_dir = paths.data_dir() / "models" / "japan" / "v0"
    else:
        out_dir = paths.outside_repo(Path(out_dir), what="out_dir")
    table = read_store_table(
        "runners",
        columns=STORE_COLUMNS,
        start=PRICE_ERA_START,
        end=date.max,  # to the store's last date
        touches_price=True,
        source=source,
    )
    frame, races_left_out, runners_dropped = _development_frame(table)
    del table
    model = dataclasses.replace(fit_win_calibration(frame, n_knots=n_knots), sources=(source,))

    scheme = load_country_scheme(COUNTRY)
    folds = walk_forward(frame["race_date"], scheme=scheme)
    oof = oof_predict(frame, folds=folds, n_knots=n_knots, chunk_races=chunk_races)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Another store copy's outputs name it, so a run on each copy keeps its own files (the fit id
    # leaves the source out: two copies with the same usable rows share it).
    name = model.fit_id if source == "jra" else f"{source}-{model.fit_id}"
    fit_path = out_dir / f"win_calibration-{name}.json"
    oof_path = out_dir / f"win_calibration_oof-{name}.parquet"
    manifest_path = out_dir / f"manifest-win_calibration-{name}.json"

    race_codes = frame["race_id"].cat.codes.to_numpy()
    fold_entries = []
    for k, fold in enumerate(folds):
        fold_entries.append(
            {
                "fold": k,
                "train_end": fold.train_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "n_train_races": int(len(np.unique(race_codes[fold.train_idx]))),
                "n_test_races": int(len(np.unique(race_codes[fold.test_idx]))),
                "n_test_rows": int(len(fold.test_idx)),
            }
        )
    manifest = {
        "fit_id": model.fit_id,
        "model": "win_calibration_v0",
        "start": model.train_start.isoformat(),
        "end": model.train_end.isoformat(),
        "n_races": model.n_races,
        "n_runners": int(len(frame)),
        "races_left_out": races_left_out,
        "runners_dropped": runners_dropped,
        "n_knots": n_knots,
        "knots": list(model.knots),
        "chunk_races": chunk_races,
        "fold_scheme": dataclasses.asdict(scheme),
        "folds": fold_entries,
        "n_oof_rows": int(len(oof)),
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit_path": str(fit_path),
        "oof_path": str(oof_path),
        "manifest_path": str(manifest_path),
    }
    if source != "jra":  # the default manifest is unchanged
        manifest["sources"] = [source]
    oof["race_id"] = oof["race_id"].astype(str)
    fit_path.write_text(model.to_json())
    pq.write_table(pa.Table.from_pandas(oof, preserve_index=False), oof_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m horsetrack.countries.japan.models.v0.calibration"
    )
    parser.add_argument("--fit", action="store_true", required=True, help="fit the calibration")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)
    print(json.dumps(run_fit(args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
