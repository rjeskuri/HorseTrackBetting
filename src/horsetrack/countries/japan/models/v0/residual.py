"""The v0 shrunk segment residual on the calibrated market (#125, #73's C4 offset form, D17).

P(i wins) is proportional to exp(f(ln q_i) + g * ln q_i), where f is #114's calibration spline
(`WinCalibration`, a fixed offset) and g = sum_s gamma_s * z_s(race): a segment-specific slope on
ln q, since a race-level constant cancels in the softmax. Equivalently p_i is proportional to
base.predict(q)_i * q_i ** g. gamma is the conditional-logit maximum likelihood with a ridge
penalty toward 0: the negative log-likelihood summed over races plus ridge / 2 * |gamma|^2, so
`ridge` is the precision of a N(0, 1/ridge) prior on each gamma_s (gamma = 0 is the market alone).

Segments (anything else is a ValueError), on when:
    `field_16_plus`  the race has 16 or more starters;
    `race_12`        it is race 12;
    `races_10_12`    it is race 10, 11 or 12;
    `venue:<CODE>`   it is at venue CODE, a JRA code from jra.yaml `venues`.
The owner's list is `segments` in countries/japan/config/residual_v0.yaml (`read_segments`). At
fit time a race's field size is its number of rows in the frame; at predict time the indicators
come from the RaceContext only (`starters`, `race_no`, `venue`).

CLI: `python -m horsetrack.countries.japan.models.v0.residual --fit --base <win_calibration file>
[--ridge R] [--out DIR]` fits on every development race of the price era (holdout days are never
read), writes the fit, its out-of-fold predictions over Japan's folds (base and residual refitted
in each fold) and a manifest under the data dir, scores the per-race winner log-loss gain over the
calibrated market with D7's meeting bootstrap, writes that report through `write_report` and
appends the run to Japan's registry (a failed run too). It never touches current.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy.optimize import minimize

from horsetrack.core import paths
from horsetrack.core.config import load_country_yaml, load_source
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import Fold, load_country_scheme, walk_forward
from horsetrack.core.eval.registry import FoldMetric, RecordRefused, RunRecord, default_registry
from horsetrack.core.eval.reports import write_report
from horsetrack.core.eval.scoring import gain, per_race_log_loss
from horsetrack.countries.japan.board.snapshot import RaceContext
from horsetrack.countries.japan.models.v0.calibration import (
    GRAD_TOL,
    STORE_COLUMNS,
    WinCalibration,
    _check_price_era,
    _development_frame,
    _group_softmax,
    _q_by_race,
    _segments,
    fit_win_calibration,
)
from horsetrack.countries.japan.models.v0.ordering import _git_sha
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START, read_store_table

COUNTRY = "japan"
TICKET = 125
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "residual_v0.yaml"
RACE_SEGMENTS = ("field_16_plus", "race_12", "races_10_12")
VENUE_PREFIX = "venue:"
RIDGE = 100.0  # the CLI default: a N(0, 0.1^2) prior on each gamma_s
BOOTSTRAP_SEED = 125
HYPOTHESIS_ID = "E4"  # #77's pre-registered segment family, cut to what the board shows
FAMILY_ID = "japan-v0-win-residual"
METRIC = "winner_logloss_gain_vs_calibrated_market"


def _check_segments(segments: Sequence[str]) -> tuple[str, ...]:
    """The names as a tuple; an unknown or repeated name is a ValueError."""
    segments = tuple(segments)
    venues = frozenset(load_source(COUNTRY, "jra").venues)
    for name in segments:
        if not isinstance(name, str):
            raise ValueError(f"a segment name must be a str, got {name!r}")
        if name in RACE_SEGMENTS:
            continue
        if name.startswith(VENUE_PREFIX) and name[len(VENUE_PREFIX) :] in venues:
            continue
        raise ValueError(
            f"unknown segment {name!r}: use {', '.join(RACE_SEGMENTS)} or venue:<CODE> with CODE "
            f"one of {', '.join(sorted(venues))}"
        )
    if len(set(segments)) != len(segments):
        raise ValueError(f"a segment is named twice in {list(segments)}")
    return segments


def _indicators(segments: tuple[str, ...], venue, race_no, starters) -> np.ndarray:
    """z_s per race, a (races x segments) array, from arrays of race attributes."""
    venue = np.asarray(venue).astype(str)
    race_no = np.asarray(race_no, dtype=np.int64)
    starters = np.asarray(starters, dtype=np.int64)
    z = np.zeros((len(race_no), len(segments)))
    for j, name in enumerate(segments):
        if name == "field_16_plus":
            z[:, j] = starters >= 16
        elif name == "race_12":
            z[:, j] = race_no == 12
        elif name == "races_10_12":
            z[:, j] = (race_no >= 10) & (race_no <= 12)
        else:
            z[:, j] = venue == name[len(VENUE_PREFIX) :]
    return z


def _scores(base: WinCalibration, q: np.ndarray, g: np.ndarray | float) -> np.ndarray:
    """f(ln q) + g * ln q for priced q (g per row, or one value)."""
    return base._score(q) + g * np.log(q)


@dataclass(frozen=True)
class ResidualFit:
    base: WinCalibration
    segments: tuple[str, ...]
    gamma: tuple[float, ...]
    ridge: float
    fit_id: str

    def __post_init__(self) -> None:
        segments = _check_segments(self.segments)
        gamma = tuple(float(g) for g in self.gamma)
        if len(gamma) != len(segments):
            raise ValueError(f"{len(gamma)} gamma value(s) for {len(segments)} segment(s)")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "ridge", float(self.ridge))

    def slope(self, race: RaceContext | None) -> float:
        """g = sum_s gamma_s * z_s(race); a None race is a ValueError when there are segments."""
        if not self.segments:
            return 0.0
        if race is None:
            raise ValueError("the residual needs the race context (venue, race_no, starters)")
        z = _indicators(self.segments, [race.venue], [race.race_no], [race.starters])[0]
        return float(z @ np.asarray(self.gamma))

    def predict(self, q: np.ndarray, race: RaceContext | None = None) -> np.ndarray:
        """P(win) for one race; NaN entries stay NaN (WinModel protocol)."""
        g = self.slope(race)
        if not self.segments:
            return self.base.predict(q)
        q = np.asarray(q, dtype=float)
        p = np.full(q.shape, np.nan)
        priced = ~np.isnan(q)
        if priced.any():
            codes = np.zeros(int(priced.sum()), dtype=np.int64)
            p[priced] = _group_softmax(_scores(self.base, q[priced], g), codes)
        return p

    def _content(self) -> dict:
        return {
            "model": "win_residual_v0",
            "segments": list(self.segments),
            "gamma": list(self.gamma),
            "ridge": self.ridge,
            "base": json.loads(self.base.to_json()),
        }

    def to_json(self) -> str:
        return json.dumps({"fit_id": self.fit_id, **self._content()}, indent=2)

    @classmethod
    def from_json(cls, s: str) -> ResidualFit:
        data = json.loads(s)
        return cls(
            base=WinCalibration.from_json(json.dumps(data["base"])),
            segments=tuple(str(name) for name in data["segments"]),
            gamma=tuple(float(g) for g in data["gamma"]),
            ridge=float(data["ridge"]),
            fit_id=str(data["fit_id"]),
        )


def _with_id(fit: ResidualFit) -> ResidualFit:
    digest = hashlib.sha256(json.dumps(fit._content()).encode()).hexdigest()[:12]
    return dataclasses.replace(fit, fit_id=digest)


def _race_attributes(frame: pd.DataFrame, codes: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per race code: venue, race number (its first row's) and field size (its rows)."""
    _, first = np.unique(codes, return_index=True)
    venue = frame["venue"].to_numpy()[first]
    race_no = frame["race_no"].to_numpy()[first]
    return venue, race_no, np.bincount(codes)


def fit_residual(
    frame: pd.DataFrame, *, base: WinCalibration, segments: Sequence[str], ridge: float
) -> ResidualFit:
    """gamma fitted on `frame` (race_id, race_date, venue, race_no, win_odds, won) with base's
    f(ln q) as a fixed offset and a ridge penalty toward 0. Races without a priced winner carry no
    likelihood; a dead heat counts each winner."""
    segments = _check_segments(segments)
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not ridge >= 0:
        raise ValueError(f"ridge must be a non-negative number, got {ridge!r}")
    if not math.isfinite(ridge):
        raise ValueError(f"ridge must be finite, got {ridge!r}")
    if not segments:
        return _with_id(ResidualFit(base=base, segments=(), gamma=(), ridge=ridge, fit_id=""))
    _check_price_era(frame["race_date"])

    codes = pd.factorize(frame["race_id"])[0]
    z_race = _indicators(segments, *_race_attributes(frame, codes))
    q = _q_by_race(frame["win_odds"].to_numpy(dtype=float), codes)
    priced = ~np.isnan(q)
    won = frame["won"].to_numpy(dtype=bool) & priced
    winners = np.bincount(codes[priced], weights=won[priced], minlength=codes.max() + 1)
    keep = priced & (winners[codes] > 0)
    if not keep.any():
        raise ValueError("no race with a priced winner to fit")

    order = np.flatnonzero(keep)
    order = order[np.argsort(codes[order], kind="stable")]
    codes, q, won = codes[order], q[order], won[order]
    starts, sizes = _segments(codes)
    n_races = len(starts)
    z = z_race[codes[starts]]
    w = winners[codes[starts]]
    offset = base._score(q)
    x = np.log(q)
    x_won = np.add.reduceat(np.where(won, x, 0.0), starts)  # sum of ln q over each race's winners

    def moments(gamma):
        s = offset + np.repeat(z @ gamma, sizes) * x
        m = np.maximum.reduceat(s, starts)
        e = np.exp(s - np.repeat(m, sizes))
        total = np.add.reduceat(e, starts)
        p = e / np.repeat(total, sizes)
        mean = np.add.reduceat(p * x, starts)
        return s, m, total, p, mean

    # Scaled per race for the optimizer's tolerance; the optimum is that of the summed objective.
    def fun(gamma):
        s, m, total, _, mean = moments(gamma)
        nll = -(s[won].sum() - (w * (np.log(total) + m)).sum())
        value = (nll + 0.5 * ridge * gamma @ gamma) / n_races
        grad = (-(z.T @ (x_won - w * mean)) + ridge * gamma) / n_races
        return value, grad

    def hess(gamma):
        _, _, _, p, mean = moments(gamma)
        var = np.add.reduceat(p * x * x, starts) - mean**2
        info = (z * (w * var)[:, None]).T @ z
        return (info + ridge * np.eye(len(gamma))) / n_races

    result = minimize(
        fun,
        np.zeros(len(segments)),
        jac=True,
        hess=hess,
        method="trust-exact",
        options={"gtol": 1e-10},
    )
    converged = result.success or np.max(np.abs(result.jac)) < GRAD_TOL
    if not converged or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"the residual fit did not converge: {result.message}")
    gamma = tuple(float(g) for g in result.x)
    return _with_id(ResidualFit(base=base, segments=segments, gamma=gamma, ridge=ridge, fit_id=""))


def oof_predict(
    frame: pd.DataFrame,
    *,
    folds: list[Fold],
    segments: Sequence[str],
    ridge: float,
    n_knots: int = 5,
) -> pd.DataFrame:
    """Out-of-fold p_cal and p_res: fold k's test rows (positions into `frame`) through the
    calibration and the residual both fitted on its training rows. A race's context is built from
    the frame (venue, race_no, starters = its rows). `won` is carried for scoring."""
    segments = _check_segments(segments)
    _check_price_era(frame["race_date"])
    race_ids = frame["race_id"].to_numpy()
    codes = pd.factorize(race_ids)[0]
    z_race = _indicators(segments, *_race_attributes(frame, codes))
    q = _q_by_race(frame["win_odds"].to_numpy(dtype=float), codes)
    horse_no = frame["horse_no"].to_numpy()
    won = frame["won"].to_numpy(dtype=bool)

    columns = ["race_id", "horse_no", "fold", "q", "p_cal", "p_res", "won"]
    parts = []
    for k, fold in enumerate(folds):
        test = np.asarray(fold.test_idx)
        if len(test) == 0:
            continue
        train = frame.iloc[fold.train_idx]
        cal = fit_win_calibration(train, n_knots=n_knots)
        res = fit_residual(train, base=cal, segments=segments, ridge=ridge)
        p_cal = cal.predict_many(q[test], codes[test])
        p_res = np.full(len(test), np.nan)
        priced = ~np.isnan(q[test])
        rows = test[priced]
        g = z_race[codes[rows]] @ np.asarray(res.gamma) if segments else 0.0
        p_res[priced] = _group_softmax(_scores(cal, q[rows], g), codes[rows])
        parts.append(
            pd.DataFrame(
                {
                    "race_id": race_ids[test],
                    "horse_no": horse_no[test],
                    "fold": np.full(len(test), k, dtype=np.int32),
                    "q": q[test],
                    "p_cal": p_cal,
                    "p_res": p_res,
                    "won": won[test],
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def read_segments(path: Path = CONFIG_PATH) -> tuple[str, ...]:
    """The owner's segment list, the `segments` key of residual_v0.yaml."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: the residual's segments are an owner decision; "
            "add `segments: [...]` there once they are confirmed"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or "segments" not in data:
        raise ValueError(f"{path} has no `segments` key")
    segments = data["segments"]
    if not isinstance(segments, list):
        raise ValueError(f"{path}: `segments` must be a list, got {segments!r}")
    return _check_segments(segments)


# --------------------------------------------------------------------------- the real-data run


def _store_build(source: str = "jra") -> dict[str, str]:
    """The store copy's build identity for the registry, as `<source>/runners@<built_utc>`."""
    try:
        build = json.loads((paths.processed_dir(source) / "_build.json").read_text())
        return {"runners": f"{source}/runners@{build['built_utc']}"}
    except (OSError, ValueError, KeyError):
        return {"runners": f"{source}/runners"}


def _bootstrap_settings() -> dict:
    settings = load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if settings.get("cluster_unit") != "meeting" or settings.get("sided") != "one-sided":
        raise ValueError(
            f"scoring.yaml bootstrap is not D7's one-sided meeting bootstrap: {settings}"
        )
    return {"n_boot": int(settings["n_boot"]), "level": float(settings["level"])}


def _score(frame: pd.DataFrame, oof: pd.DataFrame, folds: list[Fold], boot: dict) -> dict:
    """Per-race winner log-loss gain of p_res over p_cal, the meeting bootstrap, per-fold means."""
    race_id = oof["race_id"].astype(str)
    scratched = pd.Series(False, index=oof.index)
    won = oof["won"].astype(bool)
    ll_cal = per_race_log_loss(oof["p_cal"], won, race_id, scratched)
    ll_res = per_race_log_loss(oof["p_res"], won, race_id, scratched)
    g = gain(ll_res.per_race, ll_cal.per_race)

    races = frame[["race_id", "race_date", "venue"]].drop_duplicates("race_id")
    day = pd.to_datetime(races["race_date"]).dt.strftime("%Y-%m-%d")
    meeting = pd.Series(
        (day + ":" + races["venue"].astype(str)).to_numpy(),
        index=races["race_id"].astype(str).to_numpy(),
    )
    interval = paired_bootstrap(
        g.per_race, meeting, n_boot=boot["n_boot"], level=boot["level"], seed=BOOTSTRAP_SEED
    )
    fold_of = pd.Series(oof["fold"].to_numpy(), index=race_id.to_numpy()).groupby(level=0).first()
    by_fold = g.per_race.groupby(fold_of.reindex(g.per_race.index).to_numpy())
    per_fold = []
    for k, values in by_fold:
        fold = folds[int(k)]
        per_fold.append(
            {
                "fold": int(k),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "n_races": int(len(values)),
                "gain_mean": float(values.mean()),
            }
        )
    return {
        "metric": METRIC,
        "unit": "nats per race",
        "n_races": g.n_common,
        "ll_calibrated_market": float(ll_cal.per_race.mean()),
        "ll_residual": float(ll_res.per_race.mean()),
        "gain_mean": interval.mean,
        "gain_lo": interval.lo,
        "gain_hi": interval.hi,
        "level": boot["level"],
        "sided": "one-sided",
        "n_boot": boot["n_boot"],
        "seed": BOOTSTRAP_SEED,
        "n_meetings": interval.n_clusters,
        "per_fold": per_fold,
    }


def _record(
    *,
    status: str,
    scheme,
    segments,
    ridge,
    n_knots,
    base,
    fit_id,
    score: dict | None,
    source: str = "jra",
) -> RunRecord:
    metrics: tuple[FoldMetric, ...] = ()
    if score is not None:
        metrics = (
            FoldMetric(
                fold="all",
                metric=METRIC,
                value=score["gain_mean"],
                lo=score["gain_lo"],
                hi=score["gain_hi"],
                n=score["n_races"],
            ),
            *(
                FoldMetric(
                    fold=f"{f['fold']}:{f['test_start']}..{f['test_end']}",
                    metric=METRIC,
                    value=f["gain_mean"],
                    lo=None,
                    hi=None,
                    n=f["n_races"],
                )
                for f in score["per_fold"]
            ),
        )
    components = {"base": base.fit_id}
    if fit_id:
        components["residual"] = fit_id
    return RunRecord(
        hypothesis_id=HYPOTHESIS_ID,
        family_id=FAMILY_ID,
        variant=f"segments-{len(segments)}-ridge-{ridge:g}",
        ticket=TICKET,
        git_sha=_git_sha() or "unknown",
        data_builds=_store_build(source),
        sources=(source,),
        fold_scheme=json.dumps({"country": COUNTRY, **dataclasses.asdict(scheme)}, sort_keys=True),
        components=components,
        params={
            "segments": list(segments),
            "ridge": float(ridge),
            "n_knots": int(n_knots),
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        metrics=metrics,
        status=status,
    )


def _development_frame_with_race(table: pa.Table) -> tuple[pd.DataFrame, dict, dict]:
    """#114's development frame with each race's venue and race number added."""
    frame, races_left_out, runners_dropped = _development_frame(table.select(STORE_COLUMNS))
    races = (
        table.select(["race_id", "venue", "race_no"])
        .to_pandas()
        .drop_duplicates("race_id")
        .set_index("race_id")
    )
    ids = frame["race_id"].astype(str)
    frame["venue"] = ids.map(races["venue"]).astype("category")
    frame["race_no"] = ids.map(races["race_no"]).astype(np.int16)
    return frame, races_left_out, runners_dropped


def run_fit(
    base_path: Path,
    out_dir: Path | None = None,
    *,
    ridge: float = RIDGE,
    n_knots: int = 5,
    source: str = "jra",
) -> dict:
    """Fit the residual on every development race of the price era on top of the calibration in
    `base_path`, score it out of fold, and write the fit, the predictions, a manifest and the
    report under the data dir; the run (failed or scored) is appended to Japan's registry.

    `source` names the store copy read (#404): 'jra' by default, 'jravan' the JRA-VAN copy. The
    registry record, the report's `write_report` sources and (off 'jra') the manifest name it, so
    a licensed copy's figures are withheld (`core.eval.reports.WITHHELD_SOURCES`).

    An `out_dir` that resolves inside the repository raises `paths.DataDirInsideRepo` before
    anything is read or written."""
    if out_dir is None:
        out_dir = paths.data_dir() / "models" / "japan" / "v0"
    else:
        out_dir = paths.outside_repo(Path(out_dir), what="out_dir")
    base = WinCalibration.from_json(Path(base_path).read_text(encoding="utf-8"))
    segments = read_segments()
    scheme = load_country_scheme(COUNTRY)
    boot = _bootstrap_settings()
    registry = default_registry(COUNTRY)
    run = dict(
        scheme=scheme, segments=segments, ridge=ridge, n_knots=n_knots, base=base, source=source
    )

    fit_id = ""
    try:
        table = read_store_table(
            "runners",
            columns=[*STORE_COLUMNS, "venue", "race_no"],
            start=PRICE_ERA_START,
            end=date.max,  # to the store's last date
            touches_price=True,
            source=source,
        )
        frame, races_left_out, runners_dropped = _development_frame_with_race(table)
        del table
        fit = fit_residual(frame, base=base, segments=segments, ridge=ridge)
        fit_id = fit.fit_id
        folds = walk_forward(frame["race_date"], scheme=scheme)
        oof = oof_predict(frame, folds=folds, segments=segments, ridge=ridge, n_knots=n_knots)
        score = _score(frame, oof, folds, boot)
    except Exception:
        try:
            registry.append(_record(status="failed", fit_id=fit_id, score=None, **run))
        except RecordRefused:  # the same failed run is already on record
            pass
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
    fit_path = out_dir / f"residual-{fit.fit_id}.json"
    oof_path = out_dir / f"residual_oof-{fit.fit_id}.parquet"
    manifest_path = out_dir / f"manifest-residual-{fit.fit_id}.json"
    report_path = paths.reports_dir() / "japan" / f"residual_v0-{fit.fit_id}.json"
    manifest = {
        "fit_id": fit.fit_id,
        "model": "win_residual_v0",
        "base_fit_id": base.fit_id,
        "base_path": str(Path(base_path).resolve()),
        "segments": list(fit.segments),
        "gamma": dict(zip(fit.segments, fit.gamma, strict=True)),
        "ridge": fit.ridge,
        "n_knots": n_knots,
        "start": pd.Timestamp(frame["race_date"].min()).date().isoformat(),
        "end": pd.Timestamp(frame["race_date"].max()).date().isoformat(),
        "n_races": int(frame["race_id"].nunique()),
        "n_runners": int(len(frame)),
        "races_left_out": races_left_out,
        "runners_dropped": runners_dropped,
        "fold_scheme": dataclasses.asdict(scheme),
        "n_folds": len(folds),
        "n_oof_rows": int(len(oof)),
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit_path": str(fit_path),
        "oof_path": str(oof_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
    }
    if source != "jra":  # the default manifest is unchanged
        manifest["sources"] = [source]
    oof["race_id"] = oof["race_id"].astype(str)
    fit_path.write_text(fit.to_json())
    pq.write_table(pa.Table.from_pandas(oof, preserve_index=False), oof_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    report = {"fit_id": fit.fit_id, "base_fit_id": base.fit_id, "out_of_fold": score, **manifest}
    write_report(report_path, json.dumps(report, indent=2) + "\n", sources=(source,))
    manifest["run_id"] = registry.append(
        _record(status="scored", fit_id=fit.fit_id, score=score, **run)
    )
    manifest["out_of_fold"] = {k: v for k, v in score.items() if k != "per_fold"}
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.models.v0.residual")
    parser.add_argument("--fit", action="store_true", required=True, help="fit the residual")
    parser.add_argument("--base", type=Path, required=True, help="the win_calibration fit file")
    parser.add_argument("--ridge", type=float, default=RIDGE, help="ridge penalty toward 0")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)
    print(json.dumps(run_fit(args.base, args.out, ridge=args.ridge), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
