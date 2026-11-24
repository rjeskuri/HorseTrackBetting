"""The v0 ordering model: Win probabilities to Place and Wide through a discounted Harville model.

P(first = i) = p_i; P(second = j | i) = p_j^l2 / sum_{l != i} p_l^l2;
P(third = k | i, j) = p_k^l3 / sum_{l not in {i, j}} p_l^l3. With (l2, l3) = (1, 1) it is Harville.

Memory: ordered finishes are enumerated for one race at a time (an 18-runner race has 4,896
ordered triples). The fit never enumerates triples: the log-likelihood separates by stage, so each
discount is a 1-D maximum likelihood over a races x runners array.

CLI: `python -m horsetrack.countries.japan.models.v0.ordering --fit --oof PATH [--out DIR]`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from horsetrack.core.paths import data_dir

FAMILIES = ("discounted_harville",)
COLUMNS = ["race_id", "horse_no", "p_win", "finish_pos", "dead_heat"]
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ordering_v0.yaml"
# Search range for each discount: 0 is a uniform draw among the rest, above 1 favours favourites.
LAMBDA_BOUNDS = (0.0, 5.0)


def _check_family(family: str) -> None:
    if family not in FAMILIES:
        raise NotImplementedError(
            f"ordering family {family!r} is not implemented (only {', '.join(FAMILIES)})"
        )


def _check_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError(f"p must be a non-empty 1-D array, got shape {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("every Win probability must be finite and positive")
    return p / p.sum()


def _order_probs(p: np.ndarray, lambdas: tuple[float, float], places: int) -> np.ndarray:
    """P(ordered first `places`) for one race, an array with `places` axes of length n."""
    if places == 1:
        return p
    w2 = p ** lambdas[0]
    first_two = p[:, None] * w2[None, :] / (w2.sum() - w2)[:, None]
    np.fill_diagonal(first_two, 0.0)
    if places == 2:
        return first_two
    w3 = p ** lambdas[1]
    rest = w3.sum() - w3[:, None] - w3[None, :]
    np.fill_diagonal(rest, 1.0)  # i == j has zero probability already; avoid dividing by <= 0
    triples = first_two[:, :, None] * w3[None, None, :] / rest[:, :, None]
    n = len(p)
    idx = np.arange(n)
    triples[idx, :, idx] = 0.0  # k == i
    triples[:, idx, idx] = 0.0  # k == j
    return triples


@dataclass(frozen=True)
class OrderingFit:
    family: str
    lambdas: tuple[float, float]
    fit_id: str
    n_races: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambdas", tuple(float(x) for x in self.lambdas))
        if len(self.lambdas) != 2:
            raise ValueError(f"lambdas must be (lambda2, lambda3), got {self.lambdas}")

    def _orders(self, p: np.ndarray, places: int) -> tuple[np.ndarray | None, int]:
        _check_family(self.family)
        p = _check_p(p)
        if not 0 <= places <= min(3, len(p)):
            raise ValueError(f"places must be 0 to 3 and at most the {len(p)} runners: {places}")
        return (_order_probs(p, self.lambdas, places) if places else None), len(p)

    def place_probs(self, p: np.ndarray, places: int) -> np.ndarray:
        """P(runner finishes in the first `places`); 0 places (pool not offered) gives zeros."""
        probs, n = self._orders(p, places)
        if probs is None:
            return np.zeros(n)
        axes = range(probs.ndim)
        return sum(probs.sum(axis=tuple(a for a in axes if a != keep)) for keep in axes)

    def pair_probs(self, p: np.ndarray, places: int) -> np.ndarray:
        """P(both runners finish in the first `places`), k x k, symmetric, zero diagonal."""
        probs, n = self._orders(p, places)
        if probs is None or places == 1:
            return np.zeros((n, n))
        # Sum over the one position not in the pair (none when two places are paid).
        one_way = probs if places == 2 else probs.sum(axis=0) + probs.sum(axis=1) + probs.sum(2)
        return one_way + one_way.T

    def to_json(self) -> str:
        return json.dumps(
            {
                "family": self.family,
                "lambdas": list(self.lambdas),
                "fit_id": self.fit_id,
                "n_races": self.n_races,
            }
        )

    @classmethod
    def from_json(cls, s: str) -> OrderingFit:
        data = json.loads(s)
        return cls(data["family"], tuple(data["lambdas"]), data["fit_id"], int(data["n_races"]))


def top_order_loglik(p: np.ndarray, order: Sequence[int], lambdas: tuple[float, float]) -> float:
    """log P(the first finishers are `order`, in that order) under the discounted model."""
    logp = np.log(_check_p(p))
    order = [int(h) for h in order]
    if not 1 <= len(order) <= 3 or len(set(order)) != len(order):
        raise ValueError(f"order must be 1 to 3 distinct runners: {order}")
    if min(order) < 0 or max(order) >= len(logp):
        raise ValueError(f"order {order} is out of range for {len(logp)} runners")
    remaining = np.ones(len(logp), dtype=bool)
    total = 0.0
    for lam, h in zip((1.0, *lambdas), order, strict=False):
        total += lam * logp[h] - logsumexp(lam * logp[remaining])
        remaining[h] = False
    return float(total)


@dataclass(frozen=True)
class _Races:
    """The usable races as a padded races x runners array (no triples)."""

    race_ids: np.ndarray
    logp: np.ndarray  # log p_win, 0 in padding
    present: np.ndarray  # False in padding
    chosen: np.ndarray  # races x 3: column of the first, second and third finisher
    n_dead_heat: int
    n_incomplete: int


def _prepare(frame: pd.DataFrame) -> _Races:
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"the fit frame is missing columns: {missing}")
    if frame.duplicated(["race_id", "horse_no"]).any():
        raise ValueError("a horse_no appears twice in one race")
    if frame["dead_heat"].isna().any():
        raise ValueError("dead_heat has missing values")
    codes, race_ids = pd.factorize(frame["race_id"], sort=True)
    p = frame["p_win"].to_numpy(dtype=float)
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("every p_win must be finite and positive")
    pos = frame["finish_pos"].to_numpy(dtype=float)  # NaN: ran but did not finish
    finished = ~np.isnan(pos)
    if np.any(pos[finished] < 1) or np.any(pos[finished] != np.round(pos[finished])):
        raise ValueError("finish_pos must be a positive integer or missing")
    dead_heat = frame["dead_heat"].to_numpy(dtype=bool)

    n_all = len(race_ids)
    top = finished & (pos <= 3)
    slot = np.where(top, pos, 1).astype(int) - 1
    counts = np.zeros((n_all, 3), dtype=int)
    np.add.at(counts, (codes[top], slot[top]), 1)
    tied = (counts > 1).any(axis=1)
    tied[codes[top & dead_heat]] = True
    complete = (counts == 1).all(axis=1)
    used = complete & ~tied
    if not used.any():
        raise ValueError("no race has a clean first three to fit on")

    new_code = np.cumsum(used) - 1
    keep = used[codes]
    rc, rp, rtop, rslot = new_code[codes[keep]], p[keep], top[keep], slot[keep]
    sort = np.argsort(rc, kind="stable")
    rc, rp, rtop, rslot = rc[sort], rp[sort], rtop[sort], rslot[sort]
    n_races = int(used.sum())
    sizes = np.bincount(rc, minlength=n_races)
    col = np.arange(len(rc)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    logp = np.zeros((n_races, sizes.max()))
    present = np.zeros_like(logp, dtype=bool)
    logp[rc, col] = np.log(rp)
    present[rc, col] = True
    chosen = np.zeros((n_races, 3), dtype=int)
    chosen[rc[rtop], rslot[rtop]] = col[rtop]
    return _Races(
        race_ids=np.asarray(race_ids)[used],
        logp=logp,
        present=present,
        chosen=chosen,
        n_dead_heat=int(tied.sum()),
        n_incomplete=int((~complete & ~tied).sum()),
    )


def _fit_stage(races: _Races, stage: int) -> float:
    """Maximum likelihood of the stage's discount: softmax(lam * log p) over the runners left."""
    rows = np.arange(len(races.logp))
    left = races.present.copy()
    for s in range(stage):
        left[rows, races.chosen[:, s]] = False
    chosen_logp = races.logp[rows, races.chosen[:, stage]].sum()

    def nll(lam: float) -> float:
        scores = np.where(left, lam * races.logp, -np.inf)
        return -(lam * chosen_logp - logsumexp(scores, axis=1).sum())

    result = minimize_scalar(nll, bounds=LAMBDA_BOUNDS, method="bounded", options={"xatol": 1e-9})
    return float(result.x)


def _fit(races: _Races, family: str) -> OrderingFit:
    lambdas = (_fit_stage(races, 1), _fit_stage(races, 2))
    digest = hashlib.sha256()
    digest.update(json.dumps([family, list(lambdas)]).encode())
    digest.update("\n".join(map(str, races.race_ids)).encode())
    return OrderingFit(family, lambdas, digest.hexdigest()[:12], len(races.race_ids))


def fit_ordering(frame: pd.DataFrame, *, family: str) -> OrderingFit:
    """Fit (lambda2, lambda3) by maximum likelihood of each race's observed first three.

    Races with a dead heat in the first three, or without one horse each in first, second and
    third, are left out; n_races counts the races used.
    """
    _check_family(family)
    return _fit(_prepare(frame), family)


def read_family(path: Path = CONFIG_PATH) -> str:
    """The ordering family the owner confirmed, from the `family` key of ordering_v0.yaml."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: the ordering family is an owner decision; "
            "add `family: <name>` there once it is confirmed"
        )
    data = yaml.safe_load(path.read_text()) or {}
    family = data.get("family") if isinstance(data, dict) else None
    if not isinstance(family, str) or not family:
        raise ValueError(f"{path} has no `family` key naming the ordering family")
    return family


def _read_oof(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=COLUMNS)
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=COLUMNS, dtype={"race_id": str})
    raise ValueError(f"{path}: the OOF file must be .parquet or .csv")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=CONFIG_PATH.parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def run_fit(oof_path: Path, out_dir: Path | None = None) -> dict:
    """Fit on out-of-fold Win probabilities joined to finish_pos; write the fit and its manifest."""
    family = read_family()
    _check_family(family)
    oof_path = Path(oof_path)
    races = _prepare(_read_oof(oof_path))
    fit = _fit(races, family)
    out_dir = Path(out_dir) if out_dir is not None else data_dir() / "models" / "japan" / "v0"
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_path = out_dir / f"ordering-{fit.fit_id}.json"
    manifest_path = out_dir / f"manifest-ordering-{fit.fit_id}.json"
    manifest = {
        "fit_id": fit.fit_id,
        "family": fit.family,
        "lambdas": list(fit.lambdas),
        "n_races": fit.n_races,
        "races_left_out": {
            "dead_heat_in_first_three": races.n_dead_heat,
            "incomplete_first_three": races.n_incomplete,
        },
        "oof": {"path": str(oof_path), "sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest()},
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit_path": str(fit_path),
        "manifest_path": str(manifest_path),
    }
    fit_path.write_text(fit.to_json())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m horsetrack.countries.japan.models.v0.ordering")
    parser.add_argument("--fit", action="store_true", required=True, help="fit the discounts")
    parser.add_argument("--oof", type=Path, required=True, help="out-of-fold Win probabilities")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)
    print(json.dumps(run_fit(args.oof, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
