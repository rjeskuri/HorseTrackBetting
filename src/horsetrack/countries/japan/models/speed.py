"""The C3 race-level finish-speed model: par, day variant and E[top-3 time] (#96, epic #73).

    top3_time_r = par(cell_r) + v(day_r, group_r) + eps_r

- The cell is the race's values of `spec.par_keys` (track, surface, distance, layout, rail,
  class_level); class enters as a category, with no scale imposed on it.
- The day is the race's JST calendar date (from `race_utc`), the group its values of
  `spec.variant_keys` (track, surface). v is a random effect per (day, group), N(0, tau^2).
- eps is Student-t with `t_df` degrees of freedom and scale sigma (normal with SD sigma when None).

`fit` is maximum likelihood by conditional maximisation, one step of each per iteration: the
Student-t error is the usual scale mixture, so each race carries the weight
(t_df + 1) / (t_df + e^2 / sigma^2) and a slow race barely moves its par; given the weights and the
pars, (sigma^2, tau^2) maximise the likelihood of the residuals from par with the variants
integrated out (profiled over tau^2 / sigma^2, solved to machine precision; the weights count as
known precisions there, an approximation that understates tau when the day variance is small next
to the race noise); the variants are then their posterior means and each par the weighted mean of
its races' times less their variants. It starts from the cell medians and iterates to a fixed
tolerance, so the same frame gives the same fit. sigma and tau are floored at 0.01 s, a tenth of
the timing resolution. A race without a top-3 time is left out; what `top3_time_s` measures is the
caller's choice (the frame's builder). The fit always
uses the N(0, tau^2) prior, never an injected one: the level of a cell's par and the level of the
variants of the days it ran on are told apart only by the variants' prior mean, so an injected
prior with a non-zero mean would shift the par of every cell linked to that day. The par is the
cell's time on a day of average going.

`fit(races)` estimates the pars, sigma and tau and nothing else: the races it is fitted on never
feed `variant`. `observe(results)` gives it the results of races on the days to be predicted, each
one's residual taken against the frozen fitted par; the pars, sigma and tau never change. The split
is what keeps the same-day variant as of t: a par is a mean over every race of its cell in the fit
frame, whatever its `result_available_utc`, so had the variant read residuals from a par fitted on
the day itself, a race resulted after t (the target race included) would move the variant at t
through the par. For the same reason `observe` refuses a race on a (day, group) the fit frame had;
that (day, group) stays at the prior. Observations add up over calls, and a race id observed twice
is refused; a race in a cell the fit never saw has no par to take a residual from, so it is skipped
and counted in `n_unseen`; a race without a top-3 time is left out. The whole frame is checked
before anything is kept, so a refused frame leaves the model as it was. Fitting again clears the
observations.

`variant(day, *group, at_utc)` is the posterior of that day's v given the observed residuals of
that day's races in the group whose `result_available_utc` is strictly before `at_utc` (a null
time is never available), from the prior `variant_prior(race_date, *group) -> (mean, sd)` when one
is injected and N(0, tau^2) otherwise. With `same_day_variant=False` the day's own races never
count and the answer is the prior, as it is when nothing is available. Under the Student-t error
the posterior mean is the mode found by the same reweighting, started from the prior mean, and the
SD is that of the reweighted normal posterior.

`expected_top3_time(race, at_utc, pace_s=0.0)` is par(cell) + the variant's mean at t + pace_s,
the race's pace term in seconds (#97): the caller computes it (#97's `speed_horse` passes its
fitted race-level pace term), and a pace_s that is not a finite number is a ValueError.
`par` of a cell the fit never saw raises KeyError; the variant of a (day, group) with nothing
observed is the prior. No layer here is adopted (D16).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

JST = "Asia/Tokyo"
RESULT_COLUMNS = ("race_id", "race_utc", "top3_time_s", "result_available_utc")
MAX_ITER = 20_000
TOL = 1e-10  # largest change of any parameter (s) between two iterations at convergence
VAR_FLOOR = 0.01**2  # s^2: sigma and tau are at least a tenth of the timing resolution
LOG_RATIO_BOUNDS = (math.log(1e-8), math.log(1e8))  # search range of log(tau^2 / sigma^2)

__all__ = ["SpeedModel", "SpeedSpec"]


@dataclass(frozen=True)
class SpeedSpec:
    """Which columns key the par and the variant, whether the day's own results count, the error."""

    par_keys: tuple[str, ...]
    variant_keys: tuple[str, ...]
    same_day_variant: bool
    t_df: float | None

    def __post_init__(self) -> None:
        for name in ("par_keys", "variant_keys"):
            keys = getattr(self, name)
            ok = isinstance(keys, tuple) and keys and all(isinstance(k, str) and k for k in keys)
            if not ok or len(set(keys)) != len(keys):
                raise ValueError(f"{name} must be a non-empty tuple of distinct names: {keys!r}")
            reserved = set(keys) & set(RESULT_COLUMNS)
            if reserved:
                raise ValueError(f"{name} may not name {sorted(reserved)}")
        if not isinstance(self.same_day_variant, bool):
            raise TypeError(f"same_day_variant must be a bool: {self.same_day_variant!r}")
        df = self.t_df
        if df is not None and (
            isinstance(df, bool) or not isinstance(df, Real) or not math.isfinite(df) or df <= 0
        ):
            raise ValueError(f"t_df must be a positive number or None: {df!r}")


def _utc_ns(values: pd.Series, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Nanoseconds since the epoch and a known-mask; a known value must be tz-aware."""
    stamps = pd.to_datetime(values)
    known = stamps.notna().to_numpy()
    if not known.any():
        return np.zeros(len(stamps), dtype=np.int64), known
    if not isinstance(stamps.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} must be tz-aware (UTC)")
    naive = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return naive.to_numpy("datetime64[ns]").astype(np.int64), known


def _instant_ns(at: Any, name: str) -> int:
    stamp = pd.Timestamp(at)
    if stamp.tzinfo is None:
        raise ValueError(f"{name} must be tz-aware: {at!r}")
    return int(stamp.tz_convert("UTC").as_unit("ns").value)


def _jst_date(race_utc: Any) -> date:
    stamp = pd.Timestamp(race_utc)
    if stamp.tzinfo is None:
        raise ValueError(f"race_utc must be tz-aware: {race_utc!r}")
    return stamp.tz_convert(JST).date()


def _variances(
    r: np.ndarray, w: np.ndarray, group: np.ndarray, n_groups: int
) -> tuple[float, float]:
    """(sigma^2, tau^2) maximising the likelihood of the residuals r from par, each group's
    v ~ N(0, tau^2) integrated out and each race's error variance sigma^2 / w.

    With lam = tau^2 / sigma^2, W_g and S_g a group's sums of w and w * r, and Q the sum of
    w * r^2, sigma^2(lam) = (Q - sum_g lam S_g^2 / (1 + lam W_g)) / n, and the profile's derivative
    in lam is proportional to sum_g W_g / (1 + lam W_g) - sum_g S_g^2 / (1 + lam W_g)^2 / sigma^2.
    """
    big_w = np.bincount(group, w, n_groups)
    s2 = np.bincount(group, w * r, n_groups) ** 2
    q, n = float(w @ (r * r)), len(r)

    def sigma2(lam: float) -> float:
        return max((q - float(np.sum(lam * s2 / (1.0 + lam * big_w)))) / n, VAR_FLOOR)

    def slope(log_lam: float) -> float:
        lam = math.exp(log_lam)
        d = 1.0 + lam * big_w
        return float(np.sum(big_w / d)) - float(np.sum(s2 / d**2)) / sigma2(lam)

    lo, hi = LOG_RATIO_BOUNDS
    if slope(lo) >= 0:
        log_lam = lo
    elif slope(hi) <= 0:
        log_lam = hi
    else:
        log_lam = brentq(slope, lo, hi, xtol=1e-14)
    lam = math.exp(log_lam)
    out = sigma2(lam)
    return out, max(lam * out, VAR_FLOOR)


def _codes(keys: list[tuple]) -> tuple[np.ndarray, list[tuple]]:
    """Integer codes in first-seen order, and the distinct keys."""
    index: dict[tuple, int] = {}
    codes = np.fromiter((index.setdefault(k, len(index)) for k in keys), np.int64, len(keys))
    return codes, list(index)


class SpeedModel:
    """Par per cell and a (day, group) variant; `fit` and `observe` return the model itself."""

    def __init__(self, spec: SpeedSpec, *, variant_prior: Callable | None = None) -> None:
        self.spec = spec
        self._prior = variant_prior
        self._par: dict[tuple, float] | None = None
        self._fit_days: frozenset[tuple] = frozenset()  # the (day, *group) keys of the fit frame
        # observed (day, *group) -> (result times, residuals from par), in result-time order
        self._days: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._observed_ids: set = set()
        self.sigma: float | None = None  # the error's scale (its SD when t_df is None), s
        self.tau: float | None = None  # the default variant prior's SD (the random effect's), s
        self.n_races = 0
        self.n_iter = 0
        self.n_observed = 0  # observed races with a top-3 time in a fitted cell
        self.n_unseen = 0  # observed races skipped: their cell has no par

    # -- fit ------------------------------------------------------------------------------------

    def _checked(self, races: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple], list[tuple]]:
        """The races with a top-3 time, after checking the frame, and their cells and
        (day, *group) keys."""
        spec = self.spec
        keys = list(dict.fromkeys((*spec.par_keys, *spec.variant_keys)))
        missing = [c for c in (*RESULT_COLUMNS, *keys) if c not in races.columns]
        if missing:
            raise ValueError(f"races lacks column(s): {', '.join(missing)}")
        if not races["race_id"].is_unique:
            raise ValueError("races has duplicate race ids")
        frame = races[races["top3_time_s"].notna()]
        nulls = [k for k in (*keys, "race_utc") if frame[k].isna().any()]
        if nulls:
            raise ValueError(f"null value(s) in {', '.join(nulls)}")
        if not np.isfinite(frame["top3_time_s"].to_numpy(dtype=float)).all():
            raise ValueError("top3_time_s has a non-finite value")
        _utc_ns(frame["race_utc"], "race_utc")
        _utc_ns(frame["result_available_utc"], "result_available_utc")
        if frame.empty:
            return frame, [], []
        cells = list(zip(*(frame[k].tolist() for k in spec.par_keys), strict=True))
        days = pd.to_datetime(frame["race_utc"]).dt.tz_convert(JST).dt.date.tolist()
        groups = list(zip(days, *(frame[k].tolist() for k in spec.variant_keys), strict=True))
        return frame, cells, groups

    def fit(self, races: pd.DataFrame) -> SpeedModel:
        frame, cell_keys, group_keys = self._checked(races)
        if frame.empty:
            raise ValueError("no race has a top-3 time to fit on")
        y = frame["top3_time_s"].to_numpy(dtype=float)
        cell, cells = _codes(cell_keys)
        group, groups = _codes(group_keys)

        par, sigma2, tau2, n_iter = self._solve(y, cell, group, len(cells), len(groups))
        self._par = {c: float(p) for c, p in zip(cells, par, strict=True)}
        self.sigma, self.tau = math.sqrt(sigma2), math.sqrt(tau2)
        self.n_races, self.n_iter = len(y), n_iter
        self._fit_days = frozenset(groups)
        self._days, self._observed_ids = {}, set()
        self.n_observed = self.n_unseen = 0
        return self

    def observe(self, results: pd.DataFrame) -> SpeedModel:
        """Keep the residuals from the fitted par of races on days to be predicted (see the module
        docstring); added to earlier observations."""
        pars = self._fitted()
        frame, cell_keys, group_keys = self._checked(results)
        if frame.empty:
            return self
        in_fit = sorted({g for g in group_keys if g in self._fit_days}, key=repr)
        if in_fit:
            raise ValueError(
                f"{len(in_fit)} (day, {', '.join(self.spec.variant_keys)}) of the results were in "
                f"the fit frame, e.g. {in_fit[0]}: their residuals would carry their own results "
                "back through the par"
            )
        ids = frame["race_id"].tolist()
        again = self._observed_ids.intersection(ids)
        if again:
            raise ValueError(f"{len(again)} race(s) already observed, e.g. {next(iter(again))!r}")
        y = frame["top3_time_s"].to_numpy(dtype=float)
        avail, known = _utc_ns(frame["result_available_utc"], "result_available_utc")
        seen = [c in pars for c in cell_keys]
        rows: dict[tuple, list[int]] = {}
        for i, g in enumerate(group_keys):
            if seen[i] and known[i]:
                rows.setdefault(g, []).append(i)
        merged = {}
        for g, idx in rows.items():
            a = avail[idx]
            r = y[idx] - np.array([pars[cell_keys[i]] for i in idx])
            if g in self._days:
                a, r = np.r_[self._days[g][0], a], np.r_[self._days[g][1], r]
            order = np.argsort(a, kind="stable")
            merged[g] = (a[order], r[order])
        # everything is checked: keep it
        self._days.update(merged)
        self._observed_ids.update(ids)
        self.n_observed += sum(seen)
        self.n_unseen += len(seen) - sum(seen)
        return self

    def _solve(self, y, cell, group, n_cells, n_groups):
        """Pars, sigma^2 and tau^2 by conditional maximisation (see the module docstring)."""
        df = self.spec.t_df
        par = pd.Series(y).groupby(cell).median().to_numpy()
        v = np.zeros(n_groups)
        r = y - par[cell]
        mad2 = (1.4826 * np.median(np.abs(r - np.median(r)))) ** 2
        sigma2, tau2 = max(mad2 or float(np.var(r)), VAR_FLOOR), math.nan
        w_cell = np.bincount(cell, minlength=n_cells).astype(float)
        for n_iter in range(1, MAX_ITER + 1):
            w = np.ones_like(y) if df is None else (df + 1.0) / (df + (r - v[group]) ** 2 / sigma2)
            if df is not None:
                w_cell = np.bincount(cell, w, n_cells)
            sigma2_new, tau2_new = _variances(r, w, group, n_groups)
            prec = 1.0 / tau2_new + np.bincount(group, w, n_groups) / sigma2_new
            v_new = np.bincount(group, w * r, n_groups) / sigma2_new / prec
            par_new = np.bincount(cell, w * (y - v_new[group]), n_cells) / w_cell
            change = max(
                float(np.max(np.abs(par_new - par))),
                float(np.max(np.abs(v_new - v))),
                abs(math.sqrt(sigma2_new) - math.sqrt(sigma2)),
                abs(math.sqrt(tau2_new) - math.sqrt(tau2)) if n_iter > 1 else math.inf,
            )
            par, v, sigma2, tau2 = par_new, v_new, sigma2_new, tau2_new
            r = y - par[cell]
            if change <= TOL:
                return par, sigma2, tau2, n_iter
        raise RuntimeError(f"the speed model did not converge in {MAX_ITER} iterations")

    # -- reads ----------------------------------------------------------------------------------

    def _fitted(self) -> dict[tuple, float]:
        if self._par is None:
            raise RuntimeError("the speed model is not fitted")
        return self._par

    def _prior_for(self, race_date: date, group: tuple) -> tuple[float, float]:
        if self._prior is None:
            return 0.0, self.tau
        mean, sd = self._prior(race_date, *group)
        mean, sd = float(mean), float(sd)
        if not (math.isfinite(mean) and math.isfinite(sd) and sd > 0):
            raise ValueError(f"variant_prior gave {(mean, sd)} for {(race_date, *group)}")
        return mean, sd

    def par(self, **keys: Any) -> float:
        pars = self._fitted()
        if set(keys) != set(self.spec.par_keys):
            raise TypeError(
                f"par takes exactly the keywords {self.spec.par_keys}, got {tuple(keys)}"
            )
        cell = tuple(keys[k] for k in self.spec.par_keys)
        try:
            return pars[cell]
        except KeyError:
            raise KeyError(f"no par for the unseen cell {cell}") from None

    def variant(self, race_date: date, *group: Any, at_utc: datetime) -> tuple[float, float]:
        self._fitted()
        if isinstance(race_date, datetime) or not isinstance(race_date, date):
            raise TypeError(f"race_date must be a datetime.date, got {type(race_date).__name__}")
        if len(group) != len(self.spec.variant_keys):
            raise TypeError(f"variant takes the race date and {self.spec.variant_keys}")
        at_ns = _instant_ns(at_utc, "at_utc")
        m0, s0 = self._prior_for(race_date, group)
        day = self._days.get((race_date, *group)) if self.spec.same_day_variant else None
        if day is None:
            return m0, s0
        avail, resid = day
        return self._posterior(resid[: np.searchsorted(avail, at_ns, side="left")], m0, s0)

    def _posterior(self, r: np.ndarray, m0: float, s0: float) -> tuple[float, float]:
        if r.size == 0:
            return m0, s0
        df, sigma2, p0 = self.spec.t_df, self.sigma**2, 1.0 / s0**2
        mean, w = m0, np.ones_like(r)
        for _ in range(MAX_ITER):
            if df is not None:
                w = (df + 1.0) / (df + (r - mean) ** 2 / sigma2)
            prec = p0 + float(w.sum()) / sigma2
            new = (m0 * p0 + float(w @ r) / sigma2) / prec
            done = df is None or abs(new - mean) <= TOL
            mean = new
            if done:
                break
        return float(mean), float(1.0 / math.sqrt(prec))

    def expected_top3_time(self, race: Mapping, *, at_utc: datetime, pace_s: float = 0.0) -> float:
        if isinstance(pace_s, bool) or not isinstance(pace_s, Real) or not math.isfinite(pace_s):
            raise ValueError(f"pace_s must be a finite number of seconds: {pace_s!r}")
        par = self.par(**{k: race[k] for k in self.spec.par_keys})
        group = tuple(race[k] for k in self.spec.variant_keys)
        mean, _ = self.variant(_jst_date(race["race_utc"]), *group, at_utc=at_utc)
        return par + mean + pace_s
