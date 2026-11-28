"""A carry-over prior for C3's same-day variant on a day's first races (#328, epic #73).

#96's same-day variant of a (day, track, surface) starts from N(0, tau^2) and moves only once a
race of that (day, track, surface) is resulted. Before then (26% of #305's scored races at T-2) it
is its prior. `CarryOverPrior` gives `SpeedModel(variant_prior=...)` a prior that carries over what
is known before the decision time t, in one of the owner's three pre-registered kinds (#328,
2026-11-28):

- previous_day (a): the posterior variant of the same track and surface on its latest earlier race
  day with a result usable at t, decayed toward the base by rho = 0.5 ** (gap_days / half_life):
  mean rho m', sd^2 = rho^2 s'^2 + (1 - rho^2) tau^2.
- other_surface (b): the posterior variant of the same day's other surface at the same track (turf
  for dirt, dirt for turf) from its results usable at t, coupled by w = surface_weight in [0, 1]:
  mean w m_o, sd^2 = w^2 s_o^2 + (1 - w^2) tau^2. A surface with no other one (e.g. "unknown") has
  none: the base.
- both (c): (a) and (b) by their information over the base: p = 1/sd_a^2 + 1/sd_b^2 - 1/tau^2,
  mean (mean_a / sd_a^2 + mean_b / sd_b^2) / p, sd p^-0.5.

The base is N(0, model.tau^2). A result's residual is top3_time_s - model.par(cell); a result in a
cell the fit never saw is skipped. A result is usable at t when its result_available_utc is
strictly before t (a null time never is). A day's posterior from the base given its usable
residuals is the model's own (`SpeedModel`'s rule for its variant): the closed form under the
normal error (precision 1/tau^2 + n/sigma^2), the reweighted mode under the Student-t. No kind reads
the target day's own (track, surface), a later day or another track. The model's own injected prior
is never read, and the fit never reads this one (#96's ruling).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from datetime import date
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from horsetrack.countries.japan.models.speed import SpeedModel

PRIOR_KINDS = ("previous_day", "other_surface", "both")
OTHER_SURFACE = {"turf": "dirt", "dirt": "turf"}
VARIANT_KEYS = ("track", "surface")

__all__ = ["PRIOR_KINDS", "CarryOverPrior"]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number: {value!r}")
    return float(value)


def _instant_ns(at: Any) -> int:
    stamp = pd.Timestamp(at)
    if stamp.tzinfo is None:
        raise ValueError(f"at_utc must be tz-aware: {at!r}")
    return int(stamp.tz_convert("UTC").as_unit("ns").value)


class CarryOverPrior:
    """`prior(race_date, track, surface, *, at_utc) -> (mean, sd)`; `prior.at(t)` binds t."""

    def __init__(
        self,
        model: SpeedModel,
        results: pd.DataFrame,
        *,
        kind: str,
        half_life_days: float | None = None,
        surface_weight: float | None = None,
    ) -> None:
        if kind not in PRIOR_KINDS:
            raise ValueError(f"kind must be one of {PRIOR_KINDS}, got {kind!r}")
        self.kind = kind
        self.half_life_days = self.surface_weight = None
        if kind in ("previous_day", "both"):
            h = _number(half_life_days, "half_life_days")
            if h <= 0:
                raise ValueError(f"half_life_days must be > 0: {half_life_days!r}")
            self.half_life_days = h
        if kind in ("other_surface", "both"):
            w = _number(surface_weight, "surface_weight")
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"surface_weight must be in [0, 1]: {surface_weight!r}")
            self.surface_weight = w
        keys = model.spec
        if tuple(keys.variant_keys) != VARIANT_KEYS or not set(VARIANT_KEYS) <= set(keys.par_keys):
            raise ValueError(f"the model's variant keys must be {VARIANT_KEYS}, both par keys")
        if model.tau is None:
            raise RuntimeError("the speed model is not fitted")
        self._model = model
        self.tau = float(model.tau)
        self._groups, self._days = self._index(model, results)
        self._posts: dict[tuple, tuple[float, float]] = {}

    @staticmethod
    def _index(model: SpeedModel, results: pd.DataFrame) -> tuple[dict, dict]:
        """(day, track, surface) -> (result times in ns, residuals), in result-time order, and
        (track, surface) -> its sorted result days."""
        par_keys = model.spec.par_keys
        columns = ["race_date", "result_available_utc", "top3_time_s", *par_keys]
        missing = [c for c in columns if c not in results.columns]
        if missing:
            raise ValueError(f"results lacks column(s): {', '.join(missing)}")
        frame = results.loc[:, list(dict.fromkeys(columns))]
        frame = frame[frame["top3_time_s"].notna() & frame["result_available_utc"].notna()]
        groups: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        days: dict[tuple, list[date]] = {}
        if frame.empty:
            return groups, days
        avail = pd.to_datetime(frame["result_available_utc"])
        if not isinstance(avail.dtype, pd.DatetimeTZDtype):
            raise ValueError("result_available_utc must be tz-aware (UTC)")
        avail = avail.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
        avail = avail.astype(np.int64)
        race_days = pd.to_datetime(frame["race_date"]).dt.date.tolist()
        cells = list(zip(*(frame[k].tolist() for k in par_keys), strict=True))
        pars: dict[tuple, float | None] = {}
        for cell in set(cells):
            try:
                pars[cell] = model.par(**dict(zip(par_keys, cell, strict=True)))
            except KeyError:
                pars[cell] = None
        y = frame["top3_time_s"].to_numpy(dtype=float)
        rows: dict[tuple, list[int]] = {}
        for i, (day, cell) in enumerate(zip(race_days, cells, strict=True)):
            if pars[cell] is None:
                continue
            group = dict(zip(par_keys, cell, strict=True))
            rows.setdefault((day, group["track"], group["surface"]), []).append(i)
        for key, idx in rows.items():
            a = avail[idx]
            r = y[idx] - np.array([pars[cells[i]] for i in idx])
            order = np.argsort(a, kind="stable")
            groups[key] = (a[order], r[order])
            days.setdefault(key[1:], []).append(key[0])
        for value in days.values():
            value.sort()
        return groups, days

    # -- the pieces -----------------------------------------------------------------------------

    def _post(self, day: date, track: Any, surface: Any, at_ns: int) -> tuple[float, float] | None:
        """The posterior of that (day, track, surface)'s variant from the base given its results
        usable at t; None when it has none."""
        got = self._groups.get((day, track, surface))
        if got is None:
            return None
        n = int(np.searchsorted(got[0], at_ns, side="left"))
        if n == 0:
            return None
        key = (day, track, surface, n)
        if key not in self._posts:
            self._posts[key] = self._model._posterior(got[1][:n], 0.0, self.tau)
        return self._posts[key]

    def _previous_day(self, race_date: date, track: Any, surface: Any, at_ns: int):
        days = self._days.get((track, surface), [])
        for j in range(bisect.bisect_left(days, race_date) - 1, -1, -1):
            post = self._post(days[j], track, surface, at_ns)
            if post is not None:
                rho = 0.5 ** ((race_date - days[j]).days / self.half_life_days)
                return self._shrunk(post, rho)
        return 0.0, self.tau

    def _other_surface(self, race_date: date, track: Any, surface: Any, at_ns: int):
        other = OTHER_SURFACE.get(surface)
        post = None if other is None else self._post(race_date, track, other, at_ns)
        if post is None:
            return 0.0, self.tau
        return self._shrunk(post, self.surface_weight)

    def _shrunk(self, post: tuple[float, float], k: float) -> tuple[float, float]:
        m, s = post
        return k * m, math.sqrt(k * k * s * s + (1.0 - k * k) * self.tau**2)

    # -- reads ----------------------------------------------------------------------------------

    def __call__(
        self, race_date: date, track: Any, surface: Any, *, at_utc: Any
    ) -> tuple[float, float]:
        at_ns = _instant_ns(at_utc)
        if self.kind == "previous_day":
            return self._previous_day(race_date, track, surface, at_ns)
        if self.kind == "other_surface":
            return self._other_surface(race_date, track, surface, at_ns)
        (ma, sa), (mb, sb) = (
            self._previous_day(race_date, track, surface, at_ns),
            self._other_surface(race_date, track, surface, at_ns),
        )
        pa, pb = sa**-2, sb**-2
        p = pa + pb - self.tau**-2
        return (ma * pa + mb * pb) / p, p**-0.5

    def at(self, at_utc: Any) -> Callable[[date, Any, Any], tuple[float, float]]:
        """The three-argument callable `SpeedModel(variant_prior=...)` takes, at t = at_utc."""
        _instant_ns(at_utc)

        def bound(race_date: date, track: Any, surface: Any) -> tuple[float, float]:
            return self(race_date, track, surface, at_utc=at_utc)

        return bound
