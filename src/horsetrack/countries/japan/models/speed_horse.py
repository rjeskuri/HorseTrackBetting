"""The C3 horse-level finish-speed model: theta_i(t), p_speed and the same-day bias (#97, epic #73).

08 §3.2 C3: theta_i(t) + pace_r * style_i + w * weight_ir + eps_ir with Student-t eps, per-horse
relative speed and SD, and p_speed via Henery (normal) or PL strengths exp(-mu_i / s). Here

    finish_time_ir = baseline_ir + theta_i(t_r) + b[style_ir] * pace_r + w * (weight_ir - w_ref)
                     + eps_ir

`HorseSpeed(*, drift_var_per_day, t_df)` works on a fitted and observed #96 `SpeedModel`:

- Baseline. A run's baseline is #96's expected top-3 time of its race as of the run's off
  (`race_utc`, or its publication when that is earlier): the par plus the day variant from the
  day's results published strictly before then, so never the run's own race. V, the variant's
  posterior variance then, joins the run's error. This baseline is fixed: it reads no result
  published after its own run, so a run whose result is published before t brings nothing
  published at or after t. The par is the speed model's own, fitted on its frame.
- A run on a (day, group) of the speed model's fit frame (a training-window day: #96 refuses to
  observe it, so its as-of variant is the prior) has a second baseline, the par plus the fit's
  in-sample variant of that (day, group) (`SpeedModel.in_sample_variant`), whose variance is V
  there (the owner on #97, 2026-11-26, Q3 (b)). It is used at t only when every race of that
  (day, group) in the speed model's fit frame, with a top-3 time or without, had its result
  published strictly before t (a null publication: never); until then the as-of baseline stands,
  so nothing published at or after t reaches theta at t through it.
- theta_i is a random walk: its variance grows by `drift_var_per_day` (s^2) per day elapsed
  between offs, from N(m0, s0^2), the population prior, at the horse's first run.
- pace_r is the field's mean early position from earlier races (the owner on #97, 2026-11-26,
  Q2 (c)): the mean of the race's rows' non-null `early_pos_mean3` (#129's mean over the horse's
  newest 3 earlier starts of (first-corner position - 1) / (field size - 1), the caller's, as of
  t), less its mean over the fit runs (a race with no value left out). A race with no value has
  no pace term. Each style has its slope b[style] and a runner without a style none. Draw is no
  term (08's formula has none; `same_day_bias` carries the draw). w_ref is the fit runs' mean
  weight carried.
- eps is Student-t with `t_df` degrees of freedom and scale sqrt(sigma^2 + V) (normal with that SD
  when t_df is None).

The forward filter is Kalman's. Under the t error each run's update is the posterior mode of theta
found by reweighting from the prior mean, the run weighing (t_df + 1) / (t_df + r^2 / S) with r its
residual from the mode and S = sigma^2 + V, and the updated variance is that of the reweighted
normal update (as #96's variant).

`fit(runs, speed)` fits the global terms (sigma, s0, m0, w, b, with w_ref, pace_ref and gamma) on
the whole runs frame, not as of t, by maximum likelihood over the filter's one-step prediction
errors (exact under the normal error): given (sigma, s0) and the runs' weights the filter is
linear in the data, so (w, b, m0) are the generalised least squares of each column's innovations,
and (sigma, s0) maximise that profile (Nelder-Mead on their logs, each at least SCALE_FLOOR).
Under the t error the weights are then recomputed by the forward filter at the new terms, until
the terms settle. The fit reads each run as `predict` does at a t after every publication: a
fit-day run against its in-sample baseline when its (day, group) is ever fully resulted. It keeps
the runs as the horses' history and clears earlier observations. A run without a finish time is
no observation; one with a time and no publication (never resulted) is dropped, from the terms and
the history alike, and counted in `n_unpublished`; one whose cell has no par is skipped and
counted in `n_unseen`.

`observe(runs)` (fit's columns) adds later runs to the horses' history only, each residual taken
against the frozen global terms (w_ref and pace_ref included: nothing is re-centred on the
observed frame) on the baselines a fit run has; the global terms never change, as #96's
fit/observe split. A race id already fitted or observed is refused, the whole frame is checked
before anything is kept, observations add up over calls, and fitting again clears them.

The global terms read every fit run, so `predict` refuses an at_utc at or before the later of the
latest `result_available_utc` over every row of the fit frame (timed or not: a missing finish time
is a result too) and the latest instant after which a fit run's in-sample baseline is used;
observations never move that bound. The speed model's par, fitted on #96's own frame, is outside
it.

`predict(race_runners, at_utc)`: theta_i is the filter over that horse's runs, fitted or observed,
whose result was published strictly before at_utc (null: never), in off order, each against the
baseline it has at at_utc, then drifted to the card's off (no drift before a first run, none
backwards); a horse with no such run has the population prior.
gamma, the fit runs' style-weighted mean of b, is the race-level pace term: the card's expected
top-3 time is #96's at at_utc with pace_s = gamma * pace_r, and

    mu_s = that time + m0 + theta~_i + (b[style] - gamma) * pace_r + w * (weight - w_ref)
    sd_s = sqrt(Var theta_i + k * (sigma^2 + V_card))

the runner's predictive SD, V_card the card's variant variance at at_utc and sigma the error's
scale: k = 1 under the normal, t_df / (t_df - 2) under the Student-t with t_df > 2 (its variance
per scale^2), and sd_s is +inf when t_df <= 2 (that t has no finite variance; mu_s stays finite).
Result columns on the card are ignored.

`p_speed(mu, scale, method)`: one race. 'pl' is softmax(-mu / scale); its scale is for the caller
to fit, as the conditional logit (#87's `fit_conditional_logit`) of the winner on -mu, whose
coefficient is 1 / scale. 'henery' is P(i's time is the smallest) for independent N(mu_i,
scale^2) times, the trapezoid rule over a grid of HENERY_STEP scales spanning HENERY_SPAN scales
either side of the quickest mean (spectrally accurate here: the integrand is smooth and negligible
at both ends), renormalised to sum to 1.

`same_day_bias(results, at_utc, venue)`: the `venue` races of the JST day of at_utc whose result
was published strictly before at_utc. A finisher in place pos of nf (nf >= 2) scores
(nf - pos) / (nf - 1) - 0.5 (#138's score: 0 is no bias); its draw third over the race's rows is
#129's rule and its style the one given for that race. Each level's bias is the mean score of its
finishers there (0 with none: no shrinkage, `n_races` lets the caller weigh it).

No layer here is adopted (D16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from numbers import Real

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.special import log_ndtr

from horsetrack.countries.japan.handicap.factors import STYLES, _draw_group
from horsetrack.countries.japan.handicap.track_bias import DRAW_LEVELS, LEVELS
from horsetrack.countries.japan.models.speed import JST, SpeedModel, _instant_ns, _jst_date, _utc_ns

CARD_COLUMNS = ("race_id", "race_utc", "horse_id", "style", "early_pos_mean3", "weight_kg", "draw")
RUN_COLUMNS = (*CARD_COLUMNS, "finish_time_s", "result_available_utc")
BIAS_COLUMNS = (*CARD_COLUMNS[:3], "venue", "draw", "style", "finish_pos", "result_available_utc")
NS_PER_DAY = 86_400 * 10**9
NEVER = np.iinfo(np.int64).max  # the publication time of a result never published
SCALE_FLOOR = 0.01  # s: sigma and s0 are at least a tenth of the timing resolution (#96's floor)
MAX_ITER = 10_000
MODE_TOL = 1e-12  # s: the posterior mode's largest change at convergence
FIT_TOL = 1e-7  # s: the global terms' largest change between two fit rounds at convergence
SIMPLEX_TOL = 1e-10  # the profile's Nelder-Mead tolerance on log(sigma), log(s0) and the loss
HENERY_SPAN = 10.0  # scales either side of the quickest mean
HENERY_STEP = 0.1  # scales
METHODS = ("pl", "henery")

__all__ = ["HorseSpeed", "p_speed", "same_day_bias"]


def _number(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, Real) and math.isfinite(x)


def _check_frame(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks column(s): {', '.join(missing)}")


def _check_styles(style: pd.Series, name: str) -> np.ndarray:
    """The styles as an object array, None where missing; a label outside STYLES is refused."""
    bad = set(style.dropna()) - set(STYLES)
    if bad:
        raise ValueError(f"{name} has unknown style(s) {sorted(map(str, bad))}; use {STYLES}")
    return style.astype(object).where(style.notna(), None).to_numpy(object)


def _field_pace(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Each row's race's mean non-null early_pos_mean3 over its rows; NaN when it has none."""
    early = frame["early_pos_mean3"].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(early).any():
        raise ValueError(f"{name} has an infinite early_pos_mean3")
    by_race = pd.Series(early, index=frame.index).groupby(frame["race_id"].to_numpy())
    return by_race.transform("mean").to_numpy()


def _centred(pace: np.ndarray, ref: float) -> np.ndarray:
    """pace_r: the pace less the fit runs' mean, 0 (no pace term) for a race without one."""
    return np.where(np.isnan(pace), 0.0, pace - ref)


@dataclass(frozen=True)
class _Steps:
    """Runs sorted by (horse, off), taken a step at a time: step j is each horse's j-th run."""

    rows: tuple[np.ndarray, ...]  # rows[j]: those runs' rows, in horse order
    horse: tuple[np.ndarray, ...]  # horse[j]: their horses' codes
    gap: np.ndarray  # days since the horse's previous run (0 at its first)
    n_horses: int


def _steps(code: np.ndarray, tau: np.ndarray, n_horses: int) -> _Steps:
    """code (0 .. n_horses - 1) and tau (off, ns), sorted by (code, tau)."""
    n = len(code)
    if n == 0:
        return _Steps((), (), np.zeros(0), n_horses)
    first = np.r_[True, code[1:] != code[:-1]]
    start = np.flatnonzero(first)
    rank = np.arange(n) - np.repeat(start, np.diff(np.r_[start, n]))
    gap = np.where(first, 0.0, np.r_[0, np.diff(tau)] / NS_PER_DAY)
    order = np.argsort(rank, kind="stable")
    rows = tuple(np.split(order, np.cumsum(np.bincount(rank))[:-1]))
    return _Steps(rows, tuple(code[r] for r in rows), gap, n_horses)


def _linear_filter(steps: _Steps, z: np.ndarray, r: np.ndarray, q: float, p0: float):
    """The one-step innovations of every column of z (rows, c) and their variances, the state a
    random walk (variance q a day) from N(0, p0) at each horse's first run, observation variance r.
    """
    m = np.zeros((steps.n_horses, z.shape[1]))
    p = np.zeros(steps.n_horses)
    v, f = np.empty_like(z), np.empty(len(z))
    for j, (rows, h) in enumerate(zip(steps.rows, steps.horse, strict=True)):
        pp = np.full(len(rows), p0) if j == 0 else p[h] + q * steps.gap[rows]
        ff = pp + r[rows]
        vv = z[rows] - m[h]
        m[h] += (pp / ff)[:, None] * vv
        p[h] = pp * r[rows] / ff
        v[rows], f[rows] = vv, ff
    return v, f


def _mode_weights(u, s, m, p, df: float) -> np.ndarray:
    """Each run's weight at the posterior mode of theta under the t error (prior N(m, p), scale^2
    s), by reweighting from the prior mean, entry by entry."""
    mean, w = m.copy(), np.empty_like(u)
    active = np.arange(len(u))
    for _ in range(MAX_ITER):
        a = active
        wa = (df + 1.0) / (df + (u[a] - mean[a]) ** 2 / s[a])
        new = (m[a] / p[a] + wa * u[a] / s[a]) / (1.0 / p[a] + wa / s[a])
        w[a] = wa
        done = np.abs(new - mean[a]) <= MODE_TOL
        mean[a] = new
        active = a[~done]
        if active.size == 0:
            return w
    raise RuntimeError(f"the posterior mode of theta did not converge in {MAX_ITER} iterations")


def _forward(steps: _Steps, u, s, q: float, p0: float, df: float | None):
    """The forward filter of u: each run's weight and every horse's final mean and variance."""
    m, p, w = np.zeros(steps.n_horses), np.zeros(steps.n_horses), np.ones(len(u))
    for j, (rows, h) in enumerate(zip(steps.rows, steps.horse, strict=True)):
        pp = np.full(len(rows), p0) if j == 0 else p[h] + q * steps.gap[rows]
        mm, uu, ss = m[h], u[rows], s[rows]
        ww = np.ones(len(rows)) if df is None else _mode_weights(uu, ss, mm, pp, df)
        rr = ss / ww
        ff = pp + rr
        m[h] = mm + pp / ff * (uu - mm)
        p[h] = pp * rr / ff
        w[rows] = ww
    return w, m, p


@dataclass(frozen=True)
class _History:
    """One horse's usable runs in off order: off and publication (ns), residual u from the fitted
    global terms and error scale^2 sigma^2 + V against the as-of baseline, the same against the
    in-sample baseline, and the instant (ns) after which that one is used (NEVER: not at all)."""

    tau: np.ndarray
    pub: np.ndarray
    u: np.ndarray
    s: np.ndarray
    u_in: np.ndarray
    s_in: np.ndarray
    switch: np.ndarray

    def at(self, at_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(off, u, s) of the runs published strictly before at_ns, each on its baseline then."""
        avail = self.pub < at_ns
        later = self.switch[avail] < at_ns
        u = np.where(later, self.u_in[avail], self.u[avail])
        s = np.where(later, self.s_in[avail], self.s[avail])
        return self.tau[avail], u, s

    def joined(self, other: _History) -> _History:
        """Both histories' runs, in off order."""
        cols = [np.r_[getattr(self, f), getattr(other, f)] for f in self.__dataclass_fields__]
        order = np.argsort(cols[0], kind="stable")
        return _History(*(c[order] for c in cols))


@dataclass(frozen=True)
class _Runs:
    """A runs frame read against the speed model: its runs with a finish time, a publication and a
    par (frame order), each with its two baselines' errors e = time - baseline and variances."""

    horse: np.ndarray
    tau: np.ndarray
    pub: np.ndarray
    e: np.ndarray
    vv: np.ndarray
    e_in: np.ndarray
    vv_in: np.ndarray
    switch: np.ndarray
    style: np.ndarray
    weight: np.ndarray
    pace: np.ndarray  # the race's field pace, uncentred (NaN: none)
    race_ids: frozenset  # every race id of the frame
    latest_pub: int  # the latest known publication over every row of the frame (ns; -1: none)
    n_unseen: int  # runs with a time and a publication whose cell has no par
    n_unpublished: int  # runs with a time and no publication


class HorseSpeed:
    """theta_i(t) against #96's baseline, pace x style, weight; `fit` and `observe` return it."""

    def __init__(self, *, drift_var_per_day: float, t_df: float | None) -> None:
        if not _number(drift_var_per_day) or drift_var_per_day < 0:
            raise ValueError(
                f"drift_var_per_day must be a finite number >= 0: {drift_var_per_day!r}"
            )
        if t_df is not None and (not _number(t_df) or t_df <= 0):
            raise ValueError(f"t_df must be a positive number or None: {t_df!r}")
        self.drift_var_per_day = float(drift_var_per_day)
        self.t_df = None if t_df is None else float(t_df)
        self._speed: SpeedModel | None = None
        self._history: dict = {}
        self._race_ids: set = set()  # every race id fitted or observed
        # the latest publication (ns) the global terms read; predict needs an at_utc after it
        self._terms_cutoff: int | None = None
        self.sigma: float | None = None  # the error's scale (its SD under the normal), s
        self.prior_mean: float | None = None  # m0: theta at a first run, s behind the baseline
        self.prior_sd: float | None = None  # s0
        self.weight_slope: float | None = None  # w, s per kg
        self.weight_ref: float | None = None  # w_ref, kg
        self.pace_slopes: dict[str, float] = {}  # b per style, s per unit of pace_r
        self.pace_race: float | None = None  # gamma
        self.pace_ref: float | None = None  # the fit runs' mean field early position
        self.n_runs = 0  # the runs the global terms were fitted on
        self.n_observed = 0  # observed runs kept in the history
        self.n_unseen = 0  # runs, fitted or observed, with a time and no par: skipped
        self.n_unpublished = 0  # runs, fitted or observed, with a time and no publication: dropped
        self.n_iter = 0

    # -- fit ------------------------------------------------------------------------------------

    @staticmethod
    def _race_keys(speed: SpeedModel) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*speed.spec.par_keys, *speed.spec.variant_keys)))

    def fit(self, runs: pd.DataFrame, speed: SpeedModel) -> HorseSpeed:
        """Fit the global terms on runs and keep them as the horses' history; earlier
        observations are cleared (see the module docstring)."""
        r = self._read(runs, speed)
        if r.tau.size == 0:
            raise ValueError("no run has a finish time, a publication and a par to fit on")
        code, horses = pd.factorize(r.horse)
        order = np.lexsort((r.tau, code))
        code, tau = code[order], r.tau[order]
        e_in, vv_in = r.e_in[order], r.vv_in[order]
        style, weight, pace = r.style[order], r.weight[order], r.pace[order]
        known_pace = ~np.isnan(pace)
        pace_ref = float(pace[known_pace].mean()) if known_pace.any() else 0.0
        weight_ref = float(weight.mean())
        x = self._design(weight - weight_ref, style, _centred(pace, pace_ref))
        steps = _steps(code, tau, len(horses))

        # the fit reads every run as predict does after every publication (the module docstring)
        sigma, s0, coef, n_iter = self._fit_terms(steps, code, e_in, x, vv_in)
        # everything is fitted: keep it, the runs as the history
        self._speed = speed
        self.sigma, self.prior_sd, self.n_iter = sigma, s0, n_iter
        self.weight_ref, self.pace_ref = weight_ref, pace_ref
        self.weight_slope = float(coef[0])
        b = coef[1 : 1 + len(STYLES)]
        self.pace_slopes = {s: float(v) for s, v in zip(STYLES, b, strict=True)}
        counts = np.array([np.sum(style == s) for s in STYLES], dtype=float)
        self.pace_race = float(counts @ b / counts.sum()) if counts.sum() else 0.0
        self.prior_mean = float(coef[-1])
        switched = r.switch[r.switch != NEVER]
        self._terms_cutoff = max(r.latest_pub, int(switched.max()) if switched.size else -1)
        self._history, self._race_ids = {}, set()
        self.n_runs, self.n_observed = int(r.tau.size), 0
        self.n_unseen = self.n_unpublished = 0
        self._keep(r)
        return self

    def observe(self, runs: pd.DataFrame) -> HorseSpeed:
        """Add later runs to the horses' history only, against the frozen global terms; added to
        earlier observations (see the module docstring)."""
        speed = self._speed
        if speed is None:
            raise RuntimeError("the horse speed model is not fitted")
        r = self._read(runs, speed)
        again = self._race_ids & r.race_ids
        if again:
            raise ValueError(
                f"{len(again)} race(s) already fitted or observed, e.g. {sorted(again)[0]!r}"
            )
        # everything is checked: keep it
        self.n_observed += int(r.tau.size)
        self._keep(r)
        return self

    def _read(self, runs: pd.DataFrame, speed: SpeedModel) -> _Runs:
        """Check a runs frame and read its usable runs against the speed model."""
        keys = self._race_keys(speed)
        _check_frame(runs, (*RUN_COLUMNS, *keys), "runs")
        nulls = [k for k in ("race_id", "race_utc", "horse_id", *keys) if runs[k].isna().any()]
        if nulls:
            raise ValueError(f"runs has null value(s) in {', '.join(nulls)}")
        tau_all, _ = _utc_ns(runs["race_utc"], "race_utc")
        pub_all, known = _utc_ns(runs["result_available_utc"], "result_available_utc")
        style_all = _check_styles(runs["style"], "runs")
        pace_all = _field_pace(runs, "runs")  # over all the race's rows

        time = runs["finish_time_s"].to_numpy(dtype=float)
        if np.isinf(time).any():
            raise ValueError("finish_time_s has an infinite value")
        timed = ~np.isnan(time)
        weight_all = runs["weight_kg"].to_numpy(dtype=float)
        if not np.isfinite(weight_all[timed]).all():
            raise ValueError("weight_kg has a missing or non-finite value on a run with a time")

        kept = timed & known  # a result never published is never available: dropped
        as_of = np.minimum(tau_all, pub_all)
        base, vvar, base_in, vvar_in, switch = self._baselines(runs, speed, keys, as_of, kept)
        seen = kept & ~np.isnan(base)
        use = np.flatnonzero(seen)
        return _Runs(
            horse=runs["horse_id"].to_numpy(object)[use],
            tau=tau_all[use],
            pub=pub_all[use],
            e=time[use] - base[use],
            vv=vvar[use],
            e_in=time[use] - base_in[use],
            vv_in=vvar_in[use],
            switch=switch[use],
            style=style_all[use],
            weight=weight_all[use],
            pace=pace_all[use],
            race_ids=frozenset(runs["race_id"].tolist()),
            latest_pub=int(pub_all[known].max()) if known.any() else -1,
            n_unseen=int(np.sum(kept & ~seen)),
            n_unpublished=int(np.sum(timed & ~known)),
        )

    def _keep(self, r: _Runs) -> None:
        """Add r's runs to the horses' history, each residual against the global terms."""
        slopes = np.r_[self.weight_slope, [self.pace_slopes[s] for s in STYLES]]
        x = self._design(r.weight - self.weight_ref, r.style, _centred(r.pace, self.pace_ref))
        fitted = x @ slopes + self.prior_mean
        s, s_in = self.sigma**2 + r.vv, self.sigma**2 + r.vv_in
        cols = (r.tau, r.pub, r.e - fitted, s, r.e_in - fitted, s_in, r.switch)
        code, horses = pd.factorize(r.horse)
        order = np.lexsort((r.tau, code))
        code, cols = code[order], tuple(a[order] for a in cols)
        bounds = np.r_[0, np.flatnonzero(np.diff(code)) + 1, len(code)] if len(code) else [0]
        merged = {}
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            h = horses[code[lo]]
            new = _History(*(a[lo:hi] for a in cols))
            merged[h] = new if h not in self._history else self._history[h].joined(new)
        self._history.update(merged)
        self._race_ids |= r.race_ids
        self.n_unseen += r.n_unseen
        self.n_unpublished += r.n_unpublished

    @staticmethod
    def _baselines(runs, speed, keys, as_of, rows):
        """Each `rows` run's expected top-3 time as of `as_of` and the variant's variance then;
        the same against the in-sample variant of its (day, group) and the instant (ns) that
        becomes usable after (NEVER, and the as-of values, off the fit frame's days or when one of
        its races was never published). NaN where the run's cell has no par, or off `rows`."""
        n = len(runs)
        base, vvar = np.full(n, np.nan), np.full(n, np.nan)
        base_in, vvar_in = np.full(n, np.nan), np.full(n, np.nan)
        switch = np.full(n, NEVER, dtype=np.int64)
        values = {k: runs[k].tolist() for k in (*keys, "race_utc")}
        race_ids = runs["race_id"].tolist()
        cache: dict[tuple, tuple[float, float]] = {}
        in_cache: dict = {}
        for i in np.flatnonzero(rows):
            key = (race_ids[i], int(as_of[i]))
            race = None
            if key not in cache:
                race = {k: values[k][i] for k in values}
                cache[key] = _baseline(speed, race, int(as_of[i]))
            base[i], vvar[i] = cache[key]
            if race_ids[i] not in in_cache:
                race = race or {k: values[k][i] for k in values}
                in_cache[race_ids[i]] = _in_sample(speed, race)
            got = in_cache[race_ids[i]]
            base_in[i], vvar_in[i], switch[i] = (base[i], vvar[i], NEVER) if got is None else got
        return base, vvar, base_in, vvar_in, switch

    @staticmethod
    def _design(weight: np.ndarray, style: np.ndarray, pace: np.ndarray) -> np.ndarray:
        """Columns: weight carried (centred by the caller), then pace_r for each style."""
        return np.column_stack([weight, *(np.where(style == s, pace, 0.0) for s in STYLES)])

    def _fit_terms(self, steps: _Steps, code, e, x, vvar):
        """(sigma, s0, coef, rounds): coef is (w, b..., m0) (see the module docstring)."""
        q, df = self.drift_var_per_day, self.t_df
        z = np.column_stack([e, x, np.ones(len(e))])
        weights = np.ones(len(e))

        def profile(log_scale: np.ndarray) -> tuple[float, np.ndarray]:
            sigma, s0 = np.maximum(np.exp(log_scale), SCALE_FLOOR)
            r = (sigma**2 + vvar) / weights
            v, f = _linear_filter(steps, z, r, q, s0**2)
            root = 1.0 / np.sqrt(f)
            a, y = v[:, 1:] * root[:, None], v[:, 0] * root
            coef = np.linalg.lstsq(a, y, rcond=None)[0]
            res = y - a @ coef
            return 0.5 * (float(np.log(f).sum()) + float(res @ res)), coef

        log_scale = np.log(_start_scales(code, e, vvar))
        bounds = [(math.log(SCALE_FLOOR), None)] * 2
        last = None
        for n_iter in range(1, MAX_ITER + 1):
            found = minimize(
                lambda ls: profile(ls)[0],
                log_scale,
                method="Nelder-Mead",
                bounds=bounds,
                options={"xatol": SIMPLEX_TOL, "fatol": SIMPLEX_TOL, "maxiter": 4000},
            )
            log_scale = found.x
            sigma, s0 = (float(v) for v in np.maximum(np.exp(log_scale), SCALE_FLOOR))
            coef = profile(log_scale)[1]
            terms = np.r_[sigma, s0, coef]
            if df is None:
                return sigma, s0, coef, n_iter
            if last is not None and np.max(np.abs(terms - last)) <= FIT_TOL:
                return sigma, s0, coef, n_iter
            last = terms
            u = e - z[:, 1:] @ coef
            weights = _forward(steps, u, sigma**2 + vvar, q, s0**2, df)[0]
        raise RuntimeError(f"the horse speed terms did not settle in {MAX_ITER} rounds")

    # -- predict --------------------------------------------------------------------------------

    def predict(self, race_runners: pd.DataFrame, *, at_utc: datetime) -> pd.DataFrame:
        speed = self._speed
        if speed is None:
            raise RuntimeError("the horse speed model is not fitted")
        at_ns = _instant_ns(at_utc, "at_utc")
        if at_ns <= self._terms_cutoff:
            raise ValueError(
                f"at_utc {pd.Timestamp(at_ns, tz='UTC')} is not after "
                f"{pd.Timestamp(self._terms_cutoff, tz='UTC')}, the latest result the global "
                "terms were fitted on: they would carry a result published at or after at_utc"
            )
        keys = self._race_keys(speed)
        card = race_runners
        _check_frame(card, (*CARD_COLUMNS, *keys), "race_runners")
        nulls = [k for k in ("race_id", "race_utc", "horse_id", *keys) if card[k].isna().any()]
        if nulls:
            raise ValueError(f"race_runners has null value(s) in {', '.join(nulls)}")
        race_ns, _ = _utc_ns(card["race_utc"], "race_utc")
        style = _check_styles(card["style"], "race_runners")
        weight = card["weight_kg"].to_numpy(dtype=float)
        if not np.isfinite(weight).all():
            raise ValueError("race_runners has a missing or non-finite weight_kg")
        pace = _centred(_field_pace(card, "race_runners"), self.pace_ref)

        n = len(card)
        top3, vcard = np.empty(n), np.empty(n)
        values = {k: card[k].tolist() for k in (*keys, "race_utc")}
        done: dict = {}
        for i, rid in enumerate(card["race_id"].tolist()):
            if rid not in done:
                race = {k: values[k][i] for k in values}
                pace_s = float(self.pace_race * pace[i])
                mean = speed.expected_top3_time(race, at_utc=at_utc, pace_s=pace_s)
                group = tuple(race[k] for k in speed.spec.variant_keys)
                _, sd = speed.variant(_jst_date(race["race_utc"]), *group, at_utc=at_utc)
                done[rid] = (mean, sd * sd)
            top3[i], vcard[i] = done[rid]

        horse_ids = card["horse_id"].tolist()
        index = {h: i for i, h in enumerate(dict.fromkeys(horse_ids))}
        mean, var, last = self._theta(list(index), at_ns)
        k = np.array([index[h] for h in horse_ids], dtype=np.int64)
        ran = last[k] >= 0
        gap = np.where(ran, np.maximum(race_ns - last[k], 0) / NS_PER_DAY, 0.0)
        var_i = var[k] + self.drift_var_per_day * gap
        slope = np.array([self.pace_slopes.get(s, 0.0) if s else 0.0 for s in style])
        mu = (
            top3
            + self.prior_mean
            + mean[k]
            + (slope - self.pace_race) * pace
            + self.weight_slope * (weight - self.weight_ref)
        )
        df = self.t_df
        if df is not None and df <= 2.0:  # the t has no finite variance
            sd = np.full(n, math.inf)
        else:
            k = 1.0 if df is None else df / (df - 2.0)  # the error's variance per scale^2
            sd = np.sqrt(var_i + k * (self.sigma**2 + vcard))
        return pd.DataFrame({"horse_id": horse_ids, "mu_s": mu, "sd_s": sd})

    def _theta(self, horses: list, at_ns: int):
        """Each horse's filtered theta~ (mean, variance) after its runs published before at_ns,
        and its last such off in ns (-1 with none: then the prior)."""
        parts = []
        for k, h in enumerate(horses):
            hist = self._history.get(h)
            if hist is None:
                continue
            tau, u, s = hist.at(at_ns)
            if tau.size:
                parts.append((k, tau, u, s))
        mean = np.zeros(len(horses))
        var = np.full(len(horses), self.prior_sd**2)
        last = np.full(len(horses), -1, dtype=np.int64)
        if not parts:
            return mean, var, last
        code = np.concatenate([np.full(len(t), k) for k, t, _, _ in parts])
        tau = np.concatenate([t for _, t, _, _ in parts])
        u = np.concatenate([x for _, _, x, _ in parts])
        s = np.concatenate([x for _, _, _, x in parts])
        steps = _steps(code, tau, len(horses))
        _, m, p = _forward(steps, u, s, self.drift_var_per_day, self.prior_sd**2, self.t_df)
        for k, t, _, _ in parts:
            mean[k], var[k], last[k] = m[k], p[k], t[-1]
        return mean, var, last


def _baseline(speed: SpeedModel, race: dict, at_ns: int) -> tuple[float, float]:
    """(expected top-3 time, the variant's variance) of the race as of at_ns; NaN, NaN when its
    cell has no par."""
    at = pd.Timestamp(at_ns, tz="UTC")
    try:
        mean = speed.expected_top3_time(race, at_utc=at)
    except KeyError:
        return math.nan, math.nan
    group = tuple(race[k] for k in speed.spec.variant_keys)
    _, sd = speed.variant(_jst_date(race["race_utc"]), *group, at_utc=at)
    return mean, sd * sd


def _in_sample(speed: SpeedModel, race: dict) -> tuple[float, float, int] | None:
    """(par + the fit's in-sample variant, its variance, the instant in ns after which it is
    usable) of a race on a fit (day, group) that is ever fully resulted; None otherwise."""
    group = tuple(race[k] for k in speed.spec.variant_keys)
    got = speed.in_sample_variant(_jst_date(race["race_utc"]), *group)
    if got is None or got[2] is None:
        return None
    try:
        par = speed.par(**{k: race[k] for k in speed.spec.par_keys})
    except KeyError:
        return None
    mean, sd, resulted = got
    return par + mean, sd * sd, int(resulted.as_unit("ns").value)


def _start_scales(code: np.ndarray, e: np.ndarray, vvar: np.ndarray) -> np.ndarray:
    """Starting (sigma, s0): the within-horse spread less the baselines' variance, and the spread
    of the horses' mean residuals."""
    count = np.bincount(code)
    hmean = np.bincount(code, e) / np.maximum(count, 1)
    dof = max(len(e) - int(np.sum(count > 0)), 1)
    within = float(np.sum((e - hmean[code]) ** 2)) / dof - float(np.mean(vvar))
    between = float(np.std(hmean[count > 0]))
    return np.maximum([math.sqrt(max(within, 0.0)), between], SCALE_FLOOR * 10)


# -- p_speed ------------------------------------------------------------------------------------


def p_speed(mu: np.ndarray, *, scale: float, method: str = "pl") -> np.ndarray:
    """One race's win probabilities from its runners' expected times, in mu's order."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}: {method!r}")
    if not _number(scale) or scale <= 0:
        raise ValueError(f"scale must be a positive number of seconds: {scale!r}")
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 1 or mu.size == 0:
        raise ValueError(f"mu must be a non-empty 1-d array, got shape {mu.shape}")
    if not np.isfinite(mu).all():
        raise ValueError("mu has a missing or non-finite value")
    z = (mu - mu.min()) / scale
    if method == "pl":
        strength = np.exp(-z)
        return strength / strength.sum()
    grid = np.arange(-HENERY_SPAN, HENERY_SPAN + HENERY_STEP / 2, HENERY_STEP)
    d = grid[None, :] - z[:, None]  # each runner's standardised time at each grid point
    log_sf = log_ndtr(-d)
    integrand = np.exp(-0.5 * d * d + log_sf.sum(axis=0)[None, :] - log_sf)
    p = integrand.sum(axis=1)
    return p / p.sum()


# -- same-day bias ------------------------------------------------------------------------------


def same_day_bias(results: pd.DataFrame, *, at_utc: datetime, venue: str) -> pd.DataFrame:
    """The draw-third and style profile of the venue's races on at_utc's JST day whose result was
    published before at_utc: one row per (factor, level) of LEVELS with `bias` and `n_races`."""
    at_ns = _instant_ns(at_utc, "at_utc")
    _check_frame(results, BIAS_COLUMNS, "results")
    if results["race_utc"].isna().any() or results["race_id"].isna().any():
        raise ValueError("results has a null race_id or race_utc")
    _utc_ns(results["race_utc"], "race_utc")
    pub, known = _utc_ns(results["result_available_utc"], "result_available_utc")
    _check_styles(results["style"], "results")
    day = pd.Timestamp(at_ns, tz="UTC").tz_convert(JST).date()
    race_day = pd.to_datetime(results["race_utc"]).dt.tz_convert(JST).dt.date.to_numpy()
    keep = (results["venue"].to_numpy() == venue) & (race_day == day) & known & (pub < at_ns)
    frame = results.loc[keep, ["race_id", "draw", "style", "finish_pos"]]
    n_races = int(frame["race_id"].nunique())

    means: dict[tuple[str, str], float] = {}
    if n_races:
        n = pl.len().over("race_id")
        nf = pl.col("finish_pos").is_not_null().sum().over("race_id")
        scored = (
            pl.from_pandas(frame.astype({"race_id": str, "style": object}))
            .with_columns(
                _draw_group(pl.col("draw").cast(pl.Float64), n).alias("third"),
                pl.when(nf >= 2).then((nf - pl.col("finish_pos")) / (nf - 1) - 0.5).alias("y"),
            )
            .filter(pl.col("y").is_not_null())
        )
        for factor, col, names in (("draw", "third", DRAW_LEVELS), ("style", "style", STYLES)):
            got = scored.drop_nulls(col).group_by(col).agg(pl.col("y").mean())
            for level, mean in got.iter_rows():
                means[(factor, names[level] if factor == "draw" else level)] = float(mean)
    return pd.DataFrame(
        {
            "factor": [f for f, _ in LEVELS],
            "level": [lv for _, lv in LEVELS],
            "bias": [means.get(lv, 0.0) for lv in LEVELS],
            "n_races": [n_races] * len(LEVELS),
        }
    )
