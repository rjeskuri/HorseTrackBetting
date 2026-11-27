"""The C2 forward-filtered dynamic horse rating (#91, epic #73; 08 §3.2 C2, §4 E38; 05 §3.2).

Each horse has a latent ability a_h(t), a Gaussian random walk in time: N(prior_mean, prior_var) at
its first race, then between two of its races, days_off apart (elapsed time / 24 h),

    var += drift_var_per_day * days_off  (+ spell_var when days_off > spell_days, a long layoff)

with the mean unchanged. A race's update is Weng and Lin's Bradley-Terry full-pair rule (2011,
"A Bayesian approximation method for online ranking", JMLR 12; the rule behind OpenSkill): for
each pair (i, j) in the race, c = sqrt(var_i + var_j + 2 * perf_var), p_ij = 1 / (1 + exp((mu_j -
mu_i) / c)), s_ij = 1 if i finished ahead of j, 1/2 on a dead heat, 0 behind, and

    mu_i  += sum_j var_i / c * (s_ij - p_ij)
    var_i *= max(1 - sum_j (sd_i / c) * var_i / c^2 * p_ij * p_ji, KAPPA)

so a winner's mean rises, the last runner's falls and no variance widens. Races run in race_utc
order (race_id breaks a tie); every runner of a race is updated from the states before it, so
mu_before and sigma_before at a race depend only on earlier races (the filter, never the
smoothed TTT estimate, which would read later ones: 05 §3.2).

`rating_member` turns the filtered state into a within-race log-probability: the utility of a
runner is its mu_before less the race's mean mu_before, over sqrt(scale^2 + sigma_before^2), and
the probability is the softmax of the utilities within the race. scale is the performance scale
of the logit: a larger scale flattens the member, and a runner whose rating is uncertain is pulled
toward the field's mean. Only the before columns are read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

KAPPA = 1e-4  # Weng-Lin's floor on the variance shrink factor, so a variance never reaches 0
DAY_NS = 86_400 * 10**9
RESULT_COLUMNS = ("race_id", "race_utc", "horse_id", "finish_pos")
OUT_COLUMNS = ["race_id", "horse_id", "mu_before", "sigma_before", "mu_after", "sigma_after"]

__all__ = ["DynamicRating", "rating_member"]


def _positive(name: str, value: float, *, zero_ok: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero_ok):
        raise ValueError(
            f"{name} must be a finite {'non-negative' if zero_ok else 'positive'} "
            f"number, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class DynamicRating:
    drift_var_per_day: float
    perf_var: float
    prior_mean: float
    prior_var: float
    spell_var: float
    spell_days: int

    def __init__(
        self,
        *,
        drift_var_per_day: float,
        perf_var: float,
        prior_mean: float,
        prior_var: float,
        spell_var: float,
        spell_days: int,
    ) -> None:
        if not math.isfinite(float(prior_mean)):
            raise ValueError(f"prior_mean must be finite, got {prior_mean!r}")
        values = {
            "drift_var_per_day": _positive("drift_var_per_day", drift_var_per_day, zero_ok=True),
            "perf_var": _positive("perf_var", perf_var),
            "prior_mean": float(prior_mean),
            "prior_var": _positive("prior_var", prior_var),
            "spell_var": _positive("spell_var", spell_var, zero_ok=True),
            "spell_days": _positive("spell_days", spell_days, zero_ok=True),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def run(self, results: pd.DataFrame) -> pd.DataFrame:
        """Each runner's state before and after each race, one row per input row (in race_utc
        order). results: race_id, race_utc (tz-aware), horse_id, finish_pos (1 = winner)."""
        missing = [c for c in RESULT_COLUMNS if c not in results.columns]
        if missing:
            raise ValueError(f"results lack {missing}")
        stamps = pd.to_datetime(results["race_utc"])
        if not isinstance(stamps.dtype, pd.DatetimeTZDtype):
            raise ValueError("race_utc must be tz-aware (UTC)")
        if stamps.isna().any():
            raise ValueError("race_utc has missing values")
        pos = pd.to_numeric(results["finish_pos"]).to_numpy(dtype=float)
        if not np.isfinite(pos).all():
            raise ValueError("finish_pos has missing values: rank or drop non-finishers first")
        if results.duplicated(["race_id", "horse_id"]).any():
            raise ValueError("a horse appears twice in one race")
        ns = stamps.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
        frame = pd.DataFrame(
            {
                "race_id": results["race_id"].to_numpy(dtype=object),
                "horse_id": results["horse_id"].to_numpy(dtype=object),
                "t": ns.astype(np.int64),
                "pos": pos,
            }
        )
        if (frame.groupby("race_id")["t"].nunique() > 1).any():
            raise ValueError("a race has more than one race_utc")
        frame = frame.sort_values(["t", "race_id"], kind="stable")
        horse = pd.factorize(frame["horse_id"])[0]
        race = frame["race_id"].to_numpy()
        t, pos = frame["t"].to_numpy(), frame["pos"].to_numpy()
        n = len(frame)
        mu_b, var_b, mu_a, var_a = (np.empty(n) for _ in range(4))
        h_mu = np.full(horse.max() + 1 if n else 0, self.prior_mean)
        h_var = np.full(len(h_mu), self.prior_var)
        h_t = np.full(len(h_mu), -1, dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]]) if n else np.array([], int)
        for lo, hi in zip(starts, np.r_[starts[1:], n], strict=True):
            hs = horse[lo:hi]
            seen = h_t[hs] >= 0
            days = np.where(seen, (t[lo] - h_t[hs]) / DAY_NS, 0.0)
            var = h_var[hs] + self.drift_var_per_day * days
            var = var + np.where(seen & (days > self.spell_days), self.spell_var, 0.0)
            mu = h_mu[hs]
            mu_b[lo:hi], var_b[lo:hi] = mu, var
            mu_a[lo:hi], var_a[lo:hi] = self._update(mu, var, pos[lo:hi])
            h_mu[hs], h_var[hs], h_t[hs] = mu_a[lo:hi], var_a[lo:hi], t[lo]
        out = pd.DataFrame(
            {
                "race_id": frame["race_id"].to_numpy(),
                "horse_id": frame["horse_id"].to_numpy(),
                "mu_before": mu_b,
                "sigma_before": np.sqrt(var_b),
                "mu_after": mu_a,
                "sigma_after": np.sqrt(var_a),
            }
        )
        return out[OUT_COLUMNS].reset_index(drop=True)

    def _update(self, mu: np.ndarray, var: np.ndarray, pos: np.ndarray):
        """Weng-Lin Bradley-Terry full-pair update of one race."""
        c2 = var[:, None] + var[None, :] + 2.0 * self.perf_var
        c = np.sqrt(c2)
        p = 1.0 / (1.0 + np.exp((mu[None, :] - mu[:, None]) / c))  # P(i ahead of j)
        s = np.where(
            pos[:, None] < pos[None, :], 1.0, np.where(pos[:, None] == pos[None, :], 0.5, 0)
        )
        np.fill_diagonal(s, 0.0)
        np.fill_diagonal(p, 0.0)
        omega = (var[:, None] / c * (s - p)).sum(axis=1)
        q = p.T.copy()  # P(j ahead of i)
        gamma = np.sqrt(var)[:, None] / c  # Weng-Lin's gamma_q = sigma_i / c_iq
        delta = (gamma * var[:, None] / c2 * p * q).sum(axis=1)
        return mu + omega, var * np.maximum(1.0 - delta, KAPPA)


def rating_member(ratings: pd.DataFrame, *, scale: float) -> pd.Series:
    """The member's log-probability per ratings row (on its index), normalised within race_id,
    from mu_before and sigma_before only."""
    scale = _positive("scale", scale)
    race = ratings["race_id"].to_numpy(dtype=object)
    mu = ratings["mu_before"].to_numpy(dtype=float)
    sd = ratings["sigma_before"].to_numpy(dtype=float)
    centred = mu - pd.Series(mu).groupby(race).transform("mean").to_numpy()
    u = centred / np.sqrt(scale**2 + sd**2)
    top = pd.Series(u).groupby(race).transform("max").to_numpy()
    lse = top + np.log(pd.Series(np.exp(u - top)).groupby(race).transform("sum").to_numpy())
    return pd.Series(u - lse, index=ratings.index, name="rating")
