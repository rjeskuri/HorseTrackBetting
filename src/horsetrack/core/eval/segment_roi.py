"""Hierarchical shrinkage of segment ROI with favourite-longshot fixed effects (08 section 5.2, E8).

Model. A bet at final decimal odds O in a pool with takeout t wins with probability
``p = omega * q``, where ``q = (1 - t) / O`` is the takeout-adjusted implied probability, so its
expected return per unit stake is ``omega * (1 - t)``. On the log scale

    ln omega = mu[odds band] + sum over segment columns c of u_c[level of c],

with one fixed effect ``mu`` per odds band (the favourite-longshot bias, so no segment is credited
with it) and crossed random effects ``u_c[l] ~ N(0, tau_c^2)``, one variance per segment column.

Priors, a proposal for owner sign-off (08 section 9.9), set by the module constants below:

- ``mu[band] ~ N(0, MU_PRIOR_SD^2)`` with ``MU_PRIOR_SD = 1.0``, independent per band: centred on
  no bias, wide enough for any plausible band omega (95% within 0.14 to 7.1).
- ``tau_c ~ Half-Normal(scale TAU_PRIOR_SCALE)`` with ``TAU_PRIOR_SCALE = 0.1``, independent per
  segment column: a typical segment moves omega by about 10%, and ``P(tau_c > ln 1.25) = 3%``, so a
  spread of segments that beat a 20% takeout is possible but not presumed.

Fitting. For a given tau the posterior of ``theta = (mu, u)`` is approximated by a Gaussian at its
mode (Laplace). The Bernoulli log-likelihood ``won * ln p + (1 - won) * ln(1 - p)`` is concave in
``theta`` on the convex set where every bet has ``p < 1``; Newton's method with analytic gradient
and Hessian and a backtracking line search that stays inside that set finds the mode. ``p < 1`` is
never clipped: losing bets make the log-likelihood fall to minus infinity at the boundary. Bets are
aggregated by cell (band x segment levels) and implied probability first, so a fit costs no more
than the number of distinct (cell, odds, takeout) values. Tau is integrated out rather than plugged
in: its log posterior (Laplace marginal likelihood plus prior, on the ln-tau scale) is maximised
with Nelder-Mead, a multivariate t proposal is fitted at the mode, and ``TAU_SAMPLES`` importance
samples weight the conditional Gaussians. ``draws`` posterior draws of ``theta`` come from that
mixture. Every random number comes from ``np.random.default_rng`` with the given seed, so a seed
reproduces every output exactly.

The p = 1 boundary. A cell whose likeliest bet won and that no loser bounds (a small level whose
bets all won, say) has only the prior to hold it back from ``p = 1``. Once ``tau`` is large enough
the conditional supremum lies on that boundary, the Laplace approximation does not hold, and that
tau sample gets zero weight, as do samples outside ``_LOG_TAU_RANGE`` (``tau_rejected`` counts
both). This truncates the tau posterior from above where the boundary binds: for ``n`` all-won
bets at implied probability ``q e^mu`` it binds at about ``tau = sqrt(-ln(q e^mu) / n)``, e.g.
0.37 for three wins at odds 1.2, far out in the prior's tail. It matters only when many
high-probability bets of one level all won and the column has few other levels to pin tau (a 1-D
check: 20 wins at odds 1.0 beside three null levels gives a tau posterior mean of 0.070 against
0.097 exact), and then about half the tau samples are rejected. A band whose bets all won cannot
be fitted at any tau and raises.

Queries name a segment as ``{column: level}`` for one or more segment columns. Each bet's omega is
its band's and its levels' effects combined. ``expected_return``, ``p_beats_takeout`` and
``omega_mean`` read the same posterior draws of the segment's mean expected return per unit stake,
``mean(omega_i * (1 - t_i))`` over its own bets: its mean and credible interval, the probability
that it exceeds 1, and its mean over the bets' mean payout rate (the payout-weighted omega, which
clears ``1 / (1 - mean t_i)`` exactly when the expected return clears 1; with one takeout,
``P(omega_s > 1 / (1 - t))``). Every bet counts once (flat stakes): the stake column is not used.
In those draws and in ``predictive_roi``'s, each cell's ln omega is held to the bound the fit
keeps, ``ln P_MAX - ln max q`` over the fit's bets in that cell (a segment is whole cells, so these
are its own bets there), before any bet's win probability is derived, since the Gaussian draws can
cross it. Every bet of a cell shares its omega, so in every draw that omega is below
``min O_j / (1 - t_j)`` over the cell's bets: a bet returns at most ``(1 - t_i)`` times its cell's
bound, which is at most its odds, and bets at odds 1.0 and 1.2 at one takeout in one cell never
beat their stake.

Grouped input (#429). With ``count_col`` a row stands for ``count`` identical bets (same cell,
odds and takeout) of which ``won`` won, ``0 <= won <= count``. The likelihood is a sum over bets
and already aggregates them by cell and implied probability, so a grouped row fits exactly as its
bets one row each. Every query weighs each row by its count: the means over the segment's bets
behind ``expected_return``, ``p_beats_takeout`` and ``omega_mean``, and ``predictive_roi``'s mix
of future bets. A cell's bound is a maximum over its bets' implied probabilities, the same
whichever way they are grouped.

Settled wins (#430). With ``fractional_won`` (grouped input only) ``won`` may be any real of at
least 0, above ``count`` included: each hit counts for the dividend that settled it over its board
odds, below 1 when a dead heat splits the pool and above 1 when a capped board quote paid more, so
that ``omega * (1 - t)`` stays the expected return per unit stake at the dividend that pays. The
likelihood is then the Bernoulli kernel with fractional successes (a quasi-likelihood): per cell
and implied probability, the settled wins and, as losers, the bets less the settled wins, floored
at 0 (a group that settled more than its bets adds its wins and no loser term, which keeps the
kernel concave; the rest of its cell bounds it, as for a group whose bets all won). Wherever every
such group settles at most its bets, as with whole wins, it is the Bernoulli likelihood above,
unchanged. ``omega_draws`` and ``effect_draws``, the readings #430's scan takes, return the
posterior draws of omega as they are, without the cell bound above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special

MU_PRIOR_SD = 1.0
TAU_PRIOR_SCALE = 0.1
INTERVAL_LEVEL = 0.90
TAU_SAMPLES = 200
P_MAX = 1.0 - 1e-12  # a bet's win probability stays below this, in the fit and in every draw
_T_DF = 4.0  # the tau proposal: a multivariate t with these degrees of freedom,
_INFLATE = 2.0  # and its Laplace covariance at the mode widened by this factor (skewed tails)
_LOG_TAU_RANGE = (np.log(1e-6), np.log(10.0))
_BLOCK = 2_000_000  # array elements per block of the predictive simulation


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    level: float


@dataclass(frozen=True)
class Estimate:
    """A posterior mean and its central credible interval."""

    mean: float
    low: float
    high: float
    level: float


class _BoundaryMode(RuntimeError):
    """The conditional posterior's supremum lies on the ``p = 1`` boundary: no interior mode."""


class _Likelihood:
    """The Bernoulli log-likelihood aggregated by cell and implied probability; a row is
    `count` bets of which `won` won."""

    def __init__(
        self,
        cell: np.ndarray,
        q: np.ndarray,
        won: np.ndarray,
        count: np.ndarray,
        design: np.ndarray,
    ):
        self.design = design
        n_cells = design.shape[0]
        self.wins = np.bincount(cell, weights=won, minlength=n_cells).astype(float)
        # per cell and implied probability, the bets less their (settled) wins, floored at 0: with
        # whole wins no row is below 0, so this is the sum of each row's losers
        losers = pd.DataFrame({"cell": cell, "q": q, "n": count - won}).groupby(["cell", "q"])["n"]
        losers = losers.sum()
        losers = losers[losers > 0]
        self.lose_cell = losers.index.get_level_values("cell").to_numpy()
        self.lose_q = losers.index.get_level_values("q").to_numpy()
        self.lose_n = losers.to_numpy(dtype=float)
        max_q = np.zeros(n_cells)
        np.maximum.at(max_q, cell, q)
        self.eta_max = np.log(P_MAX) - np.log(max_q)  # each cell's eta must stay below this

    def log_joint(self, theta: np.ndarray, lam: np.ndarray) -> float:
        """Log-likelihood plus the Gaussian prior kernel; minus infinity where some p >= P_MAX."""
        eta = self.design @ theta
        if np.any(eta >= self.eta_max):
            return -np.inf
        p = self.lose_q * np.exp(eta[self.lose_cell])
        return float(self.wins @ eta + self.lose_n @ np.log1p(-p) - 0.5 * lam @ theta**2)

    def mode(self, theta: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Posterior mode for prior precisions `lam`: (theta, log joint, Cholesky of -Hessian)."""
        f = self.log_joint(theta, lam)
        if not np.isfinite(f):
            theta = np.zeros_like(theta)  # eta = 0 gives p = q < 1 for every bet
            f = self.log_joint(theta, lam)
        n_cells = self.design.shape[0]
        for _ in range(100):
            eta = self.design @ theta
            p = self.lose_q * np.exp(eta[self.lose_cell])
            r = self.lose_n * p / (1 - p)
            grad = self.design.T @ (self.wins - np.bincount(self.lose_cell, r, n_cells))
            grad -= lam * theta
            curv = np.bincount(self.lose_cell, r / (1 - p), n_cells)
            chol = np.linalg.cholesky((self.design.T * curv) @ self.design + np.diag(lam))
            step = linalg.cho_solve((chol, True), grad)
            decrement = float(grad @ step)
            if decrement < 1e-14:
                return theta, f, chol
            size = 1.0
            while size >= 1e-12:
                candidate = theta + size * step
                f_new = self.log_joint(candidate, lam)
                if np.isfinite(f_new) and (
                    f_new >= f + 1e-4 * size * decrement or decrement < 1e-8
                ):
                    break
                size /= 2
            else:
                break  # the line search stalled
            theta, f = candidate, f_new
        # Stalled, or out of iterations, with a Newton step that leaves the feasible set: the
        # iterate has run into p = 1 (a cell whose likeliest bet won, with no loser to bound it),
        # where the supremum is on the boundary and there is no interior mode.
        if not np.isfinite(self.log_joint(theta + step, lam)):
            raise _BoundaryMode("segment ROI fit: the mode lies on the p = 1 boundary")
        raise RuntimeError("segment ROI fit: Newton's method did not converge")


def _fd_hessian(fun, x: np.ndarray, h: float = 0.1) -> np.ndarray:
    k = len(x)
    out = np.empty((k, k))
    e = np.eye(k) * h
    for i in range(k):
        for j in range(i, k):
            out[i, j] = out[j, i] = (
                fun(x + e[i] + e[j])
                - fun(x + e[i] - e[j])
                - fun(x - e[i] + e[j])
                + fun(x - e[i] - e[j])
            ) / (4 * h * h)
    return out


class SegmentROI:
    """A fitted segment ROI model; build it with `fit_segment_roi`."""

    def __init__(
        self,
        *,
        columns: list[str],
        levels: list[dict[str, int]],
        band_levels: list[str],
        offsets: np.ndarray,
        theta_mean: np.ndarray,
        theta_draws: np.ndarray,
        design: np.ndarray,
        cell: np.ndarray,
        codes: np.ndarray,
        band: np.ndarray,
        odds: np.ndarray,
        takeout: np.ndarray,
        count: np.ndarray,
        tau: np.ndarray,
        tau_ess: float,
        tau_rejected: int,
        eta_max: np.ndarray,
        seed: int,
    ):
        self._columns = columns
        self._levels = levels
        self._band_levels = {name: i for i, name in enumerate(band_levels)}
        self._offsets = offsets
        self._mean = theta_mean
        self._draws = theta_draws
        self._design = design
        self._cell = cell
        self._codes = codes
        self._band = band
        self._odds = odds
        self._takeout = takeout
        self._count = count  # the bets each row stands for (all 1 without count_col)
        self._eta_max = eta_max
        self._seed = seed
        # Posterior means of the band effects (ln omega) and of each column's tau, the effective
        # sample size of the tau importance weights (out of TAU_SAMPLES), and how many tau samples
        # got zero weight (out of range, or conditional mode on p = 1): diagnostics, not verdicts.
        self.band_effects = {b: float(m) for b, m in zip(band_levels, theta_mean, strict=False)}
        self.tau = {c: float(t) for c, t in zip(columns, tau, strict=True)}
        self.tau_ess = tau_ess
        self.tau_rejected = tau_rejected

    def _codes_of(self, segment: dict[str, str]) -> dict[int, int]:
        if not segment:
            raise ValueError("a segment names at least one segment column")
        codes = {}
        for col, level in segment.items():
            if col not in self._columns:
                raise KeyError(f"{col!r} is not a segment column (have {self._columns})")
            c = self._columns.index(col)
            if level not in self._levels[c]:
                raise KeyError(f"{col!r} has no level {level!r}")
            codes[c] = self._levels[c][level]
        return codes

    def _rows(self, segment: dict[str, str]) -> np.ndarray:
        codes = self._codes_of(segment)
        rows = np.ones(len(self._cell), dtype=bool)
        for c, code in codes.items():
            rows &= self._codes[:, c] == code
        if not rows.any():
            raise ValueError(f"no bets in segment {segment}")
        return rows

    def omega_draws(
        self, segment: dict[str, str], *, band: str | None = None, rate_weighted: bool = False
    ) -> np.ndarray:
        """Posterior draws of the segment's omega, its band effects included: per draw, the mean
        over the segment's bets (a grouped row counts its bets) of each bet's omega. An empty
        `segment` is every bet; `band` keeps only the bets of that odds band. With
        `rate_weighted` each bet weighs by its payout rate 1 - takeout, so that a draw times the
        bets' mean payout rate is their mean expected return per unit stake, mean(omega_i *
        (1 - t_i)), even when the takeouts differ (a boosted race beside ordinary ones)."""
        rows = np.ones(len(self._cell), dtype=bool)
        if segment:
            rows &= self._rows(segment)
        if band is not None:
            if band not in self._band_levels:
                raise KeyError(f"no odds band {band!r}")
            rows &= self._band == self._band_levels[band]
        if not rows.any():
            raise ValueError(f"no bets in segment {segment} and band {band!r}")
        cells, local = np.unique(self._cell[rows], return_inverse=True)
        bets = self._count[rows] * ((1 - self._takeout[rows]) if rate_weighted else 1.0)
        weight = np.bincount(local.ravel(), weights=bets)
        weight /= weight.sum()
        design = self._design[cells]
        block = max(1, _BLOCK // len(cells))
        return np.concatenate(
            [
                np.exp(self._draws[start : start + block] @ design.T) @ weight
                for start in range(0, len(self._draws), block)
            ]
        )

    def effect_draws(self, segment: dict[str, str]) -> np.ndarray:
        """Posterior draws of exp(the sum of the named random effects): the segment effect alone,
        no band effect, whose posterior mean on the ln scale is `posterior_mean`."""
        codes = self._codes_of(segment)
        index = [self._offsets[c] + code for c, code in codes.items()]
        return np.exp(self._draws[:, index].sum(axis=1))

    def posterior_mean(self, segment: dict[str, str]) -> float:
        """Posterior mean of the sum of the named random effects (ln-omega scale)."""
        codes = self._codes_of(segment)
        return float(sum(self._mean[self._offsets[c] + code] for c, code in codes.items()))

    def _mix(self, segment: dict[str, str]) -> tuple[np.ndarray, ...]:
        """The segment's own bets grouped by (cell, odds, takeout): the cells they sit in, each
        group's index into them, its odds, its takeout and its share of the bets."""
        rows = self._rows(segment)
        mix = pd.DataFrame(
            {
                "cell": self._cell[rows],
                "odds": self._odds[rows],
                "takeout": self._takeout[rows],
                "n": self._count[rows],
            }
        )
        mix = mix.groupby(["cell", "odds", "takeout"])["n"].sum().reset_index()
        cells, local = np.unique(mix["cell"].to_numpy(), return_inverse=True)
        freq = mix["n"].to_numpy() / mix["n"].sum()
        return cells, local.ravel(), mix["odds"].to_numpy(), mix["takeout"].to_numpy(), freq

    def _win_probabilities(self, cells, local, odds, takeout):
        """Per block of posterior draws, each group's win probability ``omega (1 - t) / O``, its
        cell's ln omega first held to the fit's bound for that cell (module docstring); the clip
        at P_MAX only absorbs rounding at that bound."""
        q = (1 - takeout) / odds
        block = max(1, _BLOCK // len(odds))
        for start in range(0, len(self._draws), block):
            eta = self._draws[start : start + block] @ self._design[cells].T
            eta = np.minimum(eta, self._eta_max[cells])
            yield np.minimum(q * np.exp(eta[:, local]), P_MAX)

    def _return_draws(self, segment: dict[str, str]) -> tuple[np.ndarray, float]:
        """Posterior draws of the segment's mean expected return per unit stake,
        ``mean(O_i * p_i)`` over its own bets (``omega_i * (1 - t_i)``, each cell's omega held to
        its bound), and the bets' mean payout rate."""
        cells, local, odds, takeout, freq = self._mix(segment)
        groups = (cells, local, odds, takeout)
        draws = np.concatenate([p @ (freq * odds) for p in self._win_probabilities(*groups)])
        return draws, float(freq @ (1 - takeout))

    def omega_mean(self, segment: dict[str, str]) -> float:
        """Posterior mean of the segment's payout-weighted omega, ``sum((1 - t_i) * omega_i) /
        sum(1 - t_i)`` (each cell's omega held to its bound): the multiplier that clears
        1 / (1 - mean t_i) exactly when `expected_return` clears 1 (the same draws)."""
        draws, payout = self._return_draws(segment)
        return float(draws.mean() / payout)

    def expected_return(self, segment: dict[str, str]) -> Estimate:
        """Posterior mean and central INTERVAL_LEVEL credible interval of the segment's mean
        expected return per unit stake (break-even 1), the quantity `p_beats_takeout` judges.
        Uncertainty about the mean only; `predictive_roi` adds the outcomes of future bets."""
        draws, _ = self._return_draws(segment)
        tail = (1 - INTERVAL_LEVEL) / 2
        low, high = np.quantile(draws, [tail, 1 - tail])
        return Estimate(float(draws.mean()), float(low), float(high), INTERVAL_LEVEL)

    def p_beats_takeout(self, segment: dict[str, str]) -> float:
        """P(mean expected return per unit stake of the segment's bets > 1 | data)."""
        draws, _ = self._return_draws(segment)
        return float(np.mean(draws > 1))

    def predictive_roi(self, segment: dict[str, str], n_bets: int) -> Interval:
        """Central INTERVAL_LEVEL interval for the ROI of the segment's next `n_bets` unit bets."""
        if not isinstance(n_bets, int) or isinstance(n_bets, bool) or n_bets < 1:
            raise ValueError(f"n_bets must be a positive integer, not {n_bets!r}")
        codes = self._codes_of(segment)
        cells, local, odds, takeout, freq = self._mix(segment)
        query = [codes.get(c, -1) + 1 for c in range(len(self._columns))]
        rng = np.random.default_rng([self._seed, n_bets, *query])
        roi = []
        for p in self._win_probabilities(cells, local, odds, takeout):
            counts = rng.multinomial(n_bets, freq, size=len(p))
            roi.append(rng.binomial(counts, p) @ odds / n_bets - 1)
        tail = (1 - INTERVAL_LEVEL) / 2
        low, high = np.quantile(np.concatenate(roi), [tail, 1 - tail])
        return Interval(float(low), float(high), INTERVAL_LEVEL)


def _validate(
    bets, segment_cols, odds_band_col, takeout_col, seed, draws, count_col, fractional_won=False
) -> None:
    if not isinstance(bets, pd.DataFrame) or bets.empty:
        raise ValueError("bets must be a non-empty DataFrame")
    if not segment_cols or len(set(segment_cols)) != len(segment_cols):
        raise ValueError("segment_cols must name one or more distinct columns")
    if {odds_band_col, takeout_col} & set(segment_cols):
        raise ValueError("the band and takeout columns cannot also be segment columns")
    reserved = {"odds", "won", odds_band_col, takeout_col, *segment_cols}
    if count_col is not None and count_col in reserved:
        raise ValueError(f"the count column {count_col!r} is already used as {sorted(reserved)}")
    for name, value, low in (("seed", seed, 0), ("draws", draws, 1)):
        if not isinstance(value, int) or isinstance(value, bool) or value < low:
            raise ValueError(f"{name} must be an integer of at least {low}, not {value!r}")
    needed = ["odds", "won", odds_band_col, takeout_col, *segment_cols]
    if count_col is not None:
        needed.append(count_col)
    missing = [c for c in needed if c not in bets.columns]
    if missing:
        raise ValueError(f"bets lacks column(s): {', '.join(missing)}")
    if bets[needed].isna().any().any():
        raise ValueError("bets has missing values in " + ", ".join(needed))
    odds = pd.to_numeric(bets["odds"], errors="coerce").to_numpy(dtype=float)
    takeout = pd.to_numeric(bets[takeout_col], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(odds) & (odds >= 1)).all():
        raise ValueError("odds must be finite decimal odds of at least 1")
    if not ((takeout >= 0) & (takeout < 1)).all():
        raise ValueError(f"{takeout_col} must be in [0, 1)")
    if not ((1 - takeout) / odds < P_MAX).all():
        raise ValueError(f"(1 - {takeout_col}) / odds must be below 1 (odds 1 needs a takeout)")
    if count_col is None:
        if fractional_won:
            raise ValueError("fractional_won needs count_col (grouped input)")
        if not bets["won"].isin([0, 1]).all():
            raise ValueError("won must be 0 or 1")
        return
    count = pd.to_numeric(bets[count_col], errors="coerce").to_numpy(dtype=float)
    won = pd.to_numeric(bets["won"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(count) & (count == np.round(count)) & (count >= 1)).all():
        raise ValueError(f"{count_col} must be a positive integer")
    if fractional_won:  # settled wins: a capped quote can settle a hit above its board odds
        if not (np.isfinite(won) & (won >= 0)).all():
            raise ValueError("won must be a number of at least 0")
        return
    if not (np.isfinite(won) & (won == np.round(won)) & (won >= 0) & (won <= count)).all():
        raise ValueError(f"won must be an integer from 0 to {count_col}")


def fit_segment_roi(
    bets: pd.DataFrame,
    *,
    segment_cols: list[str],
    odds_band_col: str,
    takeout_col: str,
    seed: int,
    draws: int,
    count_col: str | None = None,
    fractional_won: bool = False,
) -> SegmentROI:
    """Fit the hierarchical omega model (module docstring) to one row per bet, or with
    `count_col` to rows of `count` identical bets of which `won` won; with `fractional_won`,
    `won` is their settled wins (module docstring, 'Settled wins')."""
    segment_cols = list(segment_cols)
    _validate(
        bets, segment_cols, odds_band_col, takeout_col, seed, draws, count_col, fractional_won
    )
    odds = bets["odds"].to_numpy(dtype=float)
    takeout = bets[takeout_col].to_numpy(dtype=float)
    won = bets["won"].to_numpy(dtype=float if fractional_won else int)
    row_bets = np.ones(len(bets), dtype=int) if count_col is None else bets[count_col].to_numpy(int)
    band, band_levels = pd.factorize(bets[odds_band_col].astype(str), sort=True)
    codes, levels = [], []
    for col in segment_cols:
        code, names = pd.factorize(bets[col].astype(str), sort=True)
        codes.append(code)
        levels.append({name: i for i, name in enumerate(names)})
    codes = np.column_stack(codes)
    sizes = [len(band_levels), *(len(lv) for lv in levels)]
    offsets = np.cumsum([0, *sizes])
    keys, cell = np.unique(np.column_stack([band, codes]), axis=0, return_inverse=True)
    cell = cell.ravel()
    design = np.zeros((len(keys), offsets[-1]))
    for k in range(keys.shape[1]):
        design[np.arange(len(keys)), offsets[k] + keys[:, k]] = 1.0
    lik = _Likelihood(cell, (1 - takeout) / odds, won, row_bets, design)

    group = np.repeat(np.arange(len(sizes)), sizes)  # 0 for mu, c + 1 for column c

    def precisions(log_tau: np.ndarray) -> np.ndarray:
        return np.concatenate([[MU_PRIOR_SD**-2], np.exp(-2 * log_tau)])[group]

    warm = np.zeros(offsets[-1])

    def log_post(log_tau: np.ndarray) -> tuple[float, np.ndarray | None]:
        """Log posterior of ln tau (Laplace marginal likelihood, prior, Jacobian) and the mode.

        Minus infinity and no mode where the conditional mode is on the p = 1 boundary: the Laplace
        approximation does not hold there, so such a tau gets no weight (module docstring).
        """
        lam = precisions(log_tau)
        try:
            theta, f, chol = lik.mode(warm, lam)
        except _BoundaryMode:
            return -np.inf, None
        warm[:] = theta
        marginal = f + 0.5 * np.log(lam).sum() - np.log(np.diag(chol)).sum()
        prior = (-0.5 * np.exp(2 * log_tau) / TAU_PRIOR_SCALE**2 + log_tau).sum()
        return float(marginal + prior), theta

    def objective(x):
        return -log_post(x)[0]

    start = np.full(len(segment_cols), np.log(TAU_PRIOR_SCALE))
    while not np.isfinite(objective(start)) and start[0] > _LOG_TAU_RANGE[0] + 1:
        start -= np.log(2)  # a smaller tau pulls the random effects back from p = 1
    with np.errstate(invalid="ignore"):  # inf - inf where a tau has its mode on p = 1
        best = optimize.minimize(
            objective, start, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8}
        ).x
        if not np.isfinite(objective(best)):
            raise RuntimeError("segment ROI fit: found no tau whose conditional mode is interior")
        hessian = _fd_hessian(objective, best)
    try:
        if not np.isfinite(hessian).all():
            raise np.linalg.LinAlgError("a finite-difference point has its mode on p = 1")
        scale = np.linalg.cholesky(np.linalg.inv(hessian) * _INFLATE**2)
    except np.linalg.LinAlgError:
        scale = np.eye(len(best))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((TAU_SAMPLES, len(best)))
    stretch = np.sqrt(_T_DF / rng.chisquare(_T_DF, TAU_SAMPLES))
    samples = best + (z @ scale.T) * stretch[:, None]
    dist = linalg.solve_triangular(scale, (samples - best).T, lower=True)
    log_q = -0.5 * (_T_DF + len(best)) * np.log1p((dist**2).sum(axis=0) / _T_DF)
    log_w = np.full(TAU_SAMPLES, -np.inf)
    modes = np.zeros((TAU_SAMPLES, offsets[-1]))
    for k, log_tau in enumerate(samples):
        if ((log_tau > _LOG_TAU_RANGE[0]) & (log_tau < _LOG_TAU_RANGE[1])).all():
            value, mode = log_post(log_tau)
            if mode is not None:
                modes[k] = mode
                log_w[k] = value - log_q[k]
    if not np.isfinite(log_w).any():
        raise RuntimeError(
            "segment ROI fit: every tau sample is out of range or has its mode on p = 1"
        )
    weights = np.exp(log_w - special.logsumexp(log_w))
    # Log weights of large magnitude (a big fit's log-likelihood) round the normalising
    # difference, so the weights can sum past 1 by more than multinomial's 1e-12 tolerance when
    # one sample dominates; renormalise them exactly (#430).
    weights = weights / weights.sum()

    theta_draws = []
    for k, count in enumerate(rng.multinomial(draws, weights)):
        if count:
            theta, _, chol = lik.mode(modes[k].copy(), precisions(samples[k]))
            noise = rng.standard_normal((offsets[-1], count))
            theta_draws.append(
                theta + linalg.solve_triangular(chol, noise, lower=True, trans="T").T
            )
    return SegmentROI(
        columns=segment_cols,
        levels=levels,
        band_levels=list(band_levels),
        offsets=offsets[1:],
        theta_mean=weights @ modes,
        theta_draws=np.concatenate(theta_draws),
        design=design,
        cell=cell,
        codes=codes,
        band=band,
        odds=odds,
        takeout=takeout,
        count=row_bets,
        tau=weights @ np.exp(samples),
        tau_ess=float(1 / (weights**2).sum()),
        tau_rejected=int((~np.isfinite(log_w)).sum()),
        eta_max=lik.eta_max,
        seed=seed,
    )
