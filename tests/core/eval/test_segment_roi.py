"""Ticket #78: hierarchical shrinkage of segment ROI with favourite-longshot (FLB) fixed effects.

Contract pinned here (the coder implements what these tests call):

- ``fit_segment_roi(bets, *, segment_cols, odds_band_col, takeout_col, seed, draws) -> SegmentROI``.
  ``bets`` has one row per bet: ``stake``, ``payout``, ``odds`` (final decimal odds O), ``won``
  (0/1), the odds-band column, the takeout column and the segment columns (string labels).
- Model: a bet's win probability is ``omega * (1 - t) / O``, so its expected return per unit stake
  is ``omega * (1 - t)``. ``ln omega = mu[band] + sum_c u_c[level of column c]`` with fixed effects
  ``mu`` per odds band (the FLB) and zero-mean random effects ``u_c ~ N(0, tau_c^2)`` for each
  segment column (crossed, not nested, when there are several columns).
- ``posterior_mean(segment)`` is the posterior mean of the segment effect on the ln-omega scale:
  the sum of the random effects named in the dict (one key -> that column's ``u`` for that level).
  It is NOT omega and NOT ROI; a segment that is like the rest has a posterior mean near 0.
- ``p_beats_takeout(segment)`` is ``P(omega_s > 1 / (1 - t) | data)``, i.e. the probability that
  the segment's expected return ``omega_s * (1 - t)`` exceeds the stake. With takeout 0.2 this is
  ``P(omega_s > 1.25)``. Where bands differ, ``omega_s`` is the segment's omega averaged over its
  own bets; these tests are built so that the verdicts do not depend on that aggregation choice.
- ``predictive_roi(segment, n_bets)`` is a central posterior-predictive interval (``Interval`` with
  ``low < high``) for the realised ROI ``sum(payout) / sum(stake) - 1`` of the segment's next
  ``n_bets`` unit-stake bets, drawn like the segment's own bets (same odds/band mix), including
  both parameter uncertainty and outcome noise. The level is the coder's choice (anything from a
  50% to a 99% central interval passes).
- The same ``seed`` gives identical outputs.
- The model's bound (#88): a bet's win probability ``omega * (1 - t) / O`` stays below 1 in every
  posterior draw that ``expected_return``, ``p_beats_takeout``, ``omega_mean`` and
  ``predictive_roi`` read, so a bet returns at most its odds and a segment's mean expected return
  per unit stake is at most the mean decimal odds of its bets. The bound is the cell's (round 4
  of PR #345): every bet of a cell (odds band x segment levels) shares one omega, so in every draw
  that omega is at most ``min O_j / (1 - t_j)`` over the cell's bets, and a bet returns at most
  ``(1 - t_i)`` times its cell's bound. Bets at odds 1.0 and 1.2, both at takeout 0.2, in one
  cell return at most 1 each: a draw cannot pay the 1.0 bet its stake and the 1.2 bet more.
- Grouped input (#429): ``fit_segment_roi(..., count_col='count')`` reads a row as ``count``
  identical bets of which ``won`` won (``0 <= won <= count``, else ValueError). It fits, queries
  and predicts exactly as the same bets one row each; without ``count_col`` nothing changes.

Every dataset is synthetic and generated here with a seeded numpy RNG. Each test first asserts the
premise it relies on (computed from the data, independently of the code under test), so the
thresholds are checked against a fixed realisation, not left to chance.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.segment_roi import Interval, SegmentROI, fit_segment_roi

TAKEOUT = 0.2
DRAWS = 1000
# Odds bands: (low, high) decimal odds, odds drawn log-uniformly inside the band.
BANDS = {"fav": (1.5, 3.0), "mid": (3.0, 8.0), "long": (8.0, 30.0)}
BAND_NAMES = tuple(BANDS)


# --- synthetic generator (independent of the module) ---------------------------------------------


def make_bets(
    rng: np.random.Generator,
    n: int,
    *,
    omega_by_band: dict[str, float],
    band_probs: tuple[float, float, float] = (0.3, 0.4, 0.3),
    takeout: float = TAKEOUT,
    labels: dict[str, str],
) -> pd.DataFrame:
    """n unit-stake bets; a bet wins with probability omega[band] * (1 - t) / O."""
    probs = np.asarray(band_probs, dtype=float)
    band_idx = rng.choice(len(BAND_NAMES), size=n, p=probs / probs.sum())
    lo = np.array([BANDS[b][0] for b in BAND_NAMES])[band_idx]
    hi = np.array([BANDS[b][1] for b in BAND_NAMES])[band_idx]
    odds = np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))), 1)
    odds = np.clip(odds, lo, hi)
    omega = np.array([omega_by_band[b] for b in BAND_NAMES])[band_idx]
    p = np.clip(omega * (1 - takeout) / odds, 0.0, 0.99)
    won = (rng.random(n) < p).astype(int)
    frame = pd.DataFrame(
        {
            "stake": 1.0,
            "payout": odds * won,
            "odds": odds,
            "won": won,
            "band": np.array(BAND_NAMES)[band_idx],
            "takeout": takeout,
        }
    )
    for col, value in labels.items():
        frame[col] = value
    return frame


def flat(omega: float) -> dict[str, float]:
    return {b: omega for b in BAND_NAMES}


def raw_log_omega(frame: pd.DataFrame, omega_by_band: dict[str, float] | None = None) -> float:
    """ln(observed wins / wins expected at the given band omegas; all bands 1 when None)."""
    w = 1.0 if omega_by_band is None else frame["band"].map(omega_by_band).to_numpy()
    expected = (w * (1 - frame["takeout"]) / frame["odds"]).sum()
    return math.log(frame["won"].sum() / expected)


def outcome_sd_of_roi(frame: pd.DataFrame, omega: float, n_bets: int) -> float:
    """SD of the ROI of n_bets resampled from frame's odds mix, omega known (no parameter noise)."""
    odds = frame["odds"].to_numpy()
    p = np.clip(omega * (1 - frame["takeout"].to_numpy()) / odds, 0.0, 0.99)
    mean_r = (odds * p).mean()
    second = (odds**2 * p).mean()
    return math.sqrt((second - mean_r**2) / n_bets)


def seg(label: str) -> dict[str, str]:
    return {"segment": label}


# --- planted segment: 20 segments x 2,000 bets, one with omega = 1.5 ------------------------------

PLANTED = "s13"
SEGMENTS = [f"s{i:02d}" for i in range(20)]


@pytest.fixture(scope="module")
def planted() -> tuple[pd.DataFrame, SegmentROI]:
    rng = np.random.default_rng(20261124)
    frames = [
        make_bets(
            rng,
            2000,
            omega_by_band=flat(1.5 if s == PLANTED else 1.0),
            labels={"segment": s},
        )
        for s in SEGMENTS
    ]
    bets = pd.concat(frames, ignore_index=True)
    fit = fit_segment_roi(
        bets,
        segment_cols=["segment"],
        odds_band_col="band",
        takeout_col="takeout",
        seed=1,
        draws=DRAWS,
    )
    return bets, fit


def test_hierarchical_omega_recovers_a_planted_segment(planted):
    """DoD: a planted segment is recovered and null segments are not flagged.

    The planted segment (omega 1.5, takeout 0.2) has the highest posterior mean effect and
    P(omega > 1.25) > 0.5; every null segment has P < 0.5. Its effect is on the ln-omega scale:
    shrunk towards 0 but well above ln(1.25) - small slack, while null effects stay near 0.
    """
    bets, fit = planted
    assert isinstance(fit, SegmentROI)

    # Premise: the planted segment really stands out in the data and the nulls do not.
    raw = {s: raw_log_omega(bets[bets["segment"] == s]) for s in SEGMENTS}
    assert raw[PLANTED] > math.log(1.35)
    assert all(abs(raw[s]) < 0.15 for s in SEGMENTS if s != PLANTED)

    means = {s: fit.posterior_mean(seg(s)) for s in SEGMENTS}
    probs = {s: fit.p_beats_takeout(seg(s)) for s in SEGMENTS}

    assert max(means, key=means.get) == PLANTED
    assert probs[PLANTED] > 0.5
    for s in SEGMENTS:
        if s != PLANTED:
            assert probs[s] < 0.5, (s, probs[s])
            assert abs(means[s]) < 0.15, (s, means[s])
    # ln-omega scale, shrunk but not erased: true effect ln 1.5 = 0.405.
    assert 0.15 < means[PLANTED] < 0.55
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_predictive_roi_of_the_next_100_bets_is_an_interval(planted):
    """The posterior predictive ROI of the next n bets is an interval that covers the truth at
    n = 100, is driven by outcome noise (so far wider at 100 bets than at 2,000) and moves with the
    segment's omega.
    """
    bets, fit = planted
    null = "s04"
    truth_roi = {PLANTED: 1.5 * (1 - TAKEOUT) - 1, null: 1.0 * (1 - TAKEOUT) - 1}

    intervals = {}
    for s, omega in ((PLANTED, 1.5), (null, 1.0)):
        i100 = fit.predictive_roi(seg(s), 100)
        i2000 = fit.predictive_roi(seg(s), 2000)
        assert isinstance(i100, Interval)
        assert math.isfinite(i100.low) and math.isfinite(i100.high)
        assert i100.low < i100.high
        assert i2000.low < i2000.high
        # Covers the true expected ROI of the next 100 bets.
        assert i100.low < truth_roi[s] < i100.high, (s, i100, truth_roi[s])
        # Outcome noise: at least ~1 SD wide (any level >= ~40%), at most ~8 SD (any level <= 99%+).
        sd100 = outcome_sd_of_roi(bets[bets["segment"] == s], omega, 100)
        width100 = i100.high - i100.low
        assert sd100 < width100 < 8 * sd100, (s, width100, sd100)
        # Fewer bets, more noise: SD falls by sqrt(20) from 100 to 2,000 bets.
        assert width100 > 2 * (i2000.high - i2000.low)
        # ROI is bounded below by -100%.
        assert i100.low >= -1.0
        intervals[s] = i100

    mid = {s: (iv.low + iv.high) / 2 for s, iv in intervals.items()}
    assert mid[PLANTED] - mid[null] > 0.15


# --- FLB: segments differ only in their odds-band mix ---------------------------------------------

FLB_OMEGA = {"fav": 1.05, "mid": 0.90, "long": 0.55}
TRACK_MIX = {
    "A": (0.7, 0.2, 0.1),
    "B": (0.4, 0.4, 0.2),
    "C": (0.2, 0.4, 0.4),
    "D": (0.1, 0.2, 0.7),
}
CLASS_MIX = {"c1": (0.6, 0.3, 0.1), "c2": (0.3, 0.4, 0.3), "c3": (0.1, 0.3, 0.6)}


def make_flb_bets(seed: int = 78, per_cell: int = 4000) -> pd.DataFrame:
    """Track x class cells whose band mix is the normalised product of the two mixes."""
    rng = np.random.default_rng(seed)
    frames = []
    for track, tmix in TRACK_MIX.items():
        for klass, cmix in CLASS_MIX.items():
            mix = tuple(a * b for a, b in zip(tmix, cmix, strict=True))
            frames.append(
                make_bets(
                    rng,
                    per_cell,
                    omega_by_band=FLB_OMEGA,
                    band_probs=mix,
                    labels={"track": track, "class": klass},
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_flb_fixed_effects_absorb_odds_band_composition():
    """DoD: odds-band composition alone creates no segment effect.

    Two crossed segment columns (track, class). Omega depends on the odds band only (favourites
    1.05, longshots 0.55); tracks and classes differ only in their band mix, so a band-blind ROI
    differs a lot between them. Every track and class effect has |posterior mean| < 0.05 and
    P(omega > 1.25) < 0.5.
    """
    bets = make_flb_bets()

    # Premise: band-blind, the longshot-heavy track looks much worse than the favourite-heavy one;
    # given the true band omegas, no track or class deviates.
    def rows(col: str, level: str) -> pd.DataFrame:
        return bets[bets[col] == level]

    assert raw_log_omega(rows("track", "A")) - raw_log_omega(rows("track", "D")) > 0.12
    assert raw_log_omega(rows("class", "c1")) - raw_log_omega(rows("class", "c3")) > 0.08
    for col, levels in (("track", TRACK_MIX), ("class", CLASS_MIX)):
        for level in levels:
            assert abs(raw_log_omega(rows(col, level), FLB_OMEGA)) < 0.04, (col, level)

    fit = fit_segment_roi(
        bets,
        segment_cols=["track", "class"],
        odds_band_col="band",
        takeout_col="takeout",
        seed=3,
        draws=DRAWS,
    )
    for col, levels in (("track", TRACK_MIX), ("class", CLASS_MIX)):
        for level in levels:
            segment = {col: level}
            assert abs(fit.posterior_mean(segment)) < 0.05, (segment, fit.posterior_mean(segment))
            assert fit.p_beats_takeout(segment) < 0.5, segment


# --- a small segment is shrunk hard --------------------------------------------------------------


def test_a_small_segment_keeps_little_of_its_raw_deviation():
    """A 50-bet segment with raw ROI +40%, among 20 null segments of 1,000 bets (omega 1).

    ROI deviation is measured against what the band effects predict (omega_band = 1 here, so an
    expected ROI of (1 - t) - 1 = -20%): raw deviation = 0.40 - (-0.20) = 0.60. The posterior mean
    ROI deviation is (1 - t) * (exp(posterior_mean) - 1); it must be below a quarter of the raw
    one, and the small segment must not be flagged.
    """
    rng = np.random.default_rng(5050)
    frames = [
        make_bets(rng, 1000, omega_by_band=flat(1.0), labels={"segment": f"n{i:02d}"})
        for i in range(20)
    ]
    small = pd.DataFrame(
        {
            "stake": 1.0,
            "odds": 5.0,
            "won": [1] * 14 + [0] * 36,
            "band": "mid",
            "takeout": TAKEOUT,
            "segment": "small",
        }
    )
    small["payout"] = small["odds"] * small["won"]
    bets = pd.concat([*frames, small[frames[0].columns]], ignore_index=True)

    raw_roi = small["payout"].sum() / small["stake"].sum() - 1
    assert raw_roi == pytest.approx(0.40)
    raw_deviation = raw_roi - ((1 - TAKEOUT) * 1.0 - 1)
    assert raw_deviation == pytest.approx(0.60)

    fit = fit_segment_roi(
        bets,
        segment_cols=["segment"],
        odds_band_col="band",
        takeout_col="takeout",
        seed=9,
        draws=DRAWS,
    )
    u = fit.posterior_mean(seg("small"))
    posterior_deviation = (1 - TAKEOUT) * (math.exp(u) - 1)
    assert abs(posterior_deviation) < raw_deviation / 4, (u, posterior_deviation)
    assert fit.p_beats_takeout(seg("small")) < 0.5


# --- reproducibility -----------------------------------------------------------------------------


def test_a_seed_makes_the_fit_reproducible():
    """DoD: results are reproducible with a seed: two fits with the same seed give identical
    posterior means, probabilities and predictive intervals (exact equality)."""
    rng = np.random.default_rng(11)
    omegas = {"r0": 1.0, "r1": 1.3, "r2": 0.9, "r3": 1.0, "r4": 1.1, "r5": 1.0}
    frames = [
        make_bets(rng, 600, omega_by_band=flat(w), labels={"segment": s}) for s, w in omegas.items()
    ]
    bets = pd.concat(frames, ignore_index=True)

    def outputs(seed: int) -> list[tuple]:
        fit = fit_segment_roi(
            bets.copy(),
            segment_cols=["segment"],
            odds_band_col="band",
            takeout_col="takeout",
            seed=seed,
            draws=DRAWS,
        )
        out = []
        for s in omegas:
            interval = fit.predictive_roi(seg(s), 100)
            out.append(
                (
                    fit.posterior_mean(seg(s)),
                    fit.p_beats_takeout(seg(s)),
                    interval.low,
                    interval.high,
                )
            )
        return out

    first = outputs(42)
    assert outputs(42) == first
    assert all(math.isfinite(x) for row in first for x in row)


# --- a level whose bets all won -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("odds", "band", "n_won"),
    [(10.0, "long", 1), (1.5, "fav", 2)],
    ids=["one-win-at-10.0", "two-wins-at-1.5"],
)
def test_a_level_whose_bets_all_won_is_fitted_and_shrunk(odds, band, n_won):
    """Review of PR #201: a tiny segment level with only winning bets (no loser) must not crash
    the fit. Among 10 null segments of 1,000 bets (omega 1), the all-win level gets a finite
    posterior mean shrunk well below its raw ln-omega deviation, is not flagged, and has a finite
    predictive ROI interval; the null segments are still not flagged.
    """
    rng = np.random.default_rng(7801)
    frames = [
        make_bets(rng, 1000, omega_by_band=flat(1.0), labels={"segment": f"n{i:02d}"})
        for i in range(10)
    ]
    tiny = pd.DataFrame(
        {
            "stake": 1.0,
            "odds": odds,
            "won": [1] * n_won,
            "band": band,
            "takeout": TAKEOUT,
            "segment": "allwin",
        }
    )
    tiny["payout"] = tiny["odds"] * tiny["won"]
    bets = pd.concat([*frames, tiny[frames[0].columns]], ignore_index=True)

    # Premise: the level's raw deviation is large and positive (every bet won).
    raw = raw_log_omega(tiny)
    assert raw == pytest.approx(math.log(odds / (1 - TAKEOUT)))
    assert raw > 0.6

    fit = fit_segment_roi(
        bets,
        segment_cols=["segment"],
        odds_band_col="band",
        takeout_col="takeout",
        seed=2,
        draws=DRAWS,
    )

    u = fit.posterior_mean(seg("allwin"))
    assert math.isfinite(u)
    assert abs(u) < raw / 2, (u, raw)
    p = fit.p_beats_takeout(seg("allwin"))
    assert 0.0 <= p < 0.5, p
    interval = fit.predictive_roi(seg("allwin"), 100)
    assert math.isfinite(interval.low) and math.isfinite(interval.high)
    assert -1.0 <= interval.low < interval.high

    for i in range(10):
        s = f"n{i:02d}"
        prob = fit.p_beats_takeout(seg(s))
        assert 0.0 <= prob < 0.5, (s, prob)
        assert math.isfinite(fit.posterior_mean(seg(s)))


# --- the model's probability bound in the posterior draws (#88) ----------------------------------


def near_certain(odds: float, takeout: float, n_won: int, n: int, level: str = "s") -> pd.DataFrame:
    """n unit-stake bets at one price and takeout, in one band, of which the first n_won won."""
    won = np.array([1] * n_won + [0] * (n - n_won))
    return pd.DataFrame(
        {
            "stake": 1.0,
            "payout": odds * won,
            "odds": odds,
            "won": won,
            "band": "fav",
            "takeout": takeout,
            "segment": level,
        }
    )


BOUND_CASES = {
    "19-of-20-at-1.0": [near_certain(1.0, 0.2, 19, 20)],
    "29-of-30-at-1.0-takeout-0.1": [near_certain(1.0, 0.1, 29, 30)],
    "19-of-20-at-1.0-beside-another-level": [
        near_certain(1.0, 0.2, 19, 20),
        near_certain(4.0, 0.2, 8, 40, level="r"),
    ],
    "19-of-20-at-1.2": [near_certain(1.2, 0.2, 19, 20)],
}


@pytest.mark.parametrize("case", list(BOUND_CASES), ids=list(BOUND_CASES))
def test_return_draws_respect_the_models_probability_bound(case):
    """Round 3 of PR #345: bets that won almost every time at a price near 1. Each wins with
    probability omega * (1 - t) / O < 1, so each returns at most its odds O per unit stake and
    the segment's mean expected return per unit stake is at most its bets' mean odds, in every
    posterior draw: the expected return's mean and upper bound stay at or below it, and so does
    omega_mean times the payout rate. At odds 1.0 every bet pays at most its stake, so the
    expected return never exceeds 1 and P(beats takeout) is 0; at odds 1.2 the bound is 1.2, not
    break-even, and the bets that returned 1.14 per unit are still likely to beat it."""
    bets = pd.concat(BOUND_CASES[case], ignore_index=True)
    query = seg("s")
    own = bets[bets["segment"] == "s"]
    bound = float(own["odds"].mean())  # the most the segment's bets can return per unit stake
    payout = float((1 - own["takeout"]).mean())
    realised = float((own["won"] * own["odds"]).mean())
    # Premise: the bets returned close to the bound, so a posterior that ignores it crosses it.
    assert bound - 0.07 <= realised < bound, (realised, bound)

    fit = fit_segment_roi(
        bets,
        segment_cols=["segment"],
        odds_band_col="band",
        takeout_col="takeout",
        seed=7802,
        draws=DRAWS,
    )
    er = fit.expected_return(query)
    assert all(math.isfinite(v) for v in (er.mean, er.low, er.high)), er
    ceiling = bound * (1 + 1e-12)  # the bound, give or take the rounding of a mean
    assert er.low <= er.mean <= er.high <= ceiling, (er, bound)
    assert fit.omega_mean(query) * payout <= ceiling, (fit.omega_mean(query), bound)
    p = fit.p_beats_takeout(query)
    if bound <= 1.0:
        assert p == 0.0, (p, er)
    else:
        assert er.high > 1.0 and p > 0.5, (p, er)


# --- the bound is the cell's: one omega for every bet of a cell (#88) ----------------------------

LONG_RUN = 1_000_000  # bets in a predictive ROI whose outcome noise is far below the bound's gap
# (odds, takeout, n_won, n) groups of segment 's' in one cell, and the cell's bound on the mean
# expected return per unit stake, worked by hand: mean over bets of (1 - t_i) * min O_j/(1 - t_j).
CELL_CASES = {
    # 1.0 binds: omega <= 1 / 0.8 = 1.25, so both prices return at most 0.8 * 1.25 = 1.
    "1.0-and-1.2-at-takeout-0.2": ([(1.0, 0.2, 4, 5), (1.2, 0.2, 9, 10)], 1.0),
    # 1.0 at 0.1 binds: omega <= 1 / 0.9; (5 * 0.9 + 10 * 0.8) / 15 / 0.9 = 12.5 / 13.5.
    "1.0-at-0.1-and-1.5-at-0.2": ([(1.0, 0.1, 4, 5), (1.5, 0.2, 9, 10)], 12.5 / 13.5),
    # The longer price binds (the lower takeout): omega <= 1.1 / 0.95, not 1.05 / 0.7;
    # (5 * 0.95 + 10 * 0.7) / 15 * 1.1 / 0.95.
    "1.1-at-0.05-binds-before-1.05-at-0.3": (
        [(1.1, 0.05, 4, 5), (1.05, 0.3, 9, 10)],
        (5 * 0.95 + 10 * 0.7) / 15 * 1.1 / 0.95,
    ),
}


def cell_bets(groups: list[tuple], *, split: bool = False) -> pd.DataFrame:
    """Segment 's''s unit-stake bets, one block per (odds, takeout, n_won, n) group, all in band
    'fav' (one cell), or with `split` each group in a band of its own (a cell of its own)."""
    frames = []
    for k, (odds, takeout, n_won, n) in enumerate(groups):
        frame = near_certain(odds, takeout, n_won, n)
        if split:
            frame["band"] = f"b{k}"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cell_bound(bets: pd.DataFrame) -> float:
    """The most the bets' mean expected return per unit stake can be when each cell's bets share
    one omega below every one of their O / (1 - t): the mean of (1 - t_i) times its cell's
    min O_j / (1 - t_j). Segment 's' only, so the cells are the bands."""
    cap = (bets["odds"] / (1 - bets["takeout"])).groupby(bets["band"]).transform("min")
    return float(((1 - bets["takeout"]) * cap).mean())


@pytest.mark.parametrize("case", list(CELL_CASES), ids=list(CELL_CASES))
def test_a_cells_shared_omega_is_bounded_by_its_likeliest_bet(case):
    """Round 4 of PR #345: bets at two prices in one cell (one band, one segment level) share one
    omega, so the bet with the highest implied probability (1 - t) / O bounds it for both: in
    every posterior draw omega <= min O_j / (1 - t_j), and no draw pays the likelier bet at most
    its odds while paying the other more than (1 - t) times that bound. Clipping each bet's
    probability apart lets a draw of omega = 1.30 return 1.00 on the 1.0 bet and 1.04 on the 1.2
    bet, beating takeout although the 1.0 bet keeps omega below 1.25. So the expected return's
    upper bound, omega_mean times the payout rate and the long-run predictive ROI stay within the
    cell's bound (below break-even in every case here), and P(beats takeout) is 0. The same bets
    with each price in a band of its own are two cells, each with its own bound, and the
    segment's expected return may then pass the one-cell bound: the bound is the cell's, not the
    segment's."""
    groups, worked = CELL_CASES[case]
    bets = cell_bets(groups)
    query = seg("s")
    bound = cell_bound(bets)
    assert bound == pytest.approx(worked, rel=1e-12)
    # Premise: one cell, two prices, each won and lost at least once; the bets returned close to
    # the bound or above it, so a posterior that ignores the shared omega crosses it.
    assert bets["band"].nunique() == 1 and bets["odds"].nunique() == 2
    assert all(0 < n_won < n for _, _, n_won, n in groups)
    assert bound <= 1.0
    realised = float((bets["won"] * bets["odds"]).mean())
    assert realised >= bound - 0.07, (realised, bound)

    fit = fit_segment_roi(
        bets,
        segment_cols=["segment"],
        odds_band_col="band",
        takeout_col="takeout",
        seed=7803,
        draws=DRAWS,
    )
    ceiling = bound * (1 + 1e-12)  # the bound, give or take the rounding of a mean
    er = fit.expected_return(query)
    assert all(math.isfinite(v) for v in (er.mean, er.low, er.high)), er
    assert er.low <= er.mean <= er.high <= ceiling, (er, bound)
    payout = float((1 - bets["takeout"]).mean())
    assert fit.omega_mean(query) * payout <= ceiling, (fit.omega_mean(query), bound)
    assert fit.p_beats_takeout(query) == 0.0, (fit.p_beats_takeout(query), er)
    # Each predictive draw is LONG_RUN bets, each returning between 0 and the top odds, with a mean
    # of at most the bound: its ROI has mean <= bound - 1 and variance <= top^2 / (4 LONG_RUN)
    # (Popoviciu). By Cantelli at a quarter of the interval's upper tail, too few draws pass the
    # slack below to move the interval's upper end above it.
    interval = fit.predictive_roi(query, LONG_RUN)
    tail = (1 - interval.level) / 2
    sd = float(bets["odds"].max()) / (2 * math.sqrt(LONG_RUN))
    slack = sd * math.sqrt(4 / tail - 1)
    assert interval.high <= bound - 1 + slack, (interval, bound - 1, slack)

    split = cell_bets(groups, split=True)
    fit = fit_segment_roi(
        split,
        segment_cols=["segment"],
        odds_band_col="band",
        takeout_col="takeout",
        seed=7803,
        draws=DRAWS,
    )
    er = fit.expected_return(query)
    assert bound < er.high <= cell_bound(split) * (1 + 1e-12), (er, bound, cell_bound(split))


# --- grouped bets: one row stands for `count` identical bets with `won` wins (#429) --------------

GROUP_KEYS = ["segment", "band", "odds", "takeout"]


def group_bets(bets: pd.DataFrame) -> pd.DataFrame:
    """One row per (segment, band, odds, takeout) with its bets' count and wins, rows shuffled."""
    grouped = bets.groupby(GROUP_KEYS, as_index=False).agg(
        count=("won", "size"), won=("won", "sum")
    )
    return grouped.sample(frac=1.0, random_state=429).reset_index(drop=True)


def fit_grouped(grouped: pd.DataFrame, seed: int) -> SegmentROI:
    return fit_segment_roi(
        grouped,
        segment_cols=["segment"],
        odds_band_col="band",
        takeout_col="takeout",
        seed=seed,
        draws=DRAWS,
        count_col="count",
    )


def test_grouped_bets_fit_like_the_same_bets_one_row_each(planted):
    """DoD: the planted-segment bets grouped by (segment, band, odds, takeout) with counts give,
    at the same seed, the per-bet fit's posterior means (within 1e-6) and P(omega beats takeout)
    (within 0.01) for every segment, and the same band effects.

    The likelihood is a sum over bets, so a binomial row is exactly its Bernoullis: the
    conditional modes, the tau posterior and the importance weights do not change, which is why
    the posterior mean (a weighted mean of modes) is pinned this tightly; P(beats takeout) comes
    from the posterior draws and is pinned to 0.01.
    """
    bets, per_bet = planted
    grouped = group_bets(bets)

    # Premise: the grouping is real (far fewer rows, rows with several wins) and exact.
    assert len(grouped) < len(bets) / 4
    assert grouped["won"].max() > 1
    assert grouped["count"].sum() == len(bets)
    assert grouped["won"].sum() == bets["won"].sum()

    fit = fit_grouped(grouped, seed=1)
    for s in SEGMENTS:
        assert fit.posterior_mean(seg(s)) == pytest.approx(
            per_bet.posterior_mean(seg(s)), abs=1e-6
        ), s
        assert abs(fit.p_beats_takeout(seg(s)) - per_bet.p_beats_takeout(seg(s))) <= 0.01, s
    for band, effect in per_bet.band_effects.items():
        assert fit.band_effects[band] == pytest.approx(effect, abs=1e-6), band
    # The verdict of the per-bet fit survives grouping.
    assert max(SEGMENTS, key=lambda s: fit.posterior_mean(seg(s))) == PLANTED


def test_grouped_bets_weigh_each_row_by_its_count():
    """A row stands for `count` bets everywhere, not only in the likelihood: P(beats takeout) and
    the predictive ROI average over bets, not over grouped rows.

    Built so that the two weightings disagree: favourites (omega 1.5, expected return 1.2) are 90%
    of the bets on a few odds values, longshots (omega 0.5, return 0.4) 10% of the bets spread over
    many odds values. Per bet the mean expected return is above 1; per grouped row it is far
    below. The grouped fit must match the per-bet fit.
    """
    rng = np.random.default_rng(4291)
    omega = {"fav": 1.5, "mid": 1.0, "long": 0.5}
    frames = [
        make_bets(rng, 2000, omega_by_band=omega, band_probs=(0.9, 0.0, 0.1), labels=seg(s))
        for s in ("g0", "g1", "g2", "g3", "g4", "g5")
    ]
    bets = pd.concat(frames, ignore_index=True)
    grouped = group_bets(bets)

    # Premise: weighting by bets and weighting by rows put the true mean return on either side of 1.
    def mean_return(frame: pd.DataFrame, weight: np.ndarray) -> float:
        ret = frame["band"].map(omega).to_numpy() * (1 - frame["takeout"].to_numpy())
        return float((ret * weight).sum() / weight.sum())

    g0 = grouped[grouped["segment"] == "g0"]
    assert mean_return(g0, g0["count"].to_numpy(dtype=float)) > 1.05
    assert mean_return(g0, np.ones(len(g0))) < 0.8
    assert grouped["won"].max() > 1

    per_bet = fit_segment_roi(
        bets,
        segment_cols=["segment"],
        odds_band_col="band",
        takeout_col="takeout",
        seed=7,
        draws=DRAWS,
    )
    fit = fit_grouped(grouped, seed=7)
    for s in ("g0", "g1", "g2", "g3", "g4", "g5"):
        assert fit.posterior_mean(seg(s)) == pytest.approx(
            per_bet.posterior_mean(seg(s)), abs=1e-6
        ), s
        assert abs(fit.p_beats_takeout(seg(s)) - per_bet.p_beats_takeout(seg(s))) <= 0.01, s
    # Per bet the segments beat the takeout; a row-weighted query would say they do not.
    assert per_bet.p_beats_takeout(seg("g0")) > 0.9
    for s in ("g0", "g3"):
        grouped_roi, per_bet_roi = (
            fit.predictive_roi(seg(s), 100),
            per_bet.predictive_roi(seg(s), 100),
        )
        assert grouped_roi.low == pytest.approx(per_bet_roi.low, abs=0.02), s
        assert grouped_roi.high == pytest.approx(per_bet_roi.high, abs=0.02), s


def test_count_col_rejects_wins_above_the_count():
    """A grouped row with more wins than bets (or negative wins) is not a set of bets: ValueError.
    So is a `count_col` the frame does not have. A row whose bets all won is valid."""
    rng = np.random.default_rng(4292)
    frames = [make_bets(rng, 400, omega_by_band=flat(1.0), labels=seg(s)) for s in ("a", "b", "c")]
    good = group_bets(pd.concat(frames, ignore_index=True))
    kwargs = dict(
        segment_cols=["segment"], odds_band_col="band", takeout_col="takeout", seed=0, draws=50
    )

    for count, won in ((1, 2), (4, 5), (3, 30), (2, -1)):
        bad = good.copy()
        bad.loc[0, "count"] = count
        bad.loc[0, "won"] = won
        with pytest.raises(ValueError):
            fit_segment_roi(bad, count_col="count", **kwargs)

    with pytest.raises(ValueError):
        fit_segment_roi(good.drop(columns="count"), count_col="count", **kwargs)

    # Premise of the cases above: the frame itself is valid grouped input.
    all_won = good.copy()
    row = all_won.index[all_won["band"] == "long"][0]
    all_won.loc[row, "won"] = all_won.loc[row, "count"]
    assert isinstance(fit_segment_roi(all_won, count_col="count", **kwargs), SegmentROI)
