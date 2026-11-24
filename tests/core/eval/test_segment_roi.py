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
