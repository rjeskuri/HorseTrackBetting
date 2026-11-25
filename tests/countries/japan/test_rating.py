"""The C2 forward-filtered dynamic horse rating (ticket #91, epic #73; 08 §3.2 C2, 05 §3.2).

Contract pinned here (`horsetrack.countries.japan.models.rating`):
- `DynamicRating(*, drift_var_per_day, perf_var, prior_mean, prior_var, spell_var, spell_days)`.
- `DynamicRating.run(results) -> pd.DataFrame`. results: race_id, race_utc (tz-aware UTC),
  horse_id, finish_pos (1 = winner), in any row order. The output has one row per input
  (race_id, horse_id) and the columns race_id, horse_id, mu_before, sigma_before, mu_after,
  sigma_after. Races are processed in race_utc order whatever the row order.
- A horse's first race starts at the prior: mu_before = prior_mean, sigma_before = sqrt(prior_var).
- Between races the ability is a random walk: at a horse's next race, days_off after its previous
  one, mu_before = the previous mu_after and
      sigma_before^2 = previous sigma_after^2 + drift_var_per_day * days_off
                       (+ spell_var when days_off > spell_days, a long layoff).
- Forward filtering only (05 §3.2): mu_before/sigma_before at a race depend on earlier races only,
  never on that race's result or later ones; mu_after on results up to and including it.
- A race's update moves the winner's mu up and the last-placed runner's mu down and does not
  widen anyone's sigma. The multi-runner update itself is the coder's ruling (not pinned).
- `rating_member(ratings, *, scale) -> pd.Series`: one log-probability per ratings row, on the
  ratings index; exp of it sums to 1 within each race_id. It reads the filtered (before) columns
  only: mu_after and sigma_after never change it. Within a race at equal sigma, a higher
  mu_before is a higher probability; equal ratings give 1/n.

Not pinned: the update rule, the exact days_off boundary at spell_days, the direction in which
`scale` sharpens the member, and the run step (drift variance chosen by inner-fold log loss,
market correlation, pseudo-R2, stage-2 alpha, registry records), which runs on local stores.
Every synthetic gap is a whole number of 24-hour days at the same clock time, so a calendar-day
and an elapsed-time reading of days_off agree. Expected values come from the parameters and the
planted abilities, never from the code under test. Every RNG is seeded.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from horsetrack.countries.japan.models.rating import DynamicRating, rating_member

T0 = pd.Timestamp("2025-01-04 04:00", tz="UTC")
OUT_COLUMNS = {"race_id", "horse_id", "mu_before", "sigma_before", "mu_after", "sigma_after"}

BASE = dict(
    drift_var_per_day=0.01,
    perf_var=1.0,
    prior_mean=0.0,
    prior_var=1.0,
    spell_var=0.5,
    spell_days=90,
)


def model(**overrides) -> DynamicRating:
    return DynamicRating(**{**BASE, **overrides})


def frame(races) -> pd.DataFrame:
    """races: (race_id, race_utc, [horse ids in finishing order]) -> the results frame."""
    rows = [
        {"race_id": rid, "race_utc": utc, "horse_id": h, "finish_pos": pos}
        for rid, utc, order in races
        for pos, h in enumerate(order, start=1)
    ]
    return pd.DataFrame(rows)


def keyed(out: pd.DataFrame) -> pd.DataFrame:
    return out.set_index(["race_id", "horse_id"]).sort_index()


def row(out: pd.DataFrame, race_id, horse_id) -> pd.Series:
    sel = out[(out["race_id"] == race_id) & (out["horse_id"] == horse_id)]
    assert len(sel) == 1, (race_id, horse_id, len(sel))
    return sel.iloc[0]


def return_after(days_off: int, **overrides):
    """h1 runs a 4-horse race at T0, then a second race days_off later against debutants."""
    results = frame(
        [
            ("r-first", T0, ["h2", "h1", "h3", "h4"]),
            ("r-back", T0 + pd.Timedelta(days=days_off), ["n1", "h1", "n2"]),
        ]
    )
    out = model(**overrides).run(results.iloc[::-1].reset_index(drop=True))
    return row(out, "r-first", "h1"), row(out, "r-back", "h1")


def season(rng, n_horses, n_days, races_per_day, field_size, *, gap_days=1):
    """Random fields each day (no horse twice a day); returns the results frame, abilities and
    the order of each race drawn from ability + N(0, 1) noise. Race ids do not sort by time."""
    ability = rng.normal(0.0, 1.0, n_horses)
    horses = np.array([f"hz{i:04d}" for i in range(n_horses)], dtype=object)
    races = []
    for day in range(n_days):
        utc0 = T0 + pd.Timedelta(days=day * gap_days)
        picks = rng.permutation(n_horses)[: races_per_day * field_size]
        for k in range(races_per_day):
            idx = picks[k * field_size : (k + 1) * field_size]
            perf = ability[idx] + rng.normal(0.0, 1.0, len(idx))
            order = horses[idx[np.argsort(-perf)]]
            rid = f"race-{rng.integers(10**9):09d}-{day}-{k}"
            races.append((rid, utc0 + pd.Timedelta(minutes=30 * k), list(order)))
    return frame(races), dict(zip(horses, ability, strict=True))


# ---------------------------------------------------------------------------------------------
# run: shape and the update direction
# ---------------------------------------------------------------------------------------------


def test_run_returns_one_row_per_runner_with_finite_positive_sigmas():
    rng = np.random.default_rng(91)
    results, _ = season(rng, n_horses=30, n_days=12, races_per_day=2, field_size=8)
    shuffled = results.sample(frac=1.0, random_state=3).reset_index(drop=True)
    out = model().run(shuffled)

    assert OUT_COLUMNS <= set(out.columns)
    assert len(out) == len(results)
    got = set(zip(out["race_id"], out["horse_id"], strict=True))
    assert got == set(zip(results["race_id"], results["horse_id"], strict=True))
    for col in ("mu_before", "sigma_before", "mu_after", "sigma_after"):
        assert np.isfinite(out[col].to_numpy(dtype=float)).all(), col
    # sigma_h(t) is exported per runner for the meld: positive at every row.
    assert (out["sigma_before"] > 0).all()
    assert (out["sigma_after"] > 0).all()


def test_an_update_lifts_the_winner_lowers_the_last_and_does_not_widen_sigma():
    rng = np.random.default_rng(7)
    results, _ = season(rng, n_horses=24, n_days=10, races_per_day=2, field_size=6)
    out = model().run(results)
    merged = out.merge(results, on=["race_id", "horse_id"])
    field_size = merged.groupby("race_id")["finish_pos"].transform("max")

    winners = merged[merged["finish_pos"] == 1]
    lasts = merged[merged["finish_pos"] == field_size]
    assert (winners["mu_after"] > winners["mu_before"]).all()
    assert (lasts["mu_after"] < lasts["mu_before"]).all()
    assert (merged["sigma_after"] <= merged["sigma_before"] + 1e-12).all()


# ---------------------------------------------------------------------------------------------
# Leakage: forward-filtered only
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cut_day", [3, 9, 15])
def test_filtered_rating_uses_no_later_race(cut_day):
    rng = np.random.default_rng(1000 + cut_day)
    results, _ = season(rng, n_horses=40, n_days=20, races_per_day=3, field_size=8, gap_days=2)
    d = (T0 + pd.Timedelta(days=cut_day * 2)).normalize() + pd.Timedelta(days=1)  # end of cut day

    changed = results.copy()
    later = changed["race_utc"] >= d
    # Reverse the finishing order of every race after d.
    size = changed.groupby("race_id")["finish_pos"].transform("max")
    changed.loc[later, "finish_pos"] = size[later] + 1 - changed.loc[later, "finish_pos"]

    m = model()
    a = keyed(m.run(results.sample(frac=1.0, random_state=11).reset_index(drop=True)))
    b = keyed(m.run(changed.sample(frac=1.0, random_state=12).reset_index(drop=True)))
    times = keyed(results)["race_utc"].reindex(a.index)
    early = (times < d).to_numpy()
    assert early.any() and (~early).any()

    cols = ["mu_before", "sigma_before", "mu_after", "sigma_after"]
    np.testing.assert_allclose(
        a.loc[early, cols].to_numpy(float), b.loc[early, cols].to_numpy(float), rtol=0, atol=1e-10
    )
    # The perturbation is real: some rating after d moves.
    assert not np.allclose(a.loc[~early, "mu_before"], b.loc[~early, "mu_before"], atol=1e-6)


def test_a_race_result_does_not_reach_its_own_before_values():
    rng = np.random.default_rng(55)
    results, _ = season(rng, n_horses=30, n_days=8, races_per_day=2, field_size=8)
    order = results.drop_duplicates("race_id").sort_values("race_utc")["race_id"].tolist()
    target = order[len(order) // 2]

    flipped = results.copy()
    sel = flipped["race_id"] == target
    flipped.loc[sel, "finish_pos"] = sel.sum() + 1 - flipped.loc[sel, "finish_pos"]

    m = model()
    a = keyed(m.run(results)).xs(target, level="race_id")
    b = keyed(m.run(flipped)).xs(target, level="race_id")
    np.testing.assert_allclose(a["mu_before"], b["mu_before"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(a["sigma_before"], b["sigma_before"], rtol=0, atol=1e-10)
    assert not np.allclose(a["mu_after"], b["mu_after"], atol=1e-6)


# ---------------------------------------------------------------------------------------------
# Prior, drift and spell
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prior_mean, prior_var, perf_var",
    [(0.0, 1.0, 1.0), (25.0, 69.4, 17.0), (-3.0, 0.25, 1.0), (1500.0, 350.0**2, 200.0**2)],
)
def test_a_debut_horse_starts_at_the_prior(prior_mean, prior_var, perf_var):
    rng = np.random.default_rng(int(prior_var) % 97)
    results, _ = season(rng, n_horses=20, n_days=6, races_per_day=2, field_size=6, gap_days=3)
    # A newcomer enters an established field on the last day.
    last = results["race_utc"].max() + pd.Timedelta(days=5)
    veterans = results["horse_id"].unique()[:5].tolist()
    results = pd.concat(
        [results, frame([("r-debut", last, [veterans[0], "newcomer", *veterans[1:]])])],
        ignore_index=True,
    )
    # Drift negligible next to the prior, so only races move a veteran's sigma.
    m = model(
        prior_mean=prior_mean,
        prior_var=prior_var,
        perf_var=perf_var,
        drift_var_per_day=prior_var * 1e-6,
    )
    out = m.run(results)
    merged = out.merge(results, on=["race_id", "horse_id"]).sort_values("race_utc")
    first = merged.groupby("horse_id", sort=False).head(1)

    assert "newcomer" in set(first["horse_id"])
    np.testing.assert_allclose(first["mu_before"], prior_mean, rtol=0, atol=1e-9)
    np.testing.assert_allclose(first["sigma_before"], math.sqrt(prior_var), rtol=1e-9)
    # A returning veteran is not reset to the prior: its races narrowed sigma.
    vet = row(out, "r-debut", veterans[0])
    assert vet["sigma_before"] < math.sqrt(prior_var)


def test_rating_sd_grows_with_days_off():
    days = [1, 3, 7, 14, 28, 56, 84]
    drift = 0.004
    sds = []
    for n in days:
        first, back = return_after(n, drift_var_per_day=drift)
        assert back["mu_before"] == pytest.approx(first["mu_after"], abs=1e-12)
        expected = math.sqrt(first["sigma_after"] ** 2 + drift * n)
        assert back["sigma_before"] == pytest.approx(expected, rel=1e-9)
        sds.append(back["sigma_before"])
    assert all(b > a for a, b in zip(sds, sds[1:], strict=False))

    # The drift rate is a real parameter: a faster walk widens the same layoff more.
    _, slow = return_after(28, drift_var_per_day=0.001)
    _, fast = return_after(28, drift_var_per_day=0.05)
    assert fast["sigma_before"] > slow["sigma_before"]


@pytest.mark.parametrize("spell_var", [0.5, 2.0])
def test_a_long_layoff_adds_the_spell_variance(spell_var):
    drift = 0.002
    params = dict(drift_var_per_day=drift, spell_var=spell_var, spell_days=90)

    first, back = return_after(60, **params)
    assert back["sigma_before"] ** 2 == pytest.approx(
        first["sigma_after"] ** 2 + drift * 60, rel=1e-9
    )

    first, back = return_after(120, **params)
    assert back["sigma_before"] ** 2 == pytest.approx(
        first["sigma_after"] ** 2 + drift * 120 + spell_var, rel=1e-9
    )
    assert back["mu_before"] == pytest.approx(first["mu_after"], abs=1e-12)

    # spell_days is a real parameter: with a 150-day threshold, 120 days off is no spell.
    first, back = return_after(120, drift_var_per_day=drift, spell_var=spell_var, spell_days=150)
    assert back["sigma_before"] ** 2 == pytest.approx(
        first["sigma_after"] ** 2 + drift * 120, rel=1e-9
    )


# ---------------------------------------------------------------------------------------------
# Recovery of planted abilities
# ---------------------------------------------------------------------------------------------


def test_ratings_recover_planted_abilities():
    rng = np.random.default_rng(2026)
    # 300 days x 10 races of 10 runners = 3,000 races over 200 horses.
    results, ability = season(rng, n_horses=200, n_days=300, races_per_day=10, field_size=10)
    assert results["race_id"].nunique() == 3000
    assert results["horse_id"].nunique() == 200

    out = model(drift_var_per_day=1e-4, perf_var=1.0, prior_var=1.0).run(results)
    merged = out.merge(results[["race_id", "horse_id", "race_utc"]], on=["race_id", "horse_id"])
    final = merged.sort_values("race_utc").groupby("horse_id")["mu_after"].last()
    rho = spearmanr(final.to_numpy(), [ability[h] for h in final.index]).statistic
    assert rho > 0.8


# ---------------------------------------------------------------------------------------------
# The member
# ---------------------------------------------------------------------------------------------


def ratings_frame(rng, sizes, *, equal_sigma=False):
    rows = []
    for r, n in enumerate(sizes):
        sig = 0.7 if equal_sigma else None
        for i in range(n):
            mb = rng.normal(0.0, 1.0)
            sb = sig if sig is not None else rng.uniform(0.3, 1.2)
            rows.append(
                {
                    "race_id": f"m{r:03d}",
                    "horse_id": f"h{r:03d}-{i:02d}",
                    "mu_before": mb,
                    "sigma_before": sb,
                    "mu_after": mb + rng.normal(0.0, 0.3),
                    "sigma_after": sb * 0.9,
                }
            )
    out = pd.DataFrame(rows).sample(frac=1.0, random_state=5)
    out.index = rng.permutation(np.arange(1000, 1000 + len(out)))
    return out


@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0])
def test_member_log_probabilities_normalise_within_each_race(scale):
    rng = np.random.default_rng(int(scale * 10))
    ratings = ratings_frame(rng, [2, 5, 8, 12, 16, 18])
    lp = rating_member(ratings, scale=scale)

    assert isinstance(lp, pd.Series)
    assert len(lp) == len(ratings)
    assert set(lp.index) == set(ratings.index)
    lp = lp.reindex(ratings.index)
    assert np.isfinite(lp.to_numpy(float)).all()
    assert (lp < 0).all()
    sums = np.exp(lp).groupby(ratings["race_id"]).sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=0, atol=1e-9)

    # Straight from run's output too.
    results, _ = season(np.random.default_rng(3), 20, 5, 2, 7)
    rated = model().run(results)
    lp2 = rating_member(rated, scale=scale).reindex(rated.index)
    sums2 = np.exp(lp2).groupby(rated["race_id"]).sum()
    np.testing.assert_allclose(sums2.to_numpy(), 1.0, rtol=0, atol=1e-9)


def test_member_ranks_by_mu_before_and_ignores_the_after_columns():
    rng = np.random.default_rng(17)
    ratings = ratings_frame(rng, [4, 9, 14], equal_sigma=True)
    lp = rating_member(ratings, scale=1.0).reindex(ratings.index)
    for _, g in ratings.groupby("race_id"):
        by_mu = lp.loc[g.sort_values("mu_before").index].to_numpy()
        assert np.all(np.diff(by_mu) > 0)

    # Equal ratings in a race of n: 1/n each.
    flat = ratings.copy()
    flat["mu_before"] = 0.4
    lp_flat = rating_member(flat, scale=1.0).reindex(flat.index)
    n = flat.groupby("race_id")["horse_id"].transform("size")
    np.testing.assert_allclose(lp_flat, -np.log(n), rtol=0, atol=1e-9)

    # Only filtered values feed the member: the after columns change nothing.
    smoothed = ratings.copy()
    smoothed["mu_after"] = rng.normal(5.0, 3.0, len(smoothed))
    smoothed["sigma_after"] = rng.uniform(0.01, 0.1, len(smoothed))
    lp_s = rating_member(smoothed, scale=1.0).reindex(smoothed.index)
    np.testing.assert_allclose(lp_s, lp, rtol=0, atol=1e-12)

    # The scale is a real parameter.
    lp_other = rating_member(ratings, scale=4.0).reindex(ratings.index)
    assert not np.allclose(lp_other, lp, atol=1e-6)
