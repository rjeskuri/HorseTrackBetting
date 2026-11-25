"""The crowd-calibration baseline at the final Win price (ticket #77; 08 §8 A3, §4 E1, E3, E4).

Contract pinned here (`horsetrack.countries.japan.studies.calibration`, not the D4 spline model in
`models.v0.calibration`):
- Inputs are pandas Series on one index; `race_id` labels each runner's race (any hashable; rows
  need not be grouped by race). Odds are decimal, stake included, and go through
  `core.pricing.clean_odds`: a missing, non-finite or sub-1.0 figure is an unpriced runner, dropped
  before any de-vig.
- `devig_normalised(odds, race_id)`, `devig_power(odds, race_id, beta)` and
  `devig_shin(odds, race_id)` return a float Series on the input's index, in its order. An unpriced
  runner takes no share (its value is NaN) and each race's priced runners sum to 1.
  - normalised: q_i = (1/O_i) / sum_j (1/O_j) over the race's priced runners;
  - power: p_i proportional to (1/O_i)^beta;
  - Shin (1993): with pi_i = 1/O_i and Pi = sum_j pi_j, p_i = (sqrt(z^2 + 4(1 - z) pi_i^2 / Pi)
    - z) / (2(1 - z)), the insider share z solved per race so the race sums to 1 (z = 0 on a book
    without overround). Equivalently a Shin bookmaker on true chances p quotes
    pi_i = sqrt(z p_i + (1 - z) p_i^2) * sum_j sqrt(z p_j + (1 - z) p_j^2); `shin_book` below.
- `fit_power_beta(odds, race_id, won) -> BetaFit(beta, se, n_races)`: the conditional-logit MLE of
  beta in P(i wins) proportional to exp(beta * ln(1/O_i)) over the race's priced runners; se is
  the observed information at beta-hat to the power -1/2; only races with a priced winner enter
  and `n_races` counts them.
- `normalised_entropy(odds, race_id)`: one value per race, indexed by race id: the Shannon entropy
  of the race's normalised probabilities over its n priced runners, divided by ln n.
- `segment_beta(frame, segment)`: `frame` has race_id, odds, won and the `segment` column (constant
  within a race); one row per segment value, with the value in a column named `segment` and
  columns beta, se, n_races, each row the beta fit on that segment's races.
- `flag_floor_bound(odds, *, floor)`: a bool Series on the input's index, True for a priced runner
  at or under `floor` (the 1e-9 tolerance of `core.pricing.floor_bound`); unpriced is never True.

Expected values come from first principles (inverse-odds shares, planted Shin chances, the
conditional-logit score and information computed here), never from the code under test. Every
frame is synthetic and every RNG is seeded. The fits' seeds were checked against a separate MLE:
each estimate sits within 1.4 standard errors of its planted beta, so the 3-se (and, for beta = 1,
the 1.96-se) bounds hold with room for any correct optimiser.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math

import numpy as np
import pandas as pd
import pytest

from horsetrack.countries.japan.studies.calibration import (
    BetaFit,
    devig_normalised,
    devig_power,
    devig_shin,
    fit_power_beta,
    flag_floor_bound,
    normalised_entropy,
    segment_beta,
)

NAN = float("nan")
INF = float("inf")
TAKE = 0.8  # the Win pool's payout share: a board quotes about TAKE / share
SHIN_ATOL = 1e-6  # Shin needs a per-race root; any solver worth using gets this close


# ---------------------------------------------------------------------------------------------
# Synthetic boards and first-principles references
# ---------------------------------------------------------------------------------------------


def priced(odds):
    odds = np.asarray(odds, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(odds) & (odds >= 1.0)


def ref_normalised(odds):
    """q over the priced runners of one race; NaN where unpriced."""
    odds = np.asarray(odds, dtype=float)
    ok = priced(odds)
    inv = np.where(ok, 1.0 / np.where(ok, odds, 1.0), 0.0)
    return np.where(ok, inv / inv.sum(), NAN)


def ref_power(odds, beta):
    odds = np.asarray(odds, dtype=float)
    ok = priced(odds)
    w = np.where(ok, (1.0 / np.where(ok, odds, 1.0)) ** beta, 0.0)
    return np.where(ok, w / w.sum(), NAN)


def ref_entropy(odds):
    q = ref_normalised(odds)
    q = q[~np.isnan(q)]
    return float(-(q * np.log(q)).sum() / math.log(len(q)))


def shin_book(p, z):
    """Decimal odds a Shin (1993) bookmaker quotes on true chances `p` with insider share `z`."""
    p = np.asarray(p, dtype=float)
    s = np.sqrt(z * p + (1.0 - z) * p**2)
    odds = 1.0 / (s * s.sum())
    assert np.all(odds >= 1.0)  # a usable book: every price is a price clean_odds keeps
    return odds


def chances(rng, n, spread=1.0):
    s = np.exp(spread * rng.standard_normal(n))
    return s / s.sum()


def random_books(rng, n_races, *, sizes=(2, 18), take=TAKE, rounded=False):
    """{race_id: odds} for pari-mutuel-like boards, field sizes drawn from `sizes`."""
    books = {}
    for k in range(n_races):
        q = chances(rng, int(rng.integers(sizes[0], sizes[1] + 1)))
        odds = take / q
        if rounded:
            odds = np.round(odds, 1)
        books[f"jra:2021-05-{k % 28 + 1:02d}:TOK:{k:03d}"] = np.maximum(odds, 1.0)
    return books


def relabel(frame, rng):
    """Rows shuffled across races, on a shuffled, non-contiguous integer index."""
    frame = frame.iloc[rng.permutation(len(frame))].copy()
    frame.index = pd.Index(7 * rng.permutation(len(frame)) + 101)
    return frame


def runners(books, rng, winners=None):
    """Runner rows (race_id, pos, odds, won) for `books`; `winners` maps race id -> position."""
    winners = winners or {}
    rows = [
        (race_id, pos, float(o), winners.get(race_id) == pos)
        for race_id, odds in books.items()
        for pos, o in enumerate(odds)
    ]
    return relabel(pd.DataFrame(rows, columns=["race_id", "pos", "odds", "won"]), rng)


def expected(frame, books, ref):
    """`ref(book)` per race, laid out on the frame's rows."""
    per_race = {race_id: ref(odds) for race_id, odds in books.items()}
    values = [per_race[r][p] for r, p in zip(frame["race_id"], frame["pos"], strict=True)]
    return pd.Series(values, index=frame.index, dtype=float)


def assert_on_index(result, frame):
    assert isinstance(result, pd.Series)
    assert result.index.equals(frame.index)


def assert_races_sum_to_one(result, race_id, atol):
    sums = result.groupby(race_id.to_numpy()).sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=0, atol=atol)


# ----- fit data: (n_races, n_max) boards, winners drawn from p proportional to q^beta -----------


def board(rng, n_races, n_max, n_min=None, spread=1.0):
    """Final Win odds rounded to 0.1 (floor 1.0); NaN where a race has fewer than n_max runners."""
    strength = rng.standard_normal((n_races, n_max)) * spread
    if n_min is None:
        absent = np.zeros(strength.shape, dtype=bool)
    else:
        size = rng.integers(n_min, n_max + 1, size=n_races)
        absent = np.arange(n_max)[None, :] >= size[:, None]
    s = np.where(absent, 0.0, np.exp(strength))
    q = s / s.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        odds = np.maximum(np.round(TAKE / q, 1), 1.0)
    odds[absent] = NAN
    return odds


def planted_chances(odds, beta):
    """p proportional to q^beta, q the normalised shares of the board's own (rounded) odds."""
    inv = 1.0 / odds
    q = inv / np.nansum(inv, axis=1, keepdims=True)
    p = q**beta
    return p / np.nansum(p, axis=1, keepdims=True)


def draw_winners(rng, p):
    p = np.nan_to_num(p)
    u = rng.random(len(p))[:, None]
    return np.minimum((np.cumsum(p, axis=1) < u).sum(axis=1), p.shape[1] - 1)


def fit_rows(odds, winners, *, prefix):
    """Long runner rows (race_id, odds, won) for a board; absent runners are left out."""
    n_races, n_max = odds.shape
    won = np.arange(n_max)[None, :] == np.asarray(winners)[:, None]
    frame = pd.DataFrame(
        {
            "race_id": np.repeat([f"{prefix}{k:05d}" for k in range(n_races)], n_max),
            "odds": odds.ravel(),
            "won": won.ravel(),
        }
    )
    return frame[~np.isnan(odds.ravel())].reset_index(drop=True)


def planted_frame(seed, n_races, n_max, beta, *, n_min=None, prefix="r"):
    rng = np.random.default_rng(seed)
    odds = board(rng, n_races, n_max, n_min)
    winners = draw_winners(rng, planted_chances(odds, beta))
    return fit_rows(odds, winners, prefix=prefix), rng


def score_and_information(frame, beta):
    """The conditional-logit score U and observed information I at `beta`, from first principles,
    over the priced runners of races with a priced winner: x = ln(1/odds), p = softmax(beta x)
    within the race, U = sum over races (x_winner - E_p x), I = sum over races Var_p x."""
    odds = frame["odds"].to_numpy(dtype=float)
    keep = priced(odds)
    d = pd.DataFrame(
        {
            "race": frame["race_id"].to_numpy()[keep],
            "x": -np.log(odds[keep]),
            "won": frame["won"].to_numpy(dtype=bool)[keep],
        }
    )
    d = d[d.groupby("race")["won"].transform("any")]
    a = beta * d["x"]
    e = np.exp(a - a.groupby(d["race"]).transform("max"))
    p = e / e.groupby(d["race"]).transform("sum")
    mean = (p * d["x"]).groupby(d["race"]).sum()
    var = (p * d["x"] ** 2).groupby(d["race"]).sum() - mean**2
    x_won = d.loc[d["won"], "x"].groupby(d.loc[d["won"], "race"]).sum()
    return float((x_won - mean).sum()), float(var.sum())


def assert_is_the_mle(beta, se, frame):
    score, information = score_and_information(frame, beta)
    assert abs(score / information) < 5e-5  # beta-hat is within a Newton step of 5e-5 of the root
    assert se == pytest.approx(information**-0.5, rel=1e-3)


def add_races_without_a_priced_winner(frame, *, prefix, extra=None):
    """A void race, a race whose winner is unpriced (NaN) and one whose winner's figure is junk
    (0.5): none of them has a priced winner, so none may enter a fit or its count."""
    extra = extra or {}
    rows = []
    for k, (won_odds, others) in enumerate(
        [(None, [2.5, 3.1, 7.7, 12.0]), (NAN, [1.9, 4.4, 9.0]), (0.5, [2.2, 2.8, 30.0])]
    ):
        for o in others:
            rows.append((f"{prefix}x{k}", o, False))
        if won_odds is not None:
            rows.append((f"{prefix}x{k}", won_odds, True))
    junk = pd.DataFrame(rows, columns=["race_id", "odds", "won"]).assign(**extra)
    return pd.concat([frame, junk], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# De-vigs: normalised and power
# ---------------------------------------------------------------------------------------------


def test_normalised_devig_sums_to_one_per_race():
    rng = np.random.default_rng(771)
    hand = runners({"A": np.array([1.6, 3.2, 3.2])}, rng)
    got = devig_normalised(hand["odds"], hand["race_id"])
    assert_on_index(got, hand)
    want = np.array([0.5, 0.25, 0.25])[hand["pos"].to_numpy()]
    np.testing.assert_allclose(got.to_numpy(), want, rtol=0, atol=1e-12)

    # Many races in one call, rows shuffled across races: each race is normalised on its own.
    for rounded in (False, True):
        books = random_books(rng, 40, rounded=rounded)
        books["flat"] = np.full(9, 7.2)
        books["two"] = np.array([1.3, 3.4])
        frame = runners(books, rng)
        got = devig_normalised(frame["odds"], frame["race_id"])
        assert_on_index(got, frame)
        np.testing.assert_allclose(got, expected(frame, books, ref_normalised), rtol=0, atol=1e-12)
        assert_races_sum_to_one(got, frame["race_id"], atol=1e-12)


def test_power_devig_with_beta_one_equals_normalised():
    rng = np.random.default_rng(772)
    books = random_books(rng, 50, rounded=True)
    frame = runners(books, rng)
    got = devig_power(frame["odds"], frame["race_id"], 1.0)
    assert_on_index(got, frame)
    np.testing.assert_allclose(got, expected(frame, books, ref_normalised), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        got, devig_normalised(frame["odds"], frame["race_id"]), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("beta", [0.5, 0.9, 1.06, 1.5, 2.0])
def test_power_devig_follows_the_power_and_beta_above_one_favours_the_favourite(beta):
    rng = np.random.default_rng([773, int(beta * 100)])
    books = random_books(rng, 40, sizes=(3, 18))  # unrounded: no tied prices
    frame = runners(books, rng)

    got = devig_power(frame["odds"], frame["race_id"], beta)

    assert_on_index(got, frame)
    np.testing.assert_allclose(
        got, expected(frame, books, lambda o: ref_power(o, beta)), rtol=0, atol=1e-12
    )
    assert_races_sum_to_one(got, frame["race_id"], atol=1e-12)
    q = expected(frame, books, ref_normalised)
    for race_id, odds in books.items():
        rows = frame["race_id"] == race_id
        fav = frame.index[rows & (frame["pos"] == int(np.argmin(odds)))][0]
        longshot = frame.index[rows & (frame["pos"] == int(np.argmax(odds)))][0]
        if beta > 1:
            assert got[fav] > q[fav] and got[longshot] < q[longshot], race_id
        else:
            assert got[fav] < q[fav] and got[longshot] > q[longshot], race_id


# ---------------------------------------------------------------------------------------------
# De-vigs: Shin
# ---------------------------------------------------------------------------------------------


def test_shin_devig_sums_to_one():
    rng = np.random.default_rng(774)
    books = random_books(rng, 30, rounded=True)  # the pari-mutuel 25% overround, board-rounded
    books.update(
        {f"bookie{k}": v for k, v in random_books(rng, 15, take=0.7, sizes=(3, 16)).items()}
    )
    books["hand"] = np.array([1.6, 3.2, 3.2])
    frame = runners(books, rng)

    got = devig_shin(frame["odds"], frame["race_id"])

    assert_on_index(got, frame)
    assert got.notna().all()
    assert ((got > 0) & (got < 1)).all()
    assert_races_sum_to_one(got, frame["race_id"], atol=1e-8)
    for race_id in books:  # a shorter price never gets a smaller chance
        p = got[frame["race_id"] == race_id].to_numpy()
        o = frame.loc[frame["race_id"] == race_id, "odds"].to_numpy()
        order = np.argsort(o, kind="stable")
        assert np.all(np.diff(p[order]) <= 1e-12), race_id


def test_shin_devig_equals_normalised_without_overround():
    rng = np.random.default_rng(775)
    fair = {
        "fair3": np.array([2.0, 4.0, 4.0]),
        "fair4": np.array([4.0, 4.0, 4.0, 4.0]),
        "fair5": np.array([2.0, 4.0, 8.0, 16.0, 16.0]),
        "fair6": np.array([4.0, 4.0, 8.0, 8.0, 8.0, 8.0]),
        "fair9": np.array([2.0, 8.0, 8.0, 16.0, 16.0, 32.0, 32.0, 32.0, 32.0]),
    }
    for odds in fair.values():
        assert (1.0 / odds).sum() == 1.0  # no overround, exactly
    books = {**fair, "overround": np.array([1.6, 3.2, 3.2]), **random_books(rng, 5)}
    frame = runners(books, rng)

    got = devig_shin(frame["odds"], frame["race_id"])

    assert_on_index(got, frame)
    rows = frame["race_id"].isin(list(fair))
    np.testing.assert_allclose(
        got[rows], expected(frame[rows], fair, ref_normalised), rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(got[rows], 1.0 / frame.loc[rows, "odds"], rtol=0, atol=1e-9)


def test_shin_devig_moves_probability_toward_the_favourite():
    rng = np.random.default_rng(776)
    books = {"hand": np.array([1.6, 3.2, 3.2])}
    books.update(random_books(rng, 25, sizes=(3, 18)))  # unrounded: no tied prices
    books.update({f"bookie{k}": v for k, v in random_books(rng, 10, take=0.7).items()})
    frame = runners(books, rng)

    got = devig_shin(frame["odds"], frame["race_id"])
    q = expected(frame, books, ref_normalised)

    hand = frame["race_id"] == "hand"
    fav_hand = frame.index[hand & (frame["pos"] == 0)][0]
    assert got[fav_hand] > 0.5  # its normalised share
    assert (got[hand & (frame["pos"] > 0)] < 0.25).all()
    for race_id, odds in books.items():
        rows = frame["race_id"] == race_id
        if len(set(odds)) < len(odds):
            continue
        fav = frame.index[rows & (frame["pos"] == int(np.argmin(odds)))][0]
        longshot = frame.index[rows & (frame["pos"] == int(np.argmax(odds)))][0]
        assert got[fav] > q[fav], race_id
        assert got[longshot] < q[longshot], race_id
        # Shin over normalised rises with the runner's chance: the overround falls on the tail.
        ratio = (got[rows] / q[rows]).to_numpy()
        by_price = np.argsort(frame.loc[rows, "odds"].to_numpy())
        assert np.all(np.diff(ratio[by_price]) < 0), race_id


@pytest.mark.parametrize("z", [0.005, 0.02, 0.05, 0.08])
def test_shin_devig_recovers_the_chances_a_shin_book_was_priced_on(z):
    rng = np.random.default_rng([777, int(z * 1000)])
    planted = {f"s{n:02d}": chances(rng, n) for n in (2, 3, 5, 8, 12, 16, 18)}
    planted["flat"] = np.full(8, 1 / 8)
    planted["two"] = np.array([0.7, 0.3])
    # Races in the same call carry other insider shares: z is solved per race.
    planted["other_z"] = chances(rng, 10)
    books = {r: shin_book(p, z) for r, p in planted.items()}
    books["other_z"] = shin_book(planted["other_z"], 0.03)
    frame = runners(books, rng)

    got = devig_shin(frame["odds"], frame["race_id"])

    assert_on_index(got, frame)
    np.testing.assert_allclose(got, expected(frame, planted, lambda p: p), rtol=0, atol=SHIN_ATOL)
    # The planted chances differ from the normalised shares, so normalising cannot pass here.
    q = expected(frame, books, ref_normalised)
    assert np.abs(q - expected(frame, planted, lambda p: p)).max() > 10 * SHIN_ATOL


# ---------------------------------------------------------------------------------------------
# Every de-vig: unpriced runners, row order, index
# ---------------------------------------------------------------------------------------------


def test_unpriced_runners_are_dropped_before_devig():
    rng = np.random.default_rng(778)
    planted = {
        "u1": np.array([0.5, 0.25, 0.15, 0.1]),
        "u2": np.array([0.4, 0.35, 0.25]),
        "clean": np.array([0.3, 0.3, 0.2, 0.1, 0.1]),
    }
    priced_books = {
        "u1": shin_book(planted["u1"], 0.03),
        "u2": shin_book(planted["u2"], 0.06),
        "clean": shin_book(planted["clean"], 0.02),
    }
    # u1: a missing price; u2: a zero, an infinite and a sub-1.0 figure around its priced runners.
    o1, o2 = priced_books["u1"], priced_books["u2"]
    books = {
        "u1": np.array([o1[0], o1[1], NAN, o1[2], o1[3]]),
        "u2": np.array([0.0, o2[0], INF, o2[1], o2[2], 0.5]),
        "clean": priced_books["clean"],
    }
    shin_expected = {
        "u1": np.array([0.5, 0.25, NAN, 0.15, 0.1]),
        "u2": np.array([NAN, 0.4, NAN, 0.35, 0.25, NAN]),
        "clean": planted["clean"],
    }
    frame = runners(books, rng)
    unpriced = ~priced(frame["odds"].to_numpy())
    assert unpriced.sum() == 4

    cases = [
        ("normalised", devig_normalised, expected(frame, books, ref_normalised), 1e-12),
        (
            "power",
            lambda o, r: devig_power(o, r, 1.2),
            expected(frame, books, lambda o: ref_power(o, 1.2)),
            1e-12,
        ),
        ("shin", devig_shin, expected(frame, shin_expected, lambda p: p), SHIN_ATOL),
    ]
    for name, devig, want, atol in cases:
        got = devig(frame["odds"], frame["race_id"])
        assert_on_index(got, frame)
        assert got[unpriced].isna().all(), name
        assert got[~unpriced].notna().all(), name
        np.testing.assert_allclose(got[~unpriced], want[~unpriced], rtol=0, atol=atol)
        assert_races_sum_to_one(got[~unpriced], frame.loc[~unpriced, "race_id"], atol=1e-8)


DEVIGS = {
    "normalised": devig_normalised,
    "power": lambda odds, race_id: devig_power(odds, race_id, 1.3),
    "shin": devig_shin,
}


@pytest.mark.parametrize("name", list(DEVIGS))
def test_devigs_keep_the_input_index_and_ignore_row_order(name):
    devig = DEVIGS[name]
    rng = np.random.default_rng(779)
    books = random_books(rng, 30, rounded=True)
    books["u"] = np.array([2.4, NAN, 3.3, 6.1, 15.0])
    frame = runners(books, rng)
    grouped = frame.sort_values(["race_id", "pos"])
    reshuffled = relabel(frame, rng)

    first = devig(frame["odds"], frame["race_id"])
    second = devig(grouped["odds"], grouped["race_id"])

    assert_on_index(first, frame)
    assert_on_index(second, grouped)
    np.testing.assert_allclose(second.reindex(frame.index), first, rtol=0, atol=1e-12)
    # Same rows under fresh labels and order: each runner keeps its value.
    third = devig(reshuffled["odds"], reshuffled["race_id"])
    assert_on_index(third, reshuffled)
    key = ["race_id", "pos"]
    a = frame.assign(v=first).set_index(key)["v"].sort_index()
    b = reshuffled.assign(v=third).set_index(key)["v"].sort_index()
    np.testing.assert_allclose(a.to_numpy(), b.reindex(a.index).to_numpy(), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# The beta fit: a conditional logit on ln(1/odds)
# ---------------------------------------------------------------------------------------------


def test_power_devig_recovers_a_planted_beta():
    frame, rng = planted_frame(106, 4_000, 12, 1.06)
    frame = relabel(frame, rng)

    fit = fit_power_beta(frame["odds"], frame["race_id"], frame["won"])

    assert isinstance(fit, BetaFit)
    assert fit.n_races == 4_000
    assert 0 < fit.se < 0.03
    assert abs(fit.beta - 1.06) < 3 * fit.se
    assert_is_the_mle(fit.beta, fit.se, frame)


@pytest.mark.parametrize("beta", [0.85, 1.0, 1.25])
def test_beta_fit_recovers_each_planted_beta_and_covers_one_on_a_calibrated_market(beta):
    frame, rng = planted_frame([77, int(round(beta * 100))], 3_000, 16, beta, n_min=8)
    frame = relabel(frame, rng)

    fit = fit_power_beta(frame["odds"], frame["race_id"], frame["won"])

    assert fit.n_races == 3_000
    assert 0 < fit.se < 0.04
    if beta == 1.0:
        assert abs(fit.beta - 1.0) <= 1.96 * fit.se  # the 95% CI covers 1
    else:
        assert abs(fit.beta - beta) < 3 * fit.se
        assert abs(fit.beta - 1.0) > 3 * fit.se  # and the identity is ruled out


def test_beta_fit_is_the_mle_over_races_with_a_priced_winner():
    frame, rng = planted_frame(780, 800, 14, 1.15, n_min=5)
    # Unpriced non-winners inside counted races: they leave the race's choice set.
    extra = frame.drop_duplicates("race_id").iloc[::7][["race_id"]]
    frame = pd.concat(
        [frame, extra.assign(odds=NAN, won=False), extra.iloc[::2].assign(odds=0.0, won=False)],
        ignore_index=True,
    )
    frame = relabel(add_races_without_a_priced_winner(frame, prefix="junk"), rng)

    fit = fit_power_beta(frame["odds"], frame["race_id"], frame["won"])

    assert fit.n_races == 800
    assert math.isfinite(fit.beta) and math.isfinite(fit.se)
    assert_is_the_mle(fit.beta, fit.se, frame)


# ---------------------------------------------------------------------------------------------
# Segment beta (E4) on one frame
# ---------------------------------------------------------------------------------------------


def segmented(parts, segment, *, seed, n_races, n_max, n_min):
    """One frame of several segments, each its own planted beta, rows shuffled together."""
    frames = []
    for value, beta in parts.items():
        frame, _ = planted_frame(
            [seed, value] if isinstance(value, int) else [seed, ord(value[0])],
            n_races,
            n_max,
            beta,
            n_min=n_min,
            prefix=f"{value}-",
        )
        frames.append(frame.assign(**{segment: value}))
    return pd.concat(frames, ignore_index=True)


def by_segment(table, segment):
    assert isinstance(table, pd.DataFrame)
    assert {segment, "beta", "se", "n_races"} <= set(table.columns)
    assert table[segment].is_unique
    return table.set_index(segment)


def test_segment_beta_is_one_on_a_calibrated_synthetic_market():
    years = (2015, 2016, 2017, 2018)
    frame = segmented(dict.fromkeys(years, 1.0), "year", seed=7, n_races=1_500, n_max=18, n_min=6)
    frame = relabel(frame, np.random.default_rng(781))

    table = by_segment(segment_beta(frame, "year"), "year")

    assert set(table.index) == set(years)
    for year in years:
        row = table.loc[year]
        assert row["n_races"] == 1_500, year
        assert 0 < row["se"] < 0.05, year  # an informative interval, not a wide one
        assert abs(row["beta"] - 1.0) <= 1.96 * row["se"], year  # the 95% CI covers 1


def test_segment_beta_fits_each_segment_on_its_own_races():
    planted = {"TOK": 0.85, "KYO": 1.0, "NAK": 1.3}
    frame = segmented(planted, "venue", seed=6, n_races=2_000, n_max=16, n_min=8)
    frame = add_races_without_a_priced_winner(frame, prefix="KYO-", extra={"venue": "KYO"})
    frame = add_races_without_a_priced_winner(frame, prefix="NAK-", extra={"venue": "NAK"})
    frame = frame.assign(horse_no=np.arange(len(frame)) % 18 + 1)  # extra columns are ignored
    frame = relabel(frame, np.random.default_rng(782))

    table = by_segment(segment_beta(frame, "venue"), "venue")

    assert set(table.index) == set(planted)
    for venue, beta in planted.items():
        row = table.loc[venue]
        assert row["n_races"] == 2_000, venue
        assert abs(row["beta"] - beta) < 3 * row["se"], venue
        assert_is_the_mle(row["beta"], row["se"], frame[frame["venue"] == venue])
    # The segments differ: one pooled beta copied to every row cannot pass.
    assert table.loc["NAK", "beta"] - table.loc["TOK", "beta"] > 0.3


# ---------------------------------------------------------------------------------------------
# Market entropy (E3)
# ---------------------------------------------------------------------------------------------


def test_entropy_is_one_for_a_flat_book_and_lower_for_a_concentrated_one():
    rng = np.random.default_rng(783)
    books = {}
    for n in (2, 3, 5, 8, 12, 16, 18):
        for take in (0.7, 0.8, 1.0):
            books[f"flat{n}-{take}"] = np.full(n, n / take)
    steps = (0.25, 0.5, 1.0, 1.5)  # each book more concentrated than the last
    for c in steps:
        q = np.exp(c * np.arange(6.0))
        books[f"conc{c}"] = TAKE / (q / q.sum())
    books["hand"] = np.array([1.6, 3.2, 3.2])
    frame = runners(books, rng)

    got = normalised_entropy(frame["odds"], frame["race_id"])

    assert isinstance(got, pd.Series)
    assert len(got) == len(books) and set(got.index) == set(books)
    for race_id in books:
        if race_id.startswith("flat"):
            assert got[race_id] == pytest.approx(1.0, abs=1e-12), race_id
    assert got["hand"] == pytest.approx(1.5 * math.log(2) / math.log(3), abs=1e-12)
    conc = np.array([got[f"conc{c}"] for c in steps])
    np.testing.assert_allclose(
        conc, [ref_entropy(books[f"conc{c}"]) for c in steps], rtol=0, atol=1e-12
    )
    assert conc[0] < 1.0 and np.all(np.diff(conc) < 0)


def test_entropy_is_the_normalised_shannon_entropy_of_each_race():
    rng = np.random.default_rng(784)
    books = random_books(rng, 60, rounded=True)
    books["with_unpriced"] = np.array([5.0, 5.0, NAN, 5.0, 0.0])  # flat over its 3 priced runners
    books["wide"] = np.array([1.1, 40.0, 95.0, 180.0, 250.0, 600.0])
    frame = runners(books, rng)

    got = normalised_entropy(frame["odds"], frame["race_id"])

    assert len(got) == len(books) and set(got.index) == set(books)
    assert ((got >= 0) & (got <= 1)).all()
    for race_id, odds in books.items():
        assert got[race_id] == pytest.approx(ref_entropy(odds), abs=1e-12), race_id
    assert got["with_unpriced"] == pytest.approx(1.0, abs=1e-12)
    assert got["wide"] < 0.5


# ---------------------------------------------------------------------------------------------
# Floor-bound runners
# ---------------------------------------------------------------------------------------------


def test_floor_bound_runners_are_flagged():
    cases = {  # floor -> [(odds, flagged)]
        1.1: [
            (1.1, True),  # at the floor
            (0.1 * 11, True),  # 1.1000000000000001: at the floor up to float error
            (1.1 + 5e-10, True),  # inside the 1e-9 tolerance
            (1.0, True),  # under the floor
            (1.2, False),
            (1.1001, False),
            (1.4, False),
            (35.0, False),
            (NAN, False),  # unpriced runners are never floor-bound
            (INF, False),
            (0.5, False),
        ],
        1.0: [(1.0, True), (1.0 + 5e-10, True), (1.1, False), (2.0, False), (0.99, False)],
        1.5: [(1.5, True), (1.4, True), (1.0, True), (1.5 + 1e-12, True), (1.6, False)],
    }
    rng = np.random.default_rng(785)
    for floor, rows in cases.items():
        odds = pd.Series([o for o, _ in rows], dtype=float)
        odds.index = pd.Index(13 * rng.permutation(len(odds)) + 5)
        want = [flag for _, flag in rows]

        got = flag_floor_bound(odds, floor=floor)

        assert isinstance(got, pd.Series)
        assert got.index.equals(odds.index)
        assert got.dtype == bool
        assert got.tolist() == want, floor
