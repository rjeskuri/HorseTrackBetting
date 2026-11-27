"""The C3 race-level finish-speed model (ticket #96, epic #73): par, day variant, E[top-3 time].

Contract pinned here (`horsetrack.countries.japan.models.speed`):
- `SpeedSpec(par_keys, variant_keys, same_day_variant, t_df)`; t_df None is a normal error.
- `SpeedModel(spec, *, variant_prior=None).fit(races) -> SpeedModel` (the returned object is the
  one used). races columns: race_id, race_utc, track, surface, distance, layout, rail,
  class_level, top3_time_s, result_available_utc, both times tz-aware UTC. The model is
  top3_time = par(cell) + v(day, track, surface) + eps, eps Student-t with t_df degrees of freedom
  (normal when None), v a random effect. A race's day is its JST calendar date; every synthetic
  race here is run between 01:00 and 04:10 UTC, so its JST and UTC dates agree.
- `fit` estimates the pars, the error scale `sigma` and the variant spread `tau` (attributes, s),
  and nothing else: the races it is fitted on never enter `variant`.
- `observe(results) -> SpeedModel` returns the model itself. results: races on the days to be
  predicted, with the columns of fit's frame. Each race's residual is taken against the fitted
  par; par, sigma and tau do not change. Before fit it raises RuntimeError; for a race on a
  (JST day, track, surface) the fit frame had, ValueError (its residual would carry its own result
  back through the par). Another track or surface on that day is accepted.
- `par(**keys)`: keyword arguments named by spec.par_keys; a cell is the tuple of their values.
  'Course' is layout (inner, outer, straight) plus rail (A-D, or 'unknown').
- `variant(race_date, track, surface, *, at_utc) -> (mean, sd)`, race_date a `datetime.date`: the
  posterior from the prior and the observed residuals of that day, track and surface whose
  result_available_utc is strictly before at_utc (a null time never is). The prior is N(0, tau^2),
  or the injected prior's (mean, sd). With same_day_variant=False the answer is the prior whatever
  was observed, as it is when nothing is available. More data moves the mean from the prior
  toward the residuals and narrows the SD.
- `variant_prior` call shape (chosen here, the ticket leaves it open):
  `variant_prior(race_date, track, surface) -> (mean, sd)`, race_date a `datetime.date` (not a
  datetime or Timestamp), track and surface strings as in the races frame, sd > 0. It may be called
  for any (date, track, surface).
- `expected_top3_time(race, *, at_utc)`: race a mapping with race_id, race_utc and the par keys (no
  result columns); it returns par(cell) + variant(day, track, surface, at_utc=at_utc)[0].
- The fit is deterministic: the same frame gives the same pars and variants.

The frame: 16 training days (the fit frame) and 8 test days after them, whose results are only
observed. Not pinned: the estimator, the shrinkage formula, whether a second `observe` adds to or
replaces the first, what an unseen cell or track returns. Expected values come from the planted
synthetic races below (pars, day variants, noise), never from the code under test. The model
reads no store: every frame is built here.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec

PAR_KEYS = ("track", "surface", "distance", "layout", "rail", "class_level")
NO_RAIL = ("track", "surface", "distance", "layout", "class_level")
VARIANT_KEYS = ("track", "surface")
COLUMNS = (
    "race_id",
    "race_utc",
    *PAR_KEYS,
    "top3_time_s",
    "result_available_utc",
)

D0 = date(2025, 1, 4)
N_TRAIN = 16
N_TEST = 8
RAIL_C_FROM = 8  # Tokyo turf runs on rail B on training days 0-7 and on rail C on days 8-15
DAYS = tuple(D0 + timedelta(days=k) for k in range(N_TRAIN))  # the training days: the fit frame
TEST_DAYS = tuple(D0 + timedelta(days=N_TRAIN + j) for j in range(N_TEST))  # observed, not fitted
ALL_DAYS = DAYS + TEST_DAYS
T0 = TEST_DAYS[0]  # every race of it is exactly par + variant
LAST = TEST_DAYS[-1]  # the sweep day
NEXT = D0 + timedelta(days=N_TRAIN + N_TEST)  # a card day: none of its races is in the frame
SIGMA = 0.3  # race-level noise, s
TOK_TURF = ("TOK", "turf")

# ---------------------------------------------------------------------------------------------
# The planted model
# ---------------------------------------------------------------------------------------------

# cell (track, surface, distance, layout, rail, class_level) -> planted par, s
PAR = {
    ("TOK", "turf", 1600, "outer", "B", 3): 96.0,
    ("TOK", "turf", 2000, "outer", "B", 3): 120.5,
    ("TOK", "turf", 1600, "outer", "B", 5): 95.3,
    ("TOK", "turf", 1600, "outer", "C", 3): 96.8,
    ("TOK", "turf", 2000, "outer", "C", 3): 121.1,
    ("TOK", "turf", 1600, "outer", "C", 5): 95.9,
    ("TOK", "dirt", 1600, "outer", "unknown", 3): 98.4,
    ("KYO", "turf", 1600, "outer", "B", 3): 95.4,
    ("KYO", "turf", 1600, "inner", "B", 3): 96.1,
    ("NII", "turf", 1600, "outer", "unknown", 3): 95.8,
    ("NII", "turf", 1000, "straight", "unknown", 3): 55.2,
    ("NII", "dirt", 1600, "outer", "unknown", 3): 97.6,
}
TOK_B = ("TOK", "turf", 1600, "outer", "B", 3)

# (the one key that differs, cell a, cell b)
ONE_KEY_PAIRS = (
    ("rail", TOK_B, ("TOK", "turf", 1600, "outer", "C", 3)),
    ("rail", ("TOK", "turf", 2000, "outer", "B", 3), ("TOK", "turf", 2000, "outer", "C", 3)),
    ("rail", ("TOK", "turf", 1600, "outer", "B", 5), ("TOK", "turf", 1600, "outer", "C", 5)),
    ("track", TOK_B, ("KYO", "turf", 1600, "outer", "B", 3)),
    (
        "track",
        ("TOK", "dirt", 1600, "outer", "unknown", 3),
        ("NII", "dirt", 1600, "outer", "unknown", 3),
    ),
    (
        "surface",
        ("NII", "turf", 1600, "outer", "unknown", 3),
        ("NII", "dirt", 1600, "outer", "unknown", 3),
    ),
    ("distance", TOK_B, ("TOK", "turf", 2000, "outer", "B", 3)),
    ("layout", ("KYO", "turf", 1600, "outer", "B", 3), ("KYO", "turf", 1600, "inner", "B", 3)),
    ("class_level", TOK_B, ("TOK", "turf", 1600, "outer", "B", 5)),
)


def rail_on(day):
    """Tokyo turf's rail: B on training days 0-7, C on 8-15; test days alternate B, C from T0."""
    k = (day - D0).days
    if k < N_TRAIN:
        return "B" if k < RAIL_C_FROM else "C"
    return "B" if (k - N_TRAIN) % 2 == 0 else "C"


def tok_turf_cells(day):
    rail = rail_on(day)
    a = ("TOK", "turf", 1600, "outer", rail, 3)
    b = ("TOK", "turf", 2000, "outer", rail, 3)
    c = ("TOK", "turf", 1600, "outer", rail, 5)
    return (a, b, c, c, b, a)  # R1..R6: R1/R6, R2/R5 and R3/R4 share a cell


def same_cell(cell, n):
    return lambda day: (cell,) * n


# (track, surface) -> (post times UTC as (hh, mm), cells of the day's races in post order)
GROUPS = {
    TOK_TURF: (((1, 5), (1, 40), (2, 15), (2, 50), (3, 25), (4, 0)), tok_turf_cells),
    ("TOK", "dirt"): (
        ((1, 20), (3, 40)),
        same_cell(("TOK", "dirt", 1600, "outer", "unknown", 3), 2),
    ),
    ("KYO", "turf"): (
        ((1, 10), (1, 45), (2, 20), (2, 55)),
        lambda day: (
            ("KYO", "turf", 1600, "outer", "B", 3),
            ("KYO", "turf", 1600, "inner", "B", 3),
            ("KYO", "turf", 1600, "inner", "B", 3),
            ("KYO", "turf", 1600, "outer", "B", 3),
        ),
    ),
    ("NII", "turf"): (
        ((1, 15), (1, 50), (2, 25)),
        lambda day: (
            ("NII", "turf", 1600, "outer", "unknown", 3),
            ("NII", "turf", 1000, "straight", "unknown", 3),
            ("NII", "turf", 1600, "outer", "unknown", 3),
        ),
    ),
    ("NII", "dirt"): (
        ((1, 30), (3, 0)),
        same_cell(("NII", "dirt", 1600, "outer", "unknown", 3), 2),
    ),
}
RESULT_DELAY = timedelta(minutes=10)
INQUIRY_DELAY = timedelta(minutes=50)  # the last day's Tokyo turf R3: published 03:05, after R4

# Day variants, s (positive = slow). On the training days Tokyo turf's sum to zero within each rail
# block and the others' over the 16 days, so a cell's mean training time is its par.
V_TOK_TURF = (
    *(1.0, -0.4, 0.3, -0.7, 0.5, -0.2, -0.6, 0.1),  # rail B days
    *(-0.5, 0.6, -0.3, 0.2, 0.9, -0.8, 0.4, -0.5),  # rail C days
)
V_TEST_TOK_TURF = (1.0, -0.6, 0.4, -0.3, 0.8, -0.9, 0.2, -0.5)  # T0 .. LAST
V_T0 = {("TOK", "dirt"): -0.8, ("KYO", "turf"): -0.6, ("NII", "turf"): 0.5, ("NII", "dirt"): 0.7}

# The last day's Tokyo turf noise R1..R6, set by hand so each result moves the variant clearly.
LAST_TOK_TURF_EPS = (0.6, -0.2, 0.9, -0.5, 0.3, -0.6)


def planted_variants():
    """(track, surface) -> one variant per day of ALL_DAYS: training days, then test days."""
    rng = np.random.default_rng(9601)
    v = {TOK_TURF: np.r_[V_TOK_TURF, V_TEST_TOK_TURF]}
    for group, v0 in V_T0.items():
        train = rng.normal(0.0, 0.5, N_TRAIN)
        train -= train.mean()
        v[group] = np.r_[train, v0, rng.normal(0.0, 0.5, N_TEST - 1)]
    return v


VARIANTS = planted_variants()


def at(day, hh, mm):
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=timezone.utc)


@lru_cache(maxsize=1)
def _base_races():
    """17 races a day. Training races have N(0, 0.3) noise, centred within each cell so a cell's
    training noise sums to 0. Test races have N(0, 0.3) noise, except T0's (none) and the last
    day's Tokyo turf (LAST_TOK_TURF_EPS)."""
    rng = np.random.default_rng(9602)
    rows = []
    for k, day in enumerate(ALL_DAYS):
        for (track, surface), (posts, cells) in GROUPS.items():
            for i, ((hh, mm), cell) in enumerate(zip(posts, cells(day), strict=True), start=1):
                post = at(day, hh, mm)
                last_tok_turf = day == LAST and (track, surface) == TOK_TURF
                delay = INQUIRY_DELAY if last_tok_turf and i == 3 else RESULT_DELAY
                eps = rng.normal(0.0, SIGMA)
                if day == T0:
                    eps = 0.0
                elif last_tok_turf:
                    eps = LAST_TOK_TURF_EPS[i - 1]
                rows.append(
                    {
                        "race_id": f"syn-{day:%Y%m%d}-{track}-{surface}-{i:02d}",
                        "race_utc": post,
                        **dict(zip(PAR_KEYS, cell, strict=True)),
                        "base": PAR[cell] + VARIANTS[(track, surface)][k],
                        "eps": eps,
                        "train": k < N_TRAIN,
                        "result_available_utc": post + delay,
                    }
                )
    frame = pd.DataFrame(rows)
    for _, idx in frame[frame["train"]].groupby(list(PAR_KEYS)).groups.items():
        frame.loc[idx, "eps"] -= frame.loc[idx, "eps"].mean()
    frame["top3_time_s"] = frame["base"] + frame["eps"]
    frame["race_utc"] = pd.to_datetime(frame["race_utc"], utc=True)
    frame["result_available_utc"] = pd.to_datetime(frame["result_available_utc"], utc=True)
    return frame[list(COLUMNS)]


def jst_day(races):
    return races["race_utc"].dt.tz_convert("Asia/Tokyo").dt.date


def training(races):
    return races[jst_day(races) < T0]


def observed(races):
    return races[jst_day(races) >= T0]


def on_day(races, day):
    return races[jst_day(races) == day].sort_values("race_utc")


def tok_turf(races, day):
    """The day's six Tokyo turf races, R1..R6 in post order, with the frame's index."""
    d = on_day(races, day)
    return d[(d["track"] == "TOK") & (d["surface"] == "turf")]


SLOW = 1.0  # s added to a single race's time


def races_frame(name="base"):
    races = _base_races().copy()
    last = tok_turf(races, LAST).index
    if name == "outlier":  # one Tokyo turf race 10 s slow (training day 3, R1: cell TOK_B)
        races.loc[tok_turf(races, DAYS[3]).index[0], "top3_time_s"] += 10.0
    elif name in ("slow_r2", "slow_r3", "slow_r5", "slow_r6"):  # one race of the last day's TT
        races.loc[last[int(name[-1]) - 1], "top3_time_s"] += SLOW
    elif name == "r2_unpublished":  # the last day's Tokyo turf R2: its result never published
        races.loc[last[1], "result_available_utc"] = pd.NaT
    elif name == "r2_dropped":
        races = races.drop(index=last[1])
    elif name == "test_days_slow":  # every test-day race 0.5 s slower, the training days as is
        races.loc[observed(races).index, "top3_time_s"] += 0.5
    else:
        assert name == "base"
    return races


def keys(cell, names=PAR_KEYS):
    full = dict(zip(PAR_KEYS, cell, strict=True))
    return {k: full[k] for k in names}


def cell_of(row):
    return (
        str(row["track"]),
        str(row["surface"]),
        int(row["distance"]),
        str(row["layout"]),
        str(row["rail"]),
        int(row["class_level"]),
    )


def card(row):
    """The race as a card shows it before the off: no result columns."""
    return {
        "race_id": str(row["race_id"]),
        "race_utc": row["race_utc"].to_pydatetime(),
        **keys(cell_of(row)),
    }


def table_prior(table, default=(0.0, 0.5)):
    def variant_prior(race_date, track, surface):
        if type(race_date) is not date:
            raise TypeError(f"race_date must be a datetime.date, got {type(race_date).__name__}")
        return table.get((race_date, str(track), str(surface)), default)

    return variant_prior


PRIORS = {
    "table": {
        (T0, "TOK", "turf"): (-0.5, 0.3),
        (T0, "TOK", "dirt"): (0.7, 0.25),
        (T0, "KYO", "turf"): (0.2, 0.6),
        (T0, "NII", "dirt"): (-0.3, 0.4),
    },
    "plus_0_4": {(T0, "TOK", "turf"): (0.4, 0.2)},
    "tight": {(T0, "TOK", "turf"): (-0.5, 0.05)},
    "loose": {(T0, "TOK", "turf"): (-0.5, 3.0)},
}


def make_spec(*, t_df=5.0, same_day=True, par_keys=PAR_KEYS):
    return SpeedSpec(
        par_keys=par_keys, variant_keys=VARIANT_KEYS, same_day_variant=same_day, t_df=t_df
    )


@pytest.fixture(scope="module")
def fit():
    """Models fitted on the training days, each once per module, and shown the test days' results
    (observe=True): fit(frame, t_df=, same_day=, par_keys=, prior=, observe=)."""
    cache = {}

    def _fit(frame="base", *, t_df=5.0, same_day=True, par_keys=PAR_KEYS, prior=None, observe=True):
        key = (frame, t_df, same_day, par_keys, prior, observe)
        if key not in cache:
            vp = None if prior is None else table_prior(PRIORS[prior])
            spec = make_spec(t_df=t_df, same_day=same_day, par_keys=par_keys)
            races = races_frame(frame)
            model = SpeedModel(spec, variant_prior=vp).fit(training(races))
            cache[key] = model.observe(observed(races)) if observe else model
        return cache[key]

    return _fit


def tt_variant(model, day, t):
    return model.variant(day, "TOK", "turf", at_utc=t)


def rmse(errors):
    return math.sqrt(sum(e * e for e in errors) / len(errors))


# The last day's Tokyo turf, R1..R6: posts 01:05 01:40 02:15 02:50 03:25 04:00, results 10 min
# later except R3's, held until 03:05 (after R4's at 03:00).
# (at_utc as (hh, mm), the races whose result is available strictly before it)
SWEEP = (
    ((0, 30), ()),
    ((1, 15), ()),  # R1 is published at exactly 01:15: not before t
    ((1, 16), (1,)),
    ((1, 50), (1,)),  # R2 at exactly 01:50
    ((1, 55), (1, 2)),
    ((2, 30), (1, 2)),  # R3 has run (02:15) but its result is held
    ((3, 0), (1, 2)),  # R4 at exactly 03:00
    ((3, 1), (1, 2, 4)),
    ((3, 4), (1, 2, 4)),
    ((3, 6), (1, 2, 3, 4)),
    ((3, 30), (1, 2, 3, 4)),  # R5 has run (03:25), result at 03:35
    ((3, 36), (1, 2, 3, 4, 5)),
    ((4, 9), (1, 2, 3, 4, 5)),
    ((4, 11), (1, 2, 3, 4, 5, 6)),
    ((23, 0), (1, 2, 3, 4, 5, 6)),
)

# One race of the last day's Tokyo turf made SLOW s slower: (race, times (hh, mm) at which its
# result is not yet in, the first time after it is). R3 runs at 02:15 but is published at 03:05;
# R5 is the target of the E[R5] checks, decided at 02:30 and at 03:24, a minute before its off.
NOT_YET = (
    (2, ((0, 30), (1, 50)), (1, 55)),
    (3, ((2, 30), (3, 4)), (3, 6)),
    (5, ((2, 30), (3, 24), (3, 35)), (3, 36)),
    (6, ((2, 30), (4, 9)), (4, 11)),
)


# ---------------------------------------------------------------------------------------------
# Broad behaviour
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t_df", [None, 5.0], ids=["normal", "student_t5"])
def test_par_recovers_every_planted_cell(fit, t_df):
    model = fit(t_df=t_df, observe=False)
    for cell, planted in PAR.items():
        assert model.par(**keys(cell)) == pytest.approx(planted, abs=0.1), cell


def test_par_keys_of_the_spec_decide_the_cells(fit):
    # Without rail in par_keys, rails B and C share one par: the mean of the two planted pars (each
    # rail runs the same number of training races).
    pooled = fit(t_df=None, par_keys=NO_RAIL, observe=False)
    for distance, class_level in ((1600, 3), (2000, 3), (1600, 5)):
        b = ("TOK", "turf", distance, "outer", "B", class_level)
        c = ("TOK", "turf", distance, "outer", "C", class_level)
        got = pooled.par(**keys(b, NO_RAIL))
        assert got == pytest.approx((PAR[b] + PAR[c]) / 2, abs=0.1), (distance, class_level)
    kyo = ("KYO", "turf", 1600, "inner", "B", 3)
    assert pooled.par(**keys(kyo, NO_RAIL)) == pytest.approx(PAR[kyo], abs=0.1)


def test_variant_follows_each_days_planted_going_by_track_and_surface(fit):
    model = fit()
    # T0, every result in: each (track, surface) gets its own day's variant, sign and size.
    for (track, surface), planted in VARIANTS.items():
        mean, sd = model.variant(T0, track, surface, at_utc=at(T0, 23, 0))
        v0 = planted[N_TRAIN]
        assert np.sign(mean) == np.sign(v0), (track, surface, mean, v0)
        assert 0.5 * abs(v0) <= abs(mean) <= abs(v0) + 0.1, (track, surface, mean, v0)
        assert sd > 0
    # Tokyo turf over the 8 test days, each at the end of its day.
    means = [tt_variant(model, d, at(d, 23, 0))[0] for d in TEST_DAYS]
    assert np.corrcoef(means, V_TEST_TOK_TURF)[0, 1] > 0.9
    assert max(abs(m - v) for m, v in zip(means, V_TEST_TOK_TURF, strict=True)) < 0.45


def test_expected_top3_time_adds_the_variant_at_t_to_the_par(fit):
    races = races_frame()
    model = fit()
    for day in (T0, TEST_DAYS[3], LAST):
        for _, row in on_day(races, day).iterrows():
            race = card(row)
            post = row["race_utc"].to_pydatetime()
            for t in (at(day, 0, 30), post - timedelta(minutes=1), at(day, 23, 0)):
                mean, _ = model.variant(day, race["track"], race["surface"], at_utc=t)
                want = model.par(**keys(cell_of(row))) + mean
                got = model.expected_top3_time(race, at_utc=t)
                assert got == pytest.approx(want, abs=1e-9), (race["race_id"], t)
    # A card day none of whose races is in either frame: the prior, so E is the par.
    for i, cell in enumerate(
        (
            ("TOK", "turf", 1600, "outer", "C", 3),
            ("TOK", "dirt", 1600, "outer", "unknown", 3),
            ("KYO", "turf", 1600, "inner", "B", 3),
            ("NII", "turf", 1000, "straight", "unknown", 3),
        ),
        start=1,
    ):
        race = {"race_id": f"syn-next-{i}", "race_utc": at(NEXT, 2, 0), **keys(cell)}
        t = at(NEXT, 1, 0)
        mean, sd = model.variant(NEXT, cell[0], cell[1], at_utc=t)
        assert mean == pytest.approx(0.0, abs=1e-12)
        assert sd == pytest.approx(model.tau, rel=1e-12)
        got = model.expected_top3_time(race, at_utc=t)
        assert got == pytest.approx(model.par(**keys(cell)), abs=1e-9)
    # The day's earlier results earn their place out of sample: Tokyo turf R4-R6 of every test
    # day, each a minute before its off, against the par alone.
    err_e, err_par = [], []
    for day in TEST_DAYS:
        for _, row in tok_turf(races, day).iloc[3:].iterrows():
            t = row["race_utc"].to_pydatetime() - timedelta(minutes=1)
            err_e.append(model.expected_top3_time(card(row), at_utc=t) - row["top3_time_s"])
            err_par.append(model.par(**keys(cell_of(row))) - row["top3_time_s"])
    assert rmse(err_e) < 0.8 * rmse(err_par)


def test_without_a_same_day_variant_the_days_own_races_never_count(fit):
    off, on = fit(same_day=False), fit()
    # Off: the prior all day long on every test day, whatever was observed.
    off_slow = fit("test_days_slow", same_day=False)
    for day in TEST_DAYS:
        for (hh, mm), _ in SWEEP:
            t = at(day, hh, mm)
            assert tt_variant(off, day, t) == pytest.approx((0.0, off.tau), abs=1e-12), (day, t)
            assert tt_variant(off_slow, day, t) == pytest.approx((0.0, off_slow.tau), abs=1e-12)
    late = tok_turf(races_frame(), LAST).iloc[5]
    t = late["race_utc"].to_pydatetime() - timedelta(minutes=1)
    assert off.expected_top3_time(card(late), at_utc=t) == pytest.approx(
        off.par(**keys(cell_of(late))), abs=1e-9
    )
    # On, out of sample (T0 is not in the fit frame, its Tokyo turf ran 1.0 s slow): the prior
    # until the day's first result is in (R1, published 01:15), then off it toward the results.
    before_on, before_off = tt_variant(on, T0, at(T0, 1, 15)), tt_variant(off, T0, at(T0, 1, 15))
    assert before_on == pytest.approx((0.0, on.tau), abs=1e-12)
    assert before_off == pytest.approx((0.0, off.tau), abs=1e-12)
    for hh, mm in ((1, 16), (2, 30), (23, 0)):
        mean_on, sd_on = tt_variant(on, T0, at(T0, hh, mm))
        assert mean_on > 0.3, (hh, mm)
        assert sd_on < on.tau, (hh, mm)
    # The observed times are what moves it: every test day 0.5 s slower moves on, never off.
    on_slow = fit("test_days_slow")
    for day in TEST_DAYS:
        t = at(day, 23, 0)
        assert tt_variant(on_slow, day, t)[0] > tt_variant(on, day, t)[0] + 0.2, day


def test_observe_takes_results_against_the_frozen_fit():
    races, slow, spec = races_frame(), races_frame("test_days_slow"), make_spec()
    model = SpeedModel(spec).fit(training(races))
    pars = {cell: model.par(**keys(cell)) for cell in PAR}
    sigma, tau = model.sigma, model.tau
    assert sigma > 0 and tau > 0
    # The fit frame never feeds the variant: before any observation every day is at the prior,
    # the training days included.
    for day in (DAYS[0], DAYS[9], T0, LAST):
        got = tt_variant(model, day, at(day, 23, 0))
        assert got == pytest.approx((0.0, tau), abs=1e-12), day
    with pytest.raises(RuntimeError):
        SpeedModel(spec).observe(observed(races))
    assert model.observe(observed(slow)) is model
    # Observed results leave the fit as it was, however slow they are...
    assert {cell: model.par(**keys(cell)) for cell in PAR} == pars
    assert (model.sigma, model.tau) == (sigma, tau)
    # ...and count only for their own days: T0 ran 1.0 s + 0.5 s slow, the training days stay put.
    assert tt_variant(model, T0, at(T0, 23, 0))[0] > 1.0
    for day in (DAYS[0], DAYS[9]):
        assert tt_variant(model, day, at(day, 23, 0)) == pytest.approx((0.0, tau), abs=1e-12)
    # A training day's race among the results is refused.
    mixed = pd.concat([observed(races), tok_turf(races, DAYS[3]).iloc[[1]]])
    with pytest.raises(ValueError):
        SpeedModel(spec).fit(training(races)).observe(mixed)
    # A model whose fit frame had the last day's Tokyo turf: that (day, track, surface) stays at
    # the prior, a result on it is refused, and the day's other groups are accepted.
    last_tt = tok_turf(races, LAST)
    in_fit = pd.concat([training(races), last_tt])
    rest = observed(races).drop(index=last_tt.index)
    accepting = SpeedModel(spec).fit(in_fit).observe(rest)
    t = at(LAST, 23, 0)
    assert tt_variant(accepting, LAST, t) == pytest.approx((0.0, accepting.tau), abs=1e-12)
    dirt_mean, _ = accepting.variant(LAST, "TOK", "dirt", at_utc=t)
    assert abs(dirt_mean) > 1e-3
    with pytest.raises(ValueError):
        SpeedModel(spec).fit(in_fit).observe(last_tt)


# ---------------------------------------------------------------------------------------------
# The ticket's named tests
# ---------------------------------------------------------------------------------------------


def test_par_is_keyed_by_track_distance_course_surface_class(fit):
    for t_df in (None, 5.0):
        model = fit(t_df=t_df, observe=False)
        for key, a, b in ONE_KEY_PAIRS:
            differs = [k for k, x, y in zip(PAR_KEYS, a, b, strict=True) if x != y]
            assert differs == [key]
            got = model.par(**keys(b)) - model.par(**keys(a))
            assert got == pytest.approx(PAR[b] - PAR[a], abs=0.15), (t_df, key, a, b)
            assert abs(got) > 0.3, (t_df, key, a, b)


def test_expected_top3_time_is_par_plus_variant(fit):
    # Par 96.0 s (planted, cell TOK_B, Tokyo turf R1 of T0) and a variant of +0.4 s: the injected
    # prior's mean, since at 00:30 on T0 no result of any race is in.
    model = fit(prior="plus_0_4")
    t = at(T0, 0, 30)
    race = card(tok_turf(races_frame(), T0).iloc[0])
    assert keys(TOK_B) == {k: race[k] for k in PAR_KEYS}
    par = model.par(**keys(TOK_B))
    mean, sd = tt_variant(model, T0, t)
    assert par == pytest.approx(96.0, abs=0.05)
    assert (mean, sd) == pytest.approx((0.4, 0.2), abs=1e-9)
    got = model.expected_top3_time(race, at_utc=t)
    assert got == pytest.approx(96.4, abs=0.05)
    assert got == pytest.approx(par + 0.4, abs=1e-9)


def test_same_day_variant_uses_only_races_resulted_before_t(fit):
    # The last day is a test day: fitted on the training days, its results observed.
    rows = tok_turf(races_frame(), LAST)
    published = list(rows["result_available_utc"])
    model = fit()
    by_set = {}
    for (hh, mm), resulted in SWEEP:
        t = at(LAST, hh, mm)
        in_frame = tuple(i for i, r in enumerate(published, start=1) if r < pd.Timestamp(t))
        assert in_frame == resulted  # the table matches the fixture
        by_set.setdefault(resulted, []).append(tt_variant(model, LAST, t))
    assert by_set[()][0] == pytest.approx((0.0, model.tau), abs=1e-12)
    # One value for each set of resulted races, whatever else has happened by t (a race run but
    # not yet published, a result published at exactly t)...
    for resulted, values in by_set.items():
        for v in values[1:]:
            assert v == pytest.approx(values[0], abs=1e-12), resulted
    # ...and a new value each time a result is published.
    for (sa, va), (sb, vb) in combinations([(s, v[0]) for s, v in by_set.items()], 2):
        assert max(abs(va[0] - vb[0]), abs(va[1] - vb[1])) > 1e-6, (sa, sb)
    # To the microsecond: a result published at exactly t is not in, one a microsecond before t is.
    for i, stamp in enumerate(published, start=1):
        before = tuple(j for j, r in enumerate(published, start=1) if r < stamp)
        t = stamp.to_pydatetime()
        got = tt_variant(model, LAST, t)
        assert got == pytest.approx(by_set[before][0], abs=1e-12), i
        got = tt_variant(model, LAST, t + timedelta(microseconds=1))
        assert got == pytest.approx(by_set[tuple(sorted((*before, i)))][0], abs=1e-12), i
    # One race SLOW s slower, nothing else changed: until its result is in, the variant at t and
    # E[R5] at t are as they were (R5's own time included); once it is in, the variant moves.
    r5 = card(rows.iloc[4])
    for i, not_yet, first_in in NOT_YET:
        slow = fit(f"slow_r{i}")
        for hh, mm in not_yet:
            t = at(LAST, hh, mm)
            assert tt_variant(slow, LAST, t) == pytest.approx(
                tt_variant(model, LAST, t), abs=1e-12
            ), (i, hh, mm)
            assert slow.expected_top3_time(r5, at_utc=t) == pytest.approx(
                model.expected_top3_time(r5, at_utc=t), abs=1e-12
            ), (i, hh, mm)
        t = at(LAST, *first_in)
        assert abs(tt_variant(slow, LAST, t)[0] - tt_variant(model, LAST, t)[0]) > 1e-3, i
    # R2 resulted at 01:50, before 02:30: its time does move the variant and E[R5] at 02:30.
    slow, t = fit("slow_r2"), at(LAST, 2, 30)
    moved = tt_variant(slow, LAST, t)[0] - tt_variant(model, LAST, t)[0]
    assert abs(moved) > 1e-3
    assert slow.expected_top3_time(r5, at_utc=t) - model.expected_top3_time(
        r5, at_utc=t
    ) == pytest.approx(moved, abs=1e-9)
    # A result never published (null time) never counts: the same as a day without that race.
    unpublished, dropped = fit("r2_unpublished"), fit("r2_dropped")
    for (hh, mm), _ in SWEEP:
        t = at(LAST, hh, mm)
        assert tt_variant(unpublished, LAST, t) == pytest.approx(
            tt_variant(dropped, LAST, t), abs=1e-12
        ), (hh, mm)
    t = at(LAST, 23, 0)
    assert abs(tt_variant(unpublished, LAST, t)[0] - tt_variant(model, LAST, t)[0]) > 1e-3
    # The fit is deterministic, so the equalities above are not luck.
    races = races_frame()
    again = SpeedModel(make_spec()).fit(training(races)).observe(observed(races))
    for (hh, mm), _ in SWEEP:
        t = at(LAST, hh, mm)
        assert tt_variant(again, LAST, t) == pytest.approx(tt_variant(model, LAST, t), abs=1e-12)


def test_student_t_downweights_an_outlier_race(fit):
    # One Tokyo turf race 10 s slow (training day 3, cell TOK_B, one of its 16 training races).
    # Under a normal error it drags the par by about 10/16 s; under a Student-t error it barely
    # moves it.
    def par_of(frame, t_df):
        return fit(frame, t_df=t_df, observe=False).par(**keys(TOK_B))

    def shift(t_df):
        return par_of("outlier", t_df) - par_of("base", t_df)

    normal = shift(None)
    assert normal > 0.3
    for t_df in (3.0, 5.0):
        assert abs(shift(t_df)) < 0.25 * normal, t_df
        assert par_of("outlier", t_df) == pytest.approx(96.0, abs=0.1)


def test_variant_shrinks_toward_its_prior_without_data(fit):
    # T0 at Tokyo turf ran exactly 1.0 s slow (no noise); the fit never saw it, and the training
    # days never feed its variant.
    rows = tok_turf(races_frame(), T0)
    for t_df in (None, 5.0):
        model = fit(t_df=t_df)
        resid = [row["top3_time_s"] - model.par(**keys(cell_of(row))) for _, row in rows.iterrows()]
        assert min(resid) > 0.8
        m0, sd0 = tt_variant(model, T0, at(T0, 0, 30))  # nothing resulted
        m1, sd1 = tt_variant(model, T0, at(T0, 1, 20))  # R1 only
        m6, sd6 = tt_variant(model, T0, at(T0, 4, 30))  # all six
        assert m0 == pytest.approx(0.0, abs=1e-12), t_df
        assert sd0 == pytest.approx(model.tau, rel=1e-12), t_df
        # the prior's SD is the random effect's spread (planted about 0.5 s), not flat, not a point
        assert 0.15 < sd0 < 1.2, (t_df, sd0)
        assert 0.0 < m1 < resid[0], (t_df, m1)
        assert m1 < m6 < max(resid), (t_df, m1, m6)
        assert sd0 > sd1 > sd6 > 0, (t_df, sd0, sd1, sd6)


def test_an_injected_variant_prior_is_used(fit):
    model = fit(prior="table")
    t0 = at(T0, 0, 30)  # no result of any race is in
    for (day, track, surface), prior in PRIORS["table"].items():
        got = model.variant(day, track, surface, at_utc=t0)
        assert got == pytest.approx(prior, abs=1e-9), (track, surface)
    assert model.variant(T0, "NII", "turf", at_utc=t0) == pytest.approx((0.0, 0.5), abs=1e-9)
    # A training day is never observed: the injected prior (here the table's default) all day.
    day = DAYS[2]
    assert tt_variant(model, day, at(day, 23, 0)) == pytest.approx((0.0, 0.5), abs=1e-9)
    # With the day's six results in (residuals about +1.0 s) the variant sits between the injected
    # mean (-0.5 s) and the data, below the variant shrunk toward 0.
    rows = tok_turf(races_frame(), T0)
    resid = [row["top3_time_s"] - model.par(**keys(cell_of(row))) for _, row in rows.iterrows()]
    t6 = at(T0, 4, 30)
    injected = tt_variant(model, T0, t6)[0]
    default = tt_variant(fit(), T0, t6)[0]
    assert -0.5 < injected < default - 0.05
    assert injected < max(resid)
    # The prior's SD sets the pull: with one result in, a tight prior holds the variant near its
    # mean and a loose one lets the race through.
    t1 = at(T0, 1, 20)
    tight = tt_variant(fit(prior="tight"), T0, t1)
    loose_model = fit(prior="loose")
    loose = tt_variant(loose_model, T0, t1)
    r1 = rows.iloc[0]["top3_time_s"] - loose_model.par(**keys(cell_of(rows.iloc[0])))
    assert tight[0] == pytest.approx(-0.5, abs=0.15)
    assert loose[0] == pytest.approx(r1, abs=0.15)
    assert tight[1] < loose[1]
