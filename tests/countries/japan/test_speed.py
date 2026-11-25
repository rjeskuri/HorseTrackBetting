"""The C3 race-level finish-speed model (ticket #96, epic #73): par, day variant, E[top-3 time].

Contract pinned here (`horsetrack.countries.japan.models.speed`):
- `SpeedSpec(par_keys, variant_keys, same_day_variant, t_df)`; t_df None is a normal error.
- `SpeedModel(spec, *, variant_prior=None).fit(races) -> SpeedModel` (the returned object is the
  one used). races columns: race_id, race_utc, track, surface, distance, layout, rail,
  class_level, top3_time_s, result_available_utc, both times tz-aware UTC. The model is
  top3_time = par(cell) + v(day, track, surface) + eps, eps Student-t with t_df degrees of freedom
  (normal when None), v a random effect. A race's day is its JST calendar date; every synthetic
  race here is run between 01:00 and 04:10 UTC, so its JST and UTC dates agree.
- `par(**keys)`: keyword arguments named by spec.par_keys; a cell is the tuple of their values.
  'Course' is layout (inner, outer, straight) plus rail (A-D, or 'unknown').
- `variant(race_date, track, surface, *, at_utc) -> (mean, sd)`, race_date a `datetime.date`: the
  variant of that day at that track and surface given only races whose result_available_utc is
  strictly before at_utc. With same_day_variant=True the day's own races count once resulted
  before t; with False they never do (earlier days only). With nothing available the answer is
  the prior: mean 0 and a positive SD (the random effect's spread) by default, or the injected
  prior's (mean, sd). More data moves the mean from the prior toward the races' residuals from
  par and narrows the SD.
- `variant_prior` call shape (chosen here, the ticket leaves it open):
  `variant_prior(race_date, track, surface) -> (mean, sd)`, race_date a `datetime.date` (not a
  datetime or Timestamp), track and surface strings as in the races frame, sd > 0. It may be called
  for any (date, track, surface), during fit too.
- `expected_top3_time(race, *, at_utc)`: race a mapping with race_id, race_utc and the par keys (no
  result columns); it returns par(cell) + variant(day, track, surface, at_utc=at_utc)[0].
- The fit is deterministic: the same frame gives the same pars and variants.

Not pinned: the estimator, the shrinkage formula, how earlier days feed a day's variant, what an
unseen cell or track returns. Expected values come from the planted synthetic races below (pars,
day variants, noise), never from the code under test. HORSETRACK_DATA_DIR points at an empty tmp
folder, so nothing can come from a store.
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
N_DAYS = 16
RAIL_C_FROM = 8  # Tokyo turf runs on rail B on days 0-7 and on rail C on days 8-15
DAYS = tuple(D0 + timedelta(days=k) for k in range(N_DAYS))
LAST = DAYS[-1]
NEXT = D0 + timedelta(days=N_DAYS)  # a card day: none of its races is in the frame
SIGMA = 0.3  # race-level noise, s

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


def tok_turf_cells(k):
    rail = "B" if k < RAIL_C_FROM else "C"
    a = ("TOK", "turf", 1600, "outer", rail, 3)
    b = ("TOK", "turf", 2000, "outer", rail, 3)
    c = ("TOK", "turf", 1600, "outer", rail, 5)
    return (a, b, c, c, b, a)  # R1..R6: R1/R6, R2/R5 and R3/R4 share a cell


def same_cell(cell, n):
    return lambda k: (cell,) * n


# (track, surface) -> (post times UTC as (hh, mm), cells of the day's races in post order)
GROUPS = {
    ("TOK", "turf"): (((1, 5), (1, 40), (2, 15), (2, 50), (3, 25), (4, 0)), tok_turf_cells),
    ("TOK", "dirt"): (
        ((1, 20), (3, 40)),
        same_cell(("TOK", "dirt", 1600, "outer", "unknown", 3), 2),
    ),
    ("KYO", "turf"): (
        ((1, 10), (1, 45), (2, 20), (2, 55)),
        lambda k: (
            ("KYO", "turf", 1600, "outer", "B", 3),
            ("KYO", "turf", 1600, "inner", "B", 3),
            ("KYO", "turf", 1600, "inner", "B", 3),
            ("KYO", "turf", 1600, "outer", "B", 3),
        ),
    ),
    ("NII", "turf"): (
        ((1, 15), (1, 50), (2, 25)),
        lambda k: (
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

# Day variants, s (positive = slow). Tokyo turf's sum to zero within each rail block, the others'
# over the 16 days, so a cell's mean time is its par.
V_TOK_TURF = (
    *(1.0, -0.4, 0.3, -0.7, 0.5, -0.2, -0.6, 0.1),  # rail B days
    *(-0.5, 0.6, -0.3, 0.2, 0.9, -0.8, 0.4, -0.5),  # rail C days
)
V_DAY0 = {("TOK", "dirt"): -0.8, ("KYO", "turf"): -0.6, ("NII", "turf"): 0.5, ("NII", "dirt"): 0.7}

# The last day's Tokyo turf noise, set by hand so the swaps below exchange clearly different times.
LAST_TOK_TURF_EPS = {1: 0.6, 6: -0.6, 3: 0.9, 4: -0.5}


def planted_variants():
    rng = np.random.default_rng(9601)
    v = {("TOK", "turf"): np.array(V_TOK_TURF)}
    for group, v0 in V_DAY0.items():
        rest = rng.normal(0.0, 0.5, N_DAYS - 1)
        rest -= (rest.sum() + v0) / (N_DAYS - 1)
        v[group] = np.r_[v0, rest]
    return v


VARIANTS = planted_variants()


def at(day, hh, mm):
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=timezone.utc)


@lru_cache(maxsize=1)
def _base_races():
    """17 races a day for 16 days. Day 0 runs without noise (every race exactly par + variant);
    every other race has N(0, 0.3) noise, centred within its cell so the cell's noise sums to 0."""
    rng = np.random.default_rng(9602)
    rows = []
    for k, day in enumerate(DAYS):
        for (track, surface), (posts, cells) in GROUPS.items():
            for i, ((hh, mm), cell) in enumerate(zip(posts, cells(k), strict=True), start=1):
                post = at(day, hh, mm)
                last_tok_turf = k == N_DAYS - 1 and (track, surface) == ("TOK", "turf")
                delay = INQUIRY_DELAY if last_tok_turf and i == 3 else RESULT_DELAY
                eps, fixed = rng.normal(0.0, SIGMA), False
                if k == 0:
                    eps, fixed = 0.0, True
                elif last_tok_turf and i in LAST_TOK_TURF_EPS:
                    eps, fixed = LAST_TOK_TURF_EPS[i], True
                rows.append(
                    {
                        "race_id": f"syn-{day:%Y%m%d}-{track}-{surface}-{i:02d}",
                        "race_utc": post,
                        **dict(zip(PAR_KEYS, cell, strict=True)),
                        "base": PAR[cell] + VARIANTS[(track, surface)][k],
                        "eps": eps,
                        "fixed": fixed,
                        "result_available_utc": post + delay,
                    }
                )
    frame = pd.DataFrame(rows)
    for _, idx in frame.groupby(list(PAR_KEYS)).groups.items():
        cell_rows = frame.loc[idx]
        free = cell_rows.index[~cell_rows["fixed"]]
        frame.loc[free, "eps"] -= cell_rows["eps"].sum() / len(free)
    frame["top3_time_s"] = frame["base"] + frame["eps"]
    frame["race_utc"] = pd.to_datetime(frame["race_utc"], utc=True)
    frame["result_available_utc"] = pd.to_datetime(frame["result_available_utc"], utc=True)
    return frame[list(COLUMNS)]


def on_day(races, day):
    jst = races["race_utc"].dt.tz_convert("Asia/Tokyo").dt.date
    return races[jst == day].sort_values("race_utc")


def tok_turf(races, day):
    """The day's six Tokyo turf races, R1..R6 in post order, with the frame's index."""
    d = on_day(races, day)
    return d[(d["track"] == "TOK") & (d["surface"] == "turf")]


def swap_times(races, day, i, j):
    rows = tok_turf(races, day)
    a, b = rows.index[i - 1], rows.index[j - 1]
    assert tuple(races.loc[a, list(PAR_KEYS)]) == tuple(races.loc[b, list(PAR_KEYS)])
    races.loc[[a, b], "top3_time_s"] = races.loc[[b, a], "top3_time_s"].to_numpy()
    return races


def races_frame(name="base"):
    races = _base_races().copy()
    if name == "swap_r3_r4":  # both resulted after 02:30 on the last day; same cell, same day
        return swap_times(races, LAST, 3, 4)
    if name == "swap_r1_r6":  # R1 resulted before 02:30, R6 after; same cell, same day
        return swap_times(races, LAST, 1, 6)
    if name == "outlier":  # one Tokyo turf race 10 s slow (day 3, R1: cell TOK_B)
        races.loc[tok_turf(races, DAYS[3]).index[0], "top3_time_s"] += 10.0
        return races
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
        (D0, "TOK", "turf"): (-0.5, 0.3),
        (D0, "TOK", "dirt"): (0.7, 0.25),
        (D0, "KYO", "turf"): (0.2, 0.6),
        (D0, "NII", "dirt"): (-0.3, 0.4),
    },
    "plus_0_4": {(D0, "TOK", "turf"): (0.4, 0.2)},
    "tight": {(D0, "TOK", "turf"): (-0.5, 0.05)},
    "loose": {(D0, "TOK", "turf"): (-0.5, 3.0)},
}


def make_spec(*, t_df=5.0, same_day=True, par_keys=PAR_KEYS):
    return SpeedSpec(
        par_keys=par_keys, variant_keys=VARIANT_KEYS, same_day_variant=same_day, t_df=t_df
    )


@pytest.fixture(scope="module")
def fit():
    """Fitted models, each once per module: fit(frame, t_df=, same_day=, par_keys=, prior=)."""
    cache = {}

    def _fit(frame="base", *, t_df=5.0, same_day=True, par_keys=PAR_KEYS, prior=None):
        key = (frame, t_df, same_day, par_keys, prior)
        if key not in cache:
            vp = None if prior is None else table_prior(PRIORS[prior])
            spec = make_spec(t_df=t_df, same_day=same_day, par_keys=par_keys)
            cache[key] = SpeedModel(spec, variant_prior=vp).fit(races_frame(frame))
        return cache[key]

    return _fit


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


# ---------------------------------------------------------------------------------------------
# Broad behaviour
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t_df", [None, 5.0], ids=["normal", "student_t5"])
def test_par_recovers_every_planted_cell(fit, t_df):
    model = fit(t_df=t_df)
    for cell, planted in PAR.items():
        assert model.par(**keys(cell)) == pytest.approx(planted, abs=0.1), cell


def test_par_keys_of_the_spec_decide_the_cells(fit):
    # Without rail in par_keys, rails B and C share one par: the mean of the two planted pars (each
    # rail runs the same number of races).
    pooled = fit(t_df=None, par_keys=NO_RAIL)
    for distance, class_level in ((1600, 3), (2000, 3), (1600, 5)):
        b = ("TOK", "turf", distance, "outer", "B", class_level)
        c = ("TOK", "turf", distance, "outer", "C", class_level)
        got = pooled.par(**keys(b, NO_RAIL))
        assert got == pytest.approx((PAR[b] + PAR[c]) / 2, abs=0.1), (distance, class_level)
    kyo = ("KYO", "turf", 1600, "inner", "B", 3)
    assert pooled.par(**keys(kyo, NO_RAIL)) == pytest.approx(PAR[kyo], abs=0.1)


def test_variant_follows_each_days_planted_going_by_track_and_surface(fit):
    model = fit()
    # Day 0, every result in: each (track, surface) gets its own day's variant, sign and size.
    for (track, surface), planted in VARIANTS.items():
        mean, sd = model.variant(D0, track, surface, at_utc=at(D0, 23, 0))
        v0 = planted[0]
        assert np.sign(mean) == np.sign(v0), (track, surface, mean, v0)
        assert 0.5 * abs(v0) <= abs(mean) <= abs(v0) + 0.1, (track, surface, mean, v0)
        assert sd > 0
    # Tokyo turf over all 16 days, each at the end of its day.
    means = [model.variant(d, "TOK", "turf", at_utc=at(d, 23, 0))[0] for d in DAYS]
    assert np.corrcoef(means, V_TOK_TURF)[0, 1] > 0.9
    assert max(abs(m - v) for m, v in zip(means, V_TOK_TURF, strict=True)) < 0.45


def test_expected_top3_time_adds_the_variant_at_t_to_the_par(fit):
    races = races_frame()
    model = fit()
    for day in (D0, DAYS[7], LAST):
        for _, row in on_day(races, day).iterrows():
            race = card(row)
            post = row["race_utc"].to_pydatetime()
            for t in (at(day, 0, 30), post - timedelta(minutes=1), at(day, 23, 0)):
                mean, _ = model.variant(day, race["track"], race["surface"], at_utc=t)
                want = model.par(**keys(cell_of(row))) + mean
                got = model.expected_top3_time(race, at_utc=t)
                assert got == pytest.approx(want, abs=1e-9), (race["race_id"], t)
    # A card day none of whose races is in the frame.
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
        got = model.expected_top3_time(race, at_utc=t)
        assert math.isfinite(got) and math.isfinite(mean) and sd > 0
        assert got == pytest.approx(model.par(**keys(cell)) + mean, abs=1e-9)
    # The day's earlier results earn their place: Tokyo turf R4-R6, each a minute before its off,
    # against the par alone.
    err_e, err_par = [], []
    for day in DAYS[1:]:
        for _, row in tok_turf(races, day).iloc[3:].iterrows():
            t = row["race_utc"].to_pydatetime() - timedelta(minutes=1)
            err_e.append(model.expected_top3_time(card(row), at_utc=t) - row["top3_time_s"])
            err_par.append(model.par(**keys(cell_of(row))) - row["top3_time_s"])
    assert rmse(err_e) < 0.8 * rmse(err_par)


def test_without_a_same_day_variant_the_days_own_races_never_count(fit):
    model = fit(same_day=False)
    # Day 0 has no earlier day: the prior all day long, however many results are in.
    for (hh, mm), _ in SWEEP:
        mean, sd = model.variant(D0, "TOK", "turf", at_utc=at(D0, hh, mm))
        assert mean == pytest.approx(0.0, abs=1e-12), (hh, mm)
        assert sd > 0
    # The last day: one value all day long.
    first = model.variant(LAST, "TOK", "turf", at_utc=at(LAST, 0, 30))
    for (hh, mm), _ in SWEEP:
        got = model.variant(LAST, "TOK", "turf", at_utc=at(LAST, hh, mm))
        assert got == pytest.approx(first, abs=1e-12), (hh, mm)
    late = tok_turf(races_frame(), LAST).iloc[5]
    t = late["race_utc"].to_pydatetime() - timedelta(minutes=1)
    assert model.expected_top3_time(card(late), at_utc=t) == pytest.approx(
        model.par(**keys(cell_of(late))) + first[0], abs=1e-9
    )
    # Exchanging R1's time (resulted at 01:15) with R6's (run at 04:00) leaves the fit as it was
    # (same cell, same day) and changes what had resulted by 02:30: without a same-day variant
    # nothing moves; with one it does, so the exchange is not idle.
    t = at(LAST, 2, 30)
    off = model.variant(LAST, "TOK", "turf", at_utc=t)
    off_swapped = fit("swap_r1_r6", same_day=False).variant(LAST, "TOK", "turf", at_utc=t)
    assert off_swapped == pytest.approx(off, abs=1e-7)
    on = fit().variant(LAST, "TOK", "turf", at_utc=t)
    on_swapped = fit("swap_r1_r6").variant(LAST, "TOK", "turf", at_utc=t)
    assert abs(on_swapped[0] - on[0]) > 1e-3


# ---------------------------------------------------------------------------------------------
# The ticket's named tests
# ---------------------------------------------------------------------------------------------


def test_par_is_keyed_by_track_distance_course_surface_class(fit):
    for t_df in (None, 5.0):
        model = fit(t_df=t_df)
        for key, a, b in ONE_KEY_PAIRS:
            differs = [k for k, x, y in zip(PAR_KEYS, a, b, strict=True) if x != y]
            assert differs == [key]
            got = model.par(**keys(b)) - model.par(**keys(a))
            assert got == pytest.approx(PAR[b] - PAR[a], abs=0.15), (t_df, key, a, b)
            assert abs(got) > 0.3, (t_df, key, a, b)


def test_expected_top3_time_is_par_plus_variant(fit):
    # Par 96.0 s (planted, cell TOK_B) and a variant of +0.4 s: the injected prior's mean, since
    # at 00:30 on day 0 no result of any race is in.
    model = fit(prior="plus_0_4")
    t = at(D0, 0, 30)
    race = card(tok_turf(races_frame(), D0).iloc[0])
    assert keys(TOK_B) == {k: race[k] for k in PAR_KEYS}
    par = model.par(**keys(TOK_B))
    mean, sd = model.variant(D0, "TOK", "turf", at_utc=t)
    assert par == pytest.approx(96.0, abs=0.05)
    assert (mean, sd) == pytest.approx((0.4, 0.2), abs=1e-9)
    got = model.expected_top3_time(race, at_utc=t)
    assert got == pytest.approx(96.4, abs=0.05)
    assert got == pytest.approx(par + 0.4, abs=1e-9)


def test_same_day_variant_uses_only_races_resulted_before_t(fit):
    rows = tok_turf(races_frame(), LAST)
    model = fit()
    by_set = {}
    for (hh, mm), resulted in SWEEP:
        t = at(LAST, hh, mm)
        in_frame = tuple(
            i for i, r in enumerate(rows["result_available_utc"], start=1) if r < pd.Timestamp(t)
        )
        assert in_frame == resulted  # the table matches the fixture
        by_set.setdefault(resulted, []).append(model.variant(LAST, "TOK", "turf", at_utc=t))
    # One value for each set of resulted races, whatever else has happened by t (a race run but
    # not yet published, a result published at exactly t)...
    for resulted, values in by_set.items():
        for v in values[1:]:
            assert v == pytest.approx(values[0], abs=1e-12), resulted
    # ...and a new value each time a result is published.
    for (sa, va), (sb, vb) in combinations([(s, v[0]) for s, v in by_set.items()], 2):
        assert max(abs(va[0] - vb[0]), abs(va[1] - vb[1])) > 1e-6, (sa, sb)
    # R3 and R4 share a cell and a day, so exchanging their times leaves the fit as it was. At
    # 02:30 both had resulted after t (R3 had run): the variant and E[R5] are unchanged.
    swapped = fit("swap_r3_r4")
    t = at(LAST, 2, 30)
    r5 = card(rows.iloc[4])
    assert swapped.variant(LAST, "TOK", "turf", at_utc=t) == pytest.approx(
        model.variant(LAST, "TOK", "turf", at_utc=t), abs=1e-7
    )
    assert swapped.expected_top3_time(r5, at_utc=t) == pytest.approx(
        model.expected_top3_time(r5, at_utc=t), abs=1e-7
    )
    # Once R4 has resulted (03:01) its time counts, and the exchange shows.
    t = at(LAST, 3, 1)
    moved = swapped.variant(LAST, "TOK", "turf", at_utc=t)[0]
    assert abs(moved - model.variant(LAST, "TOK", "turf", at_utc=t)[0]) > 1e-3
    # The fit is deterministic, so the equalities above are not luck.
    again = SpeedModel(make_spec()).fit(races_frame())
    for (hh, mm), _ in SWEEP:
        t = at(LAST, hh, mm)
        assert again.variant(LAST, "TOK", "turf", at_utc=t) == pytest.approx(
            model.variant(LAST, "TOK", "turf", at_utc=t), abs=1e-9
        )


def test_student_t_downweights_an_outlier_race(fit):
    # One Tokyo turf race 10 s slow (day 3, cell TOK_B, one of its 16 races). Under a normal error
    # it drags the par by about 10/16 s; under a Student-t error it barely moves it.
    def shift(t_df):
        return fit("outlier", t_df=t_df).par(**keys(TOK_B)) - fit(t_df=t_df).par(**keys(TOK_B))

    normal = shift(None)
    assert normal > 0.3
    for t_df in (3.0, 5.0):
        assert abs(shift(t_df)) < 0.25 * normal, t_df
        assert fit("outlier", t_df=t_df).par(**keys(TOK_B)) == pytest.approx(96.0, abs=0.1)


def test_variant_shrinks_toward_its_prior_without_data(fit):
    # Day 0 at Tokyo turf ran exactly 1.0 s slow (no noise), and no earlier day exists.
    rows = tok_turf(races_frame(), D0)
    for t_df in (None, 5.0):
        model = fit(t_df=t_df)
        resid = [row["top3_time_s"] - model.par(**keys(cell_of(row))) for _, row in rows.iterrows()]
        assert min(resid) > 0.8
        m0, sd0 = model.variant(D0, "TOK", "turf", at_utc=at(D0, 0, 30))  # nothing resulted
        m1, sd1 = model.variant(D0, "TOK", "turf", at_utc=at(D0, 1, 20))  # R1 only
        m6, sd6 = model.variant(D0, "TOK", "turf", at_utc=at(D0, 4, 30))  # all six
        assert m0 == pytest.approx(0.0, abs=1e-12), t_df
        # the prior's SD is the random effect's spread (planted about 0.5 s), not flat, not a point
        assert 0.15 < sd0 < 1.2, (t_df, sd0)
        assert 0.0 < m1 < resid[0], (t_df, m1)
        assert m1 < m6 < max(resid), (t_df, m1, m6)
        assert sd0 > sd1 > sd6 > 0, (t_df, sd0, sd1, sd6)


def test_an_injected_variant_prior_is_used(fit):
    model = fit(prior="table")
    t0 = at(D0, 0, 30)  # no result of any race is in
    for (day, track, surface), prior in PRIORS["table"].items():
        got = model.variant(day, track, surface, at_utc=t0)
        assert got == pytest.approx(prior, abs=1e-9), (track, surface)
    assert model.variant(D0, "NII", "turf", at_utc=t0) == pytest.approx((0.0, 0.5), abs=1e-9)
    # With the day's six results in (residuals about +1.0 s) the variant sits between the injected
    # mean (-0.5 s) and the data, below the variant shrunk toward 0.
    rows = tok_turf(races_frame(), D0)
    resid = [row["top3_time_s"] - model.par(**keys(cell_of(row))) for _, row in rows.iterrows()]
    t6 = at(D0, 4, 30)
    injected = model.variant(D0, "TOK", "turf", at_utc=t6)[0]
    default = fit().variant(D0, "TOK", "turf", at_utc=t6)[0]
    assert -0.5 < injected < default - 0.05
    assert injected < max(resid)
    # The prior's SD sets the pull: with one result in, a tight prior holds the variant near its
    # mean and a loose one lets the race through.
    t1 = at(D0, 1, 20)
    tight = fit(prior="tight").variant(D0, "TOK", "turf", at_utc=t1)
    loose_model = fit(prior="loose")
    loose = loose_model.variant(D0, "TOK", "turf", at_utc=t1)
    r1 = rows.iloc[0]["top3_time_s"] - loose_model.par(**keys(cell_of(rows.iloc[0])))
    assert tight[0] == pytest.approx(-0.5, abs=0.15)
    assert loose[0] == pytest.approx(r1, abs=0.15)
    assert tight[1] < loose[1]
