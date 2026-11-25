"""The day-variance estimator over many small cells, the rail setting in the cell (ticket #258).

`track_bias.estimate_day_variance` (#224, contract in test_track_bias_day_variance.py) measures
each level's day sd from deviations against the cells' prior means. #224's synthetic histories
put every race of a course in one cell of some 1500 races, where a cell's mean barely moves with
any one day. Real cells (venue, surface, distance, course_config, going, rail_setting; #222) are
small, and a cell mean fitted in-sample from a few days absorbs part of each of those days'
effects: #224's review found the sd about 12% low at some 30 races a cell.

Two estimands, both reported (owner, 2026-11-25), named apart:

(a) `DayVariance.per_level` / `.pooled`, the day sd (#224's meaning): the planted day effect's
    sd, what a (day, venue, surface) group's races share beyond their cells' TRUE means.
    Compared, as in #224's tests, with the planted effects' realised sd around each venue and
    surface's mean.
(b) `DayVariance.predictive_per_level` / `.predictive_pooled`, the predictive sd: the sd of a
    group's deviation from the shrunk prior mean fitted WITHOUT that group (leave-group-out),
    the day variance plus the prior mean's own error. This is what `fit_bias_prior`'s estimated
    modes ("pooled", None, "per_level") put in `BiasPrior.day_sd`, hence the posterior's
    `prior_sd`, while `BiasPrior.estimate` carries the whole DayVariance, (a) included.
    Compared with its value on the planted truth (`predictive_oracle`): each group's expected
    level mean given its planted u (the target's u, -c u / (nf - c) at the planted factor's other
    levels, 0 at the other factor's; plus the rail effect likewise), less #138's shrunk prior
    mean refitted by this file without the group, averaged over the group's races, squared,
    averaged over groups. Theory for the planted level: the leave-group-out cell mean of n races,
    r of them the group's in each of its q cells, carries the other groups' day effects with
    variance lam sd^2, lam = (n - r) r / (n - r + PRIOR_RACES)^2 / q (0.12 at 30 races a cell,
    0.10 at 12, 0.03 split), plus the race noise of the cell mean and, with rail effects, the
    shrinkage toward the rail-mixed parent; so (oracle / realised)^2 lies in [1 + lam,
    1 + lam + 0.25] (observed: lam + 0.02 to lam + 0.18).

This file pins that over many small cells:

- (a) the planted level's day sd within TOL of the planted one (cells of ~30 races, each day's
  races of a course in one cell or split over two, and of ~12 races); per level and pooled: the
  other levels of the planted factor at half the planted sd (a race's scores sum to 0, so they
  carry -c u / (nf - c) = -u / 2 with c = 4 of nf = 12), the other factor's levels about 0, the
  pooled sd sqrt(sum of the levels' variances / 7);
- (a) a rail-setting effect on the level's mean (constant over a group, since a course has one
  rail setting a day) stays in the cell: the day sd is still the planted one, while the same
  history read without its rail settings (a races frame without the column, #222) gives a
  clearly larger one;
- (b) the planted level, the planted factor's other levels and the pooled predictive sd within
  TOL_PRED, TOL_PRED_HALF and TOL_PRED_POOLED of the oracle, and at least (a);
- the estimated prior's day sd is (b), its estimate the same DayVariance as estimate_day_variance.

Tolerances, from simulation while writing these tests (same layouts and seeds, correct
estimators written for the purpose). (a): the between-group variance of the level's group means
within each cell, less the within-group race-to-race variance over the group's races, pooled over
cells, so the cells' true means drop out. Averaged over the four seeds of a layout, its
planted-level sd landed within 3% of the realised sd (single seeds within 8%, a relative sd near
3%); half-sd levels within 8%; pooled within 4%; the rail-free reading 1.25 to 1.33 times the
rail-split one per seed. TOL = 6% on the planted level is about four standard errors of the
four-seed mean. The current in-sample estimator lands 7% (two cells a group) to 25% (12 races a
cell) low on the planted level and 15% to 100% low pooled. Plain leave-group-out cell means land
4% to 14% high against (a), which is (b)'s excess, not an error of (b).
(b): #224's nested moment estimator (mean over groups of d^2 - s2 / r) on each race's level
mean less its cell's leave-group-out prior mean. Its four-seed mean over the oracle: planted level
0.977 to 1.012 (single seeds 0.95 to 1.03), the planted factor's other levels 0.934 to 1.017,
pooled 0.972 to 1.023 (single seeds 0.88 to 1.06). TOL_PRED = 5%, TOL_PRED_HALF = 12%,
TOL_PRED_POOLED = 7%. (a) read as (b) lands some 0.85 to 0.88 of the pooled oracle in every layout
but the split one, where lam is small and (a) and (b) nearly agree. The other factor's levels
(oracle 0.02 to 0.06, the prior mean's error alone) are too noisy per level to pin; the pooled
check carries them.

Data: synthetic only (day_variance_store.planted_race), in memory; nothing reads
$HORSETRACK_DATA_DIR.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import functools
import math
import random
from collections import Counter
from datetime import date
from typing import NamedTuple

import polars as pl
import pytest

from day_variance_store import SURFACES, VENUES, draw_group, planted_race, realized_sd, style_of
from handicap_store import Store, weekend_days
from horsetrack.countries.japan.handicap.track_bias import estimate_day_variance, fit_bias_prior

DRAW_LEVELS = ("inner", "middle", "outer")
STYLE_LEVELS = ("front", "stalker", "midpack", "closer")
LEVELS = [("draw", lv) for lv in DRAW_LEVELS] + [("style", lv) for lv in STYLE_LEVELS]
FRONT = ("style", "front")
MIDPACK = ("style", "midpack")
INNER = ("draw", "inner")

TRAIN_END = date(2025, 6, 8)
DAYS = weekend_days(date(2023, 1, 7), TRAIN_END)  # 254 days, 1016 (day, venue, surface) groups
SD = 0.10  # the planted day sd
PRIOR_RACES = 10  # #138's pseudo-races shrinking a cell's mean toward its parent's
FLOOR = 0.01  # #224: an estimated day_sd is floored here
FLOOR_ZONE = 0.02  # #224: below this a fitted prior's sd may be floored

GOINGS = ("GOOD", "YIELDING", "SOFT", "HEAVY")
RAILS = ("A", "B", "C", "D")
DISTANCES = tuple(range(1000, 3001, 200))  # 11 distances: 176 cells a course to draw from

TOL = 0.06  # planted level, relative, four-seed mean (see the module doc)
TOL_HALF = 0.12  # the planted factor's other levels (half the planted sd)
TOL_POOLED = 0.12
TOL_PRED = 0.05  # (b) against its oracle: planted level
TOL_PRED_HALF = 0.12  # (b): the planted factor's other levels
TOL_PRED_POOLED = 0.07  # (b): pooled


def null_bound(key) -> float:
    """The bound on "about 0" for a level with no day effect, as in #224's tests."""
    return 0.07 if key == FRONT else 0.035


# ---------------------------------------------------------------------------------------------
# Synthetic histories of many small cells
# ---------------------------------------------------------------------------------------------

LAYOUTS = {
    # name: (seeds, target, races a group, cells a group, groups a cell, rail effect, mode,
    #        distances drawn from)
    # every group's six races in one cell of five groups: 30 races a cell
    "cells30": ((2581, 2582, 2583, 2584), INNER, 6, 1, 5, None, "mixed", DISTANCES),
    # six races a group over two cells of three, ten groups a cell: 30 races a cell
    "cells30_split": ((2591, 2592, 2593, 2594), MIDPACK, 6, 2, 10, None, "mixed", DISTANCES),
    # four races a group in one cell of three groups: 12 races a cell
    "cells12": ((2601, 2602, 2603, 2604), INNER, 4, 1, 3, None, "mixed", DISTANCES),
    # as cells30, with the level's true mean shifted by the rail setting, over four distances so
    # that each (distance, going) recurs at several rail settings
    "rail": (
        (2611, 2612, 2613, 2614),
        INNER,
        6,
        1,
        5,
        {"A": 0.12, "B": 0.04, "C": -0.04, "D": -0.12},
        "extreme",
        DISTANCES[:4],
    ),
}


def frames(s: Store) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.DataFrame(
            s.races, infer_schema_length=None, schema_overrides={"rail_setting": pl.String}
        ),
        pl.DataFrame(s.runners, infer_schema_length=None),
    )


def small_cells_history(seed: int, name: str):
    """Every day of DAYS at both venues and surfaces, one group each with a day effect
    u ~ Uniform(-SD sqrt 3, SD sqrt 3) at the layout's target level. Each course's groups are
    dealt into cells (distance, going, rail_setting) so that every cell gets its races from
    exactly `groups a cell` groups (the last cell of a course may get fewer), `races a group /
    cells a group` races from each. A rail effect adds to the level's true mean in every race of
    the rail setting. Returns the races and runners frames, the planted u per group and each
    group's rail effect."""
    layout = LAYOUTS[name]
    _, target, per_group, cells_per_group, groups_per_cell, rail_effect, mode, distances = layout
    rng = random.Random(seed)
    s = Store()
    planted: dict[tuple, float] = {}
    shift: dict[tuple, float] = {}
    half = SD * math.sqrt(3)
    combos = [(d, g, r) for d in distances for g in GOINGS for r in RAILS]
    for venue in VENUES:
        for surface in SURFACES:
            slots = len(DAYS) * cells_per_group
            n_cells = math.ceil(slots / groups_per_cell)
            assert n_cells <= len(combos), "fixture: too few distinct cells"
            pool = combos[:]
            rng.shuffle(pool)
            deal = [c for c in range(n_cells) for _ in range(groups_per_cell)][:slots]
            rng.shuffle(deal)
            for gi, day in enumerate(DAYS):
                u = rng.uniform(-half, half)
                planted[(day, venue, surface)] = u
                for ci in deal[gi * cells_per_group : (gi + 1) * cells_per_group]:
                    distance, going, rail = pool[ci]
                    eff = rail_effect[rail] if rail_effect else 0.0
                    shift[(day, venue, surface)] = eff
                    for _ in range(per_group // cells_per_group):
                        planted_race(s, rng, day, venue=venue, surface=surface, field=12,
                                     target=target, u=u + eff, mode=mode)  # fmt: skip
                        s.races[-1].update(distance_m=distance, going=going, rail_setting=rail)
    races, runners = frames(s)
    return races, runners, planted, shift


# ---------------------------------------------------------------------------------------------
# (b) on the planted truth: the deviation from the prior fitted without the group
# ---------------------------------------------------------------------------------------------

CELL = ("venue", "surface", "distance_m", "course_config", "going", "rail_setting")
DEPTHS = (2, 3, len(CELL))  # (venue, surface) -> + distance -> the full cell (#138)


def level_coefficient(target, key, nf: int = 12) -> float:
    """A level's expected race mean per unit of the planted u (day_variance_store's doc): 1 at
    the target, -c / (nf - c) at its factor's other levels (c the target's runners), else 0."""
    if key == target:
        return 1.0
    if key[0] != target[0]:
        return 0.0
    of = draw_group if target[0] == "draw" else style_of
    c = sum(of(p, nf) == target[1] for p in range(1, nf + 1))
    return -c / (nf - c)


def predictive_oracle(races, runners, planted, shift, target) -> tuple[dict, float]:
    """(per level, pooled) predictive VARIANCES on the planted truth (see the module doc): per
    group and level, the mean over the group's races of (expected level mean given u and the
    rail effect) - (#138's shrunk prior mean of the race's cell, refitted without the group's
    races), squared; averaged over groups, and pooled over every level's groups."""
    info = {}
    cols = ("race_id", "race_date", *CELL, "field_size")
    for rid, day, *cell, nf in races.select(*cols).iter_rows():
        info[rid] = ((day, cell[0], cell[1]), tuple(cell), nf)
    acc: dict[tuple, list] = {}  # (race, level) -> [sum of scores, runners]
    for rid, draw, rp, pos in runners.select(
        "race_id", "draw", "running_positions", "finish_pos"
    ).iter_rows():
        nf = info[rid][2]
        y = (nf - pos) / (nf - 1) - 0.5
        for key in (("draw", draw_group(draw, nf)), ("style", style_of(rp[0], nf))):
            a = acc.setdefault((rid, key), [0.0, 0])
            a[0] += y
            a[1] += 1
    total: dict[tuple, list] = {}  # (cell prefix, level) -> [sum, runners, races]
    own: dict[tuple, list] = {}  # (group, cell prefix, level) -> the group's share
    for (rid, key), (sy, c) in acc.items():
        g, cell, _ = info[rid]
        for depth in DEPTHS:
            for t in (total.setdefault((cell[:depth], key), [0.0, 0, 0]),
                      own.setdefault((g, cell[:depth], key), [0.0, 0, 0])):  # fmt: skip
                t[0] += sy
                t[1] += c
                t[2] += 1

    def prior_without(g, cell, key) -> float:
        mean = 0.0
        for depth in DEPTHS:
            t = total[(cell[:depth], key)]
            h = own.get((g, cell[:depth], key), (0.0, 0, 0))
            sy, c, n = t[0] - h[0], t[1] - h[1], t[2] - h[2]
            ybar = sy / c if c else 0.0
            mean = (n * ybar + PRIOR_RACES * mean) / (n + PRIOR_RACES)
        return mean

    devs: dict[tuple, list[float]] = {}
    for rid, key in acc:
        g, cell, _ = info[rid]
        want = level_coefficient(target, key) * (planted[g] + shift.get(g, 0.0))
        devs.setdefault((g, key), []).append(want - prior_without(g, cell, key))
    sq: dict[tuple, list[float]] = {key: [] for key in LEVELS}
    for (_, key), ds in devs.items():
        if len(ds) >= 2:  # as the estimator, groups of at least two races at the level
            sq[key].append((sum(ds) / len(ds)) ** 2)
    pooled = sum(sum(v) for v in sq.values()) / sum(len(v) for v in sq.values())
    return {key: sum(v) / len(v) for key, v in sq.items()}, pooled


def lam(name: str) -> float:
    """The theory's share of the day variance in the planted level's prior error (module doc)."""
    _, _, per_group, cells_per_group, groups_per_cell, _, _, _ = LAYOUTS[name]
    r = per_group // cells_per_group
    n = r * groups_per_cell
    return (n - r) * r / (n - r + PRIOR_RACES) ** 2 / cells_per_group


class Run(NamedTuple):
    real: float  # the planted u's realised sd
    est: object  # estimate_day_variance
    no_rail: object  # the same without the rail column (rail layout only)
    sizes: Counter  # races per cell
    shift: dict  # rail effect per group
    oracle: dict  # (b)'s variance per level on the planted truth
    oracle_pooled: float


@functools.cache
def runs(name: str) -> list[Run]:
    """One Run per seed of the layout. Frames are dropped after each seed."""
    out = []
    for seed in LAYOUTS[name][0]:
        races, runners, planted, shift = small_cells_history(seed, name)
        est = estimate_day_variance(races, runners, train_end=TRAIN_END)
        no_rail = None
        if LAYOUTS[name][5] is not None:
            no_rail = estimate_day_variance(
                races.drop("rail_setting"), runners, train_end=TRAIN_END
            )
        sizes = Counter(
            races.select("venue", "surface", "distance_m", "going", "rail_setting").iter_rows()
        )
        oracle, oracle_pooled = predictive_oracle(races, runners, planted, shift,
                                                  LAYOUTS[name][1])  # fmt: skip
        out.append(Run(realized_sd(planted), est, no_rail, sizes, shift, oracle, oracle_pooled))
    return out


def mean_ratio(name: str, sd_of, scale: float = 1.0) -> float:
    """The layout's estimate, summed over its seeds, over its realised sd times `scale`."""
    rs = runs(name)
    return sum(sd_of(r) for r in rs) / (scale * sum(r.real for r in rs))


def oracle_ratio(name: str, sd_of, oracle_of) -> float:
    """The layout's estimate, summed over its seeds, over the oracle sd summed likewise."""
    rs = runs(name)
    return sum(sd_of(r) for r in rs) / sum(math.sqrt(oracle_of(r)) for r in rs)


def check_layout(name: str) -> None:
    """Fixture: many small cells, of the sizes the layout says."""
    _, _, per_group, cells_per_group, groups_per_cell, _, _, _ = LAYOUTS[name]
    want = per_group // cells_per_group * groups_per_cell
    for run in runs(name):
        assert abs(run.real - SD) <= 0.1 * SD, f"fixture: realised sd {run.real}"
        counts = Counter(run.sizes.values())
        assert len(run.sizes) >= 4 * 50, f"fixture: only {len(run.sizes)} cells"
        assert counts.most_common(1)[0][0] == want, f"fixture: cell sizes {counts}"
        assert counts[want] >= 0.95 * len(run.sizes), f"fixture: cell sizes {counts}"


def check_shape(est) -> None:
    assert set(est.per_level) == set(LEVELS), sorted(est.per_level)
    for key, sd in est.per_level.items():
        assert isinstance(sd, float) and math.isfinite(sd) and sd >= 0, (key, sd)
    assert isinstance(est.pooled, float) and math.isfinite(est.pooled) and est.pooled >= 0


def check_predictive_shape(est) -> None:
    assert set(est.predictive_per_level) == set(LEVELS), sorted(est.predictive_per_level)
    for key, sd in est.predictive_per_level.items():
        assert isinstance(sd, float) and math.isfinite(sd) and sd >= 0, (key, sd)
    pooled = est.predictive_pooled
    assert isinstance(pooled, float) and math.isfinite(pooled) and pooled >= 0, pooled


# ---------------------------------------------------------------------------------------------
# Broad: the planted day sd is recovered over many small cells
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cells30", "cells30_split", "cells12"])
def test_many_small_cells_recover_the_planted_day_sd(name):
    """Cells of 30 or 12 races, the rail setting part of each: the planted level's day sd, over
    the layout's four seeds, is the planted one within TOL, not shrunk by the cell means fitted
    from the same days."""
    check_layout(name)
    target = LAYOUTS[name][1]
    for run in runs(name):
        check_shape(run.est)
    got = mean_ratio(name, lambda r: r.est.per_level[target])
    per_seed = [round(r.est.per_level[target] / r.real, 3) for r in runs(name)]
    assert got == pytest.approx(1.0, abs=TOL), (name, target, got, per_seed)


@pytest.mark.parametrize("name", ["cells30", "cells30_split"])
def test_many_small_cells_per_level_and_pooled(name):
    """Each level its own day sd over small cells: the planted level SD, the other levels of its
    factor SD / 2, the other factor's levels about 0, and the pooled sd the root of the levels'
    mean variance, (1 + k / 4) / 7 of SD^2 with k the planted factor's other levels."""
    check_layout(name)
    target = LAYOUTS[name][1]
    same = [k for k in LEVELS if k[0] == target[0] and k != target]
    other = [k for k in LEVELS if k[0] != target[0]]
    assert mean_ratio(name, lambda r: r.est.per_level[target]) == pytest.approx(1.0, abs=TOL)
    for key in same:
        got = mean_ratio(name, lambda r, key=key: r.est.per_level[key], scale=0.5)
        assert got == pytest.approx(1.0, abs=TOL_HALF), (name, key, got)
    for run in runs(name):
        for key in other:
            assert run.est.per_level[key] <= null_bound(key), (name, key, run.est.per_level)
    pooled_scale = math.sqrt((1 + len(same) / 4) / len(LEVELS))
    got = mean_ratio(name, lambda r: r.est.pooled, scale=pooled_scale)
    assert got == pytest.approx(1.0, abs=TOL_POOLED), (name, got, pooled_scale)


# ---------------------------------------------------------------------------------------------
# The rail setting's effect stays in the cell
# ---------------------------------------------------------------------------------------------


def test_a_rail_setting_effect_is_not_read_as_day_variance():
    """The level's true mean moves by up to 0.12 with the rail setting, a between-group spread
    near the day effect's own. Cells split by rail setting keep it out of the day sd, which is
    the planted one within TOL; the same history without its rail settings reads it as day
    variance, a clearly larger sd."""
    name = "rail"
    check_layout(name)
    target = LAYOUTS[name][1]
    for run in runs(name):
        effects = list(run.shift.values())
        mean = sum(effects) / len(effects)
        spread = math.sqrt(sum((e - mean) ** 2 for e in effects) / len(effects))
        assert spread >= 0.7 * SD, f"fixture: rail effect sd {spread}"
        # without the rail setting, a (venue, surface, distance, going) mixes rail settings
        rails: dict[tuple, set] = {}
        for venue, surface, distance, going, rail in run.sizes:
            rails.setdefault((venue, surface, distance, going), set()).add(rail)
        mixed = sum(len(v) for v in rails.values()) / len(rails)
        assert mixed >= 2.5, f"fixture: {mixed} rail settings per merged cell"
    got = mean_ratio(name, lambda r: r.est.per_level[target])
    per_seed = [round(r.est.per_level[target] / r.real, 3) for r in runs(name)]
    assert got == pytest.approx(1.0, abs=TOL), (got, per_seed)
    for run in runs(name):
        check_shape(run.est)
        assert run.no_rail.per_level[target] >= 1.15 * run.est.per_level[target], run


# ---------------------------------------------------------------------------------------------
# (b): the predictive sd, the deviation from the prior fitted without the day
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_predictive_sd_is_the_deviation_from_the_prior_fitted_without_the_day(name):
    """Over many small cells (and with rail effects in the cells), the predictive sd per level
    and pooled is the sd of a day's deviation from the prior refitted without it: within
    TOL_PRED of the oracle at the planted level, TOL_PRED_HALF at its factor's other levels,
    TOL_PRED_POOLED pooled; and at least the day sd (a) at the planted level and pooled."""
    check_layout(name)
    target = LAYOUTS[name][1]
    same = [k for k in LEVELS if k[0] == target[0] and k != target]
    for run in runs(name):
        # fixture: the oracle carries the theory's prior error on top of the realised sd
        excess = run.oracle[target] / run.real**2 - 1
        assert lam(name) <= excess <= lam(name) + 0.25, f"fixture: oracle excess {excess}"
    for run in runs(name):
        check_predictive_shape(run.est)
    per_seed = [
        round(r.est.predictive_per_level[target] / math.sqrt(r.oracle[target]), 3)
        for r in runs(name)
    ]
    got = oracle_ratio(name, lambda r: r.est.predictive_per_level[target],
                       lambda r: r.oracle[target])  # fmt: skip
    assert got == pytest.approx(1.0, abs=TOL_PRED), (name, target, got, per_seed)
    for key in same:
        got = oracle_ratio(name, lambda r, key=key: r.est.predictive_per_level[key],
                           lambda r, key=key: r.oracle[key])  # fmt: skip
        assert got == pytest.approx(1.0, abs=TOL_PRED_HALF), (name, key, got)
    got = oracle_ratio(name, lambda r: r.est.predictive_pooled, lambda r: r.oracle_pooled)
    assert got == pytest.approx(1.0, abs=TOL_PRED_POOLED), (name, got)
    # (b) adds the prior mean's error to (a): never below it, over the layout's seeds
    rs = runs(name)
    b_target = sum(r.est.predictive_per_level[target] for r in rs)
    assert b_target >= sum(r.est.per_level[target] for r in rs), (name, b_target)
    b_pooled = sum(r.est.predictive_pooled for r in rs)
    assert b_pooled >= sum(r.est.pooled for r in rs), (name, b_pooled)


@functools.cache
def one_history():
    races, runners, _, _ = small_cells_history(LAYOUTS["cells12"][0][0], "cells12")
    return races, runners


@pytest.mark.parametrize("mode", ["per_level", "pooled", None])
def test_an_estimated_prior_uses_the_predictive_sd_and_keeps_both_estimates(mode):
    """fit_bias_prior's estimated modes put (b) in the prior's day sd (floored as #224 rules),
    per level or pooled, and carry the estimate as estimate_day_variance returns it, (a) and (b)
    both, for the report."""
    races, runners = one_history()
    est = estimate_day_variance(races, runners, train_end=TRAIN_END)
    check_shape(est)
    check_predictive_shape(est)
    prior = fit_bias_prior(races, runners, train_end=TRAIN_END, day_sd=mode)
    if mode == "per_level":
        want = dict(est.predictive_per_level)
    else:
        want = dict.fromkeys(LEVELS, est.predictive_pooled)
    assert set(prior.day_sd) == set(LEVELS), sorted(prior.day_sd)
    for key in LEVELS:
        sd = prior.day_sd[key]
        if want[key] >= FLOOR_ZONE:
            assert sd == pytest.approx(want[key], rel=1e-9), (mode, key, sd, want[key])
        else:
            assert FLOOR <= sd <= FLOOR_ZONE, (mode, key, sd, want[key])
    got = prior.estimate
    assert got.per_level == pytest.approx(est.per_level, abs=1e-12)
    assert got.pooled == pytest.approx(est.pooled, abs=1e-12)
    assert got.predictive_per_level == pytest.approx(est.predictive_per_level, abs=1e-12)
    assert got.predictive_pooled == pytest.approx(est.predictive_pooled, abs=1e-12)
