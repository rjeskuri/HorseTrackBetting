"""The within-day track bias's day variance, estimated per level and pooled (ticket #224).

Contract pinned here (`horsetrack.countries.japan.handicap.track_bias`), on top of #138's
(tests/countries/japan/handicap/test_track_bias.py, unchanged):

- `estimate_day_variance(races, runners, *, train_end) -> DayVariance`: the sd of a day's
  deviation from the history prior, from the race days on or before `train_end` only (later rows
  are ignored; a holdout race day on or before it is a ValueError, D14). `.per_level` maps each
  of the seven (factor, level) keys (draw inner/middle/outer, style front/stalker/midpack/closer)
  to a finite day sd >= 0; `.pooled` is one finite day sd >= 0 for every level.
  The estimate is what a (day, venue, surface) group's races share beyond the race-to-race
  spread of the level's race means within the group, so it must not lean on #138's
  race-independence (places-shuffled) null: a level whose race means barely vary in some groups
  (the "pinned" style case below, far below that null) still gets its planted day variance.
- `fit_bias_prior(..., day_sd=...)`: `None` (as before, estimated) is the same as `"pooled"`,
  every level's prior sd the pooled estimate; `"per_level"` gives each level its own estimate; a
  float > 0 is every level's prior sd (#138, unchanged); a mapping from all seven (factor, level)
  keys to floats > 0 gives each level that sd. Anything else (an unknown string, a missing level,
  a value <= 0) is a ValueError.
- `within_day_posterior` / `track_bias_features` use each level's own prior sd (`prior_sd`).
- `track_bias_report(..., day_sd=...)` passes the same values to every fold's prior.

Rulings (reversible choices this file pins):

- Ruling: the estimator is `track_bias.estimate_day_variance`, returning an object with
  `.per_level` (dict keyed by (factor, level)) and `.pooled` (float), both unfloored sds >= 0.
- Ruling: the day-sd modes are the strings "pooled" and "per_level"; None means "pooled".
- Ruling: a fitted prior's level sd equals the estimate whenever the estimate is at least 0.02;
  below that it may be floored, but stays in (0, 0.02].

Tolerances. The planted effect is compared with the planted effects' realised sd (around each
venue and surface's mean, which the prior absorbs), not the nominal sd. A per-group nested moment
estimator (mean over groups of d^2 - s^2 / r: d a group's mean race mean, s^2 the variance of its
race means, r its races), run over six seeds per layout while writing these tests, landed within
4% (sd 0.10) to 8% (sd 0.05, and unequal layouts) of the realised sd, a relative sd of about 3%
and 5%. The tolerances are 12% at sd 0.10, 20% at sd 0.05 and 25% for the unequal layouts: some
four standard errors, room for a less efficient but valid estimator, and still far from both 0 and
the other planted value. With no day effect the same estimator's variance scattered around 0 with
a standard error near 2e-4 for levels of three or more runners a race (8e-4 for front, one runner
a race); its sd, the root of a positive excursion, stayed below 0.03 (0.06 for front) over all
seeds, so "about 0" is <= 0.035 (0.07 for front).

Data: synthetic only (tests/countries/japan/handicap/day_variance_store.py), in memory or under
tmp_path; nothing reads $HORSETRACK_DATA_DIR.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import functools
import math
import random
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from day_variance_store import (
    draw_group,
    planted_history,
    planted_race,
    realized_sd,
    style_of,
)
from handicap_store import Store, T, at_jst, patch_holdout, weekend_days
from horsetrack.core.eval.folds import FoldScheme
from horsetrack.countries.japan.handicap import track_bias
from horsetrack.countries.japan.handicap.track_bias import (
    estimate_day_variance,
    fit_bias_prior,
    track_bias_features,
    within_day_posterior,
)
from horsetrack.countries.japan.handicap.track_bias_report import track_bias_report

DRAW_LEVELS = ("inner", "middle", "outer")
STYLE_LEVELS = ("front", "stalker", "midpack", "closer")
LEVELS = [("draw", lv) for lv in DRAW_LEVELS] + [("style", lv) for lv in STYLE_LEVELS]
FRONT = ("style", "front")
MIDPACK = ("style", "midpack")
INNER = ("draw", "inner")

TRAIN_END = date(2025, 6, 8)  # T (2025-06-15) is after it
DAYS = weekend_days(date(2023, 1, 7), TRAIN_END)  # 254 days, 1016 (day, venue, surface) groups

TOL = {0.05: 0.20, 0.10: 0.12}  # relative, by planted sd (see the module doc)
TOL_UNEQUAL = 0.25
FLOOR_ZONE = 0.02  # below this a fitted prior's sd may be floored


def null_bound(key) -> float:
    """The bound on "about 0" for a level with no day effect (see the module doc)."""
    return 0.07 if key == FRONT else 0.035


# ---------------------------------------------------------------------------------------------
# Synthetic histories (built once per layout; frames only)
# ---------------------------------------------------------------------------------------------


def frames(s: Store) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.DataFrame(s.races, infer_schema_length=None),
        pl.DataFrame(s.runners, infer_schema_length=None),
    )


def _pinned_small(g: int, rng: random.Random) -> int:
    return 2 if g % 2 == 0 else 8


def _pinned_mode(g: int) -> str:
    return "pinned" if g % 2 == 0 else "extreme"


def _uneven_races(g: int, rng: random.Random) -> int:
    return rng.choice([1, 2, 3, 4, 6, 8])


def _uneven_field(rng: random.Random) -> int:
    return rng.randint(8, 16)


LAYOUTS = {
    # name: (seed, target, planted sd, planted_history keywords)
    "midpack_05": (101, MIDPACK, 0.05, {}),
    "midpack_10": (102, MIDPACK, 0.10, {}),
    "inner_05": (103, INNER, 0.05, {}),
    "inner_10": (104, INNER, 0.10, {}),
    "none": (105, MIDPACK, 0.0, {}),
    "none_uneven": (106, MIDPACK, 0.0, dict(races=_uneven_races, field=_uneven_field)),
    # midpack pinned (race means barely vary) in the two-race groups, above a random order's
    # variance in the eight-race groups
    "pinned_05": (107, MIDPACK, 0.05, dict(races=_pinned_small, mode=_pinned_mode)),
    "pinned_10": (108, MIDPACK, 0.10, dict(races=_pinned_small, mode=_pinned_mode)),
    "uneven_midpack": (109, MIDPACK, 0.08, dict(races=_uneven_races, field=_uneven_field)),
    "uneven_inner": (110, INNER, 0.08, dict(races=_uneven_races, field=_uneven_field)),
}


@functools.cache
def world(name: str):
    seed, target, sd, kw = LAYOUTS[name]
    s, planted, sizes = planted_history(seed, DAYS, target=target, sd=sd, **kw)
    races, runners = frames(s)
    return races, runners, planted, sizes


@functools.cache
def estimate(name: str):
    races, runners, _, _ = world(name)
    return estimate_day_variance(races, runners, train_end=TRAIN_END)


def planted_of(name: str) -> tuple[tuple[str, str], float, float]:
    """(target, nominal sd, realised sd) of a layout."""
    _, target, sd, _ = LAYOUTS[name]
    got = realized_sd(world(name)[2])
    if sd > 0:
        assert abs(got - sd) <= 0.1 * sd, f"fixture: realised sd {got} for nominal {sd}"
    return target, sd, got


def check_shape(est) -> None:
    assert set(est.per_level) == set(LEVELS), sorted(est.per_level)
    for key, sd in est.per_level.items():
        assert isinstance(sd, float) and math.isfinite(sd) and sd >= 0, (key, sd)
    assert isinstance(est.pooled, float) and math.isfinite(est.pooled) and est.pooled >= 0


# ---------------------------------------------------------------------------------------------
# Broad: a planted day variance is recovered; none gives about 0
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["midpack_05", "midpack_10", "inner_05", "inner_10"])
def test_planted_day_variance_is_recovered(name):
    target, sd, want = planted_of(name)
    est = estimate(name)
    check_shape(est)
    got = est.per_level[target]
    assert got == pytest.approx(want, rel=TOL[sd]), (name, target, got, want)


@pytest.mark.parametrize("name", ["none", "none_uneven"])
def test_no_planted_day_variance_gives_about_zero_at_every_level(name):
    est = estimate(name)
    check_shape(est)
    for key in LEVELS:
        assert est.per_level[key] <= null_bound(key), (name, key, est.per_level[key])
    assert est.pooled <= 0.035, (name, est.pooled)


# ---------------------------------------------------------------------------------------------
# Per level: each level its own day variance; pooled in between
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["midpack_10", "inner_10"])
def test_each_level_gets_its_own_day_variance_and_pooled_sits_between(name):
    """The target level carries the planted sd; the other levels of its factor (four runners
    of twelve against the target's four of twelve) carry half of it, since a race's scores sum
    to 0; the other factor's levels carry none."""
    target, sd, want = planted_of(name)
    est = estimate(name)
    check_shape(est)
    factor = target[0]
    same = [k for k in LEVELS if k[0] == factor and k != target]
    other = [k for k in LEVELS if k[0] != factor]
    assert est.per_level[target] == pytest.approx(want, rel=TOL[sd]), est.per_level
    for key in same:
        # front: one runner a race, the noisiest level
        tol = 0.5 if key == FRONT else 0.2
        assert est.per_level[key] == pytest.approx(want / 2, rel=tol), (key, est.per_level)
    for key in other:
        assert est.per_level[key] <= null_bound(key), (key, est.per_level)
    none = max(est.per_level[k] for k in other if k != FRONT)
    assert none < est.pooled < est.per_level[target], (est.pooled, est.per_level)
    assert 0.3 * want <= est.pooled <= 0.7 * want, (est.pooled, want)


# ---------------------------------------------------------------------------------------------
# The style failure case: race means far below the places-shuffled null
# ---------------------------------------------------------------------------------------------


def race_means(runners: pl.DataFrame, key) -> dict[str, tuple[float, int, int]]:
    """race_id -> (the level's mean score, its runners c, finishers nf), by #138's rulings."""
    rows: dict[str, list[tuple[int, int, int]]] = {}
    for race_id, draw, pos, corners in runners.select(
        "race_id", "draw", "finish_pos", "running_positions"
    ).iter_rows():
        rows.setdefault(race_id, []).append((draw, pos, corners[0]))
    out = {}
    for race_id, rs in rows.items():
        n = len(rs)
        level = [
            (n - pos) / (n - 1) - 0.5
            for draw, pos, p in rs
            if (draw_group(draw, n) if key[0] == "draw" else style_of(p, n)) == key[1]
        ]
        out[race_id] = (sum(level) / len(level), len(level), n)
    return out


@pytest.mark.parametrize("name", ["pinned_05", "pinned_10"])
def test_level_far_below_the_shuffled_null_still_recovers_its_day_variance(name):
    """Midpack runners sit on a fixed block of places in every race of the two-race groups (their
    race means vary far less than a random order of the race's places allows) and spread more
    than a random order in the eight-race groups. #138's estimator, which scales the shuffled
    null to the pooled race-to-race spread, floors here; the day variance is still recovered."""
    target, sd, want = planted_of(name)
    races, runners, planted, sizes = world(name)
    # fixture: the two-race groups' race means are far below the places-shuffled null
    means = race_means(runners, target)
    group_of = {
        rid: (d, v, s) for rid, d, v, s in races.select("race_id", "race_date", "venue",
                                                        "surface").iter_rows()
    }  # fmt: skip
    by_group: dict[tuple, list[tuple[float, int, int]]] = {}
    for rid, m in means.items():
        by_group.setdefault(group_of[rid], []).append(m)
    ratio = {2: [], 8: []}
    for g, ms in by_group.items():
        r = sizes[g]
        centre = sum(m for m, _, _ in ms) / r
        spread = sum((m - centre) ** 2 for m, _, _ in ms) / (r - 1)
        # the variance of the mean of c of nf places drawn without replacement
        null = sum((nf + 1) / (12 * (nf - 1)) * (nf - c) / ((nf - 1) * c) for _, c, nf in ms) / r
        ratio[r].append((spread, null))
    pinned = sum(s for s, _ in ratio[2]) / sum(n for _, n in ratio[2])
    free = sum(s for s, _ in ratio[8]) / sum(n for _, n in ratio[8])
    assert pinned < 0.25 and free > 1.0, f"fixture: spread / null {pinned}, {free}"

    est = estimate(name)
    check_shape(est)
    assert est.per_level[target] == pytest.approx(want, rel=TOL[sd]), (est.per_level, want)
    for key in [("draw", lv) for lv in DRAW_LEVELS]:
        assert est.per_level[key] <= null_bound(key), (key, est.per_level)


# ---------------------------------------------------------------------------------------------
# Unequal races per group and runners per level
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["uneven_midpack", "uneven_inner"])
def test_unequal_races_and_runners_do_not_bias_the_estimate(name):
    """One to eight races a group (some groups a single race) and fields of 8 to 16, so the
    level's runners per race vary too."""
    target, sd, want = planted_of(name)
    _, runners, _, sizes = world(name)
    counts = {c for _, c, _ in race_means(runners, target).values()}
    assert len(set(sizes.values())) == 6 and len(counts) >= 3, f"fixture: {counts}"
    est = estimate(name)
    check_shape(est)
    assert est.per_level[target] == pytest.approx(want, rel=TOL_UNEQUAL), (est.per_level, want)
    other = [k for k in LEVELS if k[0] != target[0]]
    for key in other:
        assert est.per_level[key] <= null_bound(key), (key, est.per_level)


# ---------------------------------------------------------------------------------------------
# Train end, holdout, row order
# ---------------------------------------------------------------------------------------------


@functools.cache
def small():
    s, _, _ = planted_history(7, DAYS[-60:], target=MIDPACK, sd=0.10, races=4)
    return frames(s)


def test_estimate_uses_only_days_on_or_before_train_end():
    races, runners = small()
    base = estimate_day_variance(races, runners, train_end=TRAIN_END)
    # wildly different days after train_end (still before T) change nothing
    later = [TRAIN_END + timedelta(days=k) for k in range(1, 7)]
    s2, _, _ = planted_history(8, later, target=INNER, sd=0.15, races=8, mode="pinned")
    more_races, more_runners = frames(s2)
    races2 = pl.concat([races, more_races], how="vertical_relaxed")
    runners2 = pl.concat([runners, more_runners], how="vertical_relaxed")
    got = estimate_day_variance(races2, runners2, train_end=TRAIN_END)
    assert got.per_level == pytest.approx(base.per_level, abs=1e-12)
    assert got.pooled == pytest.approx(base.pooled, abs=1e-12)
    # ...while the same days inside the window do
    moved = estimate_day_variance(races2, runners2, train_end=later[-1])
    assert moved.per_level[INNER] > base.per_level[INNER] + 0.01, (moved.per_level, base)


def test_estimate_ignores_row_order():
    races, runners = small()
    a = estimate_day_variance(races, runners, train_end=TRAIN_END)
    b = estimate_day_variance(races.sample(fraction=1.0, shuffle=True, seed=3),
                              runners.sample(fraction=1.0, shuffle=True, seed=4),
                              train_end=TRAIN_END)  # fmt: skip
    assert b.per_level == pytest.approx(a.per_level, abs=1e-12)
    assert b.pooled == pytest.approx(a.pooled, abs=1e-12)


@pytest.mark.parametrize("entry", ["estimate", "per_level", "pooled"])
def test_new_entry_points_refuse_a_holdout_day(monkeypatch, entry):
    races, runners = small()
    held = DAYS[-3]

    def call():
        if entry == "estimate":
            return estimate_day_variance(races, runners, train_end=TRAIN_END)
        return fit_bias_prior(races, runners, train_end=TRAIN_END, day_sd=entry)

    call()  # fine before any day is held out
    patch_holdout(monkeypatch, [held], track_bias)
    with pytest.raises(ValueError):
        call()


# ---------------------------------------------------------------------------------------------
# The prior's day sd per level, and the update it drives
# ---------------------------------------------------------------------------------------------


def pub_time(day: date, race_no: int) -> datetime:
    return at_jst(day, 9) + timedelta(minutes=30 * race_no + 12)


def decision_for(day: date, race_no: int) -> datetime:
    return at_jst(day, 9) + timedelta(minutes=30 * race_no - 2)


def with_today(name: str, todays: list[tuple[date, tuple[str, str], float]], n_races: int = 9):
    """The layout's history plus, on each (day, target, u), n_races TOK turf races whose target
    level is pinned at mean score u."""
    races, runners, _, _ = world(name)
    s = Store()
    rng = random.Random(11)
    ids: dict[date, list[str]] = {}
    for day, target, u in todays:
        ids[day] = [planted_race(s, rng, day, venue="TOK", surface="turf", field=12,
                                 target=target, u=u, mode="pinned")
                    for _ in range(n_races)]  # fmt: skip
    r2, u2 = frames(s)
    return (
        pl.concat([races, r2], how="vertical_relaxed"),
        pl.concat([runners, u2], how="vertical_relaxed"),
        s,
        ids,
    )


def posterior_at(prior, races, runners, s: Store, race_id: str) -> dict:
    day = date.fromisoformat(race_id.split(":")[1])
    no = int(race_id.split(":")[3])
    published = {r["race_id"]: pub_time(r["race_date"], r["race_no"]) for r in s.races}
    frame = within_day_posterior(
        prior,
        races,
        runners,
        race_id=race_id,
        decision_time=decision_for(day, no),
        published=published,
    )
    return {(r["factor"], r["level"]): r for r in frame.to_dicts()}


def todays_scores(s: Store, race_ids, key) -> list[float]:
    out = []
    for rid in race_ids:
        rows = [r for r in s.runners if r["race_id"] == rid]
        n = len(rows)
        for r in rows:
            level = (draw_group(r["draw"], n) if key[0] == "draw"
                     else style_of(r["running_positions"][0], n))  # fmt: skip
            if level == key[1]:
                out.append((n - r["finish_pos"]) / (n - 1) - 0.5)
    return out


def share(row: dict, scores: list[float]) -> float:
    """The posterior's share of the way from the prior mean to today's mean score."""
    evidence = sum(scores) / len(scores) - row["prior_mean"]
    assert abs(evidence) > 0.15, f"fixture: today's evidence {evidence}"
    return (row["mean"] - row["prior_mean"]) / evidence


def test_per_level_prior_sd_is_each_levels_estimate():
    races, runners, _, _ = world("midpack_10")
    est = estimate("midpack_10")
    races_t, runners_t, s, ids = with_today("midpack_10", [(T, MIDPACK, 0.0)], n_races=3)
    for mode, want in (("per_level", est.per_level), ("pooled", dict.fromkeys(LEVELS, est.pooled))):
        prior = fit_bias_prior(races_t, runners_t, train_end=TRAIN_END, day_sd=mode)
        got = posterior_at(prior, races_t, runners_t, s, ids[T][0])  # nothing entered yet
        for key in LEVELS:
            sd = got[key]["prior_sd"]
            if want[key] >= FLOOR_ZONE:
                assert sd == pytest.approx(want[key], rel=1e-9), (mode, key, sd, want[key])
            else:
                assert 0 < sd <= FLOOR_ZONE, (mode, key, sd, want[key])
    # per level, the planted level's prior sd is its own: twice the other style levels', far
    # above the draw levels'
    prior = fit_bias_prior(races_t, runners_t, train_end=TRAIN_END, day_sd="per_level")
    got = posterior_at(prior, races_t, runners_t, s, ids[T][0])
    assert got[MIDPACK]["prior_sd"] > 3 * got[INNER]["prior_sd"]
    # None is the pooled mode
    none = fit_bias_prior(races_t, runners_t, train_end=TRAIN_END)
    pooled = fit_bias_prior(races_t, runners_t, train_end=TRAIN_END, day_sd="pooled")
    for rid in ids[T]:
        assert posterior_at(none, races_t, runners_t, s, rid) == posterior_at(
            pooled, races_t, runners_t, s, rid
        )


def test_level_without_day_variance_is_barely_updated_per_level():
    """History: day variance at midpack only (none at the draw thirds). Day T's midpack runners
    run far above their prior, day T+7's inner draws far above theirs. Per level, midpack moves
    most of the way to today's mean and inner barely moves; pooled, both move a middling share."""
    day_b = T + timedelta(days=7)
    races, runners, s, ids = with_today("midpack_10", [(T, MIDPACK, 0.3), (day_b, INNER, 0.3)])
    shares = {}
    for mode in ("per_level", "pooled"):
        prior = fit_bias_prior(races, runners, train_end=TRAIN_END, day_sd=mode)
        mid = posterior_at(prior, races, runners, s, ids[T][-1])[MIDPACK]
        inner = posterior_at(prior, races, runners, s, ids[day_b][-1])[INNER]
        assert mid["n_races"] == inner["n_races"] == 8
        shares[mode] = (
            share(mid, todays_scores(s, ids[T][:-1], MIDPACK)),
            share(inner, todays_scores(s, ids[day_b][:-1], INNER)),
        )
    (mid_l, inner_l), (mid_p, inner_p) = shares["per_level"], shares["pooled"]
    assert mid_l > 0.6, shares
    assert 0 <= inner_l < 0.25, shares
    assert mid_l > mid_p + 0.1 and inner_l < inner_p - 0.1, shares
    # the features carry the same per-level posterior
    prior = fit_bias_prior(races, runners, train_end=TRAIN_END, day_sd="per_level")
    target = ids[T][-1]
    feat = track_bias_features(prior, races, runners, race_id=target,
                               decision_time=decision_for(T, 9),
                               published={r["race_id"]: pub_time(r["race_date"], r["race_no"])
                                          for r in s.races})  # fmt: skip
    post = posterior_at(prior, races, runners, s, target)
    assert feat.height == 12
    for value in feat.get_column("style_adj_midpack"):
        assert value == pytest.approx(post[MIDPACK]["mean"], abs=1e-12)


def test_an_explicit_mapping_sets_each_levels_prior_sd():
    races, runners = small()
    sds = {key: 0.01 + 0.03 * i for i, key in enumerate(LEVELS)}
    s = Store()
    rng = random.Random(12)
    ids = [planted_race(s, rng, T, venue="TOK", surface="turf", field=12, target=INNER, u=0.3,
                        mode="pinned") for _ in range(6)]  # fmt: skip
    r2, u2 = frames(s)
    races = pl.concat([races, r2], how="vertical_relaxed")
    runners = pl.concat([runners, u2], how="vertical_relaxed")
    prior = fit_bias_prior(races, runners, train_end=TRAIN_END, day_sd=sds)
    first = posterior_at(prior, races, runners, s, ids[0])
    for key in LEVELS:
        assert first[key]["prior_sd"] == pytest.approx(sds[key], abs=1e-12), key
        assert first[key]["sd"] == pytest.approx(sds[key], abs=1e-12), key
    # a larger prior sd at a level lets the same evidence move it further
    narrow = fit_bias_prior(races, runners, train_end=TRAIN_END, day_sd={**sds, INNER: 0.01})
    wide = fit_bias_prior(races, runners, train_end=TRAIN_END, day_sd={**sds, INNER: 0.3})
    a = posterior_at(narrow, races, runners, s, ids[-1])[INNER]
    b = posterior_at(wide, races, runners, s, ids[-1])[INNER]
    assert b["mean"] - b["prior_mean"] > 3 * (a["mean"] - a["prior_mean"]) > 0, (a, b)


BAD_DAY_SD = {
    "unknown_mode": "per-level",
    "missing_level": {key: 0.05 for key in LEVELS if key != FRONT},
    "zero_level": {**{key: 0.05 for key in LEVELS}, INNER: 0.0},
    "negative_level": {**{key: 0.05 for key in LEVELS}, MIDPACK: -0.1},
}


@pytest.mark.parametrize("case", list(BAD_DAY_SD))
def test_a_bad_day_sd_is_refused(case):
    races, runners = small()
    with pytest.raises(ValueError):
        fit_bias_prior(races, runners, train_end=TRAIN_END, day_sd=BAD_DAY_SD[case])


@pytest.mark.parametrize("mode", ["per_level", "pooled"])
def test_a_prior_fitted_through_the_target_day_is_refused(mode):
    races, runners, s, ids = with_today("none", [(T, MIDPACK, 0.0)], n_races=3)
    prior = fit_bias_prior(races, runners, train_end=T, day_sd=mode)
    with pytest.raises(ValueError):
        posterior_at(prior, races, runners, s, ids[T][-1])


# ---------------------------------------------------------------------------------------------
# The development-data report takes the mode
# ---------------------------------------------------------------------------------------------

WEEKLY = FoldScheme(unit="week", test_length=1, step=1, min_train=4, window="expanding",
                    embargo_days=1, anchor_month=1)  # fmt: skip


def strings(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from strings(k)
            yield from strings(v)
    elif isinstance(node, list | tuple):
        for v in node:
            yield from strings(v)
    elif isinstance(node, str):
        yield node


def test_report_measures_the_pooled_and_the_per_level_modes(tmp_path, monkeypatch):
    days = weekend_days(date(2025, 1, 4), date(2025, 3, 30))
    s, _, _ = planted_history(5, days, target=MIDPACK, sd=0.10, races=4, mode="pinned")
    s.write(tmp_path, monkeypatch)
    patch_holdout(monkeypatch, [], track_bias)
    out = {
        mode: track_bias_report(days[0], days[-1], scheme=WEEKLY, n_boot=20, day_sd=mode)
        for mode in (None, "pooled", "per_level")
    }
    for mode, rep in out.items():
        assert rep["races"] > 100, mode
        gain = rep["gain_update_vs_prior"]
        assert math.isfinite(gain["mean"]) and gain["races"] == rep["races"], mode
        assert not [x for x in strings(rep) if x.startswith("jra:")], mode  # aggregates only
    # the prior alone does not depend on the day sd; the update does, and None is pooled
    assert out["per_level"]["ll_prior"] == pytest.approx(out["pooled"]["ll_prior"], rel=1e-12)
    assert abs(out["per_level"]["ll_update"] - out["pooled"]["ll_update"]) > 1e-6
    assert out[None]["ll_update"] == pytest.approx(out["pooled"]["ll_update"], rel=1e-12)
