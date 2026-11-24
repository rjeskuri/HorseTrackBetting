"""Within-day track bias from the races already run that day (ticket #138).

Contract pinned here (`horsetrack.countries.japan.handicap.track_bias`). Every function takes
store-shaped polars frames (`races` and `runners` rows as in docs/specs/store.md) and reads nothing
else; the caller scopes the store.

- `fit_bias_prior(races, runners, *, train_end, day_sd=None) -> prior`: the history prior, fitted
  on the race days on or before `train_end` only (later rows are ignored). A holdout race day
  (`horsetrack.core.eval.holdout.is_holdout_day("japan", d)`) on or before `train_end` is a
  ValueError (D14). The prior is opaque to callers; it only goes back into the two calls below.
  Per cell (venue, surface, distance, course_config, going) it holds each level's mean effect;
  `day_sd` is the prior sd of today's effect at every level (None: estimated from history; any
  estimate is > 0).
- `within_day_posterior(prior, races, runners, *, race_id, decision_time, published) -> frame`:
  seven rows, one per (factor, level): factor "draw" with levels inner, middle, outer, and factor
  "style" with levels front, stalker, midpack, closer. Columns `factor, level, prior_mean,
  prior_sd, mean, sd, n_races, n_obs`. `published` maps race_id -> the tz-aware instant its result
  was officially published; a race missing from it is unpublished.
- `track_bias_features(prior, races, runners, *, race_id, decision_time, published) -> frame`: one
  row per declared (not scratched) runner of the race: `race_id, horse_no, horse_id, draw_group,
  draw_adj, draw_adj_sd`, `style_adj_<level>` and `style_adj_<level>_sd` for the four styles, and
  `bias_races_n` (the posterior's n_races). draw_adj is the posterior mean of the runner's draw
  group.

Rulings (reversible choices this file pins):

- Ruling: the effect is measured on the percentage of rivals beaten minus one half. A finisher in
  place `pos` among `nf` finishers (nf >= 2) scores (nf - pos) / (nf - 1) - 0.5; 0 means no bias.
  Non-finishers and scratched runners give no observation; a race with fewer than 2 finishers gives
  none.
- Ruling: draw group as #129: inner if 3(draw-1) <= n-1, middle if 3(draw-1) <= 2(n-1), else
  outer; running style as #129 from the first-corner position p: front if p == 1, stalker if
  3(p-1) <= n-1, midpack if 3(p-1) <= 2(n-1), else closer. n = the race's declared runners. A
  finisher without corner positions gives a draw observation only.
- Ruling: a race enters today's update when it is on the target's race day, venue and surface,
  has a smaller race_no than the target, and was published strictly before `decision_time`
  (instants compared, whatever the time zones). Nothing else ever enters: not the target itself,
  not a later race, another venue or surface, another day, nor a race published at or after the
  decision time. `n_races` counts the races that entered, `n_obs` that level's observations.
- Ruling: the update is a conjugate normal update of today's deviation from the prior. Each
  observation's evidence is its score minus the prior mean of its own race's cell (so a race of
  another distance on the same course counts as a deviation from that distance's prior). With no
  races entered the posterior is the prior exactly; the sd never grows with more races; and the
  posterior mean lies between the prior mean and the prior mean plus the mean evidence, closer to
  the latter as observations accumulate.
- Ruling: with no history at the target's venue and surface the prior mean is 0 at every level.
- Ruling: a decision time or a publication time without a time zone is a ValueError; a target on
  a holdout day, or a prior whose train_end is on or after the target's race day, is a ValueError.

Data: synthetic only, built with tests/countries/japan/handicap/handicap_store.py, in memory.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import copy
import math
import random
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from handicap_store import R, Store, T, at_jst, patch_holdout, weekend_days
from horsetrack.countries.japan.handicap import track_bias
from horsetrack.countries.japan.handicap.track_bias import (
    fit_bias_prior,
    track_bias_features,
    within_day_posterior,
)

DRAW_LEVELS = ("inner", "middle", "outer")
STYLE_LEVELS = ("front", "stalker", "midpack", "closer")
LEVELS = [("draw", lv) for lv in DRAW_LEVELS] + [("style", lv) for lv in STYLE_LEVELS]
EXTREMES = [("draw", "inner"), ("draw", "outer"), ("style", "front"), ("style", "closer")]
POSTERIOR_COLUMNS = ["factor", "level", "prior_mean", "prior_sd", "mean", "sd", "n_races", "n_obs"]

TRAIN_END = date(2025, 6, 8)  # every history day is on or before it; T (2025-06-15) is after
HISTORY = weekend_days(date(2025, 1, 4), TRAIN_END)
CELL = dict(venue="TOK", surface="turf", distance=1600, going="GOOD", course="A")

NEUTRAL = ({}, {})
# (draw log-strengths, style log-strengths) of a planted bias
INNER_FRONT = ({"inner": 1.5, "outer": -1.5}, {"front": 1.2, "stalker": 0.4, "midpack": -0.4,
                                               "closer": -1.2})  # fmt: skip
OUTER_CLOSER = ({"inner": -1.5, "outer": 1.5}, {"front": -1.2, "stalker": -0.4, "midpack": 0.4,
                                                "closer": 1.2})  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Independent derivations (the rulings above, written out by hand)
# ---------------------------------------------------------------------------------------------


def draw_group(draw: int, n: int) -> str:
    if 3 * (draw - 1) <= n - 1:
        return "inner"
    if 3 * (draw - 1) <= 2 * (n - 1):
        return "middle"
    return "outer"


def style_of(p: int, n: int) -> str:
    if p == 1:
        return "front"
    if 3 * (p - 1) <= n - 1:
        return "stalker"
    if 3 * (p - 1) <= 2 * (n - 1):
        return "midpack"
    return "closer"


def observations(s: Store, race_ids) -> dict[tuple[str, str], list[float]]:
    """Each level's scores over the races, by the rulings above."""
    obs: dict[tuple[str, str], list[float]] = {key: [] for key in LEVELS}
    for race_id in race_ids:
        rows = [r for r in s.runners if r["race_id"] == race_id and not r["is_scratched"]]
        n = len(rows)
        fin = [r for r in rows if r["finish_pos"] is not None]
        nf = len(fin)
        if nf < 2:
            continue
        for r in fin:
            y = (nf - r["finish_pos"]) / (nf - 1) - 0.5
            obs[("draw", draw_group(r["draw"], n))].append(y)
            if len(r["running_positions"]):
                obs[("style", style_of(r["running_positions"][0], n))].append(y)
    return obs


def mean(xs) -> float:
    return sum(xs) / len(xs)


# ---------------------------------------------------------------------------------------------
# Synthetic days
# ---------------------------------------------------------------------------------------------


def biased_race(
    s: Store,
    rng: random.Random,
    day: date,
    plant=NEUTRAL,
    *,
    field: int = 12,
    race_no: int | None = None,
    scratched: int | None = None,
    dnf: bool = False,
    **cell,
) -> str:
    """A race whose finishing order is drawn Plackett-Luce from strengths exp(draw effect + style
    effect), first-corner positions a random permutation. `scratched`: that horse_no is a
    declared-then-scratched extra runner; `dnf`: the weakest runner does not finish."""
    cell = {**CELL, **cell}
    draw_beta, style_beta = plant
    corners = list(range(1, field + 1))
    rng.shuffle(corners)
    nos = [no for no in range(1, field + 2) if no != scratched][:field]
    strength = [
        math.exp(draw_beta.get(draw_group(no, field), 0.0)
                 + style_beta.get(style_of(corners[i], field), 0.0))
        for i, no in enumerate(nos)
    ]  # fmt: skip
    left, order = list(range(field)), []
    while left:
        u = rng.random() * sum(strength[i] for i in left)
        for i in left:
            u -= strength[i]
            if u <= 0:
                break
        order.append(i)
        left.remove(i)
    pos = {i: k + 1 for k, i in enumerate(order)}
    base = len(s.runners)
    runners = [
        R(f"h:{base + i}", pos=pos[i], corners=[corners[i], corners[i]], no=no)
        for i, no in enumerate(nos)
    ]
    if dnf:
        last = order[-1]
        runners[last] = R(f"h:{base + last}", pos=None, corners=[], no=nos[last])
    if scratched is not None:
        runners.append(R(f"h:{base}:scr", scratched=True, no=scratched))
    return s.race(
        day, runners, race_no=race_no, venue=cell["venue"], surface=cell["surface"],
        distance=cell["distance"], going=cell["going"], course=cell["course"],
    )  # fmt: skip


def cyclic_race(s: Store, day: date, j: int, *, n: int = 11, race_no: int | None = None) -> str:
    """Race j of a balanced cycle of n: runner i draws i+1, finishes (i+j) mod n + 1 and is
    (i+2j) mod n + 1 at the first corner. Over a full cycle every draw and every first-corner
    position takes every place once, so every level's mean score is exactly 0."""
    base = len(s.runners)
    runners = [
        R(f"h:{base + i}", pos=(i + j) % n + 1, corners=[(i + 2 * j) % n + 1] * 2, no=i + 1)
        for i in range(n)
    ]
    return s.race(
        day, runners, race_no=race_no, venue=CELL["venue"], surface=CELL["surface"],
        distance=CELL["distance"], going=CELL["going"], course=CELL["course"],
    )  # fmt: skip


def history(s: Store, rng: random.Random, plant=NEUTRAL, *, days=HISTORY[-20:], per_day=4, **cell):
    ids = []
    for day in days:
        for _ in range(per_day):
            ids.append(biased_race(s, rng, day, plant, **cell))
    return ids


def frames(s: Store) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.DataFrame(s.races, infer_schema_length=None),
        pl.DataFrame(s.runners, infer_schema_length=None),
    )


def pub_time(day: date, race_no: int) -> datetime:
    """Official result of race k: 09:00 + 30k + 12 min local, after its post."""
    return at_jst(day, 9) + timedelta(minutes=30 * race_no + 12)


def decision_for(day: date, race_no: int) -> datetime:
    """Two minutes before race k's post (09:00 + 30k local): after race k-1's result is out."""
    return at_jst(day, 9) + timedelta(minutes=30 * race_no - 2)


def published_all(s: Store) -> dict[str, datetime]:
    return {r["race_id"]: pub_time(r["race_date"], r["race_no"]) for r in s.races}


def prior_of(s: Store, *, day_sd=None, train_end=TRAIN_END):
    races, runners = frames(s)
    return fit_bias_prior(races, runners, train_end=train_end, day_sd=day_sd)


def posterior(prior, s: Store, target: str, *, published=None, decision=None) -> pl.DataFrame:
    races, runners = frames(s)
    day = date.fromisoformat(target.split(":")[1])
    no = int(target.split(":")[3])
    return within_day_posterior(
        prior, races, runners, race_id=target,
        decision_time=decision if decision is not None else decision_for(day, no),
        published=published if published is not None else published_all(s),
    )  # fmt: skip


def by_level(frame: pl.DataFrame) -> dict[tuple[str, str], dict]:
    return {(r["factor"], r["level"]): r for r in frame.to_dicts()}


def today(s: Store, rng: random.Random, plant, n_races: int, *, day=T, field=12, **cell):
    return [biased_race(s, rng, day, plant, field=field, race_no=k, **cell)
            for k in range(1, n_races + 1)]  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Broad: the posterior's shape, and the prior it starts from
# ---------------------------------------------------------------------------------------------


def test_posterior_has_one_row_per_draw_and_style_level():
    rng = random.Random(1)
    s = Store()
    history(s, rng, INNER_FRONT)
    ids = today(s, rng, NEUTRAL, 6)
    prior = prior_of(s)
    for target in ids:
        frame = posterior(prior, s, target)
        assert isinstance(frame, pl.DataFrame)
        missing = [c for c in POSTERIOR_COLUMNS if c not in frame.columns]
        assert not missing, f"within_day_posterior lacks {missing}"
        assert sorted(frame.select("factor", "level").rows()) == sorted(LEVELS)
        k = int(target.split(":")[3])
        for row in frame.to_dicts():
            assert row["n_races"] == k - 1, (target, row)
            for col in ("prior_mean", "prior_sd", "mean", "sd"):
                assert math.isfinite(row[col]), (target, col, row)
            assert row["prior_sd"] > 0 and row["sd"] > 0, row


@pytest.mark.parametrize("day_sd", [None, 0.05, 0.3, 1.0])
def test_no_race_entered_means_the_posterior_is_the_prior(day_sd):
    rng = random.Random(2)
    s = Store()
    history(s, rng, INNER_FRONT)
    ids = today(s, rng, OUTER_CLOSER, 5)
    prior = prior_of(s, day_sd=day_sd)
    first = posterior(prior, s, ids[0])  # race 1: nothing before it
    late = posterior(prior, s, ids[4], published={})  # race 5, but nothing published
    for frame in (first, late):
        for row in frame.to_dicts():
            assert row["n_races"] == 0 and row["n_obs"] == 0, row
            assert row["mean"] == pytest.approx(row["prior_mean"], abs=1e-12), row
            assert row["sd"] == pytest.approx(row["prior_sd"], abs=1e-12), row
            if day_sd is not None:
                assert row["prior_sd"] == pytest.approx(day_sd, abs=1e-12), row
    # the prior is the same object whichever race of the cell asks
    assert_frame_equal(
        first.select("factor", "level", "prior_mean", "prior_sd").sort("factor", "level"),
        late.select("factor", "level", "prior_mean", "prior_sd").sort("factor", "level"),
    )


DIMENSIONS = [
    ("venue", {"venue": "TOK"}, {"venue": "NAK"}),
    ("surface", {"surface": "turf"}, {"surface": "dirt"}),
    ("distance", {"distance": 1200}, {"distance": 2400}),
    ("rail", {"course": "A"}, {"course": "C"}),
    ("going", {"going": "GOOD"}, {"going": "HEAVY"}),
]


@pytest.mark.parametrize(("dim", "a", "b"), DIMENSIONS, ids=[d[0] for d in DIMENSIONS])
def test_prior_recovers_the_historical_bias_of_each_cell(dim, a, b):
    """Two cells that differ only in `dim` carry opposite planted biases in history; each
    cell's prior mean has the sign of its own history and most of its size."""
    rng = random.Random(3)
    s = Store()
    hist_a = history(s, rng, INNER_FRONT, **a)
    hist_b = history(s, rng, OUTER_CLOSER, **b)
    target_a = biased_race(s, rng, T, **a)
    target_b = biased_race(s, rng, T + timedelta(days=7), **b)
    prior = prior_of(s)
    for hist, target, sign in ((hist_a, target_a, 1), (hist_b, target_b, -1)):
        got = by_level(posterior(prior, s, target, published={}))
        obs = observations(s, hist)
        for key, planted in zip(EXTREMES, (1, -1, 1, -1), strict=True):
            emp = mean(obs[key])
            assert emp * planted * sign > 0.1, f"fixture: {dim} {key} history mean {emp}"
            pm = got[key]["prior_mean"]
            assert pm * emp > 0, (dim, target, key, pm, emp)
            assert abs(pm - emp) <= 0.5 * abs(emp), (dim, target, key, pm, emp)


def test_no_history_at_the_venue_and_surface_gives_a_zero_prior():
    rng = random.Random(4)
    s = Store()
    history(s, rng, INNER_FRONT, venue="NAK")
    history(s, rng, INNER_FRONT, surface="dirt")
    target = biased_race(s, rng, T)  # TOK turf: no history there
    got = posterior(prior_of(s), s, target, published={})
    for row in got.to_dicts():
        assert row["prior_mean"] == pytest.approx(0.0, abs=1e-12), row
        assert row["prior_sd"] > 0, row


def test_prior_uses_only_days_on_or_before_train_end():
    rng = random.Random(5)
    s = Store()
    history(s, rng, NEUTRAL)
    target = biased_race(s, rng, T)
    base = posterior(prior_of(s), s, target, published={})
    # strongly biased days after train_end (still before the target day) change nothing
    later = [TRAIN_END + timedelta(days=k) for k in range(1, (T - TRAIN_END).days)]
    s2 = copy.deepcopy(s)
    history(s2, random.Random(6), OUTER_CLOSER, days=later, per_day=8)
    assert_frame_equal(posterior(prior_of(s2), s2, target, published={}), base)
    # ...while the same days inside the fit window do
    moved = posterior(prior_of(s2, train_end=later[-1]), s2, target, published={})
    inner = by_level(moved)[("draw", "inner")]["prior_mean"]
    assert inner < by_level(base)[("draw", "inner")]["prior_mean"] - 0.02


# ---------------------------------------------------------------------------------------------
# Planted bias: the posterior recovers it; no bias: it stays at the prior
# ---------------------------------------------------------------------------------------------


def test_weak_prior_posterior_recovers_todays_planted_bias():
    """History says inner and front; today runs outer and closer. Under a weak prior
    (day_sd = 1) the posterior after eleven races sits at today's own mean score."""
    rng = random.Random(7)
    s = Store()
    history(s, rng, INNER_FRONT)
    ids = today(s, rng, OUTER_CLOSER, 12)
    got = by_level(posterior(prior_of(s, day_sd=1.0), s, ids[-1]))
    obs = observations(s, ids[:-1])
    for key in LEVELS:
        row = got[key]
        assert row["n_races"] == 11 and row["n_obs"] == len(obs[key]), (key, row)
        assert row["mean"] == pytest.approx(mean(obs[key]), abs=0.02), (key, row)
        assert row["sd"] < 0.5 * row["prior_sd"], (key, row)
    for key, planted in zip(EXTREMES, (-1, 1, -1, 1), strict=True):
        row = got[key]
        assert (row["prior_mean"] - row["mean"]) * planted < -0.1, (key, row)


@pytest.mark.parametrize("plant", [OUTER_CLOSER, INNER_FRONT], ids=["outer_closer", "inner_front"])
@pytest.mark.parametrize("day_sd", [None, 0.1])
def test_posterior_moves_toward_today_as_races_accumulate(plant, day_sd):
    """Race by race, from a neutral history: the sd never grows and shrinks over the day, and
    the posterior's share of the way from the prior to today's mean score never falls."""
    rng = random.Random(8)
    s = Store()
    history(s, rng, NEUTRAL)
    ids = today(s, rng, plant, 12)
    prior = prior_of(s, day_sd=day_sd)
    seq = [by_level(posterior(prior, s, target)) for target in ids]
    for key in LEVELS:
        rows = [step[key] for step in seq]
        prior_mean = rows[0]["prior_mean"]
        assert rows[0]["mean"] == pytest.approx(prior_mean, abs=1e-12)
        for k in range(1, len(rows)):
            assert rows[k]["prior_mean"] == pytest.approx(prior_mean, abs=1e-12)
            assert rows[k]["sd"] <= rows[k - 1]["sd"] + 1e-12, (key, k)
            if rows[k]["n_obs"] > rows[k - 1]["n_obs"]:
                assert rows[k]["sd"] < rows[k - 1]["sd"], (key, k)
        assert rows[-1]["sd"] < rows[0]["sd"], key
        last_w = 0.0
        for k in range(1, len(rows)):
            obs = observations(s, ids[:k])[key]
            if not obs or abs(mean(obs) - prior_mean) < 1e-3:
                continue
            w = (rows[k]["mean"] - prior_mean) / (mean(obs) - prior_mean)
            assert -1e-9 <= w <= 1 + 1e-9, (key, k, w)
            assert w >= last_w - 1e-9, (key, k, w, last_w)
            last_w = w
        assert last_w > 0, key
    # and the planted levels end on the planted side of the prior
    for key, planted in zip(EXTREMES, (1, -1, 1, -1), strict=True):
        sign = planted if plant is INNER_FRONT else -planted
        row = seq[-1][key]
        assert (row["mean"] - row["prior_mean"]) * sign > 0, (key, row)


def test_balanced_day_stays_exactly_at_the_prior():
    """History and today are full balanced cycles: every level's mean score is exactly 0 in
    both, so the posterior mean does not move while its sd shrinks."""
    s = Store()
    for day in HISTORY[-4:]:
        for j in range(11):
            cyclic_race(s, day, j)
    for j in range(11):
        cyclic_race(s, T, j, race_no=j + 1)
    target = biased_race(s, random.Random(9), T, OUTER_CLOSER, field=11, race_no=12)
    for day_sd in (None, 0.1, 1.0):
        got = by_level(posterior(prior_of(s, day_sd=day_sd), s, target))
        for key in LEVELS:
            row = got[key]
            assert row["n_races"] == 11, row
            assert row["prior_mean"] == pytest.approx(0.0, abs=1e-9), (day_sd, row)
            assert row["mean"] == pytest.approx(row["prior_mean"], abs=1e-9), (day_sd, row)
            assert row["sd"] < row["prior_sd"], (day_sd, row)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_day_like_its_history_stays_near_the_prior(seed):
    """Same planted bias in history and today (no within-day deviation): the posterior barely
    leaves the prior; a day with the bias reversed moves it several times as far."""
    rng = random.Random(seed)
    s = Store()
    history(s, rng, INNER_FRONT)
    same = today(s, rng, INNER_FRONT, 12)
    reversed_day = today(s, rng, OUTER_CLOSER, 12, day=T + timedelta(days=7))
    prior = prior_of(s, day_sd=0.1)
    stay = by_level(posterior(prior, s, same[-1]))
    move = by_level(posterior(prior, s, reversed_day[-1]))
    for key in EXTREMES:
        d_stay = abs(stay[key]["mean"] - stay[key]["prior_mean"])
        d_move = abs(move[key]["mean"] - move[key]["prior_mean"])
        assert d_move > 0.03, (key, move[key])
        assert d_stay < 0.35 * d_move, (key, d_stay, d_move)


def test_other_distance_races_count_as_deviations_from_their_own_prior():
    """Today's 1200 m races carry exactly their history's strong inner bias; the 2000 m target's
    neutral prior moves by their deviation from the 1200 m prior, not by their raw scores."""
    rng = random.Random(13)
    s = Store()
    history(s, rng, INNER_FRONT, distance=1200)
    history(s, rng, NEUTRAL, distance=2000)
    sprint = today(s, rng, INNER_FRONT, 6, distance=1200)
    target = biased_race(s, rng, T, race_no=7, distance=2000)
    sprint_prior = by_level(
        posterior(prior_of(s, day_sd=1.0), s, biased_race(s, rng, T + timedelta(days=7),
                                                          distance=1200), published={})
    )  # fmt: skip
    got = by_level(posterior(prior_of(s, day_sd=1.0), s, target))
    obs = observations(s, sprint)
    for key in LEVELS:
        evidence = mean(obs[key]) - sprint_prior[key]["prior_mean"]
        row = got[key]
        assert row["n_races"] == 6, row
        assert row["mean"] == pytest.approx(row["prior_mean"] + evidence, abs=0.03), (key, row)
    raw = mean(obs[("draw", "inner")])
    assert abs(got[("draw", "inner")]["mean"] - raw) > 0.1, "fixture: raw pooling must differ"


# ---------------------------------------------------------------------------------------------
# Leakage: only earlier races of the same course and day, published before the decision time
# ---------------------------------------------------------------------------------------------

TARGET_NO = 8
EXTREME = ({"inner": 3.0, "outer": -3.0}, {"front": 3.0, "closer": -3.0})


def leak_base() -> tuple[Store, str, dict[str, datetime]]:
    rng = random.Random(14)
    s = Store()
    history(s, rng, NEUTRAL)
    for k in range(1, 5):
        biased_race(s, rng, T, NEUTRAL, race_no=k, dnf=k == 2, scratched=5 if k == 3 else None)
    target = biased_race(s, rng, T, NEUTRAL, race_no=TARGET_NO)
    published = {r["race_id"]: pub_time(T, r["race_no"]) for r in s.races if r["race_date"] == T}
    return s, target, published


DECISION = decision_for(T, TARGET_NO)
LEAKS = {
    "published_at_decision": (dict(race_no=6), DECISION),
    "published_after_decision": (dict(race_no=6), DECISION + timedelta(seconds=1)),
    "published_hours_later": (dict(race_no=6), DECISION + timedelta(hours=3)),
    "unpublished": (dict(race_no=6), None),
    "later_race": (dict(race_no=9), DECISION - timedelta(hours=1)),
    "other_venue": (dict(race_no=6, venue="NAK"), DECISION - timedelta(hours=1)),
    "other_surface": (dict(race_no=6, surface="dirt"), DECISION - timedelta(hours=1)),
    "previous_day": (dict(race_no=6, day=T - timedelta(days=1)), DECISION - timedelta(hours=20)),
    "next_day": (dict(race_no=6, day=T + timedelta(days=1)), DECISION - timedelta(hours=1)),
}


def add_leak(s: Store, published: dict, spec: dict, when) -> str:
    spec = dict(spec)
    day = spec.pop("day", T)
    race_id = biased_race(s, random.Random(15), day, EXTREME, **spec)
    if when is not None:
        published[race_id] = when
    return race_id


@pytest.mark.parametrize("case", list(LEAKS))
def test_ineligible_race_never_enters_the_update(case):
    s, target, published = leak_base()
    prior = prior_of(s)
    base = posterior(prior, s, target, published=published, decision=DECISION)
    base_feat = track_bias_features(*_args(prior, s), race_id=target, decision_time=DECISION,
                                    published=published)  # fmt: skip
    spec, when = LEAKS[case]
    add_leak(s, published, spec, when)
    got = posterior(prior, s, target, published=published, decision=DECISION)
    assert_frame_equal(got, base)
    feat = track_bias_features(*_args(prior, s), race_id=target, decision_time=DECISION,
                               published=published)  # fmt: skip
    assert_frame_equal(feat, base_feat)


def test_target_race_own_result_never_enters():
    s, target, published = leak_base()
    prior = prior_of(s)
    base = posterior(prior, s, target, published=published, decision=DECISION)
    published = {**published, target: DECISION - timedelta(minutes=1)}
    # make the target's result as loud as possible: inner draws first, in draw order
    for r in s.runners:
        if r["race_id"] == target:
            r["finish_pos"] = r["horse_no"]
            r["placing_raw"] = str(r["horse_no"])
            r["running_positions"] = [r["horse_no"], r["horse_no"]]
    assert_frame_equal(posterior(prior, s, target, published=published, decision=DECISION), base)


@pytest.mark.parametrize(
    ("offset", "enters"),
    [(-3600, True), (-1, True), (0, False), (1, False)],
    ids=["hour_before", "second_before", "at_decision", "second_after"],
)
@pytest.mark.parametrize("zone", ["jst", "utc"])
def test_publication_boundary_is_strictly_before_the_decision_time(offset, enters, zone):
    s, target, published = leak_base()
    prior = prior_of(s)
    base = by_level(posterior(prior, s, target, published=published, decision=DECISION))
    when = DECISION + timedelta(seconds=offset)
    decision = DECISION
    if zone == "utc":  # the same instants, written in UTC
        when, decision = when.astimezone(timezone.utc), DECISION.astimezone(timezone.utc)
    leak = add_leak(s, published, dict(race_no=6), when)
    got = by_level(posterior(prior, s, target, published=published, decision=decision))
    earlier = [r["race_id"] for r in s.races if r["race_date"] == T and r["race_no"] < TARGET_NO
               and r["race_id"] != leak]  # fmt: skip
    want = observations(s, earlier + [leak] if enters else earlier)
    for key in LEVELS:
        assert got[key]["n_races"] == (5 if enters else 4), (key, got[key])
        assert got[key]["n_obs"] == len(want[key]), (key, got[key], len(want[key]))
    if enters:
        # each level moves toward the leak race's own mean score (a conjugate update's step
        # from the old posterior mean m has the sign of mean(new scores) - m)
        leak_obs = observations(s, [leak])
        for key in LEVELS:
            assert got[key]["sd"] <= base[key]["sd"] + 1e-12, key
            if not leak_obs[key]:
                continue
            assert got[key]["sd"] < base[key]["sd"], key
            pull = mean(leak_obs[key]) - base[key]["mean"]
            if abs(pull) > 1e-6:
                assert (got[key]["mean"] - base[key]["mean"]) * pull > 0, (key, got[key], pull)
        assert got[("draw", "inner")]["mean"] > base[("draw", "inner")]["mean"]
        assert got[("draw", "outer")]["mean"] < base[("draw", "outer")]["mean"]
    else:
        for key in LEVELS:
            for col in ("mean", "sd", "prior_mean", "prior_sd"):
                assert got[key][col] == pytest.approx(base[key][col], abs=1e-12), (key, col)


def test_changing_an_unpublished_race_result_changes_nothing():
    s, target, published = leak_base()
    prior = prior_of(s)
    late = [r["race_id"] for r in s.races if r["race_date"] == T and r["race_no"] == 4][0]
    published = {**published, late: DECISION}  # out exactly at the decision: too late
    base = posterior(prior, s, target, published=published, decision=DECISION)
    for r in s.runners:
        if r["race_id"] == late and r["finish_pos"] is not None:
            r["finish_pos"] = r["horse_no"] if r["horse_no"] <= 12 else r["finish_pos"]
            r["running_positions"] = [1, 1]
    assert_frame_equal(posterior(prior, s, target, published=published, decision=DECISION), base)


def test_naive_times_are_refused():
    s, target, published = leak_base()
    prior = prior_of(s)
    with pytest.raises(ValueError):
        posterior(prior, s, target, published=published, decision=DECISION.replace(tzinfo=None))
    naive = {k: v.replace(tzinfo=None) for k, v in published.items()}
    with pytest.raises(ValueError):
        posterior(prior, s, target, published=naive, decision=DECISION)


# ---------------------------------------------------------------------------------------------
# Features for #129 and the shadow page
# ---------------------------------------------------------------------------------------------


def _args(prior, s: Store):
    races, runners = frames(s)
    return prior, races, runners


FEATURE_COLUMNS = (
    ["race_id", "horse_no", "horse_id", "draw_group", "draw_adj", "draw_adj_sd"]
    + [f"style_adj_{lv}{sfx}" for lv in STYLE_LEVELS for sfx in ("", "_sd")]
    + ["bias_races_n"]
)


@pytest.mark.parametrize(
    ("field", "scratched", "race_no"), [(10, None, 3), (14, 8, 7), (16, 1, 11)]
)
def test_features_carry_each_runners_draw_adjustment_and_the_style_adjustments(
    field, scratched, race_no
):
    rng = random.Random(16)
    s = Store()
    history(s, rng, INNER_FRONT)
    today(s, rng, OUTER_CLOSER, race_no - 1)
    target = biased_race(s, rng, T, field=field, race_no=race_no, scratched=scratched)
    prior = prior_of(s)
    decision = decision_for(T, race_no)
    published = published_all(s)
    post = by_level(posterior(prior, s, target))
    feat = track_bias_features(*_args(prior, s), race_id=target, decision_time=decision,
                               published=published)  # fmt: skip
    assert isinstance(feat, pl.DataFrame)
    missing = [c for c in FEATURE_COLUMNS if c not in feat.columns]
    assert not missing, f"track_bias_features lacks {missing}"
    declared = {(r["horse_no"], r["horse_id"]) for r in s.runners
                if r["race_id"] == target and not r["is_scratched"]}  # fmt: skip
    assert len(declared) == field
    assert set(feat.select("horse_no", "horse_id").rows()) == declared
    assert feat.height == field
    groups = set()
    for row in feat.to_dicts():
        assert row["race_id"] == target
        group = draw_group(row["horse_no"], field)
        groups.add(group)
        assert row["draw_group"] == group, row
        assert row["draw_adj"] == pytest.approx(post[("draw", group)]["mean"], abs=1e-12)
        assert row["draw_adj_sd"] == pytest.approx(post[("draw", group)]["sd"], abs=1e-12)
        for lv in STYLE_LEVELS:
            assert row[f"style_adj_{lv}"] == pytest.approx(post[("style", lv)]["mean"], abs=1e-12)
            assert row[f"style_adj_{lv}_sd"] == pytest.approx(post[("style", lv)]["sd"], abs=1e-12)
        assert row["bias_races_n"] == race_no - 1
    assert groups == set(DRAW_LEVELS)
    # the three groups' adjustments are not one number
    assert len({round(post[("draw", g)]["mean"], 9) for g in DRAW_LEVELS}) == 3


# ---------------------------------------------------------------------------------------------
# Holdout (D14), in-fold priors, determinism
# ---------------------------------------------------------------------------------------------


def test_fit_refuses_a_holdout_day(monkeypatch):
    rng = random.Random(17)
    s = Store()
    history(s, rng, NEUTRAL)
    races, runners = frames(s)
    fit_bias_prior(races, runners, train_end=TRAIN_END)  # fine before any day is held out
    patch_holdout(monkeypatch, [HISTORY[-3]], track_bias)
    with pytest.raises(ValueError):
        fit_bias_prior(races, runners, train_end=TRAIN_END)


def test_posterior_and_features_refuse_a_holdout_target_day(monkeypatch):
    rng = random.Random(18)
    s = Store()
    history(s, rng, NEUTRAL)
    ids = today(s, rng, NEUTRAL, 4)
    prior = prior_of(s)
    posterior(prior, s, ids[-1])  # fine before T is held out
    patch_holdout(monkeypatch, [T], track_bias)
    with pytest.raises(ValueError):
        posterior(prior, s, ids[-1])
    with pytest.raises(ValueError):
        track_bias_features(*_args(prior, s), race_id=ids[-1], decision_time=decision_for(T, 4),
                            published=published_all(s))  # fmt: skip


@pytest.mark.parametrize("train_end", [T, T + timedelta(days=1)])
def test_prior_fitted_through_the_target_day_is_refused(train_end):
    rng = random.Random(19)
    s = Store()
    history(s, rng, NEUTRAL)
    ids = today(s, rng, NEUTRAL, 4)
    prior = prior_of(s, train_end=train_end)
    with pytest.raises(ValueError):
        posterior(prior, s, ids[-1])
    with pytest.raises(ValueError):
        track_bias_features(*_args(prior, s), race_id=ids[-1], decision_time=decision_for(T, 4),
                            published=published_all(s))  # fmt: skip


def test_output_is_deterministic_and_ignores_row_order():
    rng = random.Random(20)
    s = Store()
    history(s, rng, INNER_FRONT)
    ids = today(s, rng, OUTER_CLOSER, 7)
    target, decision, published = ids[-1], decision_for(T, 7), published_all(s)
    races, runners = frames(s)

    def run(races, runners):
        prior = fit_bias_prior(races, runners, train_end=TRAIN_END)
        kw = dict(race_id=target, decision_time=decision, published=published)
        return (within_day_posterior(prior, races, runners, **kw),
                track_bias_features(prior, races, runners, **kw))  # fmt: skip

    post_a, feat_a = run(races, runners)
    post_b, feat_b = run(races, runners)
    post_c, feat_c = run(races.sample(fraction=1.0, shuffle=True, seed=3),
                         runners.sample(fraction=1.0, shuffle=True, seed=4))  # fmt: skip
    assert_frame_equal(post_a, post_b)
    assert_frame_equal(feat_a, feat_b)
    assert_frame_equal(post_a, post_c)
    assert_frame_equal(feat_a, feat_c)
