"""The race-mean diagnostic behind #224's ledger row, with #138's choices as switches (ticket #257).

Contract pinned here (`horsetrack.countries.japan.handicap.track_bias_null`, a new module):

    null_diagnostic(races, runners, *, train_end, obs_weights=False, one_race_groups=False,
                    fitted_lam=False, null="shuffled") -> pl.DataFrame

Seven rows, one per (factor, level) in `track_bias.LEVELS` order (draw inner, middle, outer;
style front, stalker, midpack, closer), with columns `factor`, `level` (str), `groups`, `races`
(int) and `observed_var`, `null_var`, `ratio`, `lam`, `tau2` (float). It reads the race days on or
before `train_end` only (later rows are ignored; a holdout race day on or before it is a
ValueError, D14), as `track_bias.estimate_day_variance` does, and works on the same deviations:
each finisher's score minus its own race's cell prior mean at the level (the means
`fit_bias_prior` fits on the same history).

Per level and (day, venue, surface) group, the races with at least one finisher at the level are
j = 1..r; race j has c_j finishers at the level among nf_j, m_j is their mean deviation, s2_j the
population variance of all the race's finisher scores, and v_j the null variance of m_j. Under
`null="shuffled"` (the race's places shuffled among its finishers, without replacement) v_j is
w0_j = s2_j (nf_j - c_j) / ((nf_j - 1) c_j) and the null mean of m_j is 0. The race weights a_j
are 1 / r (equal race weights, the baseline) or, with `obs_weights` (#138), c_j / sum(c). Then

    d = sum a_j m_j,  q = sum (m_j - d)^2,  V0 = sum a_j^2 v_j,  E0q = sum v_j (1 - 2 a_j) + r V0

(E0q is the null expectation of q; with equal weights q = (r - 1) s^2 and E0q = (r - 1) mean(v)).

- `observed_var`: the mean over the level's groups with r >= 2 of q / (r - 1);
- `null_var`: the mean over the same groups of E0q / (r - 1);
- `ratio`: observed_var / null_var (#224's "observed / null");
- `lam`: #138's fitted scale, sum q / sum E0q over the same groups, reported in every mode;
- `tau2`: the day variance against the null, the mean over the groups used of d^2 - k V0, k = lam
  with `fitted_lam` (#138) and 1 without;
- `groups`, `races`: the groups used for tau2 and their races. The groups used are those with
  r >= 2 (the baseline) or, with `one_race_groups` (#138), every group; one-race groups never
  enter observed_var, null_var, ratio or lam (their q and E0q are 0).

With every switch on and the shuffled null this is #138's removed estimator per level (839ee72,
`_day_spread`); with every switch off it is #224's diagnostic (PR #259's table).

`null="corner"` is the conditional null: v_j and the null mean of m_j come from a model of the
level's finish places given its runners' first-corner positions (and field size), fitted on the
same history; then E0q also carries the spread of the races' null means, and d is taken about its
null mean. The model is the coder's choice; the tests pin its behaviour: where finish places do
not depend on the corner it agrees with the shuffled null, and where they do (and there is no day
effect) the level's race means vary as it says (ratio about 1, tau2 about 0) at every level, while
the shuffled null misstates them both ways. observed_var, groups and races do not depend on the
null. Any other `null` is a ValueError. A level with no observations has groups = races = 0 and
null or NaN floats.

Rulings (reversible choices this file pins):

- Ruling: the diagnostic is `track_bias_null.null_diagnostic`, a new module beside `track_bias`,
  returning one polars row per level with the columns above.
- Ruling: #138's three choices are the keyword switches `obs_weights`, `one_race_groups` and
  `fitted_lam`, each off by default (the baseline is #224's equal race weights over groups of two
  or more races, unscaled null); the null is `null="shuffled"` (default) or `"corner"`.
- Ruling: group-level quantities are averaged with equal group weights (as `estimate_day_variance`
  does); `lam` keeps #138's pooled sums (sum q / sum E0q).

Tolerances. Exact cases (small explicit histories) match a reference computed here from the
fixture's own places and corners, with the prior means read from `fit_bias_prior` (existing
code), to 1e-9 relative. The behaviour cases use 1,016 (day, venue, surface) groups of six races:
a ratio's sampling sd there is about 0.02 to 0.03, so "about 1" is within 0.12 (0.10 where the
null is exact); "tau2 about 0" is 5 standard errors of a mean of d^2 under the shuffled null at a
field of 12, sqrt(2) (w0 / 6) / sqrt(groups), per level.

Data: synthetic only, in memory; nothing reads $HORSETRACK_DATA_DIR.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import functools
import itertools
import math
import random
import statistics
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from day_variance_store import draw_group, planted_history, style_of
from handicap_store import R, Store, patch_holdout, weekend_days
from horsetrack.countries.japan.handicap import track_bias
from horsetrack.countries.japan.handicap import track_bias_null
from horsetrack.countries.japan.handicap.track_bias import fit_bias_prior
from horsetrack.countries.japan.handicap.track_bias_null import null_diagnostic

DRAW_LEVELS = ("inner", "middle", "outer")
STYLE_LEVELS = ("front", "stalker", "midpack", "closer")
LEVELS = [("draw", lv) for lv in DRAW_LEVELS] + [("style", lv) for lv in STYLE_LEVELS]
STYLE_KEYS = [key for key in LEVELS if key[0] == "style"]
FRONT = ("style", "front")
MIDPACK = ("style", "midpack")

COLUMNS = ["factor", "level", "groups", "races", "observed_var", "null_var", "ratio", "lam", "tau2"]
FLOATS = ["observed_var", "null_var", "ratio", "lam", "tau2"]
SWITCHES = ("obs_weights", "one_race_groups", "fitted_lam")
COMBOS = [
    dict(zip(SWITCHES, bits, strict=True)) for bits in itertools.product((False, True), repeat=3)
]


def combo_id(c: dict) -> str:
    return "+".join(k for k, v in c.items() if v) or "none"


TRAIN_END = date(2025, 6, 8)
DAYS = weekend_days(date(2023, 1, 7), TRAIN_END)  # 254 days, 1016 (day, venue, surface) groups
VENUES = ("TOK", "NAK")
SURFACES = ("turf", "dirt")


def frames(s: Store) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.DataFrame(s.races, infer_schema_length=None),
        pl.DataFrame(s.runners, infer_schema_length=None),
    )


def by_level(out: pl.DataFrame) -> dict[tuple[str, str], dict]:
    return {(r["factor"], r["level"]): r for r in out.iter_rows(named=True)}


def missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def check_shape(out: pl.DataFrame) -> None:
    assert isinstance(out, pl.DataFrame), type(out)
    assert set(COLUMNS) <= set(out.columns), out.columns
    assert out.height == 7, out
    assert list(zip(out["factor"], out["level"], strict=True)) == LEVELS


# ---------------------------------------------------------------------------------------------
# Exact cases: a small explicit history and a reference computed from its own places
# ---------------------------------------------------------------------------------------------

EX_DAYS = weekend_days(date(2024, 1, 6), date(2024, 6, 30))
EX_END = date(2024, 6, 30)
# races per (day, venue, surface) group: one-race groups, two-race groups and larger ones
EX_SIZES = (1, 2, 3, 4, 5, 2, 1, 3, 6, 2, 4, 1, 3, 2, 5, 1)


@dataclass
class Rec:
    """One race as built: its group, its prior cell and its runners (draw, corner, place)."""

    group: tuple[date, str, str]
    cell: tuple
    n: int
    runners: list[tuple[int, int | None, int | None]]


def exact_race(s: Store, rng: random.Random, day: date, venue: str, surface: str) -> Rec:
    n = rng.randint(6, 14)
    distance = rng.choice((1400, 1600))
    going = rng.choice(("GOOD", "SOFT"))
    corners = list(range(1, n + 1))
    rng.shuffle(corners)
    # finish leans on the corner (so the level means, and the prior, are not 0); sometimes one
    # runner does not finish (it keeps its corner, so it counts in the field but scores nothing)
    keys = [0.6 * (corners[i] - 1) / (n - 1) + rng.random() for i in range(n)]
    dnf = rng.randrange(n) if rng.random() < 0.35 else None
    order = sorted((i for i in range(n) if i != dnf), key=keys.__getitem__)
    pos: dict[int, int | None] = {i: p for p, i in enumerate(order, 1)}
    if dnf is not None:
        pos[dnf] = None
    base = len(s.runners)
    s.race(
        day,
        [R(f"x:{base + i}", pos=pos[i], corners=[corners[i]] * 2, no=i + 1) for i in range(n)],
        venue=venue, surface=surface, distance=distance, going=going, course="A",
    )  # fmt: skip
    return Rec(
        group=(day, venue, surface),
        cell=(venue, surface, distance, "A", going, None),
        n=n,
        runners=[(i + 1, corners[i], pos[i]) for i in range(n)],
    )


def exact_history(seed: int, sizes=EX_SIZES, days=EX_DAYS) -> tuple[Store, list[Rec]]:
    rng = random.Random(seed)
    s, recs = Store(), []
    for g, r in enumerate(sizes):
        day, venue, surface = days[g // 2], VENUES[g % 2], SURFACES[(g // 4) % 2]
        recs += [exact_race(s, rng, day, venue, surface) for _ in range(r)]
    return s, recs


def race_terms(rec: Rec, prior) -> dict[tuple[str, str], tuple[int, float, float]]:
    """(c, m, w0) per level present among the race's finishers."""
    fin = [(no, corner, p) for no, corner, p in rec.runners if p is not None]
    nf = len(fin)
    y = {no: (nf - p) / (nf - 1) - 0.5 for no, _, p in fin}
    s2 = statistics.pvariance(list(y.values()))
    devs: dict[tuple[str, str], list[float]] = {}
    for no, corner, _ in fin:
        keys = [("draw", draw_group(no, rec.n))]
        if corner is not None:
            keys.append(("style", style_of(corner, rec.n)))
        for key in keys:
            devs.setdefault(key, []).append(y[no] - prior.mean(rec.cell, *key))
    return {
        key: (len(e), sum(e) / len(e), s2 * (nf - len(e)) / ((nf - 1) * len(e)))
        for key, e in devs.items()
    }


def grouped(recs: list[Rec], prior) -> dict[tuple[str, str], list[list[tuple[int, float, float]]]]:
    """Per level, the groups' lists of (c, m, w0), one per race with the level."""
    per: dict[tuple[str, str], dict[tuple, list]] = {key: {} for key in LEVELS}
    for rec in recs:
        for key, t in race_terms(rec, prior).items():
            per[key].setdefault(rec.group, []).append(t)
    return {key: list(groups.values()) for key, groups in per.items()}


def reference(groups, *, obs_weights: bool, one_race_groups: bool, fitted_lam: bool) -> dict:
    """#138's construction (839ee72's `_day_spread`, per level) with each choice a switch."""
    terms = []  # (r, d, q, e0q, v0)
    for races in groups:
        r, total = len(races), sum(c for c, _, _ in races)
        a = [(c / total) if obs_weights else 1 / r for c, _, _ in races]
        d = sum(aj * m for aj, (_, m, _) in zip(a, races, strict=True))
        q = sum((m - d) ** 2 for _, m, _ in races)
        v0 = sum(aj**2 * w for aj, (_, _, w) in zip(a, races, strict=True))
        e0q = sum(w * (1 - 2 * aj) for aj, (_, _, w) in zip(a, races, strict=True)) + r * v0
        terms.append((r, d, q, e0q, v0))
    multi = [t for t in terms if t[0] >= 2]
    used = terms if one_race_groups else multi
    observed = statistics.fmean(t[2] / (t[0] - 1) for t in multi)
    null = statistics.fmean(t[3] / (t[0] - 1) for t in multi)
    lam = sum(t[2] for t in multi) / sum(t[3] for t in multi)
    k = lam if fitted_lam else 1.0
    return dict(
        groups=len(used),
        races=sum(t[0] for t in used),
        observed_var=observed,
        null_var=null,
        ratio=observed / null,
        lam=lam,
        tau2=statistics.fmean(t[1] ** 2 - k * t[4] for t in used),
    )


def baseline_reference(groups) -> dict:
    """#224's diagnostic as written in PR #259: equal race weights over groups of >= 2 races."""
    multi = [g for g in groups if len(g) >= 2]
    observed = statistics.fmean(statistics.variance([m for _, m, _ in g]) for g in multi)
    null = statistics.fmean(statistics.fmean(w for _, _, w in g) for g in multi)
    tau2 = statistics.fmean(
        statistics.fmean(m for _, m, _ in g) ** 2 - statistics.fmean(w for _, _, w in g) / len(g)
        for g in multi
    )
    return dict(groups=len(multi), races=sum(len(g) for g in multi), observed_var=observed,
                null_var=null, ratio=observed / null, tau2=tau2)  # fmt: skip


@functools.cache
def exact_world(seed: int):
    s, recs = exact_history(seed)
    races, runners = frames(s)
    prior = fit_bias_prior(races, runners, train_end=EX_END, day_sd=0.05)
    return races, runners, grouped(recs, prior)


def assert_matches(row: dict, want: dict, key) -> None:
    assert row["groups"] == want["groups"], (key, row, want)
    assert row["races"] == want["races"], (key, row, want)
    for col in FLOATS:
        if col in want:
            assert row[col] == pytest.approx(want[col], rel=1e-9, abs=1e-12), (key, col, row, want)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_baseline_is_224s_diagnostic_exactly(seed):
    races, runners, groups = exact_world(seed)
    out = null_diagnostic(races, runners, train_end=EX_END)
    check_shape(out)
    rows = by_level(out)
    for key in LEVELS:
        want = baseline_reference(groups[key])
        # the fixture's priors are not 0, its groups are uneven and some have one race
        assert want["groups"] < len(groups[key]), (key, "fixture: no one-race group")
        assert_matches(rows[key], want, key)
        # the generic reference agrees with the plain one with the switches off
        generic = reference(groups[key], obs_weights=False, one_race_groups=False, fitted_lam=False)
        assert generic["tau2"] == pytest.approx(want["tau2"], rel=1e-9, abs=1e-15)
        assert rows[key]["lam"] == pytest.approx(generic["lam"], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("switches", COMBOS, ids=combo_id)
def test_each_138_choice_switches_exactly(seed, switches):
    races, runners, groups = exact_world(seed)
    out = null_diagnostic(races, runners, train_end=EX_END, **switches)
    check_shape(out)
    rows = by_level(out)
    for key in LEVELS:
        assert_matches(rows[key], reference(groups[key], **switches), key)


def test_switches_change_what_they_should_on_the_exact_history():
    """Each switch moves tau2 at some level, so none of them is ignored (a fixture guard as well
    as a check: the reference values differ, and so must the function's)."""
    races, runners, groups = exact_world(2)
    base = by_level(null_diagnostic(races, runners, train_end=EX_END))
    for name in SWITCHES:
        got = by_level(null_diagnostic(races, runners, train_end=EX_END, **{name: True}))
        want = {key: reference(groups[key], **{s: s == name for s in SWITCHES}) for key in LEVELS}
        moved = [key for key in LEVELS if abs(want[key]["tau2"] - base[key]["tau2"]) > 1e-6]
        assert moved, (name, "fixture: the switch moves no level")
        for key in moved:
            assert got[key]["tau2"] == pytest.approx(want[key]["tau2"], rel=1e-9), (name, key)
            assert abs(got[key]["tau2"] - base[key]["tau2"]) > 1e-6, (name, key)


@pytest.mark.parametrize("switches", COMBOS, ids=combo_id)
def test_corner_null_keeps_the_observed_side(switches):
    races, runners, _ = exact_world(1)
    shuffled = by_level(null_diagnostic(races, runners, train_end=EX_END, **switches))
    out = null_diagnostic(races, runners, train_end=EX_END, null="corner", **switches)
    check_shape(out)
    for key, row in by_level(out).items():
        assert row["groups"] == shuffled[key]["groups"], key
        assert row["races"] == shuffled[key]["races"], key
        assert row["observed_var"] == pytest.approx(shuffled[key]["observed_var"], rel=1e-9)
        for col in FLOATS:
            assert math.isfinite(row[col]), (key, col, row)
        assert row["null_var"] > 0, (key, row)
        assert row["ratio"] == pytest.approx(row["observed_var"] / row["null_var"], rel=1e-9)


# ---------------------------------------------------------------------------------------------
# Behaviour: random order about 1, a pinned level far below, finish that follows the corner
# ---------------------------------------------------------------------------------------------


def w0_at_12(key) -> float:
    """w0 of a level at a field of 12, every runner finishing: s2 = 13 / 132 and c runners at
    the level (front 1, stalker 3, midpack 4, closer 4, each draw third 4)."""
    c = {FRONT: 1, ("style", "stalker"): 3}.get(key, 4)
    return 13 / 132 * (12 - c) / (11 * c)


def tau2_bound(key, groups: int) -> float:
    return 5 * math.sqrt(2) * (w0_at_12(key) / 6) / math.sqrt(groups)


@functools.cache
def random_order_world():
    # "extreme" with no day effect: every race a random order of its places
    s, _, _ = planted_history(301, DAYS, target=MIDPACK, sd=0.0, mode="extreme")
    return frames(s)


@functools.cache
def pinned_world():
    # midpack's race means pinned to the group's day effect (they barely vary within a group)
    s, _, _ = planted_history(302, DAYS, target=MIDPACK, sd=0.05, mode="pinned")
    return frames(s)


def corner_race(s: Store, rng: random.Random, day: date, venue: str, surface: str, n: int) -> None:
    """Finish follows the first corner: the leader wins or finishes last, half and half; the
    rest finish in the order of their corner position plus Gaussian noise (sd 0.3 of the field's
    span). No day effect."""
    corners = list(range(1, n + 1))
    rng.shuffle(corners)
    keys = []
    for i in range(n):
        if corners[i] == 1:
            keys.append(-math.inf if rng.random() < 0.5 else math.inf)
        else:
            keys.append((corners[i] - 1) / (n - 1) + 0.3 * rng.gauss(0.0, 1.0))
    order = sorted(range(n), key=keys.__getitem__)
    pos = {i: p for p, i in enumerate(order, 1)}
    base = len(s.runners)
    s.race(
        day,
        [R(f"c:{base + i}", pos=pos[i], corners=[corners[i]] * 2, no=i + 1) for i in range(n)],
        venue=venue, surface=surface, distance=1600, going="GOOD", course="A",
    )  # fmt: skip


@functools.cache
def corner_world():
    rng = random.Random(303)
    s = Store()
    for day in DAYS:
        for venue in VENUES:
            for surface in SURFACES:
                for _ in range(6):
                    corner_race(s, rng, day, venue, surface, rng.choice((10, 12, 14)))
    return frames(s)


@functools.cache
def diag(world: str, null: str = "shuffled", **switches) -> dict:
    races, runners = {"random": random_order_world, "pinned": pinned_world,
                      "corner": corner_world}[world]()  # fmt: skip
    out = null_diagnostic(races, runners, train_end=TRAIN_END, null=null, **switches)
    check_shape(out)
    return by_level(out)


@pytest.mark.parametrize("null", ["shuffled", "corner"])
def test_random_order_gives_ratio_about_one_at_every_level(null):
    rows = diag("random", null)
    for key in LEVELS:
        row = rows[key]
        assert row["groups"] == 1016 and row["races"] == 6096, (key, row)
        assert row["ratio"] == pytest.approx(1.0, abs=0.10), (key, row)
        assert abs(row["tau2"]) <= tau2_bound(key, 1016), (key, row)
    if null == "shuffled":  # every race a full field of 12: w0 is known
        for key in LEVELS:
            assert rows[key]["null_var"] == pytest.approx(w0_at_12(key), rel=1e-9), key


def test_corner_null_agrees_with_the_shuffled_null_when_finish_ignores_the_corner():
    shuffled, corner = diag("random"), diag("random", "corner")
    for key in LEVELS:
        assert corner[key]["null_var"] == pytest.approx(shuffled[key]["null_var"], rel=0.10), key


def test_a_pinned_level_is_far_below_the_shuffled_null():
    rows = diag("pinned")
    assert rows[MIDPACK]["ratio"] < 0.2, rows[MIDPACK]
    assert rows[MIDPACK]["lam"] < 0.2, rows[MIDPACK]
    for key in [("draw", lv) for lv in DRAW_LEVELS]:  # the draw thirds stay a random order
        assert rows[key]["ratio"] == pytest.approx(1.0, abs=0.10), (key, rows[key])


def test_shuffled_null_misstates_a_corner_led_finish_both_ways():
    rows = diag("corner")
    assert rows[FRONT]["ratio"] > 1.5, rows[FRONT]  # the leader wins or fades
    assert rows[MIDPACK]["ratio"] < 0.7, rows[MIDPACK]  # the mid-corner block finishes mid-field
    assert rows[FRONT]["tau2"] > tau2_bound(FRONT, 1016), rows[FRONT]
    assert rows[MIDPACK]["tau2"] < -tau2_bound(MIDPACK, 1016), rows[MIDPACK]
    for key in [("draw", lv) for lv in DRAW_LEVELS]:
        assert rows[key]["ratio"] == pytest.approx(1.0, abs=0.12), (key, rows[key])


@pytest.mark.parametrize("fitted_lam", [False, True])
def test_corner_null_tracks_a_corner_led_finish_at_every_level(fitted_lam):
    rows = diag("corner", "corner", fitted_lam=fitted_lam)
    for key in LEVELS:
        row = rows[key]
        assert row["ratio"] == pytest.approx(1.0, abs=0.12), (key, row)
        assert row["lam"] == pytest.approx(1.0, abs=0.12), (key, row)
        assert abs(row["tau2"]) <= tau2_bound(key, 1016), (key, row)


# ---------------------------------------------------------------------------------------------
# Train end, holdout, row order, missing levels, bad arguments
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("null", ["shuffled", "corner"])
def test_rows_after_train_end_are_ignored(null):
    races, runners, _ = exact_world(1)
    base = null_diagnostic(races, runners, train_end=EX_END, null=null)
    later_days = weekend_days(EX_END + timedelta(days=1), EX_END + timedelta(days=30))
    s2, _ = exact_history(9, sizes=(4, 4, 4, 4), days=later_days)
    more = frames(s2)
    races2 = pl.concat([races, more[0]], how="vertical_relaxed")
    runners2 = pl.concat([runners, more[1]], how="vertical_relaxed")
    got = null_diagnostic(races2, runners2, train_end=EX_END, null=null)
    assert_frame_equal(got, base, check_exact=False, rel_tol=1e-9, abs_tol=1e-12)
    later = null_diagnostic(races2, runners2, train_end=EX_END + timedelta(days=60), null=null)
    assert by_level(later)[MIDPACK]["races"] > by_level(base)[MIDPACK]["races"]


def test_row_order_does_not_matter():
    races, runners, _ = exact_world(3)
    for switches in (COMBOS[0], COMBOS[-1]):
        a = null_diagnostic(races, runners, train_end=EX_END, **switches)
        b = null_diagnostic(races.sample(fraction=1.0, shuffle=True, seed=5),
                            runners.sample(fraction=1.0, shuffle=True, seed=6),
                            train_end=EX_END, **switches)  # fmt: skip
        assert_frame_equal(b, a, check_exact=False, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize("null", ["shuffled", "corner"])
def test_a_holdout_day_is_refused(monkeypatch, null):
    races, runners, _ = exact_world(1)
    null_diagnostic(races, runners, train_end=EX_END, null=null)  # fine before
    patch_holdout(monkeypatch, [EX_DAYS[2]], track_bias, track_bias_null)
    with pytest.raises(ValueError):
        null_diagnostic(races, runners, train_end=EX_END, null=null)


@pytest.mark.parametrize("bad", ["independent", "Shuffled", "", "conditional"])
def test_an_unknown_null_is_refused(bad):
    races, runners, _ = exact_world(1)
    with pytest.raises(ValueError):
        null_diagnostic(races, runners, train_end=EX_END, null=bad)


@pytest.mark.parametrize("null", ["shuffled", "corner"])
def test_levels_without_observations_are_empty(null):
    rng = random.Random(11)
    s = Store()
    for g, r in enumerate((3, 2, 4, 3)):
        day, venue = EX_DAYS[g], VENUES[g % 2]
        for _ in range(r):
            n = rng.randint(8, 12)
            places = list(range(1, n + 1))
            rng.shuffle(places)
            base = len(s.runners)
            # no first-corner position anywhere: no style observation
            s.race(day, [R(f"n:{base + i}", pos=places[i], corners=[], no=i + 1) for i in range(n)],
                   venue=venue, surface="turf")  # fmt: skip
    races, runners = frames(s)
    rows = by_level(null_diagnostic(races, runners, train_end=EX_END, null=null))
    for key in STYLE_KEYS:
        assert rows[key]["groups"] == 0 and rows[key]["races"] == 0, rows[key]
        for col in FLOATS:
            assert missing(rows[key][col]), (key, col, rows[key])
    for key in [("draw", lv) for lv in DRAW_LEVELS]:
        assert rows[key]["groups"] == 4 and rows[key]["races"] == 12, rows[key]
        assert math.isfinite(rows[key]["ratio"]), rows[key]
