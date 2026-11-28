"""A carry-over variant prior for a day's first races (#328, epic #73; follow-up of #305).

Contract pinned here.

`horsetrack.countries.japan.models.speed_prior`:

- `PRIOR_KINDS == ("previous_day", "other_surface", "both")`: the owner's (a), (b) and (c) on
  #328 (2026-11-28), each a pre-registered variant. (d), no prior, is #305's baseline.
- `CarryOverPrior(model, results, *, kind, half_life_days=None, surface_weight=None)`. `model` is
  a fitted `SpeedModel` (its pars, sigma, tau and error are read; its own injected prior never is);
  `results` is a frame shaped like `speed_run.build_race_table`'s table (race_id, race_date,
  race_utc, result_available_utc, top3_time_s and the par keys), the non-JRDB store's results.
  A result's residual is top3_time_s - model.par(cell); a result in a cell the fit never saw is
  skipped. A result is *usable at t* when its result_available_utc is strictly before t (a null
  time never is). An unknown kind, a half_life_days that is not > 0 (previous_day, both) or a
  surface_weight outside [0, 1] (other_surface, both) is a ValueError.
- `prior(race_date, track, surface, *, at_utc) -> (mean, sd)` with base N(0, tau^2) =
  (0.0, model.tau). Post(day, track, surface; t) is the posterior of that day's variant from the
  base prior given its residuals usable at t (with a normal error: precision 1/tau^2 +
  n/sigma^2, mean sum(r)/sigma^2 / precision), the base when it has none.
  - previous_day (a): d' is the latest day strictly before race_date with a usable result of the
    same (track, surface); (m', s') = Post(d'). rho = 0.5 ** ((race_date - d').days /
    half_life_days); mean = rho m', sd^2 = rho^2 s'^2 + (1 - rho^2) tau^2 (decays to the base).
    No such day: the base.
  - other_surface (b): (m_o, s_o) = Post(race_date, track, the other surface; t) at the same track;
    w = surface_weight; mean = w m_o, sd^2 = w^2 s_o^2 + (1 - w^2) tau^2. None usable: the base.
  - both (c): the two by their information over the base: p = 1/sd_a^2 + 1/sd_b^2 - 1/tau^2,
    mean = (mean_a / sd_a^2 + mean_b / sd_b^2) / p, sd = p^-0.5. So it is (a) without a usable
    other-surface result, (b) without a previous day, and the base with neither.
  No kind reads the target day's own (track, surface), a later day or another track.
- `prior.at(at_utc)` is the three-argument callable `SpeedModel(variant_prior=...)` takes, with
  `prior.at(t)(d, track, surface) == prior(d, track, surface, at_utc=t)`. #96's ruling holds: the
  fit never reads it; only `variant` and `expected_top3_time` do.

`horsetrack.countries.japan.studies.speed_prior_run` (the study):

- `FAMILY_ID == "japan-speed-top3-c3-prior"`, `TICKET == 328`, `BASE_FAMILY_ID ==
  "japan-speed-top3-c3"` (#305's family).
- `predict_prior_folds(table, *, spec, scheme, decision_min, priors)`: #305's fold loop (holdout
  days dropped first, `walk_forward` folds, t = post_utc - decision_min). `priors` maps each kind
  to its `CarryOverPrior` keyword parameters. One row per scored race with #305's columns,
  whose par_only, variant_on and variant_off are exactly `speed_run.predict_folds`'s, plus
  f"{kind}:variant_on" and f"{kind}:variant_off" per kind: the same model with the kind's prior
  injected at the race's decision time.
- `build_spec(boot, scheme, minutes)`: the pre-registered spec, frozen before any scored read:
  family_id, ticket, base_family, sources ["jra"], and "priors", {kind: its CarryOverPrior keyword
  parameters} for the three kinds (half_life_days a number > 0 for previous_day and both,
  surface_weight a number in [0, 1] for other_surface and both).
- `compute_study(table, spec, scheme, minutes) -> (variants, tables, counts)`: per #305 error model
  ("t5", "normal"), prior kind, prediction ("variant_on", "variant_off") and baseline ("par_only",
  "variant_on": #305's variant_on), one `speed_run.Variant` whose components carry "error_model",
  "prior", "prediction" and "baseline" (failures included, never dropped). A scored variant's
  metrics: ("all", "rmse_gain_s") with D7's bounds and, per number of results seen (the races of
  the race's (day, track, surface) with a result available strictly before t, capped at 6),
  (f"results_seen:{k}", "rmse_gain_s") with bounds; gain = rmse_baseline - rmse_model.
- `run_study(out_dir)`: freezes FAMILY_ID's spec before the store is read or anything is scored,
  appends one registry record per variant (ticket 328, sources ("jra",)), writes the report through
  `write_report` and returns family_id, spec_hash and report_path. The report shows each variant's
  no-same-day-result gain over #305's variant_on with its bounds. It looks up `default_registry`,
  `read_inputs`, `store_builds`, `_git_sha`, `write_summary` and `compute_study` in its own module.

Not pinned: the posterior under a Student-t error (only the normal one has the closed form used
here), which earlier days the fold loop hands the prior beyond the leakage rule, the variant names,
params, the report's layout, counts and tables. Every frame is synthetic; expected values come from
planted residuals and closed forms, never from the code under test. No JRDB input (D21).
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import itertools
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.folds import FoldScheme
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry
from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec
from horsetrack.countries.japan.models.speed_prior import PRIOR_KINDS, CarryOverPrior
from horsetrack.countries.japan.studies import speed_prior_run as study
from horsetrack.countries.japan.studies import speed_run

JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
RESULT_LAG = timedelta(minutes=30)  # post + 30 min, the same-day results rule (availability.yaml)
PAR_KEYS = ("track", "surface", "distance", "layout", "rail", "class_level")
VARIANT_KEYS = ("track", "surface")
KINDS = ("previous_day", "other_surface", "both")
PREDICTIONS = ("variant_on", "variant_off")
BASELINES = ("par_only", "variant_on")
ERROR_MODELS = ("t5", "normal")
NORMAL = SpeedSpec(par_keys=PAR_KEYS, variant_keys=VARIANT_KEYS, same_day_variant=True, t_df=None)
NORMAL_OFF = SpeedSpec(PAR_KEYS, VARIANT_KEYS, same_day_variant=False, t_df=None)
LEVEL = float(load_country_yaml("japan", "scoring")["bootstrap"]["level"])
ATOL = 1e-9


def at(day: date, hhmm: str) -> pd.Timestamp:
    """A JST wall-clock time on `day` as a UTC instant, from first principles."""
    hh, mm = (int(x) for x in hhmm.split(":"))
    return pd.Timestamp(datetime.combine(day, time(hh, mm), tzinfo=JST).astimezone(UTC))


# ---------------------------------------------------------------------------------------------
# The closed forms (normal error)
# ---------------------------------------------------------------------------------------------


def posterior(resids, sigma, tau):
    if len(resids) == 0:
        return 0.0, tau
    prec = 1.0 / tau**2 + len(resids) / sigma**2
    return (sum(resids) / sigma**2) / prec, prec**-0.5


def decayed(m, s, gap_days, half_life, tau):
    rho = 0.5 ** (gap_days / half_life)
    return rho * m, math.sqrt(rho**2 * s**2 + (1.0 - rho**2) * tau**2)


def coupled(m, s, w, tau):
    return w * m, math.sqrt(w**2 * s**2 + (1.0 - w**2) * tau**2)


def combined(a, b, tau):
    pa, pb, p0 = a[1] ** -2, b[1] ** -2, tau**-2
    p = pa + pb - p0
    return (a[0] * pa + b[0] * pb) / p, p**-0.5


def same(got, expected, what=""):
    assert got[0] == pytest.approx(expected[0], abs=ATOL), (what, got, expected)
    assert got[1] == pytest.approx(expected[1], abs=ATOL), (what, got, expected)


# ---------------------------------------------------------------------------------------------
# A fitted model and hand-made results (the prior's unit tests)
# ---------------------------------------------------------------------------------------------

_CELL_REST = {
    ("TOK", "turf"): [(1600, "outer", "unknown", "C1"), (2000, "outer", "unknown", "C2")],
    ("TOK", "dirt"): [(1400, "inner", "unknown", "C1"), (1800, "inner", "unknown", "C2")],
    ("KYO", "turf"): [(1600, "outer", "unknown", "C1"), (2200, "outer", "unknown", "C3")],
    ("KYO", "dirt"): [(1200, "inner", "unknown", "C1"), (1900, "inner", "unknown", "C2")],
}
# (track, surface) -> its two cells (the par keys' values)
CELLS = {key: [(*key, *rest) for rest in cells] for key, cells in _CELL_REST.items()}
PARS = dict(
    zip(
        [c for cells in CELLS.values() for c in cells],
        (96.0, 121.0, 84.0, 111.5, 95.5, 134.0, 71.0, 118.0),
        strict=True,
    )
)
UNSEEN = ("TOK", "turf", 3200, "outer", "unknown", "G1")
TRAIN_DAYS = [date(2026, 9, 5) + timedelta(days=k) for k in range(30) if (k % 7) in (0, 1)]
D = date(2026, 11, 15)  # the target day, a Sunday; no fit day is near it
P = D - timedelta(days=1)  # its previous race day
E = D - timedelta(days=8)  # an earlier one, never read when P has a result
T = at(D, "12:00") - timedelta(minutes=2)  # the target race's decision time


def _race(race_id, day, hhmm, cell, top3, avail="lag"):
    post = at(day, hhmm)
    return {
        "race_id": race_id,
        "meeting_id": f"jra:{day.isoformat()}:{cell[0]}",
        "race_date": day,
        "post_utc": post,
        "race_utc": post,
        "result_available_utc": post + RESULT_LAG if avail == "lag" else avail,
        "top3_time_s": top3,
        **dict(zip(PAR_KEYS, cell, strict=True)),
    }


def frame(rows) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    for column in ("post_utc", "race_utc", "result_available_utc"):
        out[column] = pd.to_datetime(out[column], utc=True)
    return out


@pytest.fixture(scope="module")
def train():
    rng = np.random.default_rng(328)
    rows = []
    for day in TRAIN_DAYS:
        for (venue, surface), cells in CELLS.items():
            v = rng.normal(0.0, 1.0)
            for k in range(4):
                cell = cells[k % 2]
                hhmm = f"{10 + k}:{'05' if surface == 'turf' else '35'}"
                top3 = PARS[cell] + v + rng.normal(0.0, 0.3)
                rows.append(_race(f"tr:{day}:{venue}:{surface}:{k}", day, hhmm, cell, top3))
    return frame(rows)


@pytest.fixture(scope="module")
def model(train):
    return SpeedModel(NORMAL).fit(train)


class Results:
    """Results with planted residuals against the fitted pars."""

    def __init__(self, model):
        self.model, self.rows = model, []

    def add(self, day, track, surface, resid, hhmm="10:00", avail="lag", k=0):
        cell = CELLS[(track, surface)][k]
        top3 = self.model.par(**dict(zip(PAR_KEYS, cell, strict=True))) + resid
        self.rows.append(_race(f"r:{len(self.rows)}", day, hhmm, cell, top3, avail))
        return self

    def add_unseen(self, day, hhmm="10:00"):
        self.rows.append(_race(f"r:{len(self.rows)}", day, hhmm, UNSEEN, 210.0))
        return self

    def frame(self):
        return frame(self.rows)


def build(model, results, kind, *, half_life_days=7.0, surface_weight=0.8):
    params = {}
    if kind in ("previous_day", "both"):
        params["half_life_days"] = half_life_days
    if kind in ("other_surface", "both"):
        params["surface_weight"] = surface_weight
    df = results.frame() if isinstance(results, Results) else results
    return CarryOverPrior(model, df, kind=kind, **params)


PREV_TURF = [0.9, 1.3, 0.7]  # P, TOK turf
SAME_DIRT = [(0.6, "10:00"), (0.8, "10:40"), (0.2, "11:20")]  # D, TOK dirt: usable at T


def base_results(model) -> Results:
    """P's Tokyo turf (with an unseen-cell race), E's older Tokyo turf, D's Tokyo dirt, and the
    results no prior of (D, TOK, turf) may read: D's own Tokyo turf, Kyoto, a later day."""
    r = Results(model)
    for k, resid in enumerate(PREV_TURF):
        r.add(P, "TOK", "turf", resid, hhmm=f"1{k}:15", k=k % 2)
    r.add_unseen(P, "15:00")
    r.add(P, "TOK", "dirt", -0.4, hhmm="13:35")
    for resid in (-2.0, -1.5):
        r.add(E, "TOK", "turf", resid, hhmm="11:15")
    for resid, hhmm in SAME_DIRT:
        r.add(D, "TOK", "dirt", resid, hhmm=hhmm)
    r.add(D, "TOK", "turf", 5.0, hhmm="10:20")  # the same-day variant's, not the prior's
    r.add(D, "KYO", "dirt", -3.0, hhmm="10:00")
    r.add(P, "KYO", "turf", -2.5, hhmm="11:00")
    r.add(D + timedelta(days=1), "TOK", "turf", 4.0, hhmm="10:00")
    return r


def expected_base(model, kind, *, half_life=7.0, w=0.8, prev=PREV_TURF, gap=1, dirt=None):
    s, tau = model.sigma, model.tau
    dirt = [x for x, _ in SAME_DIRT] if dirt is None else dirt
    a = decayed(*posterior(prev, s, tau), gap, half_life, tau) if prev else (0.0, tau)
    b = coupled(*posterior(dirt, s, tau), w, tau)
    return {"previous_day": a, "other_surface": b, "both": combined(a, b, tau)}[kind]


# ---------------------------------------------------------------------------------------------
# (a) previous_day
# ---------------------------------------------------------------------------------------------


def test_prior_kinds_are_the_three_preregistered_variants():
    assert tuple(PRIOR_KINDS) == KINDS


@pytest.mark.parametrize("half_life", [3.5, 7.0, 14.0])
def test_previous_day_prior_is_the_last_race_days_posterior_decayed_toward_the_base(
    model, half_life
):
    s, tau = model.sigma, model.tau
    for gap in (1, 6, 13, 27):
        prev = [0.4 + 0.05 * gap, 1.1, -0.2 + 0.01 * gap][: 1 + gap % 3]
        r = Results(model)
        for k, resid in enumerate(prev):
            r.add(D - timedelta(days=gap), "TOK", "turf", resid, hhmm=f"1{k}:05", k=k % 2)
        # an older day, the other surface on the previous day, the other track: never read
        r.add(D - timedelta(days=gap + 5), "TOK", "turf", -2.0)
        r.add(D - timedelta(days=gap), "TOK", "dirt", 3.0, hhmm="15:00")
        r.add(D - timedelta(days=gap), "KYO", "turf", -1.0)
        got = build(model, r, "previous_day", half_life_days=half_life)(D, "TOK", "turf", at_utc=T)
        same(got, decayed(*posterior(prev, s, tau), gap, half_life, tau), (gap, half_life))

    # the decay reaches the base: a previous race day a year back is N(0, tau^2)
    r = Results(model).add(D - timedelta(days=400), "TOK", "turf", 1.5)
    got = build(model, r, "previous_day", half_life_days=half_life)(D, "TOK", "turf", at_utc=T)
    assert got[0] == pytest.approx(0.0, abs=1e-6) and got[1] == pytest.approx(tau, abs=1e-6)


def test_previous_day_prior_without_an_earlier_race_day_is_the_base(model):
    tau = model.tau
    r = base_results(model)
    prior = build(model, r, "previous_day")
    # no earlier day of Kyoto dirt; Tokyo's earliest results day has no earlier one
    same(prior(D, "KYO", "dirt", at_utc=T), (0.0, tau), "KYO dirt")
    same(prior(E, "TOK", "turf", at_utc=at(E, "16:00")), (0.0, tau), "E")
    # Tokyo turf on P: its previous race day is E, seven days back
    s = model.sigma
    same(
        prior(P, "TOK", "turf", at_utc=at(P, "09:00")),
        decayed(*posterior([-2.0, -1.5], s, tau), (P - E).days, 7.0, tau),
        "P",
    )
    # an empty results frame
    empty = base_results(model).frame().iloc[0:0]
    for kind in KINDS:
        same(build(model, empty, kind)(D, "TOK", "turf", at_utc=T), (0.0, tau), kind)


# ---------------------------------------------------------------------------------------------
# (b) other_surface and (c) both
# ---------------------------------------------------------------------------------------------


def test_other_surface_prior_reads_the_same_days_other_surface_at_the_same_track(model):
    s, tau = model.sigma, model.tau
    r = base_results(model)
    resids = [x for x, _ in SAME_DIRT]  # available 10:30, 11:10, 11:50
    cuts = {0: at(D, "10:29"), 1: at(D, "10:31"), 2: at(D, "11:11"), 3: at(D, "11:51")}
    for w in (0.0, 0.35, 0.8, 1.0):
        prior = build(model, r, "other_surface", surface_weight=w)
        for n, t in cuts.items():
            expected = coupled(*posterior(resids[:n], s, tau), w, tau)
            same(prior(D, "TOK", "turf", at_utc=t), expected, (w, n))
        # the other way round: Tokyo dirt reads Tokyo turf (5.0 at 10:20, available 10:50);
        # never the 4.0 of the next day, nor Kyoto's turf
        same(prior(D, "TOK", "dirt", at_utc=at(D, "10:49")), (0.0, tau), (w, "dirt early"))
        same(
            prior(D, "TOK", "dirt", at_utc=at(D, "10:51")),
            coupled(*posterior([5.0], s, tau), w, tau),
            (w, "dirt"),
        )
    # a weight of 0 is the base whatever was seen
    same(
        build(model, r, "other_surface", surface_weight=0.0)(D, "TOK", "turf", at_utc=T), (0.0, tau)
    )
    # Kyoto turf on D reads Kyoto dirt (-3.0), never Tokyo's
    same(
        build(model, r, "other_surface")(D, "KYO", "turf", at_utc=T),
        coupled(*posterior([-3.0], s, tau), 0.8, tau),
        "KYO",
    )


def test_both_combines_the_previous_day_and_the_other_surface_by_their_information(model):
    s, tau = model.sigma, model.tau
    r = base_results(model).add(D, "KYO", "turf", 1.2, hhmm="10:05")
    for half_life, w in ((7.0, 0.8), (3.0, 0.5), (20.0, 1.0)):
        prior = build(model, r, "both", half_life_days=half_life, surface_weight=w)
        a = decayed(*posterior(PREV_TURF, s, tau), 1, half_life, tau)
        b = coupled(*posterior([x for x, _ in SAME_DIRT], s, tau), w, tau)
        same(prior(D, "TOK", "turf", at_utc=T), combined(a, b, tau), (half_life, w))
        # before any other-surface result: (a) alone
        same(prior(D, "TOK", "turf", at_utc=at(D, "10:29")), a, (half_life, w, "a only"))
        # no previous day (Kyoto dirt has no earlier day here) and Kyoto turf's 1.2: (b) alone
        same(
            prior(D, "KYO", "dirt", at_utc=T),
            coupled(*posterior([1.2], s, tau), w, tau),
            (half_life, w, "b only"),
        )
        # neither
        same(prior(D, "KYO", "dirt", at_utc=at(D, "09:00")), (0.0, tau), (half_life, w, "none"))
    # the combination is tighter than either part
    prior = build(model, r, "both")
    sd_both = prior(D, "TOK", "turf", at_utc=T)[1]
    assert sd_both < build(model, r, "previous_day")(D, "TOK", "turf", at_utc=T)[1]
    assert sd_both < build(model, r, "other_surface")(D, "TOK", "turf", at_utc=T)[1]


# ---------------------------------------------------------------------------------------------
# Leakage: nothing available at or after the decision time
# ---------------------------------------------------------------------------------------------

# where the extra result goes, and the kinds that may read it when it is usable
PLACES = {
    "same_day_other_surface": ((D, "TOK", "dirt", "11:45"), {"other_surface", "both"}),
    "previous_day": ((P, "TOK", "turf", "16:00"), {"previous_day", "both"}),
}
# availability relative to T, and whether that is before T
OFFSETS = {"1 min before": (-1, True), "exactly at": (0, False), "1 min after": (1, False)}
EXTRA = 3.0


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("place", sorted(PLACES))
@pytest.mark.parametrize("offset", [*OFFSETS, "never"])
def test_no_prior_reads_a_result_available_at_or_after_the_decision_time(
    model, kind, place, offset
):
    (day, track, surface, hhmm), readers = PLACES[place]
    usable = OFFSETS[offset][1] if offset in OFFSETS else False
    avail = None if offset == "never" else T + timedelta(minutes=OFFSETS[offset][0])
    r = base_results(model).add(day, track, surface, EXTRA, hhmm=hhmm, avail=avail)
    got = build(model, r, kind)(D, "TOK", "turf", at_utc=T)

    counts = usable and kind in readers
    prev, dirt = list(PREV_TURF), [x for x, _ in SAME_DIRT]
    if counts and place == "previous_day":
        prev.append(EXTRA)
    if counts and place == "same_day_other_surface":
        dirt.append(EXTRA)
    same(got, expected_base(model, kind, prev=prev, dirt=dirt), (kind, place, offset))
    # without the extra result: the same prior unless it was usable
    clean = build(model, base_results(model), kind)(D, "TOK", "turf", at_utc=T)
    if not counts:
        same(got, clean, "unchanged")
    else:
        assert abs(got[0] - clean[0]) > 1e-3, (kind, place, offset)


@pytest.mark.parametrize("kind", KINDS)
def test_no_prior_reads_its_own_surface_that_day_a_later_day_or_another_track(model, kind):
    """At a t three days later every result here is available; the prior of (D, TOK, turf)
    still reads only P's Tokyo turf and D's Tokyo dirt."""
    late = at(D + timedelta(days=3), "09:00")
    r = base_results(model)
    r.add(D, "TOK", "turf", -4.0, hhmm="11:00")
    r.add(D + timedelta(days=1), "TOK", "dirt", 2.5)
    r.add(D + timedelta(days=2), "TOK", "turf", 2.5)
    r.add(D, "KYO", "turf", 2.0, hhmm="11:00")
    r.add(P, "KYO", "dirt", 2.0, hhmm="11:00")
    r.add(D, "TOK", "dirt", -0.7, hhmm="14:00")  # usable at `late`, not at T
    prior = build(model, r, kind)
    dirt_late = [x for x, _ in SAME_DIRT] + [-0.7]
    same(prior(D, "TOK", "turf", at_utc=late), expected_base(model, kind, dirt=dirt_late), kind)
    same(prior(D, "TOK", "turf", at_utc=T), expected_base(model, kind), kind)
    # the SpeedModel shape: at(t) binds the decision time
    for t in (T, late, at(D, "10:29")):
        bound = prior.at(t)
        same(bound(D, "TOK", "turf"), prior(D, "TOK", "turf", at_utc=t), (kind, t))


def test_carry_over_prior_refuses_an_unknown_kind_and_bad_parameters(model):
    r = base_results(model).frame()
    with pytest.raises(ValueError):
        CarryOverPrior(model, r, kind="none", half_life_days=7.0, surface_weight=0.5)
    for bad in (0.0, -1.0):
        for kind in ("previous_day", "both"):
            with pytest.raises(ValueError):
                CarryOverPrior(model, r, kind=kind, half_life_days=bad, surface_weight=0.5)
    for bad in (-0.1, 1.5):
        for kind in ("other_surface", "both"):
            with pytest.raises(ValueError):
                CarryOverPrior(model, r, kind=kind, half_life_days=7.0, surface_weight=bad)


# ---------------------------------------------------------------------------------------------
# Injected into SpeedModel: the fit is unchanged, the prediction starts from the prior
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_an_injected_prior_leaves_the_fit_and_starts_the_same_day_variant_from_it(
    train, model, kind
):
    s, tau = model.sigma, model.tau
    results = base_results(model).frame()
    prior = build(model, results, kind)
    cell = dict(zip(PAR_KEYS, CELLS[("TOK", "turf")][0], strict=True))
    par = model.par(**cell)
    # target 1: 10:20 Tokyo turf, t 10:18, no same-day Tokyo turf or dirt result usable yet;
    # target 2: 12:00 Tokyo turf, t T, after the 10:20 turf (5.0) and the three dirt results
    t1 = at(D, "10:18")
    targets = {
        "first": (t1, expected_base(model, kind, dirt=[]), []),
        "later": (T, expected_base(model, kind), [5.0]),
    }
    for name, (t, (m0, s0), same_day) in targets.items():
        race = {"race_id": f"target:{name}", "race_utc": t + timedelta(minutes=2), **cell}
        on = SpeedModel(NORMAL, variant_prior=prior.at(t)).fit(train).observe(results)
        off = SpeedModel(NORMAL_OFF, variant_prior=prior.at(t)).fit(train).observe(results)
        base = SpeedModel(NORMAL).fit(train).observe(results)
        # #96's ruling: the fit never reads the prior
        for m in (on, off):
            assert m.sigma == pytest.approx(s, abs=1e-12) and m.tau == pytest.approx(tau, abs=1e-12)
            for cells in CELLS.values():
                for c in cells:
                    keys = dict(zip(PAR_KEYS, c, strict=True))
                    assert m.par(**keys) == pytest.approx(model.par(**keys), abs=1e-12)
        # variant off: the par plus the prior's mean
        assert off.expected_top3_time(race, at_utc=t) == pytest.approx(par + m0, abs=ATOL)
        # variant on: the normal posterior from the prior given the same-day residuals
        prec = s0**-2 + len(same_day) / s**2
        mean = (m0 * s0**-2 + sum(same_day) / s**2) / prec
        assert on.expected_top3_time(race, at_utc=t) == pytest.approx(par + mean, abs=ATOL), name
        # #305's variant on, without the prior, from N(0, tau^2)
        plain = posterior(same_day, s, tau)[0]
        assert base.expected_top3_time(race, at_utc=t) == pytest.approx(par + plain, abs=ATOL)


# ---------------------------------------------------------------------------------------------
# The study: a synthetic store calendar
# ---------------------------------------------------------------------------------------------

FOLD_SCHEME = FoldScheme(
    unit="week",
    test_length=2,
    step=2,
    min_train=2,
    window="expanding",
    embargo_days=1,
    anchor_month=1,
)
FIRST_DAY = date(2026, 10, 17)  # a Saturday
LAST_DAY = date(2026, 12, 27)  # past the holdout start: some Sundays are holdout days
# 40 min apart: every earlier race's result (post + 30) is usable at the next one's T-2
POSTS = ("10:00", "10:40", "11:20", "12:00", "12:40", "13:20")
SURFACES = ("dirt", "dirt", "turf", "turf", "dirt", "turf")  # idx 0 and 2 see none of their own
STUDY_PRIORS = {
    "previous_day": {"half_life_days": 7.0},
    "other_surface": {"surface_weight": 0.8},
    "both": {"half_life_days": 7.0, "surface_weight": 0.8},
}
# two development Saturdays of the last test window, a holdout Sunday between them; on each the
# 3rd race (the first turf race) is moved so that its T-2 meets the 2nd race's result exactly
# (PROBE_AT, Tokyo) or a minute after it (PROBE_BEFORE, Kyoto)
PROBE_AT = date(2026, 12, 19)
PROBE_BEFORE = date(2026, 12, 26)
POST_OVERRIDES = {(PROBE_AT, "TOK", 2): "11:12", (PROBE_BEFORE, "KYO", 2): "11:13"}
STORE_CELLS = {
    (venue, surface): [(c[2], c[3], c[5]) for c in CELLS[(venue, surface)]]
    for venue, surface in CELLS
}


def weekend_days(first, last):
    days, d = [], first
    while d <= last:
        if d.weekday() in (5, 6):
            days.append(d)
        d += timedelta(days=1)
    return days


def planted(day: date, venue: str) -> float:
    """The (day, venue) variant, shared by both surfaces and by a weekend's two days; at least
    0.8 s either way, alternating by weekend, the venues mirrored."""
    week = (day - FIRST_DAY).days // 7
    size = (0.8, 1.2, 1.6)[week % 3]
    return (1.0 if week % 2 == 0 else -1.0) * size * (1.0 if venue == "TOK" else -1.0)


def rid(day, venue, idx):
    return f"jra:{day.isoformat()}:{venue}:{idx + 1:02d}"


@pytest.fixture(scope="module")
def store():
    """(races, runners, truth): store-shaped frames for `speed_run.build_race_table`."""
    rng = np.random.default_rng(3280)
    races, runners, truth = [], [], []
    for day in weekend_days(FIRST_DAY, LAST_DAY):
        for venue in ("TOK", "KYO"):
            g = planted(day, venue)
            seen = {"turf": 0, "dirt": 0}
            for idx, (post, surface) in enumerate(zip(POSTS, SURFACES, strict=True)):
                cells = STORE_CELLS[(venue, surface)]
                distance, layout, cls = cells[seen[surface] % 2]
                full = (venue, surface, distance, layout, "unknown", cls)
                top3 = PARS[full] + g + rng.normal(0.0, 0.25)
                race_id = rid(day, venue, idx)
                races.append(
                    {
                        "race_id": race_id,
                        "meeting_id": f"jra:{day.isoformat()}:{venue}",
                        "race_date": day,
                        "venue": venue,
                        "surface": surface,
                        "distance_m": distance,
                        "course_config": layout,
                        "rail_setting": None,
                        "race_class": cls,
                        "post_time_local": POST_OVERRIDES.get((day, venue, idx), post),
                    }
                )
                # 1st, 2nd and 3rd 0.3 s apart: the mean placed 1-3 time is top3
                for pos in (1, 2, 3):
                    runners.append(
                        {
                            "race_id": race_id,
                            "finish_pos": pos,
                            "finish_time_s": top3 - 0.3 + 0.3 * (pos - 1),
                        }
                    )
                truth.append(
                    {"race_id": race_id, "idx": idx, "venue": venue, "variant": g}
                    | {"seen_own": seen[surface], "day": day}
                )
                seen[surface] += 1
    runners = pd.DataFrame(runners)
    runners["finish_pos"] = runners["finish_pos"].astype("Int64")
    return pd.DataFrame(races), runners, pd.DataFrame(truth).set_index("race_id")


@pytest.fixture(scope="module")
def table(store):
    races, runners, _ = store
    built, _ = speed_run.build_race_table(races, runners, estimand="mean_placed_1_3")
    assert len(built) == len(races)
    return built


def run(table, decision_min=2, priors=STUDY_PRIORS, spec=NORMAL):
    out = study.predict_prior_folds(
        table, spec=spec, scheme=FOLD_SCHEME, decision_min=decision_min, priors=priors
    )
    return out.set_index("race_id")


@pytest.fixture(scope="module")
def predictions(table):
    return run(table)


def prior_columns():
    return [f"{k}:{p}" for k in KINDS for p in PREDICTIONS]


def test_the_prior_fold_loop_keeps_305s_predictions_and_adds_one_pair_per_prior(table, predictions):
    c3 = speed_run.predict_folds(table, spec=NORMAL, scheme=FOLD_SCHEME, decision_min=2)
    c3 = c3.set_index("race_id")
    assert len(predictions) > 100
    assert set(predictions.index) == set(c3.index)
    rows = predictions.loc[c3.index]
    for column in ("par_only", "variant_on", "variant_off", "top3_time_s"):
        assert np.allclose(rows[column], c3[column], rtol=0, atol=ATOL), column
    assert (rows["fold"] == c3["fold"]).all()
    assert (pd.to_datetime(rows["decision_utc"], utc=True) == c3["decision_utc"]).all()
    for column in prior_columns():
        assert column in predictions.columns, column
        assert np.isfinite(predictions[column].to_numpy(dtype=float)).all(), column
    # no holdout day is scored
    assert not any(
        is_holdout_day("japan", pd.Timestamp(d).date()) for d in predictions["race_date"]
    )


def test_other_surface_prior_moves_a_surfaces_first_race_toward_the_days_variant(
    store, predictions
):
    _, _, truth = store
    plant = truth.loc[predictions.index]
    shift = predictions["other_surface:variant_off"] - predictions["par_only"]
    # idx 2, the first turf race, follows two dirt results: toward the shared variant
    first_turf = (plant["idx"] == 2).to_numpy()
    toward = (shift[first_turf] * np.sign(plant.loc[first_turf, "variant"])).to_numpy()
    assert first_turf.sum() >= 20
    assert (toward > 0).mean() > 0.95 and toward.mean() > 0.4, toward
    # idx 0, the day's first race: no other-surface result yet, so the base prior
    first = (plant["idx"] == 0).to_numpy()
    for p in PREDICTIONS:
        assert np.allclose(
            predictions.loc[first, f"other_surface:{p}"], predictions.loc[first, p], atol=ATOL
        ), p
    # over the races with no same-day result, it beats #305's variant on
    no_own = (plant["seen_own"] == 0).to_numpy()
    actual = predictions.loc[no_own, "top3_time_s"]

    def rmse(column):
        return math.sqrt(float(((predictions.loc[no_own, column] - actual) ** 2).mean()))

    assert rmse("other_surface:variant_on") < rmse("variant_on") - 0.2


def test_previous_day_prior_carries_saturdays_variant_into_sundays_first_races(store, predictions):
    _, _, truth = store
    plant = truth.loc[predictions.index]
    fold_of = {
        (pd.Timestamp(d).date(), v): f
        for d, v, f in zip(
            predictions["race_date"], plant["venue"], predictions["fold"], strict=True
        )
    }
    day = [pd.Timestamp(d).date() for d in predictions["race_date"]]
    sunday_first = np.array(
        [
            dd.weekday() == 6 and idx in (0, 2) and fold_of.get((dd - timedelta(days=1), v)) == f
            for dd, idx, v, f in zip(
                day, plant["idx"], plant["venue"], predictions["fold"], strict=True
            )
        ]
    )
    assert sunday_first.sum() >= 8
    shift = predictions["previous_day:variant_off"] - predictions["par_only"]
    toward = (shift[sunday_first] * np.sign(plant.loc[sunday_first, "variant"])).to_numpy()
    assert (toward > 0).all() and toward.mean() > 0.5, toward
    # the day's first race has no other-surface result: (c) is (a)
    first = (plant["idx"] == 0).to_numpy()
    for p in PREDICTIONS:
        assert np.allclose(
            predictions.loc[first, f"both:{p}"],
            predictions.loc[first, f"previous_day:{p}"],
            atol=ATOL,
        ), p


ALL_COLUMNS = ["par_only", "variant_on", "variant_off", *prior_columns()]


def values(frame, rows, columns=ALL_COLUMNS) -> np.ndarray:
    return frame.loc[rows, columns].to_numpy(dtype=float)


@pytest.fixture(scope="module")
def shifted(table, predictions):
    """The fold loop again on the calendar with, all in the last test window (so no fit moves):
    the 2nd Tokyo dirt result of PROBE_AT (10:40, available 11:10, exactly the 3rd race's t) and
    the 2nd Kyoto dirt result of PROBE_BEFORE (available 11:10, a minute before its 3rd race's
    t) +5 s; every result of LAST_DAY +5 s; every holdout day's result +/- 40 s."""
    days = pd.Series([pd.Timestamp(d).date() for d in table["race_date"]], index=table.index)
    out = table.copy()
    moved = out["race_id"].isin([rid(PROBE_AT, "TOK", 1), rid(PROBE_BEFORE, "KYO", 1)])
    out.loc[moved | (days == LAST_DAY), "top3_time_s"] += 5.0
    held = days.map(lambda d: is_holdout_day("japan", d)).to_numpy(dtype=bool)
    assert held.any() and not is_holdout_day("japan", PROBE_AT)
    out.loc[held, "top3_time_s"] += np.where(out.loc[held, "surface"] == "turf", 40.0, -40.0)
    return run(out)


def test_no_prediction_reads_a_later_day_or_a_holdout_day(predictions, shifted):
    """Every race scored before PROBE_AT is unchanged: the last day's results and the holdout
    days' (a development Saturday after a holdout Sunday reads no holdout result as its
    previous race day) never reach a prediction."""
    assert set(shifted.index) == set(predictions.index)
    day = pd.Series(
        [pd.Timestamp(d).date() for d in predictions["race_date"]], index=predictions.index
    )
    before = day.index[day < PROBE_AT]
    assert len(before) > 80
    assert np.allclose(values(shifted, before), values(predictions, before), atol=ATOL)
    # PROBE_BEFORE's Kyoto races before the moved one: the next day's results never count
    early = [rid(PROBE_BEFORE, "KYO", k) for k in (0, 1)]
    assert np.allclose(values(shifted, early), values(predictions, early), atol=ATOL)


def test_a_same_day_result_moves_the_prior_only_when_available_before_t(predictions, shifted):
    """PROBE_AT: the first Tokyo turf race (11:12) has t = 11:10, exactly when the moved dirt
    result is available: nothing moves. PROBE_BEFORE: Kyoto's (11:13) has t = 11:11: the
    other-surface priors move, nothing else does."""
    readers = [f"{k}:{p}" for k in ("other_surface", "both") for p in PREDICTIONS]
    at_t = [rid(PROBE_AT, "TOK", k) for k in (0, 1, 2)]
    assert np.allclose(values(shifted, at_t), values(predictions, at_t), atol=ATOL)
    # Kyoto that day is another track
    kyoto = [rid(PROBE_AT, "KYO", k) for k in range(6)]
    assert np.allclose(values(shifted, kyoto), values(predictions, kyoto), atol=ATOL)

    race = rid(PROBE_BEFORE, "KYO", 2)
    delta = shifted.loc[race, ALL_COLUMNS].astype(float) - predictions.loc[
        race, ALL_COLUMNS
    ].astype(float)
    for column in ALL_COLUMNS:
        if column in readers:
            assert delta[column] > 0.3, (column, delta[column])
        else:
            assert delta[column] == pytest.approx(0.0, abs=ATOL), (column, delta[column])


# ---------------------------------------------------------------------------------------------
# The spec, compute_study and run_study
# ---------------------------------------------------------------------------------------------

BOOT = {"cluster_unit": "meeting", "sided": "one-sided", "level": LEVEL, "n_boot": 200, "seed": 328}


def small_boot(spec):
    return spec | {"bootstrap": dict(spec["bootstrap"]) | {"n_boot": 200}}


def test_the_spec_names_the_three_priors_and_their_parameters_before_scoring(model):
    spec = study.build_spec(BOOT, FOLD_SCHEME, 2)
    assert study.FAMILY_ID == spec["family_id"] == "japan-speed-top3-c3-prior"
    assert study.TICKET == spec["ticket"] == 328
    assert study.BASE_FAMILY_ID == spec["base_family"] == speed_run.FAMILY_ID
    assert spec["sources"] == ["jra"]
    json.dumps(spec)  # plain JSON, as freeze_family stores it
    priors = spec["priors"]
    assert tuple(sorted(priors)) == tuple(sorted(KINDS))
    for kind in ("previous_day", "both"):
        h = priors[kind]["half_life_days"]
        assert isinstance(h, (int, float)) and not isinstance(h, bool) and h > 0, kind
    for kind in ("other_surface", "both"):
        w = priors[kind]["surface_weight"]
        assert isinstance(w, (int, float)) and not isinstance(w, bool) and 0 <= w <= 1, kind
    # each kind's parameters are the prior's own keywords
    r = base_results(model).frame()
    for kind in KINDS:
        prior = CarryOverPrior(model, r, kind=kind, **priors[kind])
        mean, sd = prior(D, "TOK", "turf", at_utc=T)
        assert math.isfinite(mean) and sd > 0


def components(v):
    c = v.components
    return c["error_model"], c["prior"], c["prediction"], c["baseline"]


def metric(v, fold):
    return next((m for m in v.metrics if (m.fold, m.metric) == (fold, "rmse_gain_s")), None)


@pytest.fixture(scope="module")
def study_case(table):
    spec = small_boot(study.build_spec(BOOT, FOLD_SCHEME, 2))
    variants, tables, counts = study.compute_study(table, spec, FOLD_SCHEME, 2)
    return spec, variants, (variants, tables, counts)


def results_seen(predictions, table):
    """Per scored race, the races of its (day, track, surface) usable before t, capped at 6."""
    days = [pd.Timestamp(d).date() for d in table["race_date"]]
    avail = {}
    for d, tr, su, a in zip(
        days, table["track"], table["surface"], table["result_available_utc"], strict=True
    ):
        avail.setdefault((d, tr, su), []).append(pd.Timestamp(a))
    out = {}
    for race_id, row in predictions.iterrows():
        key = (pd.Timestamp(row["race_date"]).date(), row["track"], row["surface"])
        t = pd.Timestamp(row["decision_utc"])
        out[race_id] = min(6, sum(a < t for a in avail.get(key, [])))
    return pd.Series(out)


def rmse(err):
    return math.sqrt(float(np.mean(np.asarray(err, dtype=float) ** 2)))


def test_compute_study_records_every_prior_prediction_and_baseline_with_bounds(
    table, predictions, study_case
):
    spec, variants, _ = study_case
    assert len({v.name for v in variants}) == len(variants)
    grid = {components(v) for v in variants}
    assert grid == set(itertools.product(ERROR_MODELS, KINDS, PREDICTIONS, BASELINES))
    assert len(variants) == len(grid)
    for v in variants:
        assert v.error is None, (v.name, v.error)
        overall = metric(v, "all")
        assert overall is not None and overall.lo is not None and overall.hi is not None, v.name
        assert overall.lo <= overall.value + 1e-12 <= overall.hi + 2e-12, v.name

    # the normal error model's gains, recomputed from its fold loop
    preds = predictions if spec["priors"] == STUDY_PRIORS else run(table, priors=spec["priors"])
    seen = results_seen(preds, table)
    assert set(seen) == {0, 1, 2}
    actual = preds["top3_time_s"]
    for v in variants:
        error_model, kind, prediction, baseline = components(v)
        if error_model != "normal":
            continue
        model_err = preds[f"{kind}:{prediction}"] - actual
        base_err = preds[baseline] - actual
        overall = metric(v, "all")
        assert overall.value == pytest.approx(rmse(base_err) - rmse(model_err), abs=1e-9), v.name
        assert overall.n == len(preds)
        for k in (0, 1, 2):
            m = metric(v, f"results_seen:{k}")
            assert m is not None and m.lo is not None and m.hi is not None, (v.name, k)
            ids = seen.index[seen == k]
            expected = rmse(base_err[ids]) - rmse(model_err[ids])
            assert m.value == pytest.approx(expected, abs=1e-9), (v.name, k)
            assert m.n == len(ids), (v.name, k)
            assert m.lo <= m.value + 1e-12 <= m.hi + 2e-12, (v.name, k)
        assert metric(v, "results_seen:3") is None, v.name  # no race saw three


def test_compute_study_records_every_variant_when_the_fold_loop_fails(table):
    # one weekend: no fold has a training window, nothing is scored
    days = pd.Series([pd.Timestamp(d).date() for d in table["race_date"]])
    tiny = table[days.isin(weekend_days(FIRST_DAY, FIRST_DAY + timedelta(days=1))).to_numpy()]
    spec = small_boot(study.build_spec(BOOT, FOLD_SCHEME, 2))
    variants, _, _ = study.compute_study(tiny, spec, FOLD_SCHEME, 2)
    grid = {components(v) for v in variants}
    assert grid == set(itertools.product(ERROR_MODELS, KINDS, PREDICTIONS, BASELINES))
    for v in variants:
        assert isinstance(v.error, str) and v.error.strip(), v.name


def test_run_study_freezes_the_spec_first_and_records_each_variant(
    monkeypatch, tmp_path, store, table, study_case
):
    """The store read and the fold loop are stood in for (the fold loop's output is
    compute_study's on this calendar, `study_case`); what is pinned is the order and the records."""
    races, runners, _ = store
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str((tmp_path / "data").resolve()))
    registry = Registry(tmp_path / "registry" / "japan.jsonl", holdout_log=tmp_path / "h.jsonl")
    seen_frozen = {}

    def read_inputs():
        seen_frozen["read"] = registry.frozen_spec_hash(study.FAMILY_ID)
        return races, runners

    def compute(built, spec, scheme, minutes):
        seen_frozen["compute"] = registry.frozen_spec_hash(study.FAMILY_ID)
        seen_frozen["spec_priors"] = spec["priors"]
        assert set(built["race_id"]) == set(table["race_id"])  # the builder's table of the store
        return study_case[2]

    monkeypatch.setattr(study, "default_registry", lambda country: registry)
    monkeypatch.setattr(study, "read_inputs", read_inputs)
    monkeypatch.setattr(
        study, "store_builds", lambda: {"jra_races": "processed/jra/races@synthetic"}
    )
    monkeypatch.setattr(study, "_git_sha", lambda: "c" * 40)
    monkeypatch.setattr(study, "write_summary", lambda reg, directory: None)
    monkeypatch.setattr(study, "compute_study", compute)

    result = study.run_study(tmp_path / "out")

    frozen = registry.frozen_spec_hash(study.FAMILY_ID)
    assert frozen is not None and result["spec_hash"] == frozen
    assert seen_frozen["read"] == frozen and seen_frozen["compute"] == frozen
    assert result["family_id"] == study.FAMILY_ID
    assert set(seen_frozen["spec_priors"]) == set(KINDS)

    records = registry.records(study.FAMILY_ID)
    grid = {
        (
            r.components["error_model"],
            r.components["prior"],
            r.components["prediction"],
            r.components["baseline"],
        )
        for r in records
    }
    assert grid == set(itertools.product(ERROR_MODELS, KINDS, PREDICTIONS, BASELINES))
    assert len(records) == len(grid)
    for r in records:
        assert r.ticket == 328 and r.git_sha == "c" * 40, r.variant
        assert tuple(r.sources) == ("jra",), r.sources
        assert r.status == "scored", (r.variant, r.params)
    # #305's family is not written by this study
    assert registry.records(speed_run.FAMILY_ID) == []

    # the report: each variant's no-same-day-result gain over #305's variant on, with bounds
    lines = open(result["report_path"], encoding="utf-8").read().splitlines()
    for r in records:
        if r.components["baseline"] != "variant_on":
            continue
        m = next(m for m in r.metrics if (m.fold, m.metric) == ("results_seen:0", "rmse_gain_s"))
        cells = [f"{x:.4f}" for x in (m.value, m.lo, m.hi)]
        assert any(
            r.components["prior"] in line and all(c in line for c in cells) for line in lines
        ), (
            r.variant,
            cells,
        )
