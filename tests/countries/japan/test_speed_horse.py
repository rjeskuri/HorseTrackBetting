"""The C3 horse-level finish-speed model (ticket #97, epic #73): theta_i(t), p_speed, same-day bias.

Contract pinned here (`horsetrack.countries.japan.models.speed_horse`, and #96's `speed`):
- `HorseSpeed(*, drift_var_per_day, t_df)`: theta_i, the horse's relative speed against its race's
  expected top-3 time from #96's `SpeedModel` (par + variant), is a random walk whose variance
  grows by `drift_var_per_day` (s^2) a day; the run error is Student-t with `t_df` degrees of
  freedom, normal when None. A negative or non-finite drift, or a t_df that is not a positive
  number, is a ValueError.
- `fit(runs, speed) -> HorseSpeed` (the returned object is the one used), `speed` a fitted and
  observed `SpeedModel`. runs: one row per runner with race_id, race_utc, horse_id, finish_time_s,
  style (front, stalker, midpack, closer: the caller's, read from earlier races), early_pos_mean3
  (#129's mean early position over the horse's newest 3 earlier starts, 0 at the front, 1 at the
  back: the caller's, read from earlier races; null when there is none), weight_kg, draw,
  result_available_utc (#96's name: when the result was published, null if never) and the race's
  columns of `speed.spec.par_keys`; both times tz-aware UTC (else ValueError), a missing column a
  ValueError. fit estimates the global terms (the error scale `sigma`, the population prior
  `prior_mean` and `prior_sd`, the weight slope `weight_slope` about `weight_ref`, the pace slopes
  `pace_slopes` and `pace_race` about `pace_ref`) on these runs and keeps the runs as the horses'
  history. Fitting again starts afresh: earlier observations are gone.
- `observe(runs) -> HorseSpeed` (the returned object is the one used): runs after the fit, with
  fit's columns, join the horses' history only, each taken against the frozen global terms on the
  baseline a fit run would have (as of its off; the in-sample one below); the global terms never
  change, and observing in one call or in several is the same. A race that contributed a run to
  the fit or to an earlier observe (a run with a finish time, a par and a publication) is a
  ValueError, and the whole frame is checked first: a refused frame leaves the model as it was. A
  race all of whose runs were dropped (no finish time, or no publication: still pending) is not
  taken, fitted or observed: observed again once it has resulted, it counts, as if that were its
  first time. An observed run whose result is never published (null) is never available.
- A run's baseline, fitted or observed, is read from the speed model as it stands when predict
  needs it: a race-level result the speed model observes after a horse run was observed, published
  before that run's off, reaches the run as if the speed model had observed it first, so the same
  results predict alike in batch and incremental replay. The global terms stay as fitted.
- `predict(race_runners, *, at_utc) -> DataFrame`: race_runners is a card (race_id, race_utc,
  horse_id, style, early_pos_mean3, weight_kg, draw and the par keys; a missing one a ValueError;
  result columns are not needed and are ignored when present). One row per runner, in the card's
  order, with horse_id, mu_s (the runner's expected finish time, s: the card's expected top-3 time
  at at_utc plus the runner's terms) and sd_s. As of at_utc (tz-aware, else ValueError): theta_i
  moves only with that horse's runs, fitted or observed, whose result_available_utc is strictly
  before at_utc (null: never), and no result published at or after at_utc reaches it through a
  run's baseline either: changing a result published at or after t, observed or seen by the speed
  model, leaves every mu_s and sd_s at t exactly as they were. Nor does one reach the global terms
  or the speed model's fit: an at_utc at or before the latest of (i) the result_available_utc of
  the fit runs the terms used (a finish time, a par, a publication: an untimed run or one in a cell
  with no par, published later, does not move it), (ii) the instant after which such a run's
  in-sample baseline (below) is used, and (iii) the speed model's fit cutoff, the latest
  result_available_utc among the races with a top-3 time in the frame given to `SpeedModel.fit`
  (an untimed one, published late or never, does not move it), is a ValueError, whatever either
  model observed since. A timed race of the speed model's fit frame with no publication leaves no
  safe at_utc: refused at a fit (ValueError) or at every predict (which one is not pinned). A fit
  run with a time and no publication is either refused at fit (ValueError) or reaches no term and
  no prediction (which one is not pinned). A horse with no run gets the prior, centred on the
  population. Before fit: RuntimeError.
- sd_s is the runner's predictive SD: sqrt(Var theta_i + sigma^2 + V_card) under the normal error,
  sqrt(Var theta_i + t_df / (t_df - 2) * (sigma^2 + V_card)) under the Student-t with t_df > 2,
  and +inf when t_df <= 2 (that t has no finite variance; mu_s stays finite). V_card is the
  variance of the card's day variant at at_utc (`SpeedModel.variant`); a horse with no run has
  Var theta_i = prior_sd^2 (no drift), and a horse's Var theta_i is the same on two cards off at
  the same instant.
- The pace term (the owner, 2026-11-26, Q2 (c)): pace_r of a race is the mean of its rows'
  non-null early_pos_mean3, centred at the fit runs' mean; a runner's style chooses its slope on
  it. Two cards with the same early positions and other styles have the same pace_r; a null is
  skipped, not read as a value; a race with no early position at all has no pace term (finite,
  its unseen front runner and closer level).
- Runs on the speed model's fit days (the owner, 2026-11-26, Q3 (b)), fitted or observed: a run on
  a (day, group) of the speed model's fit frame is taken against the fit's in-sample variant of
  that (day, group) only at a t after every race of it in the frame given to `SpeedModel.fit`,
  with a top-3 time or without one, had its result published strictly before t (a null
  publication on any of them: never); until then it is on the prior path, as before.
  `SpeedModel.in_sample_variant`'s third element is that instant (None: never); its mean and SD
  come from the timed races alone, as if the untimed rows were not there. Since predict refuses a
  t at or before the speed model's fit cutoff, a (day, group) of timed races alone is fully
  resulted at every t predict takes: the wait shows at predict only through an untimed race
  published after that cutoff, or never.
- `p_speed(mu, *, scale, method='pl') -> ndarray`, one race, in mu's order: 'pl' is
  softmax(-mu / scale); 'henery' is P(runner i's time is the smallest) for independent
  N(mu_i, scale^2) times. ValueError for an unknown method, a scale that is not positive, a mu
  that is not finite.
- `same_day_bias(results, *, at_utc, venue) -> DataFrame`: results has race_id, race_utc, venue,
  horse_id, draw, style (the runner's running style in that race), finish_pos and
  result_available_utc. It reads only the `venue` races of the JST day of at_utc whose result was
  published strictly before at_utc. Seven rows, (factor, level) the draw thirds (inner, middle,
  outer) and the four styles, with `bias` (> 0: that level's runners finished ahead of their share,
  0 with nothing resulted) and `n_races` (the races read).
- `SpeedModel.expected_top3_time(race, *, at_utc, pace_s=0.0)`: #96's par + variant at t, plus
  pace_s, the race's pace term in seconds (the #97 addition); a pace_s that is not finite is a
  ValueError. Without pace_s it is #96's value.

The world: 12 days of races fit the speed model's pars; 16 days after them carry the horse runs,
their race-level results observed by the speed model. HorseSpeed's global terms are fitted on the
runs of the first 15; the runs of the last, TD, the day the as-of checks sweep, are observed. Runs
are planted as par + day variant + theta_i + 0.15 s/kg * (weight - 56) + noise, with no pace or
style effect. A second world (below) plants a pace effect by the field's mean early position and
puts observed horse runs on a fit day of the speed model. Expected values come from the planted
worlds, never from the code under test. Not pinned: the estimator of the global terms (error
scale, weight, pace x style, the prior), whether they read fit-day runs against the in-sample
variant, how a fit run with a time and no publication is kept out of them, which fit refuses a
timed speed-fit race with no publication (or whether predict does), whether observe takes runs
published before the fit's latest publication, how a run's baseline is cached, how p_speed's scale
is fitted, the bias score and its shrinkage. Every frame is synthetic; nothing reads a store.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import math
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec
from horsetrack.countries.japan.models.speed_horse import HorseSpeed, p_speed, same_day_bias

PAR_KEYS = ("track", "surface", "distance")
VARIANT_KEYS = ("track", "surface")
STYLES = ("front", "stalker", "midpack", "closer")
TOK, KYO = ("TOK", "turf"), ("KYO", "turf")
RUN_COLUMNS = (
    "race_id",
    "race_utc",
    *PAR_KEYS,
    "horse_id",
    "finish_time_s",
    "style",
    "early_pos_mean3",
    "weight_kg",
    "draw",
    "result_available_utc",
)
RACE_COLUMNS = ("race_id", "race_utc", *PAR_KEYS, "top3_time_s", "result_available_utc")

# ---------------------------------------------------------------------------------------------
# The planted world
# ---------------------------------------------------------------------------------------------

PAR = {(*TOK, 1600): 96.0, (*TOK, 2000): 120.5, (*KYO, 1600): 95.4, (*KYO, 2000): 120.0}
W_PER_KG = 0.15  # s per kg carried
REF_KG = 56.0
WEIGHTS = (54.0, 55.0, 56.0, 57.0, 58.0)
EPS_SD = 0.4  # run noise, s
V_SD = 0.5  # day variant spread, s
N_REG = 120  # regular horses, theta ~ N(0.8, 0.8^2) s behind the par, constant
THETA_MEAN, THETA_SD = 0.8, 0.8
FIELD = 8  # regulars per race
DRIFT = 0.01  # s^2 a day: the default HorseSpeed drift here
RESULT_DELAY = timedelta(minutes=10)

S0 = date(2025, 1, 4)
FIT_DAYS = tuple(S0 + timedelta(days=k) for k in range(12))  # the speed model's fit frame
H_DAYS = tuple(S0 + timedelta(days=12 + k) for k in range(16))  # the horse runs, observed
TD = H_DAYS[-1]
NEXT = TD + timedelta(days=1)  # a card day: no race of it is in any frame
LATER = NEXT + timedelta(days=60)

# A normal day: (post UTC (hh, mm), distance) of R1..R3 per (track, surface).
DAILY = {
    TOK: (((1, 5), 1600), ((2, 15), 2000), ((3, 25), 1600)),
    KYO: (((1, 10), 1600), ((2, 20), 2000), ((3, 30), 1600)),
}
# TD's Tokyo turf, R1..R7: (post, distance, publication (hh, mm) or None). R2's result is never
# published; R3 is held after an inquiry until 03:05, after R4's; R7's runners are not in the runs
# frame, only its race-level result (the speed model sees it). Planted variant: exactly 0.
TD_TOK = (
    ((1, 5), 1600, (1, 15)),
    ((1, 40), 2000, None),
    ((2, 15), 1600, (3, 5)),
    ((2, 50), 1600, (3, 0)),
    ((3, 25), 2000, (3, 35)),
    ((4, 0), 1600, (4, 10)),
    ((4, 35), 1600, (4, 45)),
)
R8_POST = (5, 10)  # a TD Tokyo turf card after R7
R7_SLOWER = 5.0  # s added to R7's top-3 time in the perturbed speed model

# Six twin horses, theta 0.3, midpack, 56 kg, draw 9 (shared, so no draw term can tell them
# apart): the same runs, dead heats, on six horse days before TD. On TD they part: a and e run R1
# (published 01:15), a 2.0 s quicker than usual, e eased 5.0 s behind it; n runs R2 (never
# published); d runs R3 (off 02:15, published 03:05); r runs R6 (off 04:00, published 04:10), each
# 2.0 s quicker. ref does not run on TD.
TWIN_THETA = 0.3
TWIN_STYLE = "midpack"
TWIN_DRAW = 9
TWINS = ("ref", "a", "n", "d", "r", "e")
TWIN_HISTORY = {1: (TOK, 1), 3: (KYO, 2), 5: (TOK, 3), 7: (KYO, 1), 9: (TOK, 2), 11: (KYO, 3)}
TWIN_TD = {"a": (1, -2.0), "e": (1, 5.0), "n": (2, -2.0), "d": (3, -2.0), "r": (6, -2.0)}

# The shifter: KYO turf R2 on eight horse days, 1.5 s slow on the first four, 1.5 s quick on the
# last four, no noise.
SHIFTER = "syn-shift"
SHIFTER_THETA = {0: 1.5, 2: 1.5, 4: 1.5, 6: 1.5, 8: -1.5, 10: -1.5, 12: -1.5, 14: -1.5}

# early_pos_mean3 here (no effect planted on it): a regular's own level, drawn once, with a little
# noise on each run and null on a tenth of its runs; the twins and the shifter one fixed value.
EARLY_SEED = 9703
EARLY_NULL = 0.1
TWIN_EARLY = 0.5
SHIFTER_EARLY = 0.4


def at(day, hh, mm, ss=0, us=0):
    return datetime(day.year, day.month, day.day, hh, mm, ss, us, tzinfo=timezone.utc)


def twin(x):
    return f"syn-tw-{x}"


def race_id(day, group, i):
    return f"syn-{day:%Y%m%d}-{group[0]}-{i:02d}"


R6_ID, R7_ID = race_id(TD, TOK, 6), race_id(TD, TOK, 7)
TWIN_PUBLISHED = {"a": at(TD, 1, 15), "n": None, "d": at(TD, 3, 5), "r": at(TD, 4, 10)}
# The latest publication among the runs before TD, the global terms' frame: KYO's R3 (off 03:30)
# on the day before TD.
FIT_CUTOFF = at(H_DAYS[-2], 3, 40)
US = timedelta(microseconds=1)


def schedule(day):
    """(group, race number, post, distance, publication) of the day's races."""
    out = []
    for group, card_ in DAILY.items():
        if day == TD and group == TOK:
            continue
        for i, ((hh, mm), distance) in enumerate(card_, start=1):
            post = at(day, hh, mm)
            out.append((group, i, post, distance, post + RESULT_DELAY))
    if day == TD:
        for i, ((hh, mm), distance, pub) in enumerate(TD_TOK, start=1):
            out.append((TOK, i, at(day, hh, mm), distance, None if pub is None else at(day, *pub)))
    return out


@lru_cache(maxsize=1)
def _world():
    rng = np.random.default_rng(9701)
    theta = rng.normal(THETA_MEAN, THETA_SD, N_REG)
    styles = [STYLES[k] for k in rng.integers(0, len(STYLES), N_REG)]
    variant = {(d, g): rng.normal(0.0, V_SD) for d in FIT_DAYS + H_DAYS for g in DAILY}
    variant[(TD, TOK)] = 0.0
    runs, races = [], []
    for day in FIT_DAYS + H_DAYS:
        k = (day - H_DAYS[0]).days  # negative on the fit days
        pool = rng.permutation(N_REG)
        for j, (group, i, post, distance, pub) in enumerate(schedule(day)):
            base = PAR[(*group, distance)] + variant[(day, group)]
            field = []
            for draw, h in enumerate(pool[FIELD * j : FIELD * (j + 1)], start=1):
                style = styles[h] if rng.random() < 0.8 else STYLES[rng.integers(0, len(STYLES))]
                weight = WEIGHTS[rng.integers(0, len(WEIGHTS))]
                time = base + theta[h] + W_PER_KG * (weight - REF_KG) + rng.normal(0.0, EPS_SD)
                field.append((f"syn-h{h:03d}", time, style, weight, draw))
            if TWIN_HISTORY.get(k) == (group, i):
                eps = rng.normal(0.0, EPS_SD)
                for x in TWINS:
                    field.append((twin(x), base + TWIN_THETA + eps, TWIN_STYLE, REF_KG, TWIN_DRAW))
            if day == TD and group == TOK:
                for x, (number, off) in TWIN_TD.items():
                    if number == i:
                        time = base + TWIN_THETA + off
                        field.append((twin(x), time, TWIN_STYLE, REF_KG, TWIN_DRAW))
            if k in SHIFTER_THETA and group == KYO and i == 2:
                field.append((SHIFTER, base + SHIFTER_THETA[k], "stalker", REF_KG, 9))
            rid = race_id(day, group, i)
            common = {
                "race_id": rid,
                "race_utc": post,
                "track": group[0],
                "surface": group[1],
                "distance": distance,
                "result_available_utc": pub,
            }
            top3 = float(np.mean(sorted(f[1] for f in field)[:3]))
            races.append({**common, "top3_time_s": top3, "horse_day": k >= 0})
            for horse_id, time, style, weight, draw in field:
                runs.append(
                    {
                        **common,
                        "horse_id": horse_id,
                        "finish_time_s": time,
                        "style": style,
                        "weight_kg": weight,
                        "draw": draw,
                        "in_runs": k >= 0 and rid != R7_ID,
                    }
                )
    runs, races = pd.DataFrame(runs), pd.DataFrame(races)
    for frame in (runs, races):
        for col in ("race_utc", "result_available_utc"):
            frame[col] = pd.to_datetime(frame[col], utc=True)
    # its own generator, so the planted times are the world's as before
    erng = np.random.default_rng(EARLY_SEED)
    early_level = {f"syn-h{h:03d}": float(v) for h, v in enumerate(erng.uniform(0.1, 0.9, N_REG))}
    early = np.empty(len(runs))
    for i, horse_id in enumerate(runs["horse_id"]):
        if horse_id in early_level:
            value = min(max(early_level[horse_id] + erng.normal(0.0, 0.05), 0.0), 1.0)
            early[i] = math.nan if erng.random() < EARLY_NULL else value
        else:
            early[i] = SHIFTER_EARLY if horse_id == SHIFTER else TWIN_EARLY
    runs["early_pos_mean3"] = early
    planted = {f"syn-h{h:03d}": float(theta[h]) for h in range(N_REG)}
    base_style = {f"syn-h{h:03d}": styles[h] for h in range(N_REG)}
    return runs, races, planted, base_style, early_level


def horse_runs():
    runs = _world()[0]
    return runs[runs["in_runs"]][list(RUN_COLUMNS)].reset_index(drop=True)


def jst_day(frame):
    return frame["race_utc"].dt.tz_convert("Asia/Tokyo").dt.date


def early_runs():
    """The horse runs before TD: the frame HorseSpeed's global terms are fitted on."""
    runs = horse_runs()
    return runs[jst_day(runs) < TD].reset_index(drop=True)


def td_runs():
    """TD's horse runs: observed after the fit."""
    runs = horse_runs()
    return runs[jst_day(runs) == TD].reset_index(drop=True)


def all_runs():
    return _world()[0]


def planted_theta():
    return _world()[2]


def style_of(horse_id):
    if horse_id.startswith("syn-tw-"):
        return TWIN_STYLE
    if horse_id == SHIFTER:
        return "stalker"
    return _world()[3].get(horse_id, TWIN_STYLE)


def early_of(horse_id):
    """A card's early_pos_mean3: the regular's own level, the twins' and the shifter's fixed
    value; null for a horse the world does not know."""
    if horse_id.startswith("syn-tw-"):
        return TWIN_EARLY
    if horse_id == SHIFTER:
        return SHIFTER_EARLY
    return _world()[4].get(horse_id, math.nan)


def fit_races():
    races = _world()[1]
    return races[~races["horse_day"]][list(RACE_COLUMNS)].reset_index(drop=True)


def observed_races(r7_slower=0.0):
    races = _world()[1]
    obs = races[races["horse_day"]][list(RACE_COLUMNS)].reset_index(drop=True)
    obs.loc[obs["race_id"] == R7_ID, "top3_time_s"] += r7_slower
    return obs


@lru_cache(maxsize=2)
def speed_model(r7_slower=0.0):
    spec = SpeedSpec(par_keys=PAR_KEYS, variant_keys=VARIANT_KEYS, same_day_variant=True, t_df=None)
    return SpeedModel(spec).fit(fit_races()).observe(observed_races(r7_slower))


def speed_on(fit_frame, observed=None):
    """A new speed model of its own, fitted on fit_frame, then observing `observed` (default: the
    horse days' race-level results, as `speed_model`)."""
    spec = SpeedSpec(par_keys=PAR_KEYS, variant_keys=VARIANT_KEYS, same_day_variant=True, t_df=None)
    model = SpeedModel(spec).fit(fit_frame)
    return model.observe(observed_races() if observed is None else observed)


def run_counts():
    return horse_runs()["horse_id"].value_counts()


@lru_cache(maxsize=1)
def regulars(min_runs=3):
    counts = run_counts()
    return tuple(sorted(h for h in planted_theta() if counts.get(h, 0) >= min_runs))


@lru_cache(maxsize=1)
def fillers():
    """Seven regulars with at least four runs, none of them on TD."""
    runs = all_runs()
    on_td = set(runs[runs["race_utc"].dt.tz_convert("Asia/Tokyo").dt.date == TD]["horse_id"])
    counts = run_counts()
    return tuple(h for h in regulars() if h not in on_td and counts[h] >= 4)[:7]


def card(
    horses,
    *,
    day=NEXT,
    post=(2, 0),
    group=TOK,
    distance=1600,
    rid=None,
    weight=None,
    style=None,
    early=None,
):
    """A card before the off: no result columns. weight, style and early map a horse to its
    weight_kg, style and early_pos_mean3 when they are not the world's."""
    weight, style, early = weight or {}, style or {}, early or {}
    rows = [
        {
            "race_id": rid or f"syn-card-{day:%Y%m%d}-{group[0]}-{post[0]:02d}{post[1]:02d}",
            "race_utc": at(day, *post),
            "track": group[0],
            "surface": group[1],
            "distance": distance,
            "horse_id": h,
            "style": style.get(h, style_of(h)),
            "early_pos_mean3": early.get(h, early_of(h)),
            "weight_kg": weight.get(h, REF_KG),
            "draw": n,
        }
        for n, h in enumerate(horses, start=1)
    ]
    frame = pd.DataFrame(rows)
    frame["race_utc"] = pd.to_datetime(frame["race_utc"], utc=True)
    frame["early_pos_mean3"] = frame["early_pos_mean3"].astype(float)
    return frame


def predicted(model, frame, t):
    out = model.predict(frame, at_utc=t)
    assert list(out["horse_id"]) == list(frame["horse_id"])
    return out.set_index("horse_id")[["mu_s", "sd_s"]]


def twin_row(model, x, t):
    return predicted(model, card([twin(x), *fillers()]), t).loc[twin(x)]


def chunks(items, n=12):
    return [items[i : i + n] for i in range(0, len(items), n)]


def fit_on(runs, *, t_df=None, drift=DRIFT, r7_slower=0.0):
    """A new HorseSpeed fitted on runs; r7_slower > 0 fits it against the speed model that saw TD's
    R7 that much slower."""
    return HorseSpeed(drift_var_per_day=drift, t_df=t_df).fit(runs, speed_model(r7_slower))


@pytest.fixture(scope="module")
def fitted():
    """HorseSpeed fitted on the runs before TD, nothing observed, each once per module:
    fitted(t_df=, drift=, r7_slower=)."""
    cache = {}

    def _fit(*, t_df=None, drift=DRIFT, r7_slower=0.0):
        key = (t_df, drift, r7_slower)
        if key not in cache:
            cache[key] = fit_on(early_runs(), t_df=t_df, drift=drift, r7_slower=r7_slower)
        return cache[key]

    return _fit


@pytest.fixture(scope="module")
def horse():
    """A model of its own fitted as `fitted`, then TD's runs observed: horse(t_df=, drift=,
    r7_slower=)."""
    cache = {}

    def _fit(*, t_df=None, drift=DRIFT, r7_slower=0.0):
        key = (t_df, drift, r7_slower)
        if key not in cache:
            model = fit_on(early_runs(), t_df=t_df, drift=drift, r7_slower=r7_slower)
            cache[key] = model.observe(td_runs())
        return cache[key]

    return _fit


GLOBAL_TERMS = ("sigma", "prior_mean", "prior_sd", "weight_slope", "weight_ref")
GLOBAL_TERMS += ("pace_race", "pace_ref")


def global_terms(model):
    """The fitted global terms, flat."""
    out = {name: float(getattr(model, name)) for name in GLOBAL_TERMS}
    out.update({f"pace_slopes[{s}]": float(model.pace_slopes[s]) for s in STYLES})
    return out


# ---------------------------------------------------------------------------------------------
# Broad behaviour
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t_df", [None, 5.0], ids=["normal", "student_t5"])
def test_predict_recovers_the_planted_horse_speeds(horse, t_df):
    # Every regular with three runs or more on a TOK turf 1600 m card the day after TD, at 56 kg:
    # its planted expected time is the par plus its theta. The horses' order and spread are the
    # model's to get right; the common level also carries the speed model's estimated pars (a
    # top-3 time is not the planted par), so it is held loosely.
    model = horse(t_df=t_df)
    horses = list(regulars())
    assert len(horses) > 80
    got = pd.concat([predicted(model, card(c), at(TD, 23, 0)) for c in chunks(horses)])
    want = np.array([PAR[(*TOK, 1600)] + planted_theta()[h] for h in horses])
    err = got.loc[horses, "mu_s"].to_numpy() - want
    assert np.corrcoef(got.loc[horses, "mu_s"], want)[0, 1] > 0.9
    assert np.mean(np.abs(err - err.mean())) < 0.3
    assert abs(np.mean(err)) < 0.6
    sd = got["sd_s"].to_numpy()
    assert np.isfinite(sd).all() and (sd > 0).all()


def test_weight_carried_moves_mu_at_the_planted_rate(fitted):
    model, horses, t = fitted(), list(fillers()), at(TD, 23, 0)
    for h in horses[:3]:
        light = predicted(model, card(horses, weight={h: 54.0}), t).loc[h, "mu_s"]
        heavy = predicted(model, card(horses, weight={h: 58.0}), t).loc[h, "mu_s"]
        assert heavy - light == pytest.approx(4 * W_PER_KG, abs=0.25), h


def test_theta_is_a_random_walk_in_the_drift(fitted):
    t = at(TD, 23, 0)
    still, base, loose = fitted(drift=0.0), fitted(), fitted(drift=0.5)
    # The shifter ran 1.5 s slow four times, then 1.5 s quick four times: a still walk weighs every
    # run alike (about its average), a loose one follows the recent runs (more than halfway to
    # the quick level).
    mu = {
        name: predicted(m, card([SHIFTER, *fillers()]), t).loc[SHIFTER, "mu_s"]
        for name, m in (("still", still), ("loose", loose))
    }
    assert mu["loose"] < mu["still"] - 0.75
    # Sixty days without a run: the mean holds and the SD widens, unless the walk is still.
    horses = [SHIFTER, twin("ref"), *fillers()]
    for m, widens in ((still, False), (base, True), (loose, True)):
        near = predicted(m, card(horses, day=NEXT), at(NEXT, 0, 0))
        far = predicted(m, card(horses, day=LATER), at(LATER, 0, 0))
        assert far["mu_s"].to_numpy() == pytest.approx(near["mu_s"].to_numpy(), abs=1e-9)
        if widens:
            assert (far["sd_s"] > near["sd_s"] + 0.05).all()
        else:
            assert far["sd_s"].to_numpy() == pytest.approx(near["sd_s"].to_numpy(), abs=1e-9)


def test_an_unseen_horse_gets_the_population_prior(fitted):
    new = ["syn-new-1", "syn-new-2"]
    t = at(TD, 23, 0)
    for model in (fitted(), fitted(t_df=5.0)):
        out = predicted(model, card([*new, *fillers()]), t)
        assert out.loc[new[0]].to_numpy() == pytest.approx(out.loc[new[1]].to_numpy(), abs=1e-9)
        assert np.isfinite(out.loc[new[0]].to_numpy()).all()
        assert out.loc[new[0], "sd_s"] > out.loc[list(fillers()), "sd_s"].max()
        population = pd.concat([predicted(model, card(c), t) for c in chunks(list(regulars()))])
        assert out.loc[new[0], "mu_s"] == pytest.approx(population["mu_s"].mean(), abs=0.35)


# ---------------------------------------------------------------------------------------------
# The ticket's named tests
# ---------------------------------------------------------------------------------------------

# (hh, mm, ss, us) on TD: around each twin's publication, and the race off before it
TWIN_SWEEP = (
    (0, 30),
    (1, 10),  # R1 has run, its result is not in
    (1, 15),  # R1 published at exactly t: not before t
    (1, 15, 0, 1),
    (1, 50),
    (2, 30),  # R3 has run (02:15), its result is held
    (3, 5),
    (3, 5, 0, 1),
    (4, 5),  # R6 has run (04:00), its result is not in
    (4, 10),
    (4, 10, 0, 1),
    (23, 0),
)


@pytest.mark.parametrize("t_df", [None, 5.0], ids=["normal", "student_t5"])
def test_relative_speed_uses_no_later_race(horse, t_df):
    # The twins share every run before TD, the fit's. Until a twin's TD result, observed, is
    # published strictly before t it is predicted exactly as ref; after, its quicker run has moved
    # it ahead of ref. n's result is never published, so n is ref all day.
    model = horse(t_df=t_df)
    for when in TWIN_SWEEP:
        t = at(TD, *when)
        ref = twin_row(model, "ref", t)
        assert np.isfinite(ref.to_numpy()).all() and ref["sd_s"] > 0
        for x, published in TWIN_PUBLISHED.items():
            got = twin_row(model, x, t)
            if published is not None and published < t:
                assert got["mu_s"] < ref["mu_s"] - 0.05, (x, when)
            else:
                assert got.to_numpy() == pytest.approx(ref.to_numpy(), abs=1e-9), (x, when)
    # R6 itself, before its result: r's own run in it (observed, published 04:10) does not count,
    # and the card's result columns, when attached, are ignored.
    runs = td_runs()
    r6 = runs[runs["race_id"] == R6_ID].reset_index(drop=True)
    assert twin("r") in set(r6["horse_id"])
    as_ref = r6.replace({"horse_id": {twin("r"): twin("ref")}})
    bare = [c for c in r6.columns if c not in ("finish_time_s", "result_available_utc")]
    for when in ((3, 59), (4, 5), (4, 10)):
        t = at(TD, *when)
        got = predicted(model, r6[bare], t)
        assert got.loc[twin("r")].to_numpy() == pytest.approx(
            predicted(model, as_ref[bare], t).loc[twin("ref")].to_numpy(), abs=1e-9
        ), when
        with_results = predicted(model, r6, t)
        assert with_results.to_numpy() == pytest.approx(got.to_numpy(), abs=1e-9), when


def test_a_result_published_at_or_after_t_never_moves_the_prediction(horse):
    # Two speed models that differ only in TD's R7 (published 04:45, its runners not in the runs
    # frame), 5 s slower in the second; the same runs, fitted and observed. Every horse that ran on
    # TD, on the TD card after R7: identical at t up to 04:45 (an earlier run's baseline never
    # reads R7), the card's own variant moved once R7 is in.
    base, slow = horse(), horse(r7_slower=R7_SLOWER)
    ran_td = sorted(set(td_runs()["horse_id"]))
    assert len(ran_td) > 50 and twin("a") in ran_td
    cards = [card(c, day=TD, post=R8_POST) for c in chunks(ran_td)]
    for when in ((4, 30), (4, 45)):
        t = at(TD, *when)
        for c in cards:
            a, b = predicted(base, c, t), predicted(slow, c, t)
            assert b.to_numpy() == pytest.approx(a.to_numpy(), abs=1e-9), when
    t = at(TD, 4, 45, 0, 1)
    c = card(list(fillers()), day=TD, post=R8_POST)
    moved = predicted(slow, c, t)["mu_s"] - predicted(base, c, t)["mu_s"]
    assert (moved > 0.3).all()


def test_student_t_downweights_eased_runners(horse):
    # e was eased 5 s behind its usual time on TD (R1); a ran 2 s quicker than usual in the same
    # race. Under a normal error the eased run drags e well behind ref; under a Student-t error it
    # barely moves it, while a's smaller departure keeps a larger share of its pull: the error
    # downweights by size, it does not just move every run less.
    t = at(TD, 23, 0)
    normal, student = horse(), horse(t_df=5.0)

    def shift(model, x):
        return twin_row(model, x, t)["mu_s"] - twin_row(model, "ref", t)["mu_s"]

    eased_n, eased_t = shift(normal, "e"), shift(student, "e")
    sharp_n, sharp_t = shift(normal, "a"), shift(student, "a")
    assert eased_n > 0.5
    assert -0.02 < eased_t < 0.3 * eased_n
    assert sharp_n < -0.1 and sharp_t < -0.05
    assert sharp_t / sharp_n > eased_t / eased_n + 0.15


# ---------------------------------------------------------------------------------------------
# Fit and observe: no result published at or after t reaches the global terms
# ---------------------------------------------------------------------------------------------


def test_predict_refuses_a_t_at_or_before_the_fit_runs_latest_publication(fitted, horse):
    # The global terms read every fit run, so a t at or before the latest publication among them
    # would let a result published at or after t in: refused. The bound is that publication (not
    # the latest off, nor the last day's), it moves with the fit frame, and it is the fit's alone:
    # TD's runs observed since leave it where it was.
    runs = early_runs()
    eight = runs[jst_day(runs) <= H_DAYS[7]].reset_index(drop=True)
    late = runs.copy()
    late_at = at(H_DAYS[-2], 9, 0)
    late.loc[late["race_id"] == race_id(H_DAYS[4], TOK, 2), "result_available_utc"] = late_at
    cases = (
        ("fitted", fitted, runs, FIT_CUTOFF),
        ("eight days", lambda: fit_on(eight), eight, at(H_DAYS[7], 3, 40)),
        ("a late result", lambda: fit_on(late), late, late_at),
        ("TD observed", horse, runs, FIT_CUTOFF),
    )
    frame = card(list(fillers()))
    for name, make, fit_runs, cutoff in cases:
        model = make()
        assert fit_runs["result_available_utc"].max() == cutoff, name  # the planted bound
        last_off = fit_runs["race_utc"].max()
        assert last_off < cutoff, name
        first_day = at(fit_runs["race_utc"].min().date(), 0, 0)
        for t in (first_day, last_off + US, cutoff - US, cutoff):
            with pytest.raises(ValueError):
                model.predict(frame, at_utc=t)
        out = predicted(model, frame, cutoff + US)
        assert np.isfinite(out.to_numpy()).all(), name


def quicker(runs, mask):
    """runs with the masked rows' finish times 1.5 to 2.0 s quicker (by draw)."""
    out = runs.copy()
    out.loc[mask, "finish_time_s"] -= 1.5 + 0.25 * (out.loc[mask, "draw"] % 3)
    return out


# TD publications (hh, mm): TOK R1, KYO R1, TOK R4, TOK R3 (held), TOK R6
OBSERVED_SWEEP = ((1, 15), (1, 20), (3, 0), (3, 5), (4, 10))


@pytest.mark.parametrize("t_df", [None, 5.0], ids=["normal", "student_t5"])
def test_an_observed_result_counts_only_once_published_before_t(horse, t_df):
    # A second model, fitted alike, observes TD's runs with every result published at or after t
    # (or never) quicker. At t every runner of TD, the fillers and a horse never seen predict
    # exactly as with the true results; a microsecond later the runners of the races published at
    # exactly t have moved, and nobody else.
    base = horse(t_df=t_df)
    obs = td_runs()
    pub = obs["result_available_utc"]
    horses = [*sorted(set(obs["horse_id"])), *fillers(), "syn-new-1"]
    cards = [card(c, day=TD, post=R8_POST) for c in chunks(horses, 20)]

    def read(model, t):
        return pd.concat([predicted(model, c, t) for c in cards])

    for when in OBSERVED_SWEEP:
        t = at(TD, *when)
        changed = fit_on(early_runs(), t_df=t_df).observe(quicker(obs, pub.isna() | (pub >= t)))
        assert read(changed, t).to_numpy() == pytest.approx(read(base, t).to_numpy(), abs=1e-9)
        a, b = read(base, t + US), read(changed, t + US)
        now_in = sorted(set(obs.loc[pub == t, "horse_id"]))
        assert len(now_in) >= FIELD, when
        rest = [h for h in a.index if h not in now_in]
        assert b.loc[rest].to_numpy() == pytest.approx(a.loc[rest].to_numpy(), abs=1e-9), when
        step = b.loc[now_in, "mu_s"] - a.loc[now_in, "mu_s"]
        assert (step.abs() > 1e-6).all(), when
        if t_df is None:  # the normal filter is linear: a quicker result, a quicker horse
            assert (step < -0.05).all(), when


def test_observe_refuses_a_race_already_fitted_or_observed(fitted, horse):
    obs = td_runs()
    tok, kyo = (obs[obs["track"] == g[0]].reset_index(drop=True) for g in (TOK, KYO))
    horses = [*sorted(set(obs["horse_id"])), *fillers()]
    cards = [card(c, day=TD, post=R8_POST) for c in chunks(horses, 20)]
    times = (at(TD, 1, 30), at(TD, 3, 50), at(TD, 23, 0))  # KYO's results are all in at 03:40

    def state(model):
        return np.concatenate([predicted(model, c, t).to_numpy() for t in times for c in cards])

    model = fit_on(early_runs()).observe(tok)
    kept = state(model)
    r1 = tok[tok["race_id"] == race_id(TD, TOK, 1)]
    runs = early_runs()
    fit_race = runs[runs["race_id"] == race_id(H_DAYS[-2], KYO, 2)]
    assert len(r1) >= FIELD and len(fit_race) >= FIELD
    moved_fit_race = fit_race.assign(finish_time_s=fit_race["finish_time_s"] - 2.0)
    # KYO's runs, new, with a race already observed or one of the fit frame: refused whole, so
    # KYO's runs are not kept either.
    for again in (r1, fit_race, moved_fit_race):
        with pytest.raises(ValueError):
            model.observe(pd.concat([kyo, again], ignore_index=True))
        assert state(model) == pytest.approx(kept, abs=1e-12)
    # Observations add up: TOK's runs, then KYO's, are TD's runs observed at once.
    model = model.observe(kyo)
    assert state(model) == pytest.approx(state(horse()), abs=1e-9)
    with pytest.raises(ValueError):
        model.observe(kyo)
    # Fitting again clears them: the model is the one fitted alone, and TD's runs observe again.
    model = model.fit(early_runs(), speed_model())
    assert state(model) == pytest.approx(state(fitted()), abs=1e-9)
    model = model.observe(obs)
    assert state(model) == pytest.approx(state(horse()), abs=1e-9)


@pytest.mark.parametrize("t_df", [None, 5.0], ids=["normal", "student_t5"])
def test_observe_never_changes_the_global_terms(fitted, t_df):
    # TD's runs observed 4 s slower, every one at 60 kg and early_pos_mean3 0.9: had observe
    # refitted a term or re-centred the weight or the pace on its own frame, a term, or a new
    # horse's prediction, would move.
    model = fit_on(early_runs(), t_df=t_df)
    before = global_terms(model)
    odd = td_runs()
    odd = odd.assign(finish_time_s=odd["finish_time_s"] + 4.0, weight_kg=60.0, early_pos_mean3=0.9)
    model = model.observe(odd)
    assert global_terms(model) == before
    alone = fitted(t_df=t_df)
    new = ["syn-new-f", "syn-new-s", "syn-new-m", "syn-new-c"]
    styles = dict(zip(new, STYLES, strict=True))
    for level, kg, day in ((0.2, 54.0, NEXT), (0.8, 58.0, TD)):
        c = card(
            new,
            day=day,
            post=R8_POST,
            style=styles,
            early=dict.fromkeys(new, level),
            weight={new[0]: kg, new[3]: 112.0 - kg},
        )
        for t in (at(TD, 23, 0), at(NEXT, 1, 0)):
            got, want = predicted(model, c, t), predicted(alone, c, t)
            assert got.to_numpy() == pytest.approx(want.to_numpy(), abs=1e-9), (level, t)
    # observe did keep the runs: a horse of TD's R1 has moved
    t, c = at(TD, 23, 0), card([twin("a"), *fillers()])
    moved = predicted(model, c, t)["mu_s"] - predicted(alone, c, t)["mu_s"]
    assert abs(moved[twin("a")]) > 0.01


def test_a_fit_run_never_published_reaches_no_prediction():
    # A race of the fit frame whose result is never published. Either fit refuses the frame, or
    # the race reaches no global term (nor, never published, any theta): its times changed, every
    # term and every prediction after the frame's other results is as it was.
    runs = early_runs()
    hit = runs["race_id"] == race_id(H_DAYS[6], TOK, 2)
    assert hit.sum() >= FIELD
    unpublished = runs.copy()
    unpublished.loc[hit, "result_available_utc"] = pd.NaT
    moved = unpublished.copy()
    moved.loc[hit, "finish_time_s"] += 3.0
    fits = []
    for frame in (unpublished, moved):
        try:
            fits.append(fit_on(frame))
        except ValueError:
            fits.append(None)
    if fits[0] is None or fits[1] is None:
        assert fits[0] is None and fits[1] is None  # refused at fit, whatever the times
        return
    a, b = fits
    assert global_terms(b) == pytest.approx(global_terms(a), abs=1e-9)
    horses = [*sorted(set(runs.loc[hit, "horse_id"])), *fillers()]
    t = FIT_CUTOFF + US  # the latest publication among the published runs
    for c in (card(horses), card(horses, day=TD, post=R8_POST)):
        assert predicted(b, c, t).to_numpy() == pytest.approx(
            predicted(a, c, t).to_numpy(), abs=1e-9
        )


def late_race(tag, *, distance, timed, off, pub):
    """A TOK turf race of eight new horses, each 1.0 s (plus 0.1 s a draw) behind the 1600 m par
    when timed, no finish time otherwise; 1800 m is a cell with no par."""
    frame = pd.DataFrame(
        [
            {
                "race_id": f"syn-late-{tag}",
                "race_utc": off,
                "track": TOK[0],
                "surface": TOK[1],
                "distance": distance,
                "horse_id": f"syn-late-{tag}-{k}",
                "finish_time_s": PAR[(*TOK, 1600)] + 1.0 + 0.1 * k if timed else math.nan,
                "style": STYLES[k % len(STYLES)],
                "early_pos_mean3": 0.5,
                "weight_kg": REF_KG,
                "draw": k + 1,
                "result_available_utc": pub,
            }
            for k in range(FIELD)
        ]
    )[list(RUN_COLUMNS)]
    for col in ("race_utc", "result_available_utc"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    frame["finish_time_s"] = frame["finish_time_s"].astype(float)
    return frame


# Two races of the fit frame published hours after FIT_CUTOFF whose runs no term reads
LATE_RUNS = {
    "untimed": {"distance": 1600, "timed": False, "off": (8, 0), "pub": (9, 0)},
    "unseen_cell": {"distance": 1800, "timed": True, "off": (8, 30), "pub": (9, 30)},
}


def test_the_runs_cutoff_counts_only_the_runs_the_terms_used(fitted):
    # The bound from the runs is the latest publication among the runs the global terms read (a
    # finish time, a par, a publication), with the instants their in-sample baselines take over. A
    # race of the fit frame published later whose runs have no finish time (no observation), or
    # whose cell has no par (skipped), moves no bound, no term and no prediction.
    frame = card([twin("ref"), *fillers(), "syn-new-1"])
    t = FIT_CUTOFF + US
    for tag, plan in LATE_RUNS.items():
        extra = late_race(
            tag,
            distance=plan["distance"],
            timed=plan["timed"],
            off=at(H_DAYS[-2], *plan["off"]),
            pub=at(H_DAYS[-2], *plan["pub"]),
        )
        runs = pd.concat([early_runs(), extra], ignore_index=True)
        assert runs["result_available_utc"].max() == at(H_DAYS[-2], *plan["pub"]) > FIT_CUTOFF
        model = fit_on(runs)
        assert global_terms(model) == pytest.approx(global_terms(fitted()), abs=1e-9), tag
        with pytest.raises(ValueError):
            model.predict(frame, at_utc=FIT_CUTOFF)
        assert predicted(model, frame, t).to_numpy() == pytest.approx(
            predicted(fitted(), frame, t).to_numpy(), abs=1e-9
        ), tag
    # The second world (below): P's and U's runs (GOING_DAY, published 01:15) among the terms'
    # runs. NAK's day is fully resulted at 04:00 (its untimed R3), after the speed model's fit
    # cutoff (03:00): until then P's in-sample baseline may not be used, so a t at or before 04:00
    # is refused. HAN's day never completes: U's run never takes its in-sample baseline, and holds
    # no bound back.
    going = w2_going_runs()
    pu = going[going["horse_id"].isin([sp(k, j) for k in "PU" for j in range(len(SP_THETA))])]
    assert len(pu) == 2 * len(SP_THETA)
    model = HorseSpeed(drift_var_per_day=DRIFT, t_df=None).fit(
        pd.concat([w2_fit_runs(), pu], ignore_index=True), w2_speed_model()
    )
    w2_frame = special_card("FGPU")
    for when in ((3, 0, 0, 1), (3, 30), (4, 0)):
        with pytest.raises(ValueError):
            model.predict(w2_frame, at_utc=at(GOING_DAY, *when))
    assert np.isfinite(predicted(model, w2_frame, at(GOING_DAY, 4, 0, 0, 1)).to_numpy()).all()


# A timed race of the speed model's fit frame. No horse run is on a fit day of this world, so its
# publication reaches no run's baseline: only the speed model's fit cutoff.
SPEED_LATE = race_id(FIT_DAYS[3], TOK, 2)


def with_publication(races, rid, pub):
    """races with race rid published at pub (None: never)."""
    out = races.copy()
    stamp = pd.NaT if pub is None else pd.Timestamp(pub)
    out.loc[out["race_id"] == rid, "result_available_utc"] = stamp
    return out


def with_untimed_race(races, pub):
    """races and one more TOK turf 1600 m race on SPEED_LATE's day, after the day's others, with no
    top-3 time, published at pub (None: never)."""
    extra = pd.DataFrame(
        [
            {
                "race_id": f"syn-{FIT_DAYS[3]:%Y%m%d}-TOK-untimed",
                "race_utc": at(FIT_DAYS[3], 4, 0),
                "track": TOK[0],
                "surface": TOK[1],
                "distance": 1600,
                "top3_time_s": math.nan,
                "result_available_utc": pub,
            }
        ]
    )
    for col in ("race_utc", "result_available_utc"):
        extra[col] = pd.to_datetime(extra[col], utc=True)
    return pd.concat([races, extra[list(RACE_COLUMNS)]], ignore_index=True)


# The speed model's fit frame, and the bound predict then holds
SPEED_CUTOFF_CASES = {
    "timed_published_on_td": (
        lambda r: with_publication(r, SPEED_LATE, at(TD, 9, 0)),
        at(TD, 9, 0),
    ),
    "timed_published_after_the_runs": (
        lambda r: with_publication(r, SPEED_LATE, at(H_DAYS[-2], 5, 0)),
        at(H_DAYS[-2], 5, 0),
    ),
    "timed_published_before_the_runs_bound": (
        lambda r: with_publication(r, SPEED_LATE, at(H_DAYS[5], 6, 0)),
        FIT_CUTOFF,
    ),
    "untimed_published_on_td": (lambda r: with_untimed_race(r, at(TD, 9, 0)), FIT_CUTOFF),
    "untimed_never_published": (lambda r: with_untimed_race(r, None), FIT_CUTOFF),
}


def test_predict_refuses_a_t_at_or_before_the_speed_models_fit_cutoff(fitted):
    # The speed model's pars, sigma and tau are fitted on its own frame whatever the publications,
    # and reach every run's baseline and every card: a t at or before the latest publication among
    # that frame's timed races would let a result published at or after t in, as a t at or before
    # the runs' own bound (FIT_CUTOFF) would. The bound is the later of the two. An untimed race
    # brings no time to the fit: published late or never, it moves nothing. Only a publication
    # differs from the world's speed model, so just after the bound every prediction is that of
    # the model fitted against the world's.
    frame = card([twin("ref"), *fillers(), "syn-new-1"])
    for case, (make, bound) in SPEED_CUTOFF_CASES.items():
        races = make(fit_races())
        timed = races[races["top3_time_s"].notna()]
        assert timed["result_available_utc"].notna().all(), case
        assert max(timed["result_available_utc"].max(), FIT_CUTOFF) == bound, case  # planted
        model = HorseSpeed(drift_var_per_day=DRIFT, t_df=None).fit(early_runs(), speed_on(races))
        for t in (FIT_CUTOFF, bound - US, bound):
            try:
                model.predict(frame, at_utc=t)
            except ValueError:
                continue
            pytest.fail(f"{case}: predict took {t}, at or before the bound {bound}")
        t = bound + US
        assert predicted(model, frame, t).to_numpy() == pytest.approx(
            predicted(fitted(), frame, t).to_numpy(), abs=1e-9
        ), case


def test_a_timed_speed_fit_race_never_published_leaves_no_safe_t():
    # SPEED_LATE's result never published: its time is in the pars, sigma and tau, and no at_utc
    # is after its publication. Refused at a fit (the speed model's or the horse model's) or at
    # every predict, however late.
    races = with_publication(fit_races(), SPEED_LATE, None)
    try:
        model = HorseSpeed(drift_var_per_day=DRIFT, t_df=None).fit(early_runs(), speed_on(races))
    except ValueError:
        return
    frame = card([twin("ref"), *fillers(), "syn-new-1"])
    far = LATER + timedelta(days=3650)
    for t in (FIT_CUTOFF + US, at(TD, 23, 0), at(LATER, 0, 0), at(far, 0, 0)):
        with pytest.raises(ValueError):
            model.predict(frame, at_utc=t)


def td_state(model, times):
    """Every TD runner's and filler's prediction on a TD card after R7, at each (hh, mm, ...) of
    times on TD."""
    horses = [*sorted(set(td_runs()["horse_id"])), *fillers()]
    cards = [card(c, day=TD, post=R8_POST) for c in chunks(horses, 20)]
    return np.concatenate(
        [predicted(model, c, at(TD, *w)).to_numpy() for w in times for c in cards]
    )


R2_ID, R3_ID = race_id(TD, TOK, 2), race_id(TD, TOK, 3)
R2_LATE = at(TD, 5, 30)  # R2's result, held, published after all
TD_TIMES = ((1, 30), (3, 50), (4, 10), (4, 10, 0, 1), (5, 30), (5, 30, 0, 1), (23, 0))


def test_observe_reserves_only_races_that_contributed(horse):
    obs = td_runs()

    def of(rid):
        return (obs["race_id"] == rid).to_numpy()

    # R6 pending: observed before its result, its runners with no finish time and no publication.
    # It contributed nothing, so its result, observed once in, counts: the model is the one that
    # observed TD's runs at once.
    r6 = of(R6_ID)
    pending = obs.copy()
    pending.loc[r6, "finish_time_s"] = math.nan
    pending.loc[r6, "result_available_utc"] = pd.NaT
    model = fit_on(early_runs()).observe(pending).observe(obs[r6])
    assert td_state(model, TD_TIMES) == pytest.approx(td_state(horse(), TD_TIMES), abs=1e-9)
    # R2 held: its times in, its publication not (as observed, never). Published at 05:30 after all
    # and observed again, it counts from then on, as if observed so the first time: n, ref's twin
    # until then, is ahead of ref after.
    r2 = of(R2_ID)
    published = obs[r2].assign(result_available_utc=pd.Timestamp(R2_LATE))
    model = fit_on(early_runs()).observe(obs).observe(published)
    want = fit_on(early_runs()).observe(pd.concat([obs[~r2], published], ignore_index=True))
    assert td_state(model, TD_TIMES) == pytest.approx(td_state(want, TD_TIMES), abs=1e-9)
    n, ref = twin_row(model, "n", R2_LATE), twin_row(model, "ref", R2_LATE)
    assert n.to_numpy() == pytest.approx(ref.to_numpy(), abs=1e-9)
    after = R2_LATE + US
    assert twin_row(model, "n", after)["mu_s"] < twin_row(model, "ref", after)["mu_s"] - 0.05
    # R3 with one runner timed contributed that run: its full result is refused, and the refusal
    # leaves the model as it was.
    r3 = of(R3_ID)
    partial = obs.copy()
    partial.loc[np.flatnonzero(r3)[1:], "finish_time_s"] = math.nan
    model = fit_on(early_runs()).observe(partial)
    kept = td_state(model, TD_TIMES)
    with pytest.raises(ValueError):
        model.observe(obs[r3])
    assert td_state(model, TD_TIMES) == pytest.approx(kept, abs=1e-12)
    # A race of the fit frame whose runners all lack a finish time (H_DAYS[-3]'s TOK R2) is not
    # taken either: its result, observed later (published 06:00 on TD), joins its horses' history,
    # and they move at 06:00, nobody else.
    runs = early_runs()
    x = (runs["race_id"] == race_id(H_DAYS[-3], TOK, 2)).to_numpy()
    assert x.sum() >= FIELD
    untimed = runs.copy()
    untimed.loc[x, "finish_time_s"] = math.nan
    in_at = at(TD, 6, 0)
    resulted = runs[x].assign(result_available_utc=pd.Timestamp(in_at))
    base, model = fit_on(untimed), fit_on(untimed).observe(resulted)
    ran = sorted(set(runs.loc[x, "horse_id"]))
    others = [h for h in fillers() if h not in ran]
    c = card([*ran, *others])
    assert predicted(model, c, in_at).to_numpy() == pytest.approx(
        predicted(base, c, in_at).to_numpy(), abs=1e-9
    )
    a, b = predicted(base, c, in_at + US), predicted(model, c, in_at + US)
    assert b.loc[others].to_numpy() == pytest.approx(a.loc[others].to_numpy(), abs=1e-9)
    assert ((b.loc[ran, "mu_s"] - a.loc[ran, "mu_s"]).abs() > 1e-6).all()


def test_a_race_result_the_speed_model_observes_later_reaches_runs_observed_before(horse):
    # The speed model observes TD's TOK R1 (published 01:15) and R4 (03:00) only after HorseSpeed
    # observed TD's runs: R3..R6 (offs 02:15 to 04:00) among them, whose baselines read R1, and R5's
    # and R6's R4 too. Read from the speed model as it stands, each baseline is the one it would
    # have had had R1 and R4 been observed first: every prediction is the batch model's, whatever
    # was predicted in between, and the global terms stay as fitted.
    races = observed_races()
    r1, r4 = (races["race_id"] == race_id(TD, TOK, i) for i in (1, 4))
    speed = speed_on(fit_races(), races[~(r1 | r4)])
    model = HorseSpeed(drift_var_per_day=DRIFT, t_df=None).fit(early_runs(), speed)
    model = model.observe(td_runs())
    terms = global_terms(model)
    assert terms == pytest.approx(global_terms(horse()), abs=1e-9)  # no fit run reads TD
    times = ((1, 30), (3, 10), (3, 50), (4, 10, 0, 1), (23, 0))
    td_state(model, times)  # predicted before the speed model observes R1
    speed.observe(races[r1])
    td_state(model, times)  # and between R1 and R4
    speed.observe(races[r4])
    assert td_state(model, times) == pytest.approx(td_state(horse(), times), abs=1e-9)
    assert global_terms(model) == terms


# ---------------------------------------------------------------------------------------------
# sd_s: the runner's predictive SD
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t_df", [None, 2.5, 5.0, 30.0], ids=["normal", "t2.5", "t5", "t30"])
def test_sd_s_is_the_runners_predictive_sd(fitted, t_df):
    # The run error's variance is sigma^2 + V_card under the normal and t_df / (t_df - 2) times it
    # under the t. Two cards off at the same instant (TD 05:10), so a horse's theta is as uncertain
    # on both: KYO's read at 01:00, before any KYO result of the day (its variant is the prior),
    # TOK's at 23:00, after the day's TOK results.
    model, speed = fitted(t_df=t_df), speed_model()
    k = 1.0 if t_df is None else t_df / (t_df - 2.0)
    new, seen = ["syn-new-1", "syn-new-2"], list(fillers())
    var, v_card = {}, {}
    for group, t in ((KYO, at(TD, 1, 0)), (TOK, at(TD, 23, 0))):
        out = predicted(model, card([*seen, *new], day=TD, post=R8_POST, group=group), t)
        v_card[group] = speed.variant(TD, *group, at_utc=t)[1] ** 2
        var[group] = out["sd_s"] ** 2
        want = model.prior_sd**2 + k * (model.sigma**2 + v_card[group])
        assert var[group][new].to_numpy() == pytest.approx([want, want], rel=1e-9), group
    assert v_card[KYO] > v_card[TOK] + 0.05  # the day's TOK results narrowed TOK's variant
    gap = (var[KYO] - var[TOK])[seen].to_numpy()
    assert gap == pytest.approx(np.full(len(seen), k * (v_card[KYO] - v_card[TOK])), rel=1e-9)
    theta_var = var[TOK][seen] - k * (model.sigma**2 + v_card[TOK])
    assert (theta_var > 0).all() and (theta_var < model.prior_sd**2).all()


@pytest.mark.parametrize("t_df", [2.0, 1.5])
def test_sd_s_is_infinite_when_the_t_error_has_no_variance(fitted, t_df):
    model = fitted(t_df=t_df)
    horses = [*fillers(), "syn-new-1"]
    for c in (card(horses), card(horses, day=TD, post=R8_POST)):
        out = predicted(model, c, at(TD, 23, 0))
        assert np.isfinite(out["mu_s"].to_numpy()).all()
        assert np.isposinf(out["sd_s"].to_numpy()).all()


def pl_reference(mu, s):
    lo = min(mu)
    w = [math.exp(-(m - lo) / s) for m in mu]
    return [x / sum(w) for x in w]


def henery_reference(mu, s):
    """P(i fastest) for independent N(mu_i, s^2) times, by quadrature over runner i's z."""
    out = []
    for i, mi in enumerate(mu):
        others = [mj for j, mj in enumerate(mu) if j != i]

        def f(z, mi=mi, others=others):
            surv = np.prod([stats.norm.sf(z + (mi - mj) / s) for mj in others])
            return stats.norm.pdf(z) * surv

        out.append(integrate.quad(f, -12.0, 12.0, epsabs=1e-13, epsrel=1e-12, limit=200)[0])
    return out


SOFTMAX_CASES = (
    ([95.0, 96.0], 1.0, [0.7311, 0.2689], 1e-4),  # the ticket's
    ([95.0, 96.0, 97.0], 1.0, [0.66524, 0.24473, 0.09003], 1e-5),
    ([97.0, 95.0, 96.0], 1.0, [0.09003, 0.66524, 0.24473], 1e-5),  # in mu's order
    ([95.0, 96.0], 0.5, [0.880797, 0.119203], 1e-6),
    ([95.0, 96.0], 2.0, [0.622459, 0.377541], 1e-6),
    ([100.2, 99.8, 101.0, 99.8], 0.8, pl_reference([100.2, 99.8, 101.0, 99.8], 0.8), 1e-12),
    ([96.4], 1.0, [1.0], 1e-12),
    ([1000.0, 1000.5], 0.01, [1.0, math.exp(-50.0)], 1e-15),  # no overflow at a small scale
)


@pytest.mark.parametrize(("mu", "s", "want", "tol"), SOFTMAX_CASES)
def test_pl_strengths_are_softmax_of_minus_mu_over_s(mu, s, want, tol):
    got = p_speed(np.array(mu), scale=s)
    assert isinstance(got, np.ndarray) and got.shape == (len(mu),)
    assert got == pytest.approx(want, abs=tol)
    assert p_speed(np.array(mu), scale=s, method="pl") == pytest.approx(got, abs=1e-15)
    # only the differences of mu count
    assert p_speed(np.array(mu) + 37.5, scale=s) == pytest.approx(got, abs=1e-12)


def test_p_speed_sums_to_one_per_race():
    rng = np.random.default_rng(9702)
    for n in (1, 2, 3, 5, 8, 12, 16, 18):
        for s in (0.25, 1.0, 4.0):
            mu = rng.uniform(92.0, 100.0, n)
            for method in ("pl", "henery"):
                p = p_speed(mu, scale=s, method=method)
                assert p.shape == (n,), (n, s, method)
                assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                assert p.sum() == pytest.approx(1.0, abs=1e-9), (n, s, method)
                order = np.argsort(mu)
                assert (np.diff(p[order]) <= 1e-12).all(), (n, s, method)  # quicker, likelier
                big = p[order][:-1] > 1e-6
                assert (np.diff(p[order])[big] < 0).all(), (n, s, method)
    # Henery: independent normal times with SD s. Two runners have the closed form
    # Phi((mu_2 - mu_1) / (s * sqrt 2)); more runners match the quadrature.
    two = p_speed(np.array([95.0, 96.0]), scale=1.0, method="henery")
    assert two == pytest.approx([0.760250, 0.239750], abs=1e-6)
    assert two[0] == pytest.approx(stats.norm.cdf(1.0 / math.sqrt(2.0)), abs=1e-9)
    for mu, s in (([95.0, 96.0, 97.0], 1.0), ([96.3, 95.1, 95.9, 97.4, 95.1], 0.6)):
        got = p_speed(np.array(mu), scale=s, method="henery")
        assert got == pytest.approx(henery_reference(mu, s), abs=1e-6), mu
        assert got != pytest.approx(pl_reference(mu, s), abs=1e-3)  # not PL by another name
    # tied mu, tied p
    for method in ("pl", "henery"):
        p = p_speed(np.array([96.0, 95.5, 96.0]), scale=0.7, method=method)
        assert p[0] == pytest.approx(p[2], abs=1e-12) and p[1] > p[0]


# The same-day bias frame: one JST day at TOK and KYO and the day before at TOK, nine runners a
# race, the draws 1-9 and a fixed style per draw, so each style is spread evenly over the draws.
# Each race's finish follows a pattern: inner (draw 1 wins), outer (draw 9 wins), front (front
# runners first, closers last) or closer.
BIAS_DAY = date(2025, 2, 15)
STYLE_OF_DRAW = {
    1: "front",
    2: "closer",
    3: "stalker",
    4: "midpack",
    5: "midpack",
    6: "midpack",
    7: "stalker",
    8: "closer",
    9: "front",
}
BIAS_TOK = (  # R1..R7: (post, pattern, publication or None)
    ((1, 5), "inner", (1, 15)),
    ((1, 40), "front", (1, 50)),
    ((2, 15), "outer", (3, 5)),  # held: published after R4
    ((2, 50), "closer", (3, 0)),
    ((3, 25), "inner", (3, 35)),
    ((4, 0), "outer", (4, 10)),
    ((4, 35), "front", None),  # never published
)
BIAS_KYO = (((1, 10), "outer", (1, 20)), ((1, 45), "closer", (1, 55)), ((2, 20), "outer", (2, 30)))
BIAS_EVE = (  # TOK the day before, every result long in
    ((1, 5), "outer", (1, 15)),
    ((1, 40), "closer", (1, 50)),
    ((2, 15), "outer", (2, 25)),
    ((2, 50), "closer", (3, 0)),
)
# (UTC time on BIAS_DAY, TOK races of its JST day resulted strictly before it)
BIAS_SWEEP = (
    ((0, 30), 0),
    ((1, 15), 0),
    ((1, 16), 1),
    ((1, 50), 1),
    ((1, 51), 2),
    ((2, 30), 2),  # R3 has run, its result is held
    ((3, 0), 2),
    ((3, 1), 3),
    ((3, 5), 3),
    ((3, 6), 4),
    ((3, 36), 5),
    ((4, 10), 5),
    ((4, 11), 6),
    ((14, 59), 6),  # 23:59 JST, the end of BIAS_DAY
    ((15, 0), 0),  # 00:00 JST the next day: a new JST day, nothing of BIAS_DAY read
)
LEVELS = {("draw", lv) for lv in ("inner", "middle", "outer")} | {("style", s) for s in STYLES}


def finish_of(pattern):
    """draw -> finish_pos for nine runners."""
    rank = {s: k for k, s in enumerate(STYLES)}
    keys = {
        "inner": lambda d: d,
        "outer": lambda d: -d,
        "front": lambda d: (rank[STYLE_OF_DRAW[d]], d),
        "closer": lambda d: (-rank[STYLE_OF_DRAW[d]], d),
    }
    order = sorted(range(1, 10), key=keys[pattern])
    return {d: pos for pos, d in enumerate(order, start=1)}


def bias_results(*, r3="outer", tok_day_only=False):
    rows = []
    day_before = BIAS_DAY - timedelta(days=1)
    meetings = [(BIAS_DAY, "TOK", BIAS_TOK)]
    if not tok_day_only:
        meetings += [(BIAS_DAY, "KYO", BIAS_KYO), (day_before, "TOK", BIAS_EVE)]
    for day, venue, races in meetings:
        for i, (post, pattern, pub) in enumerate(races, start=1):
            if (day, venue, i) == (BIAS_DAY, "TOK", 3):
                pattern = r3
            finish = finish_of(pattern)
            for d in range(1, 10):
                rows.append(
                    {
                        "race_id": f"syn-b-{day:%Y%m%d}-{venue}-{i:02d}",
                        "race_utc": at(day, *post),
                        "venue": venue,
                        "horse_id": f"syn-b-{venue}-{i}-{d}",
                        "draw": d,
                        "style": STYLE_OF_DRAW[d],
                        "finish_pos": finish[d],
                        "result_available_utc": None if pub is None else at(day, *pub),
                    }
                )
    frame = pd.DataFrame(rows)
    for col in ("race_utc", "result_available_utc"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def bias(results, t, venue="TOK"):
    out = same_day_bias(results, at_utc=t, venue=venue)
    assert set(zip(out["factor"], out["level"], strict=True)) == LEVELS and len(out) == len(LEVELS)
    return out.sort_values(["factor", "level"]).reset_index(drop=True)


def profile(out):
    return {(f, lv): b for f, lv, b in zip(out["factor"], out["level"], out["bias"], strict=True)}


def assert_same(a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=0, atol=1e-12)


def test_same_day_bias_uses_only_races_resulted_before_t():
    base = bias_results()
    by_n = {}
    for (hh, mm), n in BIAS_SWEEP:
        out = bias(base, at(BIAS_DAY, hh, mm))
        assert (out["n_races"] == n).all(), (hh, mm)
        by_n.setdefault(n, []).append(out)
    # One profile for each set of resulted races, whatever has run or been published elsewhere by
    # t, and a new one each time a result of the day's venue is published.
    for outs in by_n.values():
        for out in outs[1:]:
            assert_same(out, outs[0])
    for n in range(1, 7):
        assert (by_n[n][0]["bias"] - by_n[n - 1][0]["bias"]).abs().max() > 1e-6, n
    # Nothing resulted: no bias. R1 alone (inner draw won): inner ahead, outer behind. R1 and R2
    # (front runners won R2; R1's styles are even): front ahead, closers behind.
    assert (by_n[0][0]["bias"] == 0.0).all()
    one, two = profile(by_n[1][0]), profile(by_n[2][0])
    assert one[("draw", "inner")] > 0 > one[("draw", "outer")]
    assert two[("style", "front")] > 0 > two[("style", "closer")]
    # Another venue reads its own races only (KYO's outer draws won).
    kyo = profile(bias(base, at(BIAS_DAY, 14, 59), "KYO"))
    assert kyo[("draw", "outer")] > 0 > kyo[("draw", "inner")]
    assert (bias(base, at(BIAS_DAY, 14, 59), "KYO")["n_races"] == 3).all()
    # R3 finished the other way round: no change until its result is in (03:05), then a change.
    flipped = bias_results(r3="inner")
    for when in ((2, 30), (3, 0), (3, 5)):
        t = at(BIAS_DAY, *when)
        assert_same(bias(flipped, t), bias(base, t))
    t = at(BIAS_DAY, 3, 6)
    assert profile(bias(flipped, t))[("draw", "inner")] > profile(bias(base, t))[("draw", "inner")]
    # The other venue's races and the day before's never count.
    alone = bias_results(tok_day_only=True)
    for (hh, mm), _ in BIAS_SWEEP:
        t = at(BIAS_DAY, hh, mm)
        assert_same(bias(alone, t), bias(base, t))


# The pace term: one cell (TOK turf 1600 m) at exactly 96.0 s plus each day's variant, the
# variants symmetric so the par is 96.0; the card day has an injected variant prior of +0.4 s.
PACE_DAYS = tuple(date(2025, 3, 1) + timedelta(days=k) for k in range(8))
PACE_V = (0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1)
PACE_CARD_DAY = date(2025, 3, 20)


def flat_races():
    rows = []
    for day, v in zip(PACE_DAYS, PACE_V, strict=True):
        for i, (hh, mm) in enumerate(((1, 5), (2, 15)), start=1):
            post = at(day, hh, mm)
            rows.append(
                {
                    "race_id": f"syn-pace-{day:%Y%m%d}-{i:02d}",
                    "race_utc": post,
                    "track": "TOK",
                    "surface": "turf",
                    "distance": 1600,
                    "top3_time_s": 96.0 + v,
                    "result_available_utc": post + RESULT_DELAY,
                }
            )
    frame = pd.DataFrame(rows)
    for col in ("race_utc", "result_available_utc"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def test_expected_top3_time_adds_the_pace_term():
    spec = SpeedSpec(par_keys=PAR_KEYS, variant_keys=VARIANT_KEYS, same_day_variant=True, t_df=None)

    def prior(race_date, track, surface):
        return 0.4, 0.2

    model = SpeedModel(spec, variant_prior=prior).fit(flat_races())
    race = {
        "race_id": "syn-pace-card-01",
        "race_utc": at(PACE_CARD_DAY, 2, 0),
        "track": "TOK",
        "surface": "turf",
        "distance": 1600,
    }
    t = at(PACE_CARD_DAY, 0, 30)
    par = model.par(track="TOK", surface="turf", distance=1600)
    assert par == pytest.approx(96.0, abs=0.05)
    assert model.variant(PACE_CARD_DAY, "TOK", "turf", at_utc=t) == pytest.approx((0.4, 0.2))
    default = model.expected_top3_time(race, at_utc=t)
    assert default == pytest.approx(par + 0.4, abs=1e-9)  # #96's value, unchanged
    assert model.expected_top3_time(race, at_utc=t, pace_s=0.0) == default
    # par 96.0 + variant 0.4 + pace 0.3
    assert model.expected_top3_time(race, at_utc=t, pace_s=0.3) == pytest.approx(96.7, abs=0.05)
    for pace in (-0.8, 0.3, 1.25):
        got = model.expected_top3_time(race, at_utc=t, pace_s=pace)
        assert got == pytest.approx(default + pace, abs=1e-9), pace
    # On an observed day the pace term adds to the variant at t as well (TD's R6, before and
    # after the day's first results).
    world = speed_model()
    r6 = {"race_id": R6_ID, "race_utc": at(TD, 4, 0), "track": "TOK", "surface": "turf"}
    r6["distance"] = 1600
    for when in ((0, 30), (3, 59)):
        t_td = at(TD, *when)
        plain = world.expected_top3_time(r6, at_utc=t_td)
        got = world.expected_top3_time(r6, at_utc=t_td, pace_s=-0.5)
        assert got == pytest.approx(plain - 0.5, abs=1e-9), when
    assert world.expected_top3_time(r6, at_utc=at(TD, 3, 59)) != pytest.approx(
        world.expected_top3_time(r6, at_utc=at(TD, 0, 30)), abs=1e-6
    )  # the day's results did move the variant between the two times
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            model.expected_top3_time(race, at_utc=t, pace_s=bad)


# ---------------------------------------------------------------------------------------------
# The second world: pace by the field's mean early position, horse runs on a fit day
# ---------------------------------------------------------------------------------------------

# The owner on #97 (2026-11-26): pace_r is the field's mean early position from earlier races
# (#129's early_pos_mean3, Q2 (c)), and a fit day's in-sample variant is used only once that
# (day, group) is fully resulted before t (Q3 (b)).
#
# One par per (surface, distance), shared by every track, the variant per (track, surface): a TOK
# and a KYO race of the same distance share a par but not a variant. The speed model's top-3 times
# are planted directly, par + variant + N(0, 0.15^2), apart from the runners. 7 fit days, then 12
# horse days whose race-level results the speed model observes, then GOING_DAY, the speed model's
# last fit day. HorseSpeed's global terms are fitted on the horse days' runs; GOING_DAY's runs are
# observed. (The world is drawn with GOING_DAY in the eighth place, so its random draws are those
# of the draft that had GOING_DAY before the horse days.)
#
# Regulars (horse days only): theta ~ N(0.8, 1.2^2), style drawn afresh each run and apart from the
# early positions, so the field's front-runner share carries no pace. A race's runners have
# early_pos_mean3 around a race level drawn from U(0.15, 0.85) (null on a tenth), and the race's
# pace is the mean of its non-null values; each run is planted as
#     par + variant + theta + PACE_SLOPE[style] * (field mean - 0.5) + 0.15 s/kg * (kg - 56)
#     + noise.
# One race (DEBUT) has no early position at all (planted pace effect 0).
#
# GOING_DAY is planted 2.0 s slow at every track. Five sets of five special horses, the same five
# thetas and styles, 56 kg, early_pos_mean3 0.5 in a field of their own (so the same pace in every
# special race), one noise-free run each:
# - F_j in GOING_DAY's TOK R1 (01:05, published 01:15): par + 2.0 + theta_j. TOK's R3 is in at
#   02:25 and its R2 held until 03:00: the TOK group's three races are all resulted at 03:00.
# - G_j in GOING_DAY's KYO R1, at the same off and publication: par + 2.0 + theta_j. KYO's R3 has
#   no top-3 time in the speed model's frame and is never published, so the KYO group is never
#   fully resulted. (Untimed: a timed race never published would leave predict no safe t. Its
#   random draw is kept, so the world's other draws are as before.)
# - P_j in GOING_DAY's NAK R1, at the same off and publication: par + 2.0 + theta_j. NAK's R1 and
#   R2 have top-3 times, both in by 01:50; its R3 has none in the speed model's frame and is
#   published at 04:00, so the NAK group is fully resulted at 04:00, not at 01:50.
# - U_j in GOING_DAY's HAN R1, as P_j, but HAN's R3, untimed, is never published: never fully
#   resulted.
# - O_j in O_DAY's TOK R4 (05:00), a horse day planted at exactly 0: par + theta_j. Its baseline
#   reads the day's three TOK results before it, as F's in-sample variant reads GOING_DAY's three:
#   the same evidence on the variant, without the going.
# NAK's and HAN's timed races are planted at exactly par + 2.0, from no random draw. The speed
# model's fit cutoff, the latest publication among its fit frame's timed races, is TOK's R2 on
# GOING_DAY (03:00): predict refuses every t at or before it, when TOK is fully resulted too.
W2_PAR_KEYS = ("surface", "distance")
W2_PAR = {("turf", 1600): 96.0, ("turf", 2000): 120.5}
W2_SEED = 9704
W2_N_REG = 80
W2_THETA_SD = 1.2
W2_V_SD = 0.3
W2_RACE_SD = 0.15
PACE_SLOPE = {"front": -3.0, "stalker": -1.0, "midpack": 1.0, "closer": 3.0}  # s per unit
PACE_MID = 0.5
W2_PRE_DAYS = tuple(date(2025, 6, 7) + timedelta(days=k) for k in range(7))
W2_H_DAYS = tuple(date(2025, 6, 15) + timedelta(days=k) for k in range(12))
GOING_DAY = W2_H_DAYS[-1] + timedelta(days=1)  # the speed model's last fit day
W2_FIT_DAYS = (*W2_PRE_DAYS, GOING_DAY)
W2_DRAW_ORDER = (*W2_PRE_DAYS, GOING_DAY, *W2_H_DAYS)
GOING = 2.0
O_DAY = W2_H_DAYS[0]
DEBUT = (W2_H_DAYS[1], TOK, 1)
W2_CARD_DAY = GOING_DAY + timedelta(days=1)  # in no frame: its variant is the prior
NAK, HAN = ("NAK", "turf"), ("HAN", "turf")
QUIET = (NAK, HAN)  # planted exactly, from no random draw
# GOING_DAY, R1..R3 at each track: (post, distance, publication or None)
GOING_TOK = (((1, 5), 1600, (1, 15)), ((1, 40), 2000, (3, 0)), ((2, 15), 1600, (2, 25)))
GOING_KYO = (((1, 5), 1600, (1, 15)), ((1, 40), 2000, (1, 50)), ((2, 15), 1600, None))
GOING_NAK = (((1, 5), 1600, (1, 15)), ((1, 40), 2000, (1, 50)), ((2, 15), 1600, (4, 0)))
GOING_HAN = (((1, 5), 1600, (1, 15)), ((1, 40), 2000, (1, 50)), ((2, 15), 1600, None))
UNTIMED = {(KYO, 3), (NAK, 3), (HAN, 3)}  # no top-3 time in the speed model's frame
W2_SPEED_CUTOFF = at(GOING_DAY, 3, 0)  # TOK's R2 on GOING_DAY
O_RACE = ((5, 0), 1600, (5, 10))  # after O_DAY's TOK R1..R3 are in (03:35)
SP_THETA = tuple(THETA_MEAN + d for d in (-0.4, -0.2, 0.0, 0.2, 0.4))
SP_STYLE = ("front", "stalker", "midpack", "closer", "midpack")
SP_EARLY = 0.5
SP_RACES = {
    "F": (GOING_DAY, TOK, 1),
    "G": (GOING_DAY, KYO, 1),
    "P": (GOING_DAY, NAK, 1),
    "U": (GOING_DAY, HAN, 1),
    "O": (O_DAY, TOK, 4),
}


def sp(kind, j):
    return f"syn-w2-{kind}{j}"


def w2_race_id(day, group, i):
    return f"syn-w2-{day:%Y%m%d}-{group[0]}-{i:02d}"


def w2_schedule(day):
    """(group, race number, post, distance, publication) of a second-world day's races."""
    if day == GOING_DAY:
        plans = ((TOK, GOING_TOK), (KYO, GOING_KYO), (NAK, GOING_NAK), (HAN, GOING_HAN))
        return [
            (group, i, at(day, *post), dist, None if pub is None else at(day, *pub))
            for group, plan in plans
            for i, (post, dist, pub) in enumerate(plan, start=1)
        ]
    out = schedule(day)
    if day == O_DAY:
        (post, dist, pub) = O_RACE
        out = [*out, (TOK, 4, at(day, *post), dist, at(day, *pub))]
    return out


@lru_cache(maxsize=1)
def _world2():
    rng = np.random.default_rng(W2_SEED)
    theta = rng.normal(THETA_MEAN, W2_THETA_SD, W2_N_REG)
    days = W2_DRAW_ORDER
    variant = {(d, g): rng.normal(0.0, W2_V_SD) for d in days for g in DAILY}
    for group in (TOK, KYO, *QUIET):
        variant[(GOING_DAY, group)] = GOING
    variant[(O_DAY, TOK)] = 0.0
    runs, races = [], []
    for day in days:
        pool = rng.permutation(W2_N_REG)
        slot = 0
        for group, i, post, distance, pub in w2_schedule(day):
            base = W2_PAR[(group[1], distance)] + variant[(day, group)]
            rid = w2_race_id(day, group, i)
            common = {
                "race_id": rid,
                "race_utc": post,
                "track": group[0],
                "surface": group[1],
                "distance": distance,
                "result_available_utc": pub,
            }
            if group in QUIET:
                top3 = math.nan if (group, i) in UNTIMED else base
            else:
                top3 = base + rng.normal(0.0, W2_RACE_SD)  # drawn even when untimed
                if (group, i) in UNTIMED:
                    top3 = math.nan
            races.append({**common, "top3_time_s": top3, "fit": day in W2_FIT_DAYS})
            field = []  # (horse_id, style, early_pos_mean3, weight, time)
            if day in W2_H_DAYS and (day, group, i) not in SP_RACES.values():
                horses = pool[FIELD * slot : FIELD * (slot + 1)]
                slot += 1
                level = rng.uniform(0.15, 0.85)
                early = []
                for _ in horses:
                    value = float(np.clip(level + rng.normal(0.0, 0.08), 0.0, 1.0))
                    null = (day, group, i) == DEBUT or rng.random() < EARLY_NULL
                    early.append(math.nan if null else value)
                known = [e for e in early if not math.isnan(e)]
                pace = float(np.mean(known)) if known else PACE_MID
                for h, e in zip(horses, early, strict=True):
                    style = STYLES[rng.integers(0, len(STYLES))]
                    weight = WEIGHTS[rng.integers(0, len(WEIGHTS))]
                    time = (
                        base
                        + theta[h]
                        + PACE_SLOPE[style] * (pace - PACE_MID)
                        + W_PER_KG * (weight - REF_KG)
                        + rng.normal(0.0, EPS_SD)
                    )
                    field.append((f"syn-w2-h{h:03d}", style, e, weight, time))
            for kind, where in SP_RACES.items():
                if (day, group, i) == where:
                    for j, (th, style) in enumerate(zip(SP_THETA, SP_STYLE, strict=True)):
                        field.append((sp(kind, j), style, SP_EARLY, REF_KG, base + th))
            for draw, (horse_id, style, e, weight, time) in enumerate(field, start=1):
                runs.append(
                    {
                        **common,
                        "horse_id": horse_id,
                        "finish_time_s": time,
                        "style": style,
                        "early_pos_mean3": e,
                        "weight_kg": weight,
                        "draw": draw,
                    }
                )
    runs, races = pd.DataFrame(runs), pd.DataFrame(races)
    for frame in (runs, races):
        for col in ("race_utc", "result_available_utc"):
            frame[col] = pd.to_datetime(frame[col], utc=True)
    runs["early_pos_mean3"] = runs["early_pos_mean3"].astype(float)
    return runs[list(RUN_COLUMNS)], races


def w2_runs():
    return _world2()[0].copy()


def w2_fit_runs():
    """The horse days' runs: the frame HorseSpeed's global terms are fitted on."""
    runs = w2_runs()
    return runs[jst_day(runs) != GOING_DAY].reset_index(drop=True)


def w2_going_runs():
    """GOING_DAY's runs (F, G, P and U): observed after the fit."""
    runs = w2_runs()
    return runs[jst_day(runs) == GOING_DAY].reset_index(drop=True)


@lru_cache(maxsize=1)
def w2_speed_model():
    spec = SpeedSpec(
        par_keys=W2_PAR_KEYS, variant_keys=VARIANT_KEYS, same_day_variant=True, t_df=None
    )
    races = _world2()[1]
    fit, seen = (races[m][list(RACE_COLUMNS)] for m in (races["fit"], ~races["fit"]))
    return SpeedModel(spec).fit(fit.reset_index(drop=True)).observe(seen.reset_index(drop=True))


@pytest.fixture(scope="module")
def w2_fitted():
    """HorseSpeed fitted on the horse days' runs, nothing observed."""
    return HorseSpeed(drift_var_per_day=DRIFT, t_df=None).fit(w2_fit_runs(), w2_speed_model())


@pytest.fixture(scope="module")
def w2_horse():
    """A model of its own fitted as w2_fitted, then GOING_DAY's runs observed."""
    model = HorseSpeed(drift_var_per_day=DRIFT, t_df=None).fit(w2_fit_runs(), w2_speed_model())
    return model.observe(w2_going_runs())


@lru_cache(maxsize=1)
def w2_regulars():
    counts = w2_runs()["horse_id"].value_counts()
    return tuple(sorted(h for h, n in counts.items() if h.startswith("syn-w2-h") and n >= 4))


def w2_pace_ref():
    """The fit runs' mean pace, from the planted frame: each run's race's mean non-null
    early_pos_mean3, averaged over the runs (a race with none has no pace and is left out)."""
    runs = w2_fit_runs()
    pace = runs.groupby("race_id")["early_pos_mean3"].transform("mean")
    return float(pace.dropna().mean())


def w2_card(rows, *, day=W2_CARD_DAY, post=(2, 0)):
    """A TOK turf 1600 m card of (horse_id, style, early_pos_mean3) rows, every runner at 56 kg."""
    frame = pd.DataFrame(
        [
            {
                "race_id": f"syn-w2-card-{day:%Y%m%d}-{post[0]:02d}{post[1]:02d}",
                "race_utc": at(day, *post),
                "track": TOK[0],
                "surface": TOK[1],
                "distance": 1600,
                "horse_id": h,
                "style": style,
                "early_pos_mean3": math.nan if early is None else early,
                "weight_kg": REF_KG,
                "draw": n,
            }
            for n, (h, style, early) in enumerate(rows, start=1)
        ]
    )
    frame["race_utc"] = pd.to_datetime(frame["race_utc"], utc=True)
    frame["early_pos_mean3"] = frame["early_pos_mean3"].astype(float)
    return frame


W2_T = at(W2_CARD_DAY, 0, 0)  # after every race of both frames
CARD_STYLES = ("front", "front", "stalker", "stalker", "midpack", "midpack", "closer", "closer")


def test_pace_is_the_fields_mean_early_position(w2_fitted):
    # The same eight runners and styles, every early_pos_mean3 at 0.3 and then at 0.7: each runner
    # moves by its style's planted slope times 0.4. The styles do not change, so a pace read off
    # them (the front-runner share) would move nobody.
    horses = w2_regulars()[:8]
    assert len(w2_regulars()) > 60

    def mu(level):
        rows = [(h, s, level) for h, s in zip(horses, CARD_STYLES, strict=True)]
        return predicted(w2_fitted, w2_card(rows), W2_T)["mu_s"]

    moved = mu(0.7) - mu(0.3)
    for h, style in zip(horses, CARD_STYLES, strict=True):
        assert moved[h] == pytest.approx(PACE_SLOPE[style] * 0.4, abs=0.3), (h, style)
    by_style = {
        s: np.mean([moved[h] for h, x in zip(horses, CARD_STYLES, strict=True) if x == s])
        for s in STYLES
    }
    assert by_style["front"] < by_style["stalker"] < by_style["midpack"] < by_style["closer"]
    # pace_r is centred at the fit runs' mean: there an unseen front runner and an unseen closer
    # are level; either side of it they part by the planted slopes' difference.
    ref = w2_pace_ref()
    spread = PACE_SLOPE["closer"] - PACE_SLOPE["front"]
    for level in (ref - 0.25, ref, ref + 0.25):
        rows = [("syn-w2-new-f", "front", level), ("syn-w2-new-c", "closer", level)]
        out = predicted(w2_fitted, w2_card(rows), W2_T)["mu_s"]
        gap = out["syn-w2-new-c"] - out["syn-w2-new-f"]
        tol = 0.1 + 1.2 * abs(level - ref)
        assert gap == pytest.approx(spread * (level - ref), abs=tol), level


def test_pace_reads_the_early_positions_not_the_styles(w2_fitted):
    horses = list(w2_regulars()[8:16])
    rows_of = lambda styles, early: list(zip(horses, styles, early, strict=True))  # noqa: E731

    def out(rows):
        return predicted(w2_fitted, w2_card(rows), W2_T)

    # The same early positions, the other seven all closers or all front runners: the field's
    # mean early position, the race's pace, is the same, so the front runner is the same.
    early = (0.5, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.5)
    closers = out(rows_of(("front", *["closer"] * 7), early)).loc[horses[0]]
    fronts = out(rows_of(("front", *["front"] * 7), early)).loc[horses[0]]
    assert fronts.to_numpy() == pytest.approx(closers.to_numpy(), abs=1e-9)

    # A null early_pos_mean3 is skipped in the field mean: seven values with mean 0.5 and a null
    # are the race of those seven and a 0.5. A value that is there counts: 1.0 for the last
    # runner moves the field mean to 0.5625, the front runners quicker and the closers slower.
    seven = (0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.5)
    skipped = out(rows_of(CARD_STYLES, (*seven, None)))
    filled = out(rows_of(CARD_STYLES, (*seven, 0.5)))
    assert skipped.to_numpy() == pytest.approx(filled.to_numpy(), abs=1e-9)
    later = out(rows_of(CARD_STYLES, (*seven, 1.0)))
    for h, style in zip(horses, CARD_STYLES, strict=True):
        step = later.loc[h, "mu_s"] - filled.loc[h, "mu_s"]
        if style == "front":
            assert step < -0.05, h
        if style == "closer":
            assert step > 0.05, h

    # A race whose runners have no early position at all (a debut race) has no pace term: an
    # unseen front runner and an unseen closer are level, and every value is finite.
    rows = [("syn-w2-new-f", "front", None), ("syn-w2-new-c", "closer", None)]
    rows += [(h, s, None) for h, s in zip(horses[2:], CARD_STYLES[2:], strict=True)]
    debut = out(rows)
    assert np.isfinite(debut.to_numpy()).all() and (debut["sd_s"] > 0).all()
    assert debut.loc["syn-w2-new-f", "mu_s"] == pytest.approx(
        debut.loc["syn-w2-new-c", "mu_s"], abs=1e-9
    )


def special_card(kinds):
    return w2_card([(sp(k, j), s, SP_EARLY) for k in kinds for j, s in enumerate(SP_STYLE)])


# (hh, mm, ss, us) on GOING_DAY, and whether TOK's races of the day are all resulted before it
GOING_SWEEP = (
    ((0, 30), False),  # nothing has run
    ((1, 15), False),  # R1 published at exactly t: not before t
    ((1, 15, 0, 1), False),  # F's and G's runs are in; TOK's R2 and R3 are not
    ((1, 50), False),
    ((2, 30), False),  # every race has run and R3 is in; R2 is held
    ((3, 0), False),  # R2 published at exactly t
    ((3, 0, 0, 1), True),
    ((5, 0), True),
    ((23, 0), True),
)


def w2_speed_cutoff():
    """The latest publication among the timed races of the second world's speed fit frame, from
    the planted frame."""
    races = _world2()[1]
    timed = races[races["fit"] & races["top3_time_s"].notna()]
    assert timed["result_available_utc"].notna().all()
    return timed["result_available_utc"].max()


def test_a_fit_days_in_sample_variant_waits_until_the_day_is_fully_resulted(w2_horse):
    # F and G are twins (the same par, off, publication, theta, style, weight and pace, and the
    # same 2.0 s of going in their time) apart from their (day, group)'s completion. TOK's races
    # on GOING_DAY are all timed, so TOK is fully resulted (03:00) by the speed model's fit cutoff
    # (TOK's R2, 03:00): predict refuses every t at or before it, and at every t after it the
    # in-sample variant takes the going out of F's run: F is well ahead of G, which stays on the
    # prior path (KYO's untimed R3 is never published), the same from t to t. (F as G's twin
    # before TOK is fully resulted is no longer seen at predict: that wait is the untimed sweep's
    # below, and the completion rule `SpeedModel.in_sample_variant`'s.)
    assert w2_speed_cutoff() == W2_SPEED_CUTOFF
    frame = special_card("FG")
    allowed = []
    for when, done in GOING_SWEEP:
        t = at(GOING_DAY, *when)
        if t <= W2_SPEED_CUTOFF:
            with pytest.raises(ValueError):
                w2_horse.predict(frame, at_utc=t)
            continue
        assert done, when  # the planted sweep: TOK is fully resulted at every allowed t
        out = predicted(w2_horse, frame, t)
        assert np.isfinite(out.to_numpy()).all(), when
        for j in range(len(SP_THETA)):
            f, g = out.loc[sp("F", j)], out.loc[sp("G", j)]
            assert f["mu_s"] < g["mu_s"] - 0.6, (when, j)
        allowed.append(out.loc[[sp("G", j) for j in range(len(SP_THETA))]])
    assert len(allowed) == 3
    for g_rows in allowed[1:]:  # G stays on the prior path
        assert g_rows.to_numpy() == pytest.approx(allowed[0].to_numpy(), abs=1e-9)


def test_a_fit_days_in_sample_variant_takes_its_going_out(w2_horse):
    # After everything: F, whose fit day is fully resulted, is centred as O, its twin that ran on
    # a day of no going (the 2.0 s removed); G, never fully resulted, still carries the going.
    out = predicted(w2_horse, special_card("FGO"), W2_T)["mu_s"]
    for j in range(len(SP_THETA)):
        f, g, o = out[sp("F", j)], out[sp("G", j)], out[sp("O", j)]
        assert f == pytest.approx(o, abs=0.4), j
        assert g - o > 0.6, j
    # theta's own spread survives: F's order is the planted one
    f_mu = [out[sp("F", j)] for j in range(len(SP_THETA))]
    assert f_mu[0] < f_mu[2] < f_mu[4]


# (hh, mm, ss, us) on GOING_DAY, and whether NAK's races of the day, the untimed R3 with them, are
# all resulted before it
UNTIMED_SWEEP = (
    ((1, 15, 0, 1), False),  # P's, U's and G's runs are in
    ((1, 50), False),
    ((1, 50, 0, 1), False),  # NAK's and HAN's timed races are all in; their R3 is not
    ((2, 30), False),
    ((3, 0, 0, 1), False),  # TOK is fully resulted (F has its in-sample variant)
    ((4, 0), False),  # NAK's R3 published at exactly t
    ((4, 0, 0, 1), True),
    ((23, 0), True),
)


def test_a_fit_days_untimed_race_holds_back_its_in_sample_variant(w2_horse):
    # P (NAK) and U (HAN) are G's twins until NAK's untimed R3 is published, after its timed races
    # and after the speed model's fit cutoff (03:00, before which predict refuses every t): its
    # result counts in the day's completion though it brings no top-3 time, so P stays on the
    # prior path until 04:00. HAN's R3 is never published: U is G's twin all day.
    frame = special_card("FGPU")
    seen = []
    for when, done in UNTIMED_SWEEP:
        t = at(GOING_DAY, *when)
        if t <= W2_SPEED_CUTOFF:
            with pytest.raises(ValueError):
                w2_horse.predict(frame, at_utc=t)
            continue
        out = predicted(w2_horse, frame, t)
        seen.append(done)
        for j in range(len(SP_THETA)):
            p, u, g = (out.loc[sp(k, j)] for k in "PUG")
            assert u.to_numpy() == pytest.approx(g.to_numpy(), abs=1e-9), (when, j)
            if done:
                assert p["mu_s"] < g["mu_s"] - 0.6, (when, j)
            else:
                assert p.to_numpy() == pytest.approx(g.to_numpy(), abs=1e-9), (when, j)
    assert seen == [False, False, True, True]  # the wait is seen after the cutoff


# A fit day of six groups, each a track's turf 1600 m races on COMPLETION_DAY, beside the flat
# TOK frame above (one par, turf 1600 m, shared by every track): each race's (publication (hh, mm)
# or None, whether it has a top-3 time), and the instant (hh, mm) the (day, group) is fully
# resulted at (None: never).
COMPLETION_DAY = date(2025, 4, 5)
COMPLETION_POSTS = ((1, 5), (1, 40), (2, 15))
COMPLETION_CASES = (
    ("TOK", (((1, 15), True), ((1, 50), True), ((4, 0), False)), (4, 0)),  # untimed, in last
    ("KYO", (((1, 15), True), ((1, 50), True), (None, False)), None),  # untimed, never in
    ("NAK", (((1, 15), True), ((1, 30), False), ((2, 25), True)), (2, 25)),  # untimed, in earlier
    ("HAN", (((1, 15), True), ((1, 50), True), ((2, 25), True)), (2, 25)),  # every race timed
    ("SAP", (((1, 15), True), (None, True)), None),  # a timed race never in
    ("CHU", (((5, 0), False), ((1, 50), True)), (5, 0)),  # the first race untimed, in last
)
COMPLETION_GOING = {"TOK": 0.6, "KYO": -0.4, "NAK": 0.3, "HAN": 0.8, "SAP": -0.2, "CHU": 0.5}


def completion_races():
    rows = []
    for track, plan, _ in COMPLETION_CASES:
        for i, (pub, timed) in enumerate(plan, start=1):
            rows.append(
                {
                    "race_id": f"syn-done-{track}-{i:02d}",
                    "race_utc": at(COMPLETION_DAY, *COMPLETION_POSTS[i - 1]),
                    "track": track,
                    "surface": "turf",
                    "distance": 1600,
                    "top3_time_s": 96.0 + COMPLETION_GOING[track] + 0.1 * i if timed else math.nan,
                    "result_available_utc": None if pub is None else at(COMPLETION_DAY, *pub),
                }
            )
    probe = pd.DataFrame(rows)
    for col in ("race_utc", "result_available_utc"):
        probe[col] = pd.to_datetime(probe[col], utc=True)
    return pd.concat([flat_races(), probe], ignore_index=True)


@pytest.mark.parametrize("t_df", [None, 5.0], ids=["normal", "student_t5"])
def test_a_fit_day_is_fully_resulted_once_its_untimed_races_are_too(t_df):
    # SpeedModel.fit on the frame with the untimed rows and without them: the same pars, scales
    # and in-sample variants (estimated from the timed races alone), but the instant a (day,
    # group) is fully resulted is the latest publication over all its races, timed or not.
    spec = SpeedSpec(
        par_keys=W2_PAR_KEYS, variant_keys=VARIANT_KEYS, same_day_variant=True, t_df=t_df
    )
    races = completion_races()
    model = SpeedModel(spec).fit(races)
    timed = SpeedModel(spec).fit(races[races["top3_time_s"].notna()].reset_index(drop=True))
    cell = {"surface": "turf", "distance": 1600}
    assert model.par(**cell) == pytest.approx(timed.par(**cell), abs=1e-12)
    assert (model.sigma, model.tau) == pytest.approx((timed.sigma, timed.tau), abs=1e-12)
    for track, _, done in COMPLETION_CASES:
        got = model.in_sample_variant(COMPLETION_DAY, track, "turf")
        ref = timed.in_sample_variant(COMPLETION_DAY, track, "turf")
        assert got is not None and ref is not None, track
        assert got[:2] == pytest.approx(ref[:2], abs=1e-12), track
        if done is None:
            assert got[2] is None, track
        else:
            assert got[2] == pd.Timestamp(at(COMPLETION_DAY, *done)), track


# ---------------------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------------------

NAIVE = datetime(2025, 2, 1, 3, 0)  # no tzinfo

BAD_INPUTS = (
    "negative_drift",
    "nan_drift",
    "zero_t_df",
    "predict_before_fit",
    "runs_missing_a_column",
    "runs_missing_early_pos_mean3",
    "card_missing_early_pos_mean3",
    "runs_naive_race_utc",
    "predict_naive_at_utc",
    "p_speed_unknown_method",
    "p_speed_zero_scale",
    "p_speed_nan_mu",
    "same_day_bias_naive_at_utc",
)


@pytest.mark.parametrize("case", BAD_INPUTS)
def test_bad_inputs_are_refused(fitted, case):
    mu = np.array([95.0, 96.0])
    calls = {
        "negative_drift": (ValueError, lambda: HorseSpeed(drift_var_per_day=-0.01, t_df=None)),
        "nan_drift": (ValueError, lambda: HorseSpeed(drift_var_per_day=math.nan, t_df=None)),
        "zero_t_df": (ValueError, lambda: HorseSpeed(drift_var_per_day=DRIFT, t_df=0.0)),
        "predict_before_fit": (
            RuntimeError,
            lambda: HorseSpeed(drift_var_per_day=DRIFT, t_df=None).predict(
                card(list(fillers())), at_utc=at(TD, 23, 0)
            ),
        ),
        "runs_missing_a_column": (
            ValueError,
            lambda: HorseSpeed(drift_var_per_day=DRIFT, t_df=None).fit(
                early_runs().drop(columns="result_available_utc"), speed_model()
            ),
        ),
        "runs_missing_early_pos_mean3": (
            ValueError,
            lambda: HorseSpeed(drift_var_per_day=DRIFT, t_df=None).fit(
                early_runs().drop(columns="early_pos_mean3"), speed_model()
            ),
        ),
        "card_missing_early_pos_mean3": (
            ValueError,
            lambda: fitted().predict(
                card(list(fillers())).drop(columns="early_pos_mean3"), at_utc=at(TD, 23, 0)
            ),
        ),
        "runs_naive_race_utc": (
            ValueError,
            lambda: HorseSpeed(drift_var_per_day=DRIFT, t_df=None).fit(
                early_runs().assign(race_utc=lambda f: f["race_utc"].dt.tz_localize(None)),
                speed_model(),
            ),
        ),
        "predict_naive_at_utc": (
            ValueError,
            lambda: fitted().predict(card(list(fillers())), at_utc=NAIVE),
        ),
        "p_speed_unknown_method": (ValueError, lambda: p_speed(mu, scale=1.0, method="bogus")),
        "p_speed_zero_scale": (ValueError, lambda: p_speed(mu, scale=0.0)),
        "p_speed_nan_mu": (ValueError, lambda: p_speed(np.array([95.0, math.nan]), scale=1.0)),
        "same_day_bias_naive_at_utc": (
            ValueError,
            lambda: same_day_bias(bias_results(), at_utc=NAIVE, venue="TOK"),
        ),
    }
    error, call = calls[case]
    with pytest.raises(error):
        call()
