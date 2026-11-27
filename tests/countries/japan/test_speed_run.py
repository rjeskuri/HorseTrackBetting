"""#96's expected top-3 time on development folds against the par-only baseline (#305, epic #73).

Contract pinned here (`horsetrack.countries.japan.studies.speed_run`):

- `build_race_table(races, runners, *, estimand) -> (table, counts)`. `races` and `runners` are
  store-shaped (docs/specs/store.md). Columns read: races race_id, meeting_id, race_date (a
  `datetime.date`, the JST race day), venue, surface, distance_m, course_config, rail_setting,
  race_class, post_time_local ("H:MM" or "HH:MM" on the JST wall clock); runners race_id,
  finish_pos (Int64, null when not placed), finish_time_s. Nothing else is read: no JRDB table
  (D21).
  - `estimand` is keyword-only with no default. The one name it knows is "mean_placed_1_3", the
    owner's answer on #96 and its dead-heat reading on #305 (2026-11-25): top3_time_s is the
    mean finish time of the first three past the post, per runner. Each runner placed 1, 2 or 3
    counts once; only a dead heat for 3rd, which puts more than three runners in the top 3, is
    averaged into one third slot. So 1,1,3 -> (t1 + t1' + t3)/3; 1,2,2 -> (t1 + t2 + t2')/3;
    1,2,3,3 -> (t1 + t2 + mean(t3, t3'))/3; 1,1,3,3 -> (t1 + t1' + mean(t3, t3'))/3. A race with
    no 3rd-placed runner uses the runners placed 1-2 present, each once (1,2 -> (t1 + t2)/2).
    Any other value (None included) is a ValueError.
  - `table`: one row per kept race with race_id, meeting_id, race_date, post_utc, race_utc
    (= post_utc), result_available_utc (= post_utc + 30 min, the same-day dividends rule of
    availability.yaml), top3_time_s, and the par and variant keys of `SpeedModel`: track (venue),
    surface, distance (distance_m), layout (course_config) and rail (rail_setting), each null
    -> "unknown", and class_level (race_class). post_utc is post_time_local on the Asia/Tokyo
    wall clock of race_date, as a tz-aware UTC instant (docs/specs/as-of.md, #227).
  - `counts`, a dict: n_races (races in), n_jump (surface "jump", excluded), n_no_post_time
    (excluded), n_no_placed_time (no runner placed 1-3 with a finish time, excluded), n_kept
    (= len(table)), and the coverage of the kept races (#416), each counted on its own, so a race
    may be in more than one: n_missing_timing (a runner placed 1-3 has no finish time),
    n_short_order (fewer than three runners placed 1-3: placings not awarded, e.g. only 1 and 2;
    a dead heat for 2nd, 1,2,2, is three runners and not short) and n_tie (two or more runners
    share a place in 1-3). Who placed comes from finish_pos alone, before any time is looked at.
    In these fixtures a race has at most one reason to be excluded, so the exclusions and n_kept
    add up to n_races.
  - The label is unchanged by #416 (changing it is an owner decision): it is still computed from
    the placed runners that have a time, so 1,2,3 with only 3rd timed is t3. Such a race is now
    flagged: the table's boolean column complete_timing is True when every runner placed 1-3 has
    a finish time (a genuine short order with its placings timed is complete).
- `predict_folds(table, *, spec, scheme, decision_min) -> predictions`: the #70 fold loop. Races
  on holdout days (`core.eval.holdout.is_holdout_day("japan", race_date)`) are dropped first, so
  they are in no fold, training or test; `core.eval.folds.walk_forward(race_date, scheme=scheme)`
  over the development races gives the folds. Per fold a `SpeedModel` is fitted on the training
  races and observes the test races' results; each test race is predicted at its decision time
  t = post_utc - `decision_min` minutes with `expected_top3_time(race, at_utc=t)`, with
  `spec.same_day_variant` True (variant_on) and False (variant_off), and the par of its cell is
  the baseline (par_only). One row per scored race: race_id, meeting_id, race_date, fold (one
  label per test window), post_utc, decision_utc, top3_time_s and the three predictions, plus
  `spec.par_keys`. A test race whose cell its fold's fit never saw is not scored and is counted
  in `predictions.attrs["n_unseen_cell"]`.
- `rmse_diff_bootstrap(model_err, baseline_err, clusters, *, n_boot, level, seed) -> RmseDiff`:
  the errors (prediction - actual, s) are Series indexed by race id, `clusters` maps race id to
  meeting (venue x day, D7). `RmseDiff` has rmse_model, rmse_baseline, gain (= rmse_baseline -
  rmse_model, positive when the model is better, as `core.eval.scoring.gain`), lo, hi, n_races
  and n_clusters. Each replicate draws as many meetings as there are, with replacement, keeps
  every race of each drawn meeting, and recomputes both RMSEs over the pooled races, model and
  baseline on the same draw (paired). lo is the (1 - level) quantile of the replicates' gain and
  hi the level quantile, as `core.eval.bootstrap.paired_bootstrap`.
- `rmse_table(predictions, *, model, baseline, by, n_boot, level, seed) -> DataFrame`: the
  columns scope ("overall" or "cell"), the `by` columns (null on the overall row), n_races,
  n_meetings, rmse_model, rmse_baseline, gain, lo, hi; one overall row and one row per distinct
  `by` value, each with the meeting-clustered bounds of its own races (clusters from meeting_id).
- `timing_sensitivity(table, *, spec, scheme, decision_min, n_boot, level, seed) -> dict` (#416):
  the fold loop twice for one error model (`spec`), on the same development fold windows (those of
  the full table: dropping races never moves a window). "existing": `predict_folds` on the full
  table, the published comparison, unchanged. "complete": the races with complete_timing only;
  the others are in no fit and are not scored, and its attrs["folds"] has the same fold labels in
  the same order. "gains": per prediction ("variant_on", "variant_off") a dict with "existing"
  and "complete" (`RmseDiff` against par_only over each arm's scored races) and "paired" (a dict
  over the races scored in both arms: n_races, n_meetings, gain_existing, gain_complete, each arm's
  gain on those races, delta = gain_complete - gain_existing, and meeting-clustered one-sided
  bounds lo, hi on delta). A table without complete_timing is a ValueError.
- `compute_study(table, spec, scheme, minutes)` (#416 part): next to each existing-label variant
  `N`, a variant `N:complete_timing` (a registry record each, failures included) whose metrics
  include ("all", "rmse_gain_s"), the complete arm's gain, and ("all", "rmse_gain_change_s"), the
  paired delta with its bounds. `render_report` shows the builder's coverage counts and, for each
  error model and prediction, the existing gain, the complete-timing gain and the paired delta
  side by side on one table row (floats as `markdown_table` writes them, 4 decimals).
- `read_inputs() -> (races, runners)`: the JRA store's races and runners
  (`processed_dir("jra")`), holdout days dropped, shaped for `build_race_table`; it reads no
  JRDB table.

Not pinned: the order of exclusions when a race has two reasons, more than three runners placed
1-2 (1,1,1,1 or 1,2,2,2, which the owner's answer leaves open), the fold label's format, the
registry records' other fields, the report's wording and layout beyond the row above, the run's
CLI, and whether the complete arm's same-day variant reads the dropped races' results. Every
frame is synthetic and built here; expected values come from the planted races (hand-computed
means, zoneinfo instants, planted pars and day variants), never from the code under test. The
holdout days are the committed definition's (holdout.yaml), so the synthetic calendar reaches
past 2026-11-28 to hold some.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import inspect
import math
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.models.speed import SpeedSpec
from horsetrack.countries.japan.studies import speed_run as study
from horsetrack.countries.japan.studies.speed_run import (
    build_race_table,
    predict_folds,
    read_inputs,
    rmse_diff_bootstrap,
    rmse_table,
)

ESTIMAND = "mean_placed_1_3"
JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
RESULT_LAG = timedelta(minutes=30)  # the same-day dividends rule (availability.yaml, #227)
PAR_KEYS = ("track", "surface", "distance", "layout", "rail", "class_level")
VARIANT_KEYS = ("track", "surface")
TABLE_COLUMNS = {
    "race_id",
    "meeting_id",
    "race_date",
    "post_utc",
    "race_utc",
    "result_available_utc",
    "top3_time_s",
    "complete_timing",
    *PAR_KEYS,
}
COUNT_KEYS = (
    "n_races",
    "n_jump",
    "n_no_post_time",
    "n_no_placed_time",
    "n_kept",
    "n_missing_timing",
    "n_short_order",
    "n_tie",
)
PREDICTIONS = ("variant_on", "variant_off")
BOOT = load_country_yaml("japan", "scoring")["bootstrap"]
LEVEL = float(BOOT["level"])  # D7: 95%, one-sided
N_BOOT = 2000  # a test's precision, far under the run's n_boot

# ---------------------------------------------------------------------------------------------
# Store-shaped cards
# ---------------------------------------------------------------------------------------------

DAY = date(2026, 10, 10)


def rid(day: date, venue: str, race_no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{race_no:02d}"


def post_instant(day: date, hhmm: str) -> pd.Timestamp:
    """The post as a UTC instant, from first principles: the JST wall clock of the race day."""
    hh, mm = (int(x) for x in hhmm.split(":"))
    return pd.Timestamp(datetime.combine(day, time(hh, mm), tzinfo=JST).astimezone(UTC))


def card(specs):
    """Store-shaped races and runners frames. Each spec is a dict: race_no, and optionally day,
    venue, surface, distance, course, rail, cls, post (None: no post time), and `runners`, a list
    of (finish_pos or None, finish_time_s or None)."""
    races, runners = [], []
    for spec in specs:
        day, venue, no = spec.get("day", DAY), spec.get("venue", "TOK"), spec["race_no"]
        race_id = rid(day, venue, no)
        placed = [p for p, _ in spec["runners"] if p is not None]
        winner = [t for p, t in spec["runners"] if p == 1 and t is not None]
        races.append(
            {
                "jurisdiction": "jra",
                "race_id": race_id,
                "meeting_id": f"jra:{day.isoformat()}:{venue}",
                "race_date": day,
                "venue": venue,
                "race_no": no,
                "race_class": spec.get("cls", "C1"),
                "distance_m": spec.get("distance", 1600),
                "surface": spec.get("surface", "turf"),
                "course_config": spec.get("course", "outer"),
                "rail_setting": spec.get("rail"),
                "post_time_local": spec.get("post", f"{10 + (no - 1) % 12}:05"),
                "field_size": len(spec["runners"]),
                "finish_time_s": winner[0] if winner else None,
                "status": "run" if placed else "abandoned",
            }
        )
        for horse_no, (pos, t) in enumerate(spec["runners"], start=1):
            runners.append(
                {
                    "jurisdiction": "jra",
                    "race_id": race_id,
                    "meeting_id": f"jra:{day.isoformat()}:{venue}",
                    "race_date": day,
                    "venue": venue,
                    "race_no": no,
                    "horse_id": f"h:{race_id}:{horse_no}",
                    "horse_no": horse_no,
                    "finish_pos": pos,
                    "dead_heat": pos is not None and placed.count(pos) > 1,
                    "finish_time_s": t,
                    "is_scratched": False,
                }
            )
    races = pd.DataFrame(races)
    runners = pd.DataFrame(runners)
    runners["finish_pos"] = runners["finish_pos"].astype("Int64")
    runners["finish_time_s"] = runners["finish_time_s"].astype(float)
    runners["dead_heat"] = runners["dead_heat"].astype("boolean")
    # nothing may rely on the runners being in finishing order
    return races, runners.sample(frac=1.0, random_state=305).reset_index(drop=True)


def full_field(winner: float, n: int = 8):
    """Placings 1..n, 0.3 s apart."""
    return [(p, round(winner + 0.3 * (p - 1), 1)) for p in range(1, n + 1)]


def by_race(table):
    return table.set_index("race_id")


def test_builder_requires_the_estimand():
    races, runners = card(
        [{"race_no": 1, "runners": full_field(95.0)}, {"race_no": 2, "runners": full_field(70.5)}]
    )
    param = inspect.signature(build_race_table).parameters["estimand"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty

    with pytest.raises(TypeError):
        build_race_table(races, runners)
    with pytest.raises(TypeError):
        build_race_table(races, runners, ESTIMAND)
    for bad in (None, "", "no_such_estimand"):
        with pytest.raises(ValueError):
            build_race_table(races, runners, estimand=bad)

    table, counts = build_race_table(races, runners, estimand=ESTIMAND)
    assert sorted(table["race_id"]) == [rid(DAY, "TOK", 1), rid(DAY, "TOK", 2)]
    assert by_race(table).loc[rid(DAY, "TOK", 1), "top3_time_s"] == pytest.approx(95.3)
    assert by_race(table).loc[rid(DAY, "TOK", 2), "top3_time_s"] == pytest.approx(70.8)
    assert set(COUNT_KEYS) <= set(counts)


def test_top3_time_is_mean_of_placed_1_to_3_with_dead_heats_averaged():
    # (runners, expected top3_time_s by hand); each race has its own scale
    cases = [
        # no dead heat: the mean of the three placed times; 4th and later never count
        ([(1, 95.0), (2, 95.2), (3, 95.7), (4, 96.0), (5, 96.5)], (95.0 + 95.2 + 95.7) / 3),
        # a dead heat for 3rd, equal times: the 3rd-placed runners fill one third slot
        ([(1, 120.1), (2, 120.4), (3, 120.7), (3, 120.7), (5, 121.0)], (120.1 + 120.4 + 120.7) / 3),
        # a dead heat for 3rd with unequal times: their mean is the third slot's time
        ([(1, 70.0), (2, 70.3), (3, 70.5), (3, 70.7), (5, 71.0)], (70.0 + 70.3 + 70.6) / 3),
        # a triple dead heat for 3rd
        ([(1, 83.0), (2, 83.2), (3, 83.5), (3, 83.5), (3, 83.8)], (83.0 + 83.2 + 83.6) / 3),
        # only two placed (the rest did not finish): the placed runners present, each once
        ([(1, 101.0), (2, 101.6), (None, None), (None, None)], (101.0 + 101.6) / 2),
        # only the winner placed
        ([(1, 60.0), (None, None), (None, None)], 60.0),
        # an unplaced runner's time never counts, however fast
        ([(None, 50.0), (1, 110.0), (2, 110.5), (3, 111.0), (4, 111.2)], 110.5),
        # placings given out of order on the page
        ([(3, 58.9), (1, 58.1), (5, 59.5), (2, 58.4), (4, 59.0)], (58.1 + 58.4 + 58.9) / 3),
    ]
    races, runners = card(
        [{"race_no": k + 1, "runners": rows} for k, (rows, _) in enumerate(cases)]
    )
    table, counts = build_race_table(races, runners, estimand=ESTIMAND)
    got = by_race(table)["top3_time_s"]
    assert len(got) == len(cases)
    for k, (_, expected) in enumerate(cases):
        assert got[rid(DAY, "TOK", k + 1)] == pytest.approx(expected, abs=1e-9), k + 1
    # the two races with fewer than three runners placed are kept and counted as short orders,
    # the three dead heats for 3rd as ties; every placed runner here is timed
    assert counts["n_short_order"] == 2
    assert counts["n_tie"] == 3
    assert counts["n_missing_timing"] == 0
    assert by_race(table)["complete_timing"].astype(bool).all()


def test_a_dead_heat_above_3rd_counts_each_runner_once():
    """The owner's 'dead heats averaged' read per runner, the first three past the post (#305,
    2026-11-25): each runner placed 1 or 2 counts once, a dead heat there included, and only the
    runners placed 3rd are averaged into one third slot. The 4th and later never count."""
    cases = [
        # dead heat for 1st, equal times: both winners count; it has a 3rd, so not partly placed
        ([(1, 96.0), (1, 96.0), (3, 96.4), (4, 96.6)], (96.0 + 96.0 + 96.4) / 3),
        # dead heat for 2nd: both count; no 3rd-placed runner, so partly placed
        ([(1, 84.0), (2, 84.3), (2, 84.3), (4, 84.6)], (84.0 + 84.3 + 84.3) / 3),
        # dead heat for 1st with unequal times: each winner's own time counts
        ([(1, 72.0), (1, 72.2), (3, 72.7), (4, 73.0)], (72.0 + 72.2 + 72.7) / 3),
        # dead heats for 1st and for 3rd: both winners count, the two 3rds share the third slot
        ([(1, 88.0), (1, 88.0), (3, 88.5), (3, 88.9), (5, 89.2)], (88.0 + 88.0 + 88.7) / 3),
    ]
    races, runners = card(
        [{"race_no": k + 1, "runners": rows} for k, (rows, _) in enumerate(cases)]
    )
    table, counts = build_race_table(races, runners, estimand=ESTIMAND)
    got = by_race(table)["top3_time_s"]
    assert len(got) == len(cases)
    for k, (_, expected) in enumerate(cases):
        assert got[rid(DAY, "TOK", k + 1)] == pytest.approx(expected, abs=1e-9), k + 1
    # all four are ties; none is a short order: the dead heat for 2nd puts three runners in the
    # top 3 although nobody is placed 3rd
    assert counts["n_tie"] == 4
    assert counts["n_short_order"] == 0
    assert counts["n_missing_timing"] == 0


# (runners, top3_time_s under the unchanged signed label, complete_timing, missing timing,
#  short order, tie). Who placed is read from finish_pos alone; the label still averages the
# placed runners that have a time (#416 keeps it; changing it is the owner's decision).
COVERAGE_CASES = {
    "timed_top3": ([(1, 95.0), (2, 95.2), (3, 95.7), (4, 96.0)], (95.0 + 95.2 + 95.7) / 3)
    + (True, False, False, False),
    # 1st and 2nd placed without a time, only 3rd timed: missing timing, not a short order
    "only_3rd_timed": ([(1, None), (2, None), (3, 101.7), (4, 102.0)], 101.7)
    + (False, True, False, False),
    "1st_untimed": ([(1, None), (2, 88.4), (3, 88.9), (4, 89.1)], (88.4 + 88.9) / 2)
    + (False, True, False, False),
    "2nd_untimed": ([(1, 70.0), (2, None), (3, 70.6), (5, 71.2)], (70.0 + 70.6) / 2)
    + (False, True, False, False),
    # an untimed 3rd is a missing time, not the short order #305's n_partly_placed called it
    "3rd_untimed": ([(1, 110.0), (2, 110.5), (3, None), (4, 111.4)], (110.0 + 110.5) / 2)
    + (False, True, False, False),
    # two genuine finishers: only 1st and 2nd were placed, both timed
    "two_finishers": ([(1, 101.0), (2, 101.6), (None, None), (None, None)], (101.0 + 101.6) / 2)
    + (True, False, True, False),
    "winner_only": ([(1, 60.0), (None, None), (None, None)], 60.0) + (True, False, True, False),
    # ties: today's convention, each runner placed 1 or 2 once, the 3rds as one slot
    "dead_heat_1st": ([(1, 96.0), (1, 96.0), (3, 96.4), (4, 96.6)], (96.0 + 96.0 + 96.4) / 3)
    + (True, False, False, True),
    "dead_heat_3rd": (
        [(1, 70.0), (2, 70.3), (3, 70.5), (3, 70.7), (5, 71.0)],
        (70.0 + 70.3 + 70.6) / 3,
    )
    + (True, False, False, True),
    # three runners in the top 3 with nobody placed 3rd: a tie, not a short order
    "dead_heat_2nd": ([(1, 84.0), (2, 84.3), (2, 84.3), (4, 84.6)], (84.0 + 84.3 + 84.3) / 3)
    + (True, False, False, True),
    # a runner placed 4th or later without a time is not part of the top 3
    "4th_untimed": (
        [(1, 83.0), (2, 83.2), (3, 83.5), (4, None), (5, None)],
        (83.0 + 83.2 + 83.5) / 3,
    )
    + (True, False, False, False),
    # an unplaced runner with a time, however fast, is not an official finisher
    "unplaced_timed": ([(None, 50.0), (1, 110.0), (2, 110.5), (3, 111.0)], 110.5)
    + (True, False, False, False),
    # both conditions at once, each counted on its own
    "short_and_untimed": ([(1, 77.0), (2, None), (None, None)], 77.0) + (False, True, True, False),
    "dead_heat_3rd_one_untimed": (
        [(1, 90.0), (2, 90.2), (3, 90.5), (3, None)],
        (90.0 + 90.2 + 90.5) / 3,
    )
    + (False, True, False, True),
    "dead_heat_1st_alone": ([(1, 64.0), (1, 64.2), (None, None)], 64.1) + (True, False, True, True),
}


@pytest.mark.parametrize("seed", [416, 7, 2026])
def test_builder_counts_missing_timing_short_orders_and_ties_separately(seed):
    rng = np.random.default_rng(seed)
    names = list(COVERAGE_CASES)
    multiplicity = dict(zip(names, rng.integers(0, 4, len(names)).tolist(), strict=True))
    multiplicity["only_3rd_timed"] = max(1, multiplicity["only_3rd_timed"])
    specs, case_of = [], {}
    for name in names:
        rows = COVERAGE_CASES[name][0]
        for _ in range(multiplicity[name]):
            no = len(specs) + 1
            specs.append({"race_no": no, "runners": rows})
            case_of[rid(DAY, "TOK", no)] = name
    races, runners = card(specs)

    table, counts = build_race_table(races, runners, estimand=ESTIMAND)
    rows = by_race(table)
    assert set(rows.index) == set(case_of)  # every race here has a placed, timed runner: kept
    assert pd.api.types.is_bool_dtype(rows["complete_timing"]), rows["complete_timing"].dtype
    for race_id, name in case_of.items():
        _, label, complete, *_ = COVERAGE_CASES[name]
        assert rows.loc[race_id, "top3_time_s"] == pytest.approx(label, abs=1e-6), name
        assert bool(rows.loc[race_id, "complete_timing"]) is complete, name

    def n(flag: int) -> int:
        return sum(multiplicity[k] for k in names if COVERAGE_CASES[k][flag])

    assert {k: counts[k] for k in ("n_missing_timing", "n_short_order", "n_tie")} == {
        "n_missing_timing": n(3),
        "n_short_order": n(4),
        "n_tie": n(5),
    }, multiplicity
    assert counts["n_kept"] == len(specs) and counts["n_no_placed_time"] == 0
    assert int(rows["complete_timing"].sum()) == n(2)


def test_builder_excludes_jump_races():
    for n_jump in (0, 1, 3):
        specs = [
            {"race_no": 1, "runners": full_field(96.0)},
            {"race_no": 2, "runners": full_field(121.0), "distance": 2000},
            {"race_no": 3, "runners": full_field(98.0), "surface": "dirt", "course": None},
        ]
        specs += [
            # a jump race with a full, timed top 3 is still excluded
            {
                "race_no": 4 + j,
                "runners": full_field(200.0 + j),
                "surface": "jump",
                "distance": 3000,
            }
            for j in range(n_jump)
        ]
        races, runners = card(specs)
        table, counts = build_race_table(races, runners, estimand=ESTIMAND)
        assert sorted(table["race_id"]) == [rid(DAY, "TOK", k) for k in (1, 2, 3)], n_jump
        assert (table["surface"] != "jump").all()
        assert counts["n_jump"] == n_jump
        assert counts["n_races"] == 3 + n_jump
        assert counts["n_kept"] == 3


@pytest.mark.parametrize(
    "n_complete, n_jump, n_two_placed, n_dh_second, n_abandoned, n_untimed, n_no_post, n_missing",
    [(3, 1, 1, 1, 1, 1, 1, 2), (5, 3, 2, 0, 0, 2, 0, 0), (1, 0, 0, 2, 3, 0, 2, 3)],
)
def test_builder_counts_the_races_dropped_and_their_coverage(
    n_complete, n_jump, n_two_placed, n_dh_second, n_abandoned, n_untimed, n_no_post, n_missing
):
    specs, kept = [], []

    def add(kind, n, runners, **race):
        for _ in range(n):
            no = len(specs) + 1
            specs.append({"race_no": no, "runners": runners(no), **race})
            if kind == "kept":
                kept.append(rid(DAY, "TOK", no))

    add("kept", n_complete, lambda no: full_field(90.0 + no))
    # a dead heat for 1st has a 3rd-placed runner: complete
    add("kept", 1, lambda no: [(1, 80.0), (1, 80.0), (3, 80.4), (4, 80.9)])
    add("jump", n_jump, lambda no: full_field(200.0 + no), surface="jump", distance=3000)
    add("kept", n_two_placed, lambda no: [(1, 100.0 + no), (2, 100.4 + no), (None, None)])
    add("kept", n_dh_second, lambda no: [(1, 84.0), (2, 84.3), (2, 84.3), (4, 84.6)])
    # placed 1, 2 and 3, the 3rd without a time: kept, missing timing, not a short order
    add("kept", n_missing, lambda no: [(1, 70.0 + no), (2, 70.2 + no), (3, None), (4, 71.0 + no)])
    # abandoned: nobody placed, nobody timed
    add("drop", n_abandoned, lambda no: [(None, None)] * 6)
    # placed, but without a finish time
    add("drop", n_untimed, lambda no: [(1, None), (2, None), (3, None), (4, None)])
    add("drop", n_no_post, lambda no: full_field(95.0 + no), post=None)
    races, runners = card(specs)

    table, counts = build_race_table(races, runners, estimand=ESTIMAND)
    assert sorted(table["race_id"]) == sorted(kept)
    assert table["race_id"].is_unique
    assert {k: counts[k] for k in COUNT_KEYS} == {
        "n_races": len(specs),
        "n_jump": n_jump,
        "n_no_post_time": n_no_post,
        "n_no_placed_time": n_abandoned + n_untimed,
        "n_kept": len(kept),
        # the untimed races left out are not counted again among the kept races' coverage
        "n_missing_timing": n_missing,
        "n_short_order": n_two_placed,
        "n_tie": 1 + n_dh_second,
    }
    excluded = counts["n_jump"] + counts["n_no_post_time"] + counts["n_no_placed_time"]
    assert excluded + counts["n_kept"] == counts["n_races"]


def test_result_available_is_post_plus_30_min():
    # (race day, post_time_local): single- and two-digit hours, and a post whose UTC date is the
    # day before its JST race day
    posts = [
        (date(2026, 11, 28), "15:40"),
        (date(2026, 11, 28), "9:50"),
        (date(2025, 6, 15), "12:05"),
        (date(2026, 1, 4), "10:01"),
        (date(2026, 3, 1), "8:30"),
    ]
    specs = [
        {"race_no": k + 1, "day": d, "venue": "NAK", "post": hhmm, "runners": full_field(95.0)}
        for k, (d, hhmm) in enumerate(posts)
    ]
    # a race without a post time never has a result time: it is left out and counted
    specs.append(
        {"race_no": 9, "day": DAY, "venue": "NAK", "post": None, "runners": full_field(95)}
    )
    races, runners = card(specs)

    table, counts = build_race_table(races, runners, estimand=ESTIMAND)
    assert TABLE_COLUMNS <= set(table.columns)
    rows = by_race(table)
    assert len(rows) == len(posts)
    assert counts["n_no_post_time"] == 1
    for column in ("post_utc", "race_utc", "result_available_utc"):
        assert isinstance(rows[column].dtype, pd.DatetimeTZDtype), column
    for k, (d, hhmm) in enumerate(posts):
        row = rows.loc[rid(d, "NAK", k + 1)]
        post = post_instant(d, hhmm)
        assert pd.Timestamp(row["post_utc"]) == post, (d, hhmm)
        assert pd.Timestamp(row["race_utc"]) == post, (d, hhmm)
        assert pd.Timestamp(row["result_available_utc"]) == post + RESULT_LAG, (d, hhmm)
        assert pd.Timestamp(row["race_date"]).date() == d
    # 15:40 JST on 2026-11-28 is 06:40 UTC, its result at 07:10 UTC (docs/specs/as-of.md)
    first = rows.loc[rid(date(2026, 11, 28), "NAK", 1), "result_available_utc"]
    assert pd.Timestamp(first) == pd.Timestamp("2026-11-28 07:10", tz="UTC")


def test_the_table_keys_each_race_by_track_distance_course_and_class():
    names = ("race_no", "venue", "surface", "distance", "course", "rail", "cls")
    cards = [
        (1, "TOK", "turf", 1600, "outer", "B", "C1"),
        (2, "TOK", "turf", 2400, "inner", None, "G1"),
        (3, "KYO", "dirt", 1800, None, None, "OP"),
        (4, "NII", "turf", 1000, None, "A", "DEBUT"),
    ]
    specs = [dict(zip(names, values, strict=True)) for values in cards]
    for spec in specs:
        spec["runners"] = full_field(60.0 + 20.0 * spec["race_no"])
    races, runners = card(specs)
    table, _ = build_race_table(races, runners, estimand=ESTIMAND)
    rows = by_race(table)
    expected = {
        1: ("TOK", "turf", 1600, "outer", "B", "C1"),
        2: ("TOK", "turf", 2400, "inner", "unknown", "G1"),
        3: ("KYO", "dirt", 1800, "unknown", "unknown", "OP"),
        4: ("NII", "turf", 1000, "unknown", "A", "DEBUT"),
    }
    for spec in specs:
        race_id = rid(DAY, spec["venue"], spec["race_no"])
        got = tuple(rows.loc[race_id, k] for k in PAR_KEYS)
        assert got == expected[spec["race_no"]], race_id
        assert rows.loc[race_id, "meeting_id"] == f"jra:{DAY.isoformat()}:{spec['venue']}"
    assert table[list(PAR_KEYS)].notna().all().all()


# ---------------------------------------------------------------------------------------------
# The fold loop: a planted calendar of races
# ---------------------------------------------------------------------------------------------

# two-week test windows from 2026-11-02: four folds, each training on every earlier week
FOLD_SCHEME = FoldScheme(
    unit="week",
    test_length=2,
    step=2,
    min_train=2,
    window="expanding",
    embargo_days=1,
    anchor_month=1,
)
SPEC = SpeedSpec(par_keys=PAR_KEYS, variant_keys=VARIANT_KEYS, same_day_variant=True, t_df=None)
FIRST_DAY = date(2026, 10, 17)  # a Saturday
LAST_DAY = date(2026, 12, 27)  # past the holdout start (2026-11-28): some days are holdout
HOLDOUT_START = date(2026, 11, 28)
# 32 min apart: a race's result (post + 30 min) is available exactly at the next race's
# decision time when decision_min is 2, and one minute before it when decision_min is 1
POSTS = ("10:00", "10:32", "11:04", "11:36", "12:08", "12:40")
CELLS = {
    "TOK": (
        ("TOK", "turf", 1600, "outer", "unknown", "C1"),
        ("TOK", "turf", 2000, "outer", "unknown", "C2"),
        ("TOK", "turf", 1400, "inner", "unknown", "C1"),
    ),
    "KYO": (
        ("KYO", "turf", 1600, "outer", "unknown", "C1"),
        ("KYO", "turf", 1800, "inner", "unknown", "C3"),
        ("KYO", "turf", 1200, "inner", "unknown", "C1"),
    ),
}
PLANTED_PAR = dict(
    zip(
        (*CELLS["TOK"], *CELLS["KYO"]),
        (96.0, 121.0, 83.5, 95.5, 108.0, 70.5),
        strict=True,
    )
)
UNSEEN_CELL = ("KYO", "turf", 3000, "outer", "unknown", "G1")  # one race, on the last Saturday
UNSEEN_DAY = date(2026, 12, 26)
NOISE_SD = 0.3  # race noise, s; next to day variants of 0.8-1.6 s the fits converge fast
POISON_S = 40.0  # added to (and taken from) two cells' races on holdout days
PROBE_DAY = date(2026, 11, 7)  # a development test day whose results the leakage test moves
PROBE_SHIFT_S = 5.0


def weekend_days(first: date, last: date) -> list[date]:
    days, d = [], first
    while d <= last:
        if d.weekday() in (5, 6):
            days.append(d)
        d += timedelta(days=1)
    return days


def planted_variant(day: date, venue: str) -> float:
    """At least 0.8 s either way; each week's Saturday and Sunday cancel, and the venues mirror."""
    week = (day - FIRST_DAY).days // 7
    size = (0.8, 1.2, 1.6)[week % 3]
    sign = 1.0 if (week % 2 == 0) == (day.weekday() == 5) else -1.0
    return sign * size * (1.0 if venue == "TOK" else -1.0)


def _row(day, venue, idx, post, cell, top3):
    post_utc = post_instant(day, post)
    return {
        "race_id": rid(day, venue, idx + 1),
        "meeting_id": f"jra:{day.isoformat()}:{venue}",
        "race_date": day,
        "post_utc": post_utc,
        "race_utc": post_utc,
        "result_available_utc": post_utc + RESULT_LAG,
        "top3_time_s": top3,
        **dict(zip(PAR_KEYS, cell, strict=True)),
    }


@pytest.fixture(scope="module")
def calendar():
    """(table, truth): a builder-shaped table of two venues' six races on every weekend day from
    FIRST_DAY to LAST_DAY, and the planted values per race (idx, venue, par, variant)."""
    rng = np.random.default_rng(305)
    rows, truth = [], []
    for day in weekend_days(FIRST_DAY, LAST_DAY):
        held = is_holdout_day("japan", day)
        for venue, cells in CELLS.items():
            v = planted_variant(day, venue)
            for idx, post in enumerate(POSTS):
                cell = cells[idx % 3]
                top3 = PLANTED_PAR[cell] + v + rng.normal(0.0, NOISE_SD)
                if held and venue == "TOK" and idx % 3 in (0, 1):
                    top3 += POISON_S if idx % 3 == 0 else -POISON_S
                rows.append(_row(day, venue, idx, post, cell, top3))
                truth.append(
                    {"race_id": rows[-1]["race_id"], "idx": idx, "venue": venue}
                    | {"par": PLANTED_PAR[cell], "variant": v, "holdout": held}
                )
    unseen = _row(UNSEEN_DAY, "KYO", 6, "13:12", UNSEEN_CELL, 185.0 + rng.normal(0.0, NOISE_SD))
    rows.append(unseen)
    truth.append(
        {"race_id": unseen["race_id"], "idx": 6, "venue": "KYO", "par": 185.0, "variant": 0.0}
        | {"holdout": False}
    )
    table = pd.DataFrame(rows)
    for column in ("post_utc", "race_utc", "result_available_utc"):
        table[column] = pd.to_datetime(table[column], utc=True)
    truth = pd.DataFrame(truth).set_index("race_id")

    # the calendar holds holdout days and development days on both sides of the holdout start
    days = sorted(set(table["race_date"]))
    held = {d for d in days if is_holdout_day("japan", d)}
    assert held and all(d >= HOLDOUT_START for d in held)
    assert any(d >= HOLDOUT_START and d not in held for d in days)
    assert not is_holdout_day("japan", PROBE_DAY) and not is_holdout_day("japan", UNSEEN_DAY)
    return table.sample(frac=1.0, random_state=96).reset_index(drop=True), truth


def run(table, decision_min):
    return predict_folds(table, spec=SPEC, scheme=FOLD_SCHEME, decision_min=decision_min)


@pytest.fixture(scope="module")
def predictions(calendar):
    table, _ = calendar
    return run(table, 2)


def development_folds(table):
    """The races of each test window of walk_forward over the development races."""
    dev = table[[not is_holdout_day("japan", d) for d in table["race_date"]]]
    dev = dev.reset_index(drop=True)
    folds = walk_forward(dev["race_date"], scheme=FOLD_SCHEME)
    return [set(dev["race_id"].iloc[f.test_idx]) for f in folds]


def test_the_fold_loop_scores_each_development_test_race_once_and_never_a_holdout_day(
    calendar, predictions
):
    table, truth = calendar
    unseen_id = rid(UNSEEN_DAY, "KYO", 7)
    windows = [w - {unseen_id} for w in development_folds(table)]
    assert len(windows) >= 3

    assert predictions["race_id"].is_unique
    assert set(predictions["race_id"]) == set().union(*windows)
    # one fold label per test window
    labelled = predictions.groupby("fold")["race_id"].agg(frozenset)
    assert set(labelled) == {frozenset(w) for w in windows}

    held = {d for d in table["race_date"].unique() if is_holdout_day("japan", d)}
    scored_days = {pd.Timestamp(d).date() for d in predictions["race_date"]}
    assert not scored_days & held
    # development days after the holdout start are scored: only the holdout days are dropped
    assert any(d >= HOLDOUT_START for d in scored_days)

    # the race whose cell no fit has seen is not scored, and it is counted
    assert unseen_id not in set(predictions["race_id"])
    assert predictions.attrs["n_unseen_cell"] == 1

    # no fit trained on a holdout day: its races carry +/- 40 s in two Tokyo cells, and every
    # scored par stays at the planted one
    rows = predictions.set_index("race_id")
    error = (rows["par_only"] - truth.loc[rows.index, "par"]).abs()
    assert error.max() < 0.4, error.sort_values().tail()


def test_variant_off_is_the_par_only_baseline_and_variant_on_reads_earlier_results(
    calendar, predictions
):
    table, truth = calendar
    rows = predictions.set_index("race_id")
    plant = truth.loc[rows.index]

    assert set(PAR_KEYS) <= set(predictions.columns)
    actual = table.set_index("race_id").loc[rows.index, "top3_time_s"]
    assert np.allclose(rows["top3_time_s"], actual, rtol=0, atol=1e-12)
    expected_t = table.set_index("race_id").loc[rows.index, "post_utc"] - timedelta(minutes=2)
    assert (pd.to_datetime(rows["decision_utc"], utc=True) == expected_t).all()

    # without an injected prior the day's prior is N(0, tau^2): variant off is the par
    assert np.allclose(rows["variant_off"], rows["par_only"], rtol=0, atol=1e-9)

    # the 1st and 2nd races of a day have no result before t (the 1st race's result is available
    # exactly at the 2nd race's decision time): the prior, so the par
    early = plant["idx"] <= 1
    assert early.sum() > 40
    assert np.allclose(rows.loc[early, "variant_on"], rows.loc[early, "par_only"], atol=1e-9)

    # from the 3rd race on, the day's earlier results move the prediction toward the planted
    # variant of that day and track
    late = ~early
    shift = rows.loc[late, "variant_on"] - rows.loc[late, "par_only"]
    toward = shift * np.sign(plant.loc[late, "variant"])
    assert late.sum() > 80
    assert (toward > 0).mean() > 0.95
    assert toward.mean() > 0.7
    # and it beats the par on those races
    err_on = (rows.loc[late, "variant_on"] - rows.loc[late, "top3_time_s"]).abs()
    err_par = (rows.loc[late, "par_only"] - rows.loc[late, "top3_time_s"]).abs()
    assert err_on.mean() < 0.6 * err_par.mean()


@pytest.mark.parametrize("decision_min", [2, 1])
def test_no_test_race_variant_reads_a_result_available_at_or_after_its_decision_time(
    calendar, decision_min
):
    """Move every result of the probe day's Tokyo races from the 2nd on, the target's own
    included; a race's prediction may change only if one of those results is available strictly
    before its decision time t = post - decision_min."""
    table, _ = calendar
    probe = (table["race_date"] == PROBE_DAY) & (table["track"] == "TOK")
    races = table[probe].sort_values("post_utc")
    moved = races["race_id"].iloc[1:]
    shifted = table.copy()
    shifted.loc[shifted["race_id"].isin(moved), "top3_time_s"] += PROBE_SHIFT_S

    base = run(table, decision_min).set_index("race_id")
    after = run(shifted, decision_min).set_index("race_id")

    t = races["post_utc"] - timedelta(minutes=decision_min)
    first_moved_available = races["result_available_utc"].iloc[1]
    can_see = (t > first_moved_available).to_numpy()
    expected_unchanged = {0, 1, 2} if decision_min == 2 else {0, 1}
    assert {k for k, seen in enumerate(can_see) if not seen} == expected_unchanged

    for k, race_id in enumerate(races["race_id"]):
        assert race_id in base.index and race_id in after.index, race_id
        # the par is fitted on training days only: the test day's own results never move it
        assert after.loc[race_id, "par_only"] == pytest.approx(base.loc[race_id, "par_only"])
        delta = after.loc[race_id, "variant_on"] - base.loc[race_id, "variant_on"]
        if can_see[k]:
            assert delta > 0.5, (k, delta)
        else:
            assert delta == pytest.approx(0.0, abs=1e-9), (k, delta)

    # Kyoto's races that day are another track: untouched
    other = table[(table["race_date"] == PROBE_DAY) & (table["track"] == "KYO")]["race_id"]
    assert np.allclose(after.loc[other, "variant_on"], base.loc[other, "variant_on"], atol=1e-9)


# ---------------------------------------------------------------------------------------------
# The RMSE difference and its meeting-clustered paired bootstrap
# ---------------------------------------------------------------------------------------------

GAIN_AB = math.sqrt(3.0) - 1.5  # the pooled gain over meetings A and B below


def two_meetings(n_a=30, n_b=10):
    """Meeting A: the model is exact and the baseline 2 s off; meeting B: the model 3 s off and
    the baseline exact. Pooled, the model's RMSE is 1.5 and the baseline's sqrt(3)."""
    ids = [f"jra:2026-10-10:TOK:{k:02d}" for k in range(n_a)]
    ids += [f"jra:2026-10-11:KYO:{k:02d}" for k in range(n_b)]
    model = pd.Series([0.0] * n_a + [3.0] * n_b, index=ids)
    baseline = pd.Series([2.0] * n_a + [0.0] * n_b, index=ids)
    meeting = pd.Series(["jra:2026-10-10:TOK"] * n_a + ["jra:2026-10-11:KYO"] * n_b, index=ids)
    return model, baseline, meeting


def test_bootstrap_resamples_whole_meetings():
    model, baseline, meeting = two_meetings()
    got = rmse_diff_bootstrap(model, baseline, meeting, n_boot=N_BOOT, level=LEVEL, seed=1)
    assert got.rmse_model == pytest.approx(1.5)
    assert got.rmse_baseline == pytest.approx(math.sqrt(3.0))
    assert got.gain == pytest.approx(GAIN_AB)
    assert (got.n_races, got.n_clusters) == (40, 2)
    # A replicate draws two whole meetings: AA (gain 2 - 0), AB or BA (the pooled gain) or BB
    # (0 - 3), each a quarter, a half and a quarter of the time. The one-sided 95% bounds are
    # the BB and AA values.
    assert got.lo == pytest.approx(-3.0, abs=1e-9)
    assert got.hi == pytest.approx(2.0, abs=1e-9)
    # At level 0.6 both bounds fall in the mixed draws, so each replicate pools its races
    # (the mean of the two meetings' gains would be -0.5)
    mid = rmse_diff_bootstrap(model, baseline, meeting, n_boot=N_BOOT, level=0.6, seed=1)
    assert mid.lo == pytest.approx(GAIN_AB, abs=1e-9)
    assert mid.hi == pytest.approx(GAIN_AB, abs=1e-9)
    # Without the meetings (each race its own cluster) the same errors give a far narrower
    # interval: the clusters passed are the ones resampled
    alone = pd.Series(model.index, index=model.index)
    per_race = rmse_diff_bootstrap(model, baseline, alone, n_boot=N_BOOT, level=LEVEL, seed=1)
    assert per_race.n_clusters == 40
    assert -1.0 < per_race.lo < GAIN_AB < per_race.hi < 1.0
    # Paired: with the model's errors equal to the baseline's, every replicate's gain is 0
    rng = np.random.default_rng(7)
    same = pd.Series(rng.normal(0.0, 1.0, len(model)), index=model.index)
    paired = rmse_diff_bootstrap(same, same.copy(), meeting, n_boot=N_BOOT, level=LEVEL, seed=1)
    assert paired.gain == pytest.approx(0.0, abs=1e-12)
    assert paired.lo == pytest.approx(0.0, abs=1e-12)
    assert paired.hi == pytest.approx(0.0, abs=1e-12)


def _rmse(err) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(err, dtype=float)))))


def test_rmse_table_has_an_overall_row_and_a_row_per_cell_with_meeting_bounds():
    rng = np.random.default_rng(3050)
    better = ("TOK", "turf", 1600, "outer", "unknown", "C1")  # the model wins here
    worse = ("KYO", "turf", 1800, "inner", "unknown", "C3")  # the baseline wins here
    exact = ("NII", "turf", 1000, "straight", "unknown", "C1")  # the two-meeting cell
    rows = []
    for m in range(40):
        day = date(2026, 1, 3) + timedelta(days=7 * (m // 2))
        venue = ("TOK", "KYO")[m % 2]
        for k in range(6):
            cell, (sd_model, sd_base) = (better, (0.2, 1.0)) if k < 3 else (worse, (1.0, 0.2))
            rows.append(
                {"race_id": rid(day, venue, k + 1), "meeting_id": f"jra:{day.isoformat()}:{venue}"}
                | dict(zip(PAR_KEYS, cell, strict=True))
                | {"err_model": rng.normal(0, sd_model), "err_base": rng.normal(0, sd_base)}
            )
    model, baseline, meeting = two_meetings()
    for race_id in model.index:
        rows.append(
            {"race_id": race_id.replace("TOK", "NII").replace("KYO", "NIX")}
            | {"meeting_id": meeting[race_id].replace("TOK", "NII").replace("KYO", "NIX")}
            | dict(zip(PAR_KEYS, exact, strict=True))
            | {"err_model": model[race_id], "err_base": baseline[race_id]}
        )
    frame = pd.DataFrame(rows)
    actual = rng.normal(100.0, 10.0, len(frame))
    preds = frame.assign(
        top3_time_s=actual,
        variant_on=actual + frame["err_model"],
        par_only=actual + frame["err_base"],
    ).drop(columns=["err_model", "err_base"])
    preds = preds.sample(frac=1.0, random_state=1).reset_index(drop=True)

    out = rmse_table(
        preds,
        model="variant_on",
        baseline="par_only",
        by=PAR_KEYS,
        n_boot=N_BOOT,
        level=LEVEL,
        seed=11,
    )
    columns = {"scope", *PAR_KEYS, "n_races", "n_meetings"}
    columns |= {"rmse_model", "rmse_baseline", "gain", "lo", "hi"}
    assert columns <= set(out.columns)
    assert sorted(out["scope"]) == ["cell", "cell", "cell", "overall"]

    overall = out[out["scope"] == "overall"].iloc[0]
    assert overall[list(PAR_KEYS)].isna().all()
    cells = out[out["scope"] == "cell"].set_index(list(PAR_KEYS))
    assert set(cells.index) == {better, worse, exact}

    cell_of = list(zip(*(frame[k] for k in PAR_KEYS), strict=True))
    groups = {"overall": frame}
    groups |= {c: frame[[x == c for x in cell_of]] for c in (better, worse, exact)}
    for key, part in groups.items():
        row = overall if key == "overall" else cells.loc[key]
        rmse_m, rmse_b = _rmse(part["err_model"]), _rmse(part["err_base"])
        assert row["n_races"] == len(part), key
        assert row["n_meetings"] == part["meeting_id"].nunique(), key
        assert row["rmse_model"] == pytest.approx(rmse_m, abs=1e-9), key
        assert row["rmse_baseline"] == pytest.approx(rmse_b, abs=1e-9), key
        assert row["gain"] == pytest.approx(rmse_b - rmse_m, abs=1e-9), key
        assert row["lo"] <= row["gain"] <= row["hi"], key

    assert cells.loc[better, "lo"] > 0.0  # a gain the lower bound supports
    assert cells.loc[worse, "hi"] < 0.0  # a loss
    assert overall["lo"] < overall["hi"]
    # the two-meeting cell's bounds are its whole meetings' (BB and AA), not its races'
    assert cells.loc[exact, "gain"] == pytest.approx(GAIN_AB)
    assert cells.loc[exact, "lo"] == pytest.approx(-3.0, abs=1e-9)
    assert cells.loc[exact, "hi"] == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------------------------
# The store read: races and runners only (D21), never a holdout day
# ---------------------------------------------------------------------------------------------


def write_store(tmp_path, monkeypatch, races: pd.DataFrame, runners: pd.DataFrame) -> None:
    """The two tables as the store writes them, under a fresh data directory with no JRDB table."""
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    jra = data / "processed" / "jra"
    jra.mkdir(parents=True)
    for name, frame in (("races", races), ("runners", runners)):
        spec = schemas.TABLES[name].columns
        schema = pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in spec.items()])
        records = frame.astype(object).where(frame.notna(), None).to_dict("records")
        pq.write_table(pa.Table.from_pylist(records, schema=schema), jra / f"{name}.parquet")
    assert not any("jrdb" in p.name for p in data.rglob("*"))


def test_read_inputs_reads_the_store_races_and_runners_without_holdout_days(tmp_path, monkeypatch):
    dev_days = (date(2026, 11, 21), date(2026, 11, 28))
    held_day = date(2026, 11, 29)
    assert is_holdout_day("japan", held_day)
    assert not any(is_holdout_day("japan", d) for d in dev_days)
    specs = []
    for j, day in enumerate((*dev_days, held_day)):
        for no in (1, 2):
            winner = 90.0 + 10.0 * j + no
            specs.append({"race_no": no, "day": day, "runners": full_field(winner)})
    races, runners = card(specs)
    write_store(tmp_path, monkeypatch, races, runners)

    got_races, got_runners = read_inputs()
    for frame in (got_races, got_runners):
        frame_days = {pd.Timestamp(d).date() for d in frame["race_date"]}
        assert frame_days == set(dev_days)

    table, counts = build_race_table(got_races, got_runners, estimand=ESTIMAND)
    assert counts["n_races"] == 4
    expected = {
        rid(day, "TOK", no): 90.0 + 10.0 * j + no + 0.3
        for j, day in enumerate(dev_days)
        for no in (1, 2)
    }
    got = by_race(table)["top3_time_s"]
    assert set(got.index) == set(expected)
    for race_id, value in expected.items():
        assert got[race_id] == pytest.approx(value, abs=1e-9), race_id


# ---------------------------------------------------------------------------------------------
# The complete-timing sensitivity (#416): the same fold windows, with and without the races
# whose official top 3 lacks a finish time
# ---------------------------------------------------------------------------------------------

TIMING_POISON_S = 8.0  # added to the incomplete races' labels: a top-3 time mismeasured
FIRST_WEEKEND = (FIRST_DAY, FIRST_DAY + timedelta(days=1))
STUDY_SPEC = {
    "bootstrap": {"cluster_unit": "meeting", "sided": "one-sided", "level": LEVEL}
    | {"n_boot": 200, "seed": 305}
}


def timing_table(calendar, *, all_complete=False):
    """(table, incomplete race ids): the calendar with complete_timing. Unless `all_complete`,
    it is False on every race of the first weekend (dropping those races alone would move every
    fold window a week later) and on Tokyo's first race of each day, whose label carries
    +TIMING_POISON_S on development days; the same cell's 4th race is complete and clean."""
    table, truth = calendar
    table = table.copy()
    plant = truth.loc[table["race_id"]]
    tok_first = ((plant["venue"] == "TOK") & (plant["idx"] == 0)).to_numpy()
    first_weekend = table["race_date"].isin(FIRST_WEEKEND).to_numpy()
    incomplete = np.zeros(len(table), dtype=bool)
    if not all_complete:
        incomplete = tok_first | first_weekend
        poisoned = tok_first & ~plant["holdout"].to_numpy()
        table.loc[poisoned, "top3_time_s"] += TIMING_POISON_S
    table["complete_timing"] = ~incomplete
    return table, set(table.loc[incomplete, "race_id"])


def sensitivity(table, *, seed=416):
    return study.timing_sensitivity(
        table,
        spec=SPEC,
        scheme=FOLD_SCHEME,
        decision_min=2,
        n_boot=N_BOOT,
        level=LEVEL,
        seed=seed,
    )


def gain_of(frame: pd.DataFrame, model: str) -> float:
    """RMSE(par only) - RMSE(model) over the frame's races, from first principles."""
    actual = frame["top3_time_s"].to_numpy(dtype=float)
    return _rmse(frame["par_only"].to_numpy(dtype=float) - actual) - _rmse(
        frame[model].to_numpy(dtype=float) - actual
    )


def development(table: pd.DataFrame) -> pd.DataFrame:
    keep = [not is_holdout_day("japan", d) for d in table["race_date"]]
    return table[keep].reset_index(drop=True)


@pytest.fixture(scope="module")
def timing_case(calendar):
    table, incomplete = timing_table(calendar)
    return table, incomplete, sensitivity(table)


def test_timing_sensitivity_existing_arm_is_the_published_fold_loop(timing_case):
    table, _, result = timing_case
    expected = run(table, 2).set_index("race_id")
    got = result["existing"].set_index("race_id")
    assert set(got.index) == set(expected.index)
    got = got.loc[expected.index]
    assert (got["fold"] == expected["fold"]).all()
    for column in ("top3_time_s", "par_only", *PREDICTIONS):
        assert np.allclose(got[column], expected[column], rtol=0, atol=1e-12), column


def test_timing_sensitivity_scores_both_arms_on_the_same_fold_windows(timing_case):
    table, _, result = timing_case
    existing, complete = result["existing"], result["complete"]
    dev = development(table)
    windows = [
        (f.test_start, f.test_end) for f in walk_forward(dev["race_date"], scheme=FOLD_SCHEME)
    ]
    # the fixture bites: folds over the complete races alone would start a week later
    kept = dev[dev["complete_timing"].to_numpy(dtype=bool)].reset_index(drop=True)
    naive = [
        (f.test_start, f.test_end) for f in walk_forward(kept["race_date"], scheme=FOLD_SCHEME)
    ]
    assert naive != windows

    labels = [f["fold"] for f in existing.attrs["folds"]]
    assert len(labels) == len(windows)
    assert [f["fold"] for f in complete.attrs["folds"]] == labels
    assert set(complete["fold"]) <= set(labels)
    both = existing.merge(complete, on="race_id", suffixes=("_existing", "_complete"))
    assert len(both) > 100
    assert (both["fold_existing"] == both["fold_complete"]).all()


def test_timing_sensitivity_drops_incomplete_races_from_the_fit_and_the_scoring(
    calendar, timing_case
):
    table, incomplete, result = timing_case
    _, truth = calendar
    existing, complete = result["existing"], result["complete"]
    # the published arm scores the incomplete races; the complete arm never does
    assert incomplete & set(existing["race_id"])
    assert not incomplete & set(complete["race_id"])
    assert set(complete["race_id"]) == set(existing["race_id"]) - incomplete

    # each fold of the complete arm trains on the complete development races before its window
    dev = development(table)
    folds = walk_forward(dev["race_date"], scheme=FOLD_SCHEME)
    is_complete = dev["complete_timing"].to_numpy(dtype=bool)
    assert [f["n_train"] for f in existing.attrs["folds"]] == [len(f.train_idx) for f in folds]
    assert [f["n_train"] for f in complete.attrs["folds"]] == [
        int(is_complete[f.train_idx].sum()) for f in folds
    ]

    # the poisoned labels pull the published arm's par of Tokyo's first cell, and never the
    # complete arm's: its pars stay at the planted ones
    rows = complete.set_index("race_id")
    assert (rows["par_only"] - truth.loc[rows.index, "par"]).abs().max() < 0.4
    rows = existing.set_index("race_id")
    plant = truth.loc[rows.index]
    sibling = ((plant["venue"] == "TOK") & (plant["idx"] == 3)).to_numpy()
    assert sibling.sum() >= 8
    assert ((rows.loc[sibling, "par_only"] - plant.loc[sibling, "par"]) > 2.0).all()


def test_timing_sensitivity_reports_each_arm_gain_and_the_paired_change(timing_case):
    _, _, result = timing_case
    existing = result["existing"].set_index("race_id")
    complete = result["complete"].set_index("race_id")
    common = existing.index.intersection(complete.index)
    assert set(result["gains"]) == set(PREDICTIONS)
    for model in PREDICTIONS:
        gains = result["gains"][model]
        for arm, frame in (("existing", existing), ("complete", complete)):
            diff = gains[arm]
            assert diff.n_races == len(frame), (model, arm)
            assert diff.n_clusters == frame["meeting_id"].nunique(), (model, arm)
            assert diff.gain == pytest.approx(gain_of(frame, model), abs=1e-9), (model, arm)
            assert diff.lo <= diff.gain + 1e-12 and diff.gain <= diff.hi + 1e-12, (model, arm)
        paired = gains["paired"]
        before = gain_of(existing.loc[common], model)
        after = gain_of(complete.loc[common], model)
        assert paired["n_races"] == len(common), model
        assert paired["n_meetings"] == existing.loc[common, "meeting_id"].nunique(), model
        assert paired["gain_existing"] == pytest.approx(before, abs=1e-9), model
        assert paired["gain_complete"] == pytest.approx(after, abs=1e-9), model
        assert paired["delta"] == pytest.approx(after - before, abs=1e-9), model
        assert paired["lo"] <= paired["delta"] + 1e-12, model
        assert paired["delta"] <= paired["hi"] + 1e-12, model
    # the dropped races change the fits, so the change on the same races is not zero
    assert abs(result["gains"]["variant_on"]["paired"]["delta"]) > 1e-3
    # variant off is the par in both arms (no injected prior)
    for arm in ("existing", "complete"):
        assert result["gains"]["variant_off"][arm].gain == pytest.approx(0.0, abs=1e-9), arm


def test_timing_sensitivity_is_exactly_zero_when_every_race_is_complete(calendar):
    table, incomplete = timing_table(calendar, all_complete=True)
    assert not incomplete
    result = sensitivity(table)
    existing = result["existing"].set_index("race_id")
    complete = result["complete"].set_index("race_id")
    assert set(complete.index) == set(existing.index)
    complete = complete.loc[existing.index]
    assert (complete["fold"] == existing["fold"]).all()
    for column in ("top3_time_s", "par_only", *PREDICTIONS):
        assert np.allclose(complete[column], existing[column], rtol=0, atol=1e-12), column
    for model in PREDICTIONS:
        gains = result["gains"][model]
        assert gains["complete"].gain == pytest.approx(gains["existing"].gain, abs=1e-12)
        assert gains["complete"].n_races == gains["existing"].n_races == len(existing)
        paired = gains["paired"]
        assert paired["n_races"] == len(existing), model
        for key in ("delta", "lo", "hi"):
            assert paired[key] == pytest.approx(0.0, abs=1e-12), (model, key)


def test_timing_sensitivity_requires_the_complete_timing_flag(calendar):
    table, _ = calendar
    assert "complete_timing" not in table.columns
    with pytest.raises(ValueError):
        sensitivity(table)


# compute_study and render_report: the sensitivity is a registry record beside each variant


def metrics_of(variant) -> dict:
    return {(m.fold, m.metric): m for m in variant.metrics}


@pytest.fixture(scope="module")
def study_run(timing_case):
    table, _, _ = timing_case
    return study.compute_study(table, STUDY_SPEC, FOLD_SCHEME, 2)


def test_compute_study_records_a_complete_timing_variant_beside_each_existing_one(
    timing_case, study_run
):
    _, _, result = timing_case  # the normal error model's arms (SPEC: t_df None)
    variants, _, _ = study_run
    by_name = {v.name: v for v in variants}
    assert len(by_name) == len(variants)
    existing = [v for v in variants if not v.name.endswith(":complete_timing")]
    assert len(existing) == 4 and all(v.error is None for v in existing)
    assert set(by_name) == {v.name for v in existing} | {
        f"{v.name}:complete_timing" for v in existing
    }
    for v in variants:
        assert v.error is None and ("all", "rmse_gain_s") in metrics_of(v), v.name

    for model in PREDICTIONS:
        name = f"normal:{model}_vs_par_only"
        gains = result["gains"][model]
        old = metrics_of(by_name[name])[("all", "rmse_gain_s")]
        assert old.value == pytest.approx(gains["existing"].gain, abs=1e-9), model
        assert old.n == gains["existing"].n_races, model
        new = metrics_of(by_name[f"{name}:complete_timing"])
        gain = new[("all", "rmse_gain_s")]
        assert gain.value == pytest.approx(gains["complete"].gain, abs=1e-9), model
        assert gain.n == gains["complete"].n_races, model
        change = new[("all", "rmse_gain_change_s")]
        assert change.value == pytest.approx(gains["paired"]["delta"], abs=1e-9), model
        assert change.n == gains["paired"]["n_races"], model
        assert change.lo is not None and change.hi is not None, model
        assert change.lo <= change.value + 1e-12 and change.value <= change.hi + 1e-12, model


def test_compute_study_records_a_failed_complete_timing_arm_and_keeps_the_existing_one(calendar):
    table, _ = calendar
    # one venue, no race with a complete timing: the complete arm has nothing to score
    table = table[table["track"] == "KYO"].assign(complete_timing=False)
    variants, _, _ = study.compute_study(table, STUDY_SPEC, FOLD_SCHEME, 2)
    failed = [v for v in variants if v.name.endswith(":complete_timing")]
    scored = [v for v in variants if not v.name.endswith(":complete_timing")]
    assert len(failed) == len(scored) == 4
    assert {f"{v.name}:complete_timing" for v in scored} == {v.name for v in failed}
    for v in scored:
        assert v.error is None and ("all", "rmse_gain_s") in metrics_of(v), v.name
    for v in failed:
        assert isinstance(v.error, str) and v.error.strip(), v.name


def test_render_report_shows_the_coverage_and_the_sensitivity_side_by_side(study_run):
    variants, tables, counts = study_run
    races, runners = card(
        [{"race_no": k + 1, "runners": case[0]} for k, case in enumerate(COVERAGE_CASES.values())]
    )
    _, builder = build_race_table(races, runners, estimand=ESTIMAND)
    report = study.render_report(variants, tables, {"builder": builder} | counts, {"family": "x"})
    for key in ("n_missing_timing", "n_short_order", "n_tie"):
        assert key in report, key

    by_name = {v.name: v for v in variants}
    lines = report.splitlines()
    for error_name in ("t5", "normal"):
        name = f"{error_name}:variant_on_vs_par_only"
        old = metrics_of(by_name[name])[("all", "rmse_gain_s")].value
        new = metrics_of(by_name[f"{name}:complete_timing"])
        cells = [
            f"{x:.4f}"
            for x in (
                old,
                new[("all", "rmse_gain_s")].value,
                new[("all", "rmse_gain_change_s")].value,
            )
        ]
        assert len(set(cells)) == 3, cells
        assert any(all(c in line for c in cells) for line in lines), (error_name, cells)
