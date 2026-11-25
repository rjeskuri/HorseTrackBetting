"""Handicap factors per runner as of decision time (ticket #129).

Contract pinned here (`horsetrack.countries.japan.handicap.factors`):

- `handicap_factors(race_id, *, as_of, train_end=None) -> polars.DataFrame`: one row per declared
  (not scratched) runner of the race, keys `race_id`, `horse_no`, `horse_id`. History comes from
  #128's `pp_lines(race_id, as_of=as_of)` (default n = 10 lines). The in-fold pieces (the class par
  of the speed figure and the draw bias) are fitted on the non-holdout race days on or before
  `train_end`, which defaults to the as-of cut: the last race day before the race whose result is
  published before `as_of` (#128's ruling: at the end of its local race day). A `train_end` after
  the cut is a ValueError, as is a naive `as_of`. Every read goes through #114's scoped reader.
  Today's card (the race's venue, surface, distance, course, going, class; each runner's draw,
  weight carried, body-weight change, jockey and trainer) is read from the store's row of the race;
  its result never is.
- Missing values: every output column other than the keys (and `race_date`, if present) has a
  Boolean companion `<column>_missing`, True exactly when the value is null. Nothing is imputed.
  Every call returns the same schema, whatever the race.

Rulings (reversible choices this file pins; the per-test comments compute each value by hand):

- Ruling: speed figure of a past start = class par - the runner's finish_time_s (seconds faster
  than par; higher is better). Class par = median (mean of the middle two for an even count) of
  races.finish_time_s over the races with the same (venue, surface, distance_m, race_class) on the
  in-fold days; no par or no time -> no figure. No day variant until #96.
- Ruling: `speed_best` = max figure, `speed_last` = the newest start that has a figure,
  `speed_mean3` = mean of the newest 3 figures, `speed_n` = figures among the PP lines (0 for a
  debutant, never null).
- Ruling: the race's last 600 m = median last_3f_s of that race's declared runners that have one;
  late figure = race last 600 m - own last_3f_s. Early position = (first corner position - 1) /
  (field_size - 1); late gain = final-corner position - finish_pos. `late_mean3`,
  `early_pos_mean3`, `late_gain_mean3` = means over the newest 3 starts that have the value.
- Ruling: style of a start from its first corner position p in a field of n: front if p == 1,
  stalker if 3(p-1) <= n-1, midpack if 3(p-1) <= 2(n-1), else closer. A horse's `style` is the most
  frequent over its newest 5 starts with corner positions, ties to the more forward style; values
  "front", "stalker", "midpack", "closer" (逃げ/先行/差し/追込). `field_front_n` = front-styled
  runners in today's field, `field_front_share` = that / declared runners, `front_rivals` = the
  other front-styled runners.
- Ruling: class level DEBUT 0, MAIDEN 0, C1 1, C2 2, C3 3, OP 4, L 5, G3 6, G2 7, G1 8, anything
  else null (prize is not in the store). `class_change` = today's level - mean level of the newest
  3 starts with a level; `speed_best_at_class` = best figure at a level >= today's.
- Ruling: `speed_slope3` = least-squares slope of the newest 3 figures (at least 2) against start
  order, oldest to newest (positive = improving). A layoff is a gap of 60 days or more; a debut also
  starts a run. `runs_since_layoff` = starts since the latest layoff or debut, that start counted;
  0 when today's race comes 60+ days after the last start. `starts_90d` counts starts at most 90
  days before the race day.
- Ruling: fit dimensions: distance band (<=1400, 1401-1800, 1801-2200, >=2201 m), surface, going
  band (GOOD and GOOD TO YIELDING / SOFT and HEAVY), venue, course_config. For each `<d>` in dist,
  surface, going, venue, course: `fit_<d>_n` starts matching today, `fit_<d>_top3` of them in the
  first three, `fit_<d>` = m / (m + 2) * (mean figure at matching starts - mean figure overall),
  m the matching starts with a figure (0.0 with no match; null with no figure at all).
- Ruling: `body_wt_change` = today's horse_wt_change_kg; `wt_carried_change` = today's
  actual_wt_kg - the newest start's; `jockey_change` = today's jockey_key != the newest start's.
  `jockey_rate`, `trainer_rate` = (wins + 20 p0) / (rides + 20) over the declared runners on the
  race days within 365 days ending at the cut (cut - 364 .. cut), p0 = wins / runners over all of
  them.
- Ruling: draw bias at today's (venue, surface, distance_m): a runner's draw group is inner if
  3(draw-1) <= n-1, middle if 3(draw-1) <= 2(n-1), else outer (n = declared runners);
  `draw_bias` = (wins + 2) / (sum of 1/n + 2) over the in-fold runners of that group there (an
  impact value shrunk to 1); null with no in-fold runner at that course and distance.

Data: synthetic only (tests/countries/japan/handicap/handicap_store.py), written under tmp_path.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

from datetime import date, datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from handicap_store import R, Store, T, at_jst, patch_holdout, perturb, random_store, rid
from handicap_store import weekend_days
from horsetrack.countries.japan.handicap import factors
from horsetrack.countries.japan.handicap.factors import handicap_factors

TARGET = rid(T, "TOK", 11)
AS_OF = at_jst(T, 15)  # race day, mid-afternoon; the cut is T - 1

KEYS = {"race_id", "horse_no", "horse_id", "race_date"}
SPEED = ["speed_best", "speed_last", "speed_mean3", "speed_n"]
PACE = ["late_mean3", "early_pos_mean3", "late_gain_mean3"]
STYLE = ["style", "field_front_n", "field_front_share", "front_rivals"]
CLASS = ["class_today", "class_change", "speed_best_at_class"]
FORM = ["speed_slope3", "days_since_last", "runs_since_layoff", "starts_90d"]
FIT_DIMS = ["dist", "surface", "going", "venue", "course"]
FIT = [f"fit_{d}{s}" for d in FIT_DIMS for s in ("_n", "_top3", "")]
CONDITION = ["body_wt_change", "wt_carried_change", "jockey_change", "jockey_rate", "trainer_rate"]
DRAW = ["draw_bias"]
FACTOR_COLUMNS = SPEED + PACE + STYLE + CLASS + FORM + FIT + CONDITION + DRAW


def row(frame: pl.DataFrame, horse_id: str) -> dict:
    rows = frame.filter(pl.col("horse_id") == horse_id).to_dicts()
    assert len(rows) == 1, f"{horse_id}: {len(rows)} rows"
    return rows[0]


def assert_values(frame: pl.DataFrame, horse_id: str, want: dict) -> None:
    got = row(frame, horse_id)
    for col, value in want.items():
        assert col in got, f"no column {col}"
        if value is None:
            assert got[col] is None, (horse_id, col, got[col])
        elif isinstance(value, float):
            assert got[col] == pytest.approx(value, abs=1e-6), (horse_id, col, got[col], value)
        else:
            assert got[col] == value, (horse_id, col, got[col], value)


def write(store: Store, tmp_path, monkeypatch, sub="data"):
    return store.write(tmp_path, monkeypatch, sub)


# ---------------------------------------------------------------------------------------------
# Broad: one row per declared runner, every factor column
# ---------------------------------------------------------------------------------------------

LEAK_DAYS = weekend_days(date(2025, 3, 1), date(2025, 6, 22))


def leak_store() -> Store:
    return random_store(LEAK_DAYS, target=TARGET)


def test_one_row_per_declared_runner_with_every_factor_column(tmp_path, monkeypatch):
    s = leak_store()
    write(s, tmp_path, monkeypatch)
    races = [TARGET, rid(date(2025, 3, 1), "TOK", 1), rid(date(2025, 4, 19), "NAK", 2),
             rid(date(2025, 6, 14), "TOK", 1)]  # fmt: skip
    for race_id in races:
        day = date.fromisoformat(race_id.split(":")[1])
        frame = handicap_factors(race_id, as_of=at_jst(day, 9))
        assert isinstance(frame, pl.DataFrame)
        missing = [c for c in FACTOR_COLUMNS if c not in frame.columns]
        assert not missing, f"handicap_factors lacks {missing}"
        declared = {
            (r["horse_no"], r["horse_id"])
            for r in s.runners
            if r["race_id"] == race_id and not r["is_scratched"]
        }
        got = set(frame.select("horse_no", "horse_id").iter_rows())
        assert got == declared, race_id
        assert frame.height == len(declared)
        assert set(frame["race_id"].to_list()) == {race_id}
    target = handicap_factors(TARGET, as_of=AS_OF)
    assert "h:out" not in target["horse_id"].to_list()  # scratched from the race
    assert "h:new" in target["horse_id"].to_list()


# ---------------------------------------------------------------------------------------------
# 1. Speed: figures against the class par from the in-fold days
# ---------------------------------------------------------------------------------------------


def speed_store() -> Store:
    s = Store()
    tok = {"venue": "TOK", "surface": "turf", "distance": 1600, "cls": "C2"}  # K1
    nak = {"venue": "NAK", "surface": "turf", "distance": 2000, "cls": "C1"}  # K2
    dirt = {"venue": "TOK", "surface": "dirt", "distance": 1400, "cls": "C1"}  # K3
    # par-only races (no tracked horse)
    s.race(date(2025, 2, 1), [], winner_time=94.6, fill_to=4, **tok)
    s.race(date(2025, 2, 8), [], winner_time=93.0, fill_to=4, **tok)
    s.race(date(2025, 1, 12), [], winner_time=122.0, fill_to=4, **nak)
    s.race(date(2025, 1, 19), [], winner_time=121.6, fill_to=4, **nak)
    s.race(date(2025, 3, 1), [], winner_time=86.0, fill_to=4, **dirt)
    s.race(date(2025, 3, 8), [], winner_time=85.4, fill_to=4, **dirt)
    # A: four starts, newest first S1 .. S4
    s.race(date(2025, 3, 30), [R("h:a", pos=4, time=95.0)], winner_time=94.0, fill_to=6, **tok)
    s.race(date(2025, 4, 20), [R("h:a", pos=3, time=85.6)], winner_time=85.0, fill_to=6, **dirt)
    s.race(date(2025, 5, 11), [R("h:a", pos=1, time=121.0)], fill_to=6, **nak)
    s.race(date(2025, 6, 1), [R("h:a", pos=2, time=94.4)], winner_time=94.2, fill_to=6, **tok)
    # C: an older start with a figure, then a newer one it did not finish
    s.race(date(2025, 5, 25), [R("h:c", pos=3, time=69.5)], venue="HSN", distance=1200,
           winner_time=69.0, fill_to=6)  # fmt: skip
    s.race(date(2025, 6, 8), [R("h:c")], venue="KYO", distance=1800, winner_time=108.0,
           fill_to=6)  # fmt: skip
    # bait: a K1 race earlier on T, one the day after (with A in it), and the target's own result
    s.race(T, [], race_no=2, winner_time=99.9, fill_to=4, **tok)
    s.race(T + timedelta(days=1), [R("h:a", pos=1, time=70.0)], fill_to=4, **tok)
    s.race(T, [R("h:a", pos=1, time=80.0), R("h:b", pos=2), R("h:c", pos=3)], race_no=11, **tok)
    return s


# Default (cut 06-14). Par sets: K1 {94.6, 93.0, 94.0, 94.2} -> 94.1; K2 {122.0, 121.6, 121.0}
# -> 121.6; K3 {86.0, 85.4, 85.0} -> 85.4; HSN 1200 {69.0} -> 69.0.
#   A: S1 94.1-94.4 = -0.3, S2 121.6-121.0 = 0.6, S3 85.4-85.6 = -0.2, S4 94.1-95.0 = -0.9
#      best 0.6, last -0.3, mean3 (-0.3+0.6-0.2)/3, n 4
#   C: newest start unfinished (no figure); 69.0-69.5 = -0.5 -> best = last = mean3 = -0.5, n 1
# train_end 2025-03-31: K1 {94.6, 93.0, 94.0} -> 94.0; K2 {122.0, 121.6} -> 121.8;
# K3 {86.0, 85.4} -> 85.7; HSN 1200 none.
#   A: S1 94.0-94.4 = -0.4, S2 121.8-121.0 = 0.8, S3 85.7-85.6 = 0.1, S4 94.0-95.0 = -1.0
#      best 0.8, last -0.4, mean3 (-0.4+0.8+0.1)/3, n 4; C: no figure
SPEED_CASES = {
    None: {
        "h:a": {"speed_best": 0.6, "speed_last": -0.3, "speed_mean3": 0.1 / 3, "speed_n": 4},
        "h:c": {"speed_best": -0.5, "speed_last": -0.5, "speed_mean3": -0.5, "speed_n": 1},
        "h:b": {"speed_best": None, "speed_last": None, "speed_mean3": None, "speed_n": 0},
    },
    date(2025, 3, 31): {
        "h:a": {"speed_best": 0.8, "speed_last": -0.4, "speed_mean3": 0.5 / 3, "speed_n": 4},
        "h:c": {"speed_best": None, "speed_last": None, "speed_mean3": None, "speed_n": 0},
        "h:b": {"speed_best": None, "speed_last": None, "speed_mean3": None, "speed_n": 0},
    },
}


@pytest.mark.parametrize("train_end", list(SPEED_CASES), ids=["as-of-cut", "train-end-0331"])
def test_speed_figures_against_the_in_fold_class_par(tmp_path, monkeypatch, train_end):
    write(speed_store(), tmp_path, monkeypatch)
    kwargs = {} if train_end is None else {"train_end": train_end}
    frame = handicap_factors(TARGET, as_of=AS_OF, **kwargs)
    for horse_id, want in SPEED_CASES[train_end].items():
        assert_values(frame, horse_id, want)
        assert_values(frame, horse_id, {f"{c}_missing": want[c] is None for c in SPEED})


def test_train_end_defaults_to_the_cut_and_may_not_pass_it(tmp_path, monkeypatch):
    write(speed_store(), tmp_path, monkeypatch)
    default = handicap_factors(TARGET, as_of=AS_OF)
    assert_frame_equal(default, handicap_factors(TARGET, as_of=AS_OF, train_end=T - timedelta(1)))
    with pytest.raises(ValueError):
        handicap_factors(TARGET, as_of=AS_OF, train_end=T)
    with pytest.raises(ValueError):  # 06-14 is not yet published at 23:00 on 06-14
        handicap_factors(TARGET, as_of=at_jst(T - timedelta(1), 23), train_end=T - timedelta(1))
    with pytest.raises(ValueError):
        handicap_factors(TARGET, as_of=datetime(2025, 6, 15, 15, 0))


# ---------------------------------------------------------------------------------------------
# 2. Pace and late speed: last 600 m against the race's own, corner profile
# ---------------------------------------------------------------------------------------------


def others(tag: str, places: list[int], l3fs: list[float]) -> list[R]:
    return [R(f"o:{tag}:{p}", pos=p, l3f=v) for p, v in zip(places, l3fs, strict=True)]


def pace_store() -> Store:
    s = Store()
    # E4 (oldest): field 5, median {33.0, 36.0, 36.0, 36.5, 35.5} = 36.0 -> late 3.0;
    #   early (5-1)/4 = 1.0; gain 5-1 = 4. Beyond the newest three with values: not averaged.
    s.race(date(2025, 4, 12), [R("h:e", pos=1, l3f=33.0, corners=[5, 5]),
           *others("e4", [2, 3, 4, 5], [36.0, 36.0, 36.5, 35.5])], distance=2000)  # fmt: skip
    # E3: field 6 (one did not finish, no last 600 m); median {36.0, 35.0, 35.4, 35.8, 34.8}
    #   = 35.4 -> late -0.6; early (1-1)/5 = 0.0; gain 1-3 = -2
    s.race(date(2025, 4, 26), [R("h:e", pos=3, l3f=36.0, corners=[1, 1, 1]),
           *others("e3", [1, 2, 4, 5], [35.0, 35.4, 35.8, 34.8]),
           R("o:e3:dnf", corners=[6])], distance=2000)  # fmt: skip
    # E2: field 5 (a scratched runner is not in it); median {35.0, 34.0, 36.0, 34.6, 35.4}
    #   = 35.0 -> late 0.0; early (5-1)/4 = 1.0; gain 4-2 = 2
    s.race(date(2025, 5, 10), [R("h:e", pos=2, l3f=35.0, corners=[5, 4]),
           *others("e2", [1, 3, 4, 5], [34.0, 36.0, 34.6, 35.4]),
           R("o:e2:scr", scratched=True)], distance=2000)  # fmt: skip
    # E1: field 5; median {34.0, 34.5, 35.0, 33.8, 35.2} = 34.5 -> late 0.5;
    #   early (2-1)/4 = 0.25; gain 3-1 = 2
    s.race(date(2025, 5, 24), [R("h:e", pos=1, l3f=34.0, corners=[2, 2, 3]),
           *others("e1", [2, 3, 4, 5], [34.5, 35.0, 33.8, 35.2])], distance=2000)  # fmt: skip
    # E0 (newest): did not finish, no corners: no value for any of the three
    s.race(date(2025, 6, 7), [R("h:e", corners=[])], fill_to=5, distance=2000)
    # a horse whose only start has no value at all
    s.race(date(2025, 5, 31), [R("h:dnf", corners=[])], fill_to=5)
    s.race(T, [R("h:e", pos=1, l3f=30.0, corners=[1, 1]), R("h:dnf", pos=2), R("h:b", pos=3)],
           race_no=11)  # fmt: skip
    return s


def test_pace_and_late_speed_relative_to_the_race(tmp_path, monkeypatch):
    write(pace_store(), tmp_path, monkeypatch)
    frame = handicap_factors(TARGET, as_of=AS_OF)
    # newest three starts with values: E1, E2, E3
    assert_values(frame, "h:e", {
        "late_mean3": (0.5 + 0.0 - 0.6) / 3,
        "early_pos_mean3": (0.25 + 1.0 + 0.0) / 3,
        "late_gain_mean3": (2 + 2 - 2) / 3,
    })  # fmt: skip
    for horse_id in ("h:dnf", "h:b"):
        assert_values(frame, horse_id, dict.fromkeys(PACE))
        assert_values(frame, horse_id, {f"{c}_missing": True for c in PACE})
    assert_values(frame, "h:e", {f"{c}_missing": False for c in PACE})


# ---------------------------------------------------------------------------------------------
# 3. Running style and today's pace scenario
# ---------------------------------------------------------------------------------------------

# (first corner position, field size) per start, newest first; the start's style by the ruling
STYLE_HISTORY = {
    "h:f1": [(1, 10), (1, 12), (2, 10)],  # front, front, stalker (3*1 <= 9) -> front
    "h:f2": [(1, 8), (4, 10), (4, 10)],  # front, stalker (3*3 <= 9, boundary) x2 -> stalker
    "h:f3": [(5, 10), (7, 10), (10, 10)],  # midpack (12 <= 18), midpack (18 <= 18), closer
    "h:f4": [(8, 10), (12, 16), (1, 16)],  # closer (21 > 18), closer (33 > 30), front
    "h:f5": [(1, 10), (10, 10)],  # front, closer: a tie -> the more forward, front
    # newest 5: closer, front, closer, front, closer -> closer (all 8 would make it front)
    "h:f6": [(9, 10), (1, 10), (10, 10), (1, 10), (16, 16), (1, 12), (1, 12), (1, 12)],
    "h:f7": [],  # debutant
    "h:f8": None,  # one start without corner positions
}
STYLE_WANT = {"h:f1": "front", "h:f2": "stalker", "h:f3": "midpack", "h:f4": "closer",
              "h:f5": "front", "h:f6": "closer", "h:f7": None, "h:f8": None}  # fmt: skip


def style_store() -> Store:
    s = Store()
    for h, (horse, starts) in enumerate(STYLE_HISTORY.items()):
        if starts is None:
            s.race(T - timedelta(days=1 + h), [R(horse, pos=3, corners=[])], fill_to=10)
            continue
        for j, (p, n) in enumerate(starts):
            # later corners mirror the first, so only the first corner gives the ruling's style
            s.race(T - timedelta(days=1 + 10 * j + h),
                   [R(horse, pos=n - p + 1, corners=[p, n - p + 1])], fill_to=n)  # fmt: skip
    s.race(T, [R(horse, pos=k + 1) for k, horse in enumerate(STYLE_HISTORY)], race_no=11)
    return s


def test_running_style_and_todays_pace_scenario(tmp_path, monkeypatch):
    write(style_store(), tmp_path, monkeypatch)
    frame = handicap_factors(TARGET, as_of=AS_OF)
    fronts = [h for h, st in STYLE_WANT.items() if st == "front"]
    assert fronts == ["h:f1", "h:f5"]
    for horse_id, style in STYLE_WANT.items():
        assert_values(frame, horse_id, {
            "style": style, "style_missing": style is None,
            "field_front_n": 2, "field_front_share": 2 / 8,
            "front_rivals": 1 if horse_id in fronts else 2,
        })  # fmt: skip


# ---------------------------------------------------------------------------------------------
# 4. Class: today's level, the move up or down, best figure at today's class or higher
# ---------------------------------------------------------------------------------------------


# Each start in its own (venue, surface, distance, class): par = its winner's time, so a figure
# is -0.2 s per place behind the winner. (day, class, place, distance), newest first.
CLASS_STARTS = {
    # G: C3 2nd (-0.2), C1 1st (0.0), OP 3rd (-0.4), MAIDEN 1st (0.0)
    "h:g": ("TOK", [(date(2025, 6, 1), "C3", 2, 1200), (date(2025, 5, 18), "C1", 1, 1400),
                    (date(2025, 5, 4), "OP", 3, 1800), (date(2025, 4, 20), "MAIDEN", 1, 2000)]),
    # H: C1 2nd (-0.2), MAIDEN 1st (0.0)
    "h:h": ("NAK", [(date(2025, 6, 7), "C1", 2, 1200), (date(2025, 5, 24), "MAIDEN", 1, 1400)]),
    # I: G1 5th (-0.8), an unclassed start it did not finish, DEBUT 1st (0.0), C1 1st (0.0)
    "h:i": ("KYO", [(date(2025, 6, 8), "G1", 5, 1200), (date(2025, 5, 25), None, None, 1400),
                    (date(2025, 5, 11), "DEBUT", 1, 1800), (date(2025, 4, 27), "C1", 1, 2000)]),
}  # fmt: skip


def class_store() -> Store:
    s = Store()
    for horse, (venue, starts) in CLASS_STARTS.items():
        for day, cls, pos, dist in starts:
            s.race(day, [R(horse, pos=pos)], venue=venue, cls=cls, distance=dist, fill_to=6)
    s.race(T, [R("h:g", pos=1), R("h:h", pos=2), R("h:i", pos=3), R("h:b", pos=4)], race_no=11)
    return s


def test_class_move_and_best_figure_at_todays_class(tmp_path, monkeypatch):
    write(class_store(), tmp_path, monkeypatch)
    frame = handicap_factors(TARGET, as_of=AS_OF)  # today: C2, level 2
    # G: 2 - (3+1+4)/3; best at level >= 2 among C3 (-0.2) and OP (-0.4)
    assert_values(frame, "h:g", {"class_today": 2, "class_change": 2 - 8 / 3,
                                 "speed_best_at_class": -0.2})  # fmt: skip
    # H: 2 - (1+0)/2; no start at level >= 2
    assert_values(frame, "h:h", {"class_change": 1.5, "speed_best_at_class": None,
                                 "speed_best_at_class_missing": True})  # fmt: skip
    # I: newest three with a level: G1 8, DEBUT 0, C1 1 -> 2 - 3; best at >= 2: G1's -0.8
    assert_values(frame, "h:i", {"class_change": -1.0, "speed_best_at_class": -0.8})
    assert_values(frame, "h:b", {"class_today": 2, "class_change": None,
                                 "class_change_missing": True,
                                 "speed_best_at_class": None})  # fmt: skip


CLASS_LADDER = {"DEBUT": 0, "MAIDEN": 0, "C1": 1, "C2": 2, "C3": 3, "OP": 4, "L": 5, "G3": 6,
                "G2": 7, "G1": 8, "XYZ": None, None: None}  # fmt: skip


def test_class_level_of_todays_race(tmp_path, monkeypatch):
    s = Store()
    race_ids = {}
    for k, cls in enumerate(CLASS_LADDER):
        race_ids[cls] = s.race(T, [R(f"h:lad{k}", pos=1)], venue="SAP", race_no=k + 1, cls=cls)
    write(s, tmp_path, monkeypatch)
    for cls, level in CLASS_LADDER.items():
        frame = handicap_factors(race_ids[cls], as_of=AS_OF)
        got = frame.select("class_today", "class_today_missing").row(0)
        assert got == (level, level is None), cls


# ---------------------------------------------------------------------------------------------
# 5. Form cycle: trend, days since the last start, runs since a layoff, starts in 90 days
# ---------------------------------------------------------------------------------------------


def form_store() -> Store:
    s = Store()
    starts = {  # newest first: (day, place); each in its own par key, figure -0.2 per place
        "h:k": ("TOK", [(date(2025, 6, 1), 2), (date(2025, 5, 18), 4), (date(2025, 3, 2), 1),
                        (date(2025, 2, 10), 1)]),
        "h:l": ("NAK", [(date(2025, 3, 17), 1)]),
        "h:m": ("KYO", [(date(2025, 5, 25), 3), (date(2025, 5, 11), 1)]),
        "h:n": ("HSN", [(date(2025, 6, 1), 1), (date(2025, 4, 2), 2)]),
    }  # fmt: skip
    for horse, (venue, lines) in starts.items():
        for (day, pos), dist in zip(lines, [1200, 1400, 1800, 2000], strict=False):
            s.race(day, [R(horse, pos=pos)], venue=venue, distance=dist, cls="C1", fill_to=6)
    s.race(T, [R(h, pos=k + 1) for k, h in enumerate(["h:k", "h:l", "h:m", "h:n", "h:b"])],
           race_no=11)  # fmt: skip
    return s


# K: figures newest first -0.2, -0.6, 0.0 (and 0.0): oldest->newest [0.0, -0.6, -0.2], slope
#    (-0.2 - 0.0)/2 = -0.1; last start 06-01: 14 days; 05-18 came 77 days after 03-02 (a layoff):
#    runs 2; within 90 days (from 03-17): 06-01, 05-18 -> 2
# L: one start exactly 90 days before (counts); today comes 90 days after it: runs 0; no slope
# M: [0.0, -0.4] -> slope -0.4; 21 days; no gap of 60+, the debut starts the run: runs 2
# N: [-0.2, 0.0] -> slope 0.2; 06-01 came exactly 60 days after 04-02 (a layoff): runs 1; 2 starts
FORM_WANT = {
    "h:k": {"speed_slope3": -0.1, "days_since_last": 14, "runs_since_layoff": 2, "starts_90d": 2},
    "h:l": {"speed_slope3": None, "days_since_last": 90, "runs_since_layoff": 0, "starts_90d": 1},
    "h:m": {"speed_slope3": -0.4, "days_since_last": 21, "runs_since_layoff": 2, "starts_90d": 2},
    "h:n": {"speed_slope3": 0.2, "days_since_last": 14, "runs_since_layoff": 1, "starts_90d": 2},
    "h:b": {"speed_slope3": None, "days_since_last": None, "runs_since_layoff": None,
            "starts_90d": 0},
}  # fmt: skip


def test_form_cycle(tmp_path, monkeypatch):
    write(form_store(), tmp_path, monkeypatch)
    frame = handicap_factors(TARGET, as_of=AS_OF)
    for horse_id, want in FORM_WANT.items():
        assert_values(frame, horse_id, want)
        assert_values(frame, horse_id, {f"{c}_missing": want[c] is None for c in FORM})


# ---------------------------------------------------------------------------------------------
# 6. Fit: record and shrunk figure at today's distance band, surface, going, venue, course
# ---------------------------------------------------------------------------------------------


def fit_store() -> Store:
    s = Store()
    # P, newest first (figure -0.2 per place behind; each start in its own par key):
    #   P1 TOK turf 1800 outer GOOD TO YIELDING 3rd -0.4
    #   P2 NAK dirt 1600 outer SOFT             1st  0.0
    #   P3 TOK turf 2000 inner HEAVY            6th -1.0
    #   P4 KYO turf 1400 outer GOOD             5th -0.8
    for day, venue, surface, dist, course, going, pos in [
        (date(2025, 6, 1), "TOK", "turf", 1800, "outer", "GOOD TO YIELDING", 3),
        (date(2025, 5, 18), "NAK", "dirt", 1600, "outer", "SOFT", 1),
        (date(2025, 5, 4), "TOK", "turf", 2000, "inner", "HEAVY", 6),
        (date(2025, 4, 20), "KYO", "turf", 1400, "outer", "GOOD", 5),
    ]:
        s.race(day, [R("h:p", pos=pos)], venue=venue, surface=surface, distance=dist,
               course=course, going=going, fill_to=6)  # fmt: skip
    # Q: one start matching today on every dimension, not finished (no figure)
    s.race(date(2025, 5, 25), [R("h:q")], fill_to=6)
    # R: one win matching today on no dimension
    s.race(date(2025, 5, 11), [R("h:r", pos=1)], venue="KYO", surface="dirt", distance=2400,
           course="inner", going="HEAVY", fill_to=6)  # fmt: skip
    # today: TOK turf 1600 outer GOOD
    s.race(T, [R("h:p", pos=1), R("h:q", pos=2), R("h:r", pos=3), R("h:b", pos=4)], race_no=11)
    return s


# P overall mean figure (-0.4 + 0.0 - 1.0 - 0.8)/4 = -0.55
#   dist 1401-1800: P1, P2 -> n 2, top3 2, mean -0.2:  2/4 * (-0.2 + 0.55) = 0.175
#   surface turf: P1, P3, P4 -> n 3, top3 1, mean -2.2/3: 3/5 * (-2.2/3 + 0.55) = -0.11
#   going GOOD/GTY: P1, P4 -> n 2, top3 1, mean -0.6: 2/4 * (-0.6 + 0.55) = -0.025
#   venue TOK: P1, P3 -> n 2, top3 1, mean -0.7: 2/4 * (-0.7 + 0.55) = -0.075
#   course outer: P1, P2, P4 -> n 3, top3 2, mean -0.4: 3/5 * (-0.4 + 0.55) = 0.09
FIT_P = {"dist": (2, 2, 0.175), "surface": (3, 1, -0.11), "going": (2, 1, -0.025),
         "venue": (2, 1, -0.075), "course": (3, 2, 0.09)}  # fmt: skip


def fit_want(per_dim: dict) -> dict:
    want = {}
    for d, (n, top3, fit) in per_dim.items():
        want.update({f"fit_{d}_n": n, f"fit_{d}_top3": top3, f"fit_{d}": fit})
    return want


def test_fit_record_and_shrunk_figure_by_dimension(tmp_path, monkeypatch):
    write(fit_store(), tmp_path, monkeypatch)
    frame = handicap_factors(TARGET, as_of=AS_OF)
    assert_values(frame, "h:p", fit_want(FIT_P))
    # Q: matches everywhere but has no figure -> null fit, counted record
    assert_values(frame, "h:q", fit_want(dict.fromkeys(FIT_DIMS, (1, 0, None))))
    # R: a figure but no match -> shrunk fully to its own level: 0.0, not null
    assert_values(frame, "h:r", fit_want(dict.fromkeys(FIT_DIMS, (0, 0, 0.0))))
    assert_values(frame, "h:b", fit_want(dict.fromkeys(FIT_DIMS, (0, 0, None))))
    for d in FIT_DIMS:
        assert_values(frame, "h:q", {f"fit_{d}_missing": True, f"fit_{d}_n_missing": False})
        assert_values(frame, "h:r", {f"fit_{d}_missing": False})


# ---------------------------------------------------------------------------------------------
# 7. Condition and connections
# ---------------------------------------------------------------------------------------------


def connections_store() -> Store:
    s = Store()

    def run(horse, pos, jockey, trainer, **kw):
        return R(horse, pos=pos, jockey=jockey, trainer=trainer, **kw)

    # inside the window (2024-06-15 .. 2025-06-14)
    s.race(date(2025, 5, 1), [run("h:s1", 2, "J2", "T1", carried=55.0), run("r1b", 1, "J1", "T1"),
           run("r1c", 3, "J3", "T2"), run("r1d", 4, "J4", "T2"),
           R("r1x", scratched=True, jockey="J1", trainer="T1")])  # fmt: skip
    s.race(date(2025, 4, 1), [run("h:s2", 1, "J2", "T2", carried=56.0), run("r2b", 2, "J1", "T1"),
           run("r2c", 3, "J3", "T2"), run("r2d", 4, "J4", "T2")])  # fmt: skip
    s.race(date(2024, 6, 15), [run("r3a", 1, "J1", "T2"), run("r3b", 2, "J2", "T1"),
           run("r3c", 3, "J3", "T1"), run("r3d", 4, "J4", "T1")])  # fmt: skip
    # bait: the day before the window, the race day itself and the day after
    s.race(date(2024, 6, 14), [run("r4a", 1, "J1", "T1"), run("r4b", 2, "J5", "T3"),
           run("r4c", 3, "J2", "T2"), run("r4d", 4, "J3", "T2")])  # fmt: skip
    s.race(T, [run("r6a", 1, "J1", "T1"), run("r6b", 2, "J5", "T3")], race_no=3)
    s.race(T + timedelta(days=1), [run("r5a", 1, "J1", "T1"), run("r5b", 2, "J5", "T3")])
    s.race(T, [run("h:s1", 1, "J1", "T1", carried=57.0, body=486, body_chg=6),
               run("h:s2", 2, "J2", "T2", carried=54.0, body=470, body_chg=-4),
               run("h:s3", 3, "J5", "T3", carried=55.0, body=None, body_chg=None)],
           race_no=11)  # fmt: skip
    return s


def test_condition_and_connections(tmp_path, monkeypatch):
    write(connections_store(), tmp_path, monkeypatch)
    frame = handicap_factors(TARGET, as_of=AS_OF)
    # window: 12 declared runners (the scratch is not one), 3 winners -> p0 = 0.25, 20 * p0 = 5
    #   J1 3 rides 2 wins (2+5)/23; J2 3 rides 1 win 6/23; J5 none 5/20
    #   T1 6 runners 1 win 6/26; T2 6 runners 2 wins 7/26; T3 none 5/20
    assert_values(frame, "h:s1", {
        "body_wt_change": 6, "wt_carried_change": 2.0, "jockey_change": True,
        "jockey_rate": 7 / 23, "trainer_rate": 6 / 26,
    })  # fmt: skip
    assert_values(frame, "h:s2", {
        "body_wt_change": -4, "wt_carried_change": -2.0, "jockey_change": False,
        "jockey_rate": 6 / 23, "trainer_rate": 7 / 26,
    })  # fmt: skip
    assert_values(frame, "h:s3", {
        "body_wt_change": None, "body_wt_change_missing": True, "wt_carried_change": None,
        "jockey_change": None, "jockey_change_missing": True,
        "jockey_rate": 0.25, "trainer_rate": 0.25, "jockey_rate_missing": False,
    })  # fmt: skip


# ---------------------------------------------------------------------------------------------
# 8. Draw: today's draw group against the in-fold course-and-distance bias
# ---------------------------------------------------------------------------------------------


def draw_race(s: Store, day: date, n: int, winner_draw: int, **race) -> str:
    places = iter(range(2, n + 1))
    runners = [R(f"dr:{day}:{race.get('venue', 'TOK')}:{race.get('surface', 'turf')}:"
                 f"{race.get('distance', 1600)}:{d}", no=d,
                 pos=1 if d == winner_draw else next(places)) for d in range(1, n + 1)]  # fmt: skip
    return s.race(day, runners, **race)


def draw_store() -> Store:
    s = Store()
    draw_race(s, date(2025, 3, 2), 7, 1)  # TOK turf 1600: inner d<=3, middle d<=5, outer 6-7
    draw_race(s, date(2025, 4, 6), 7, 2)
    draw_race(s, date(2025, 5, 11), 4, 4)  # inner d<=2, middle d=3, outer d=4
    # other courses or distances, each won from the outside: not today's course and distance
    draw_race(s, date(2025, 5, 18), 7, 7, distance=1400)
    draw_race(s, date(2025, 5, 25), 7, 7, venue="NAK")
    draw_race(s, date(2025, 5, 25), 7, 7, surface="dirt")
    # bait: earlier on T and the day after, won from the middle
    draw_race(s, T, 7, 5, race_no=3)
    draw_race(s, T + timedelta(days=1), 7, 5)
    today = [R(f"d:{d}", pos=d, no=d) for d in range(1, 10)] + [R("d:10", no=10, scratched=True)]
    s.race(T, today, race_no=11)  # 9 declared: inner 1-3, middle 4-6, outer 7-9
    s.race(T, [R(f"e:{d}", pos=d) for d in (1, 2, 3)], venue="NAK", race_no=11, distance=2400)
    return s


# Expected wins per runner = 1/n. Default (all three TOK turf 1600 races):
#   inner  W 2 (D1 d1, D2 d2), E 3/7 + 3/7 + 2/4  -> (2+2)/(E+2)
#   middle W 0,                E 2/7 + 2/7 + 1/4  -> (0+2)/(E+2)
#   outer  W 1 (D3 d4),        E 2/7 + 2/7 + 1/4  -> (1+2)/(E+2)
# train_end 2025-04-30 (D1, D2 only): inner 4/(6/7+2) = 1.4, middle 2/(4/7+2), outer 2/(4/7+2)
DRAW_CASES = {
    None: {"inner": 4 / (6 / 7 + 0.5 + 2), "middle": 2 / (4 / 7 + 0.25 + 2),
           "outer": 3 / (4 / 7 + 0.25 + 2)},
    date(2025, 4, 30): {"inner": 1.4, "middle": 2 / (4 / 7 + 2), "outer": 2 / (4 / 7 + 2)},
}  # fmt: skip
TODAY_GROUP = {1: "inner", 3: "inner", 4: "middle", 6: "middle", 7: "outer", 9: "outer"}


@pytest.mark.parametrize("train_end", list(DRAW_CASES), ids=["as-of-cut", "train-end-0430"])
def test_draw_against_the_in_fold_course_and_distance_bias(tmp_path, monkeypatch, train_end):
    write(draw_store(), tmp_path, monkeypatch)
    kwargs = {} if train_end is None else {"train_end": train_end}
    frame = handicap_factors(TARGET, as_of=AS_OF, **kwargs)
    for d, group in TODAY_GROUP.items():
        assert_values(frame, f"d:{d}", {"draw_bias": DRAW_CASES[train_end][group]})
    other = handicap_factors(rid(T, "NAK", 11), as_of=AS_OF, **kwargs)
    for d in (1, 2, 3):  # no history at NAK turf 2400
        assert_values(other, f"e:{d}", {"draw_bias": None, "draw_bias_missing": True})


# ---------------------------------------------------------------------------------------------
# Leakage: no result on or after the cut changes any factor
# ---------------------------------------------------------------------------------------------

LEAK_AS_OF = {
    "race-day": (AS_OF, T - timedelta(days=1)),
    "a-month-before": (at_jst(date(2025, 5, 20), 12), date(2025, 5, 19)),
}


def with_a_future_race(s: Store, after: date) -> Store:
    """A race after the cut at today's course, distance and class, won in a record time by
    jockey:0 and trainer:0, with every pool horse in it."""
    s.race(after + timedelta(days=1), [R(f"h:{i:02d}", pos=i + 1, time=50.0 + i,
           jockey="jockey:0", trainer="trainer:0") for i in range(12)])  # fmt: skip
    return s


@pytest.mark.parametrize("case", list(LEAK_AS_OF))
def test_no_future_result_changes_any_factor(tmp_path, monkeypatch, case):
    as_of, cut = LEAK_AS_OF[case]
    s = leak_store()
    write(s, tmp_path, monkeypatch)
    base = handicap_factors(TARGET, as_of=as_of)
    changed = with_a_future_race(perturb(s, lambda d: d > cut, keep_card=TARGET), cut)
    write(changed, tmp_path, monkeypatch, sub="changed")
    assert_frame_equal(handicap_factors(TARGET, as_of=as_of), base)


def test_the_leakage_check_can_fail(tmp_path, monkeypatch):
    """The same perturbation on the three weeks before the cut does change the factors."""
    as_of, cut = LEAK_AS_OF["race-day"]
    s = leak_store()
    write(s, tmp_path, monkeypatch)
    base = handicap_factors(TARGET, as_of=as_of)
    changed = perturb(s, lambda d: cut - timedelta(days=21) < d <= cut)
    write(changed, tmp_path, monkeypatch, sub="changed")
    again = handicap_factors(TARGET, as_of=as_of)
    assert not again.equals(base)


def test_a_holdout_day_is_never_used(tmp_path, monkeypatch):
    """A past day marked holdout gives the same factors as a store without that day."""
    hidden = date(2025, 6, 7)
    s = leak_store()
    write(s, tmp_path, monkeypatch)
    full = handicap_factors(TARGET, as_of=AS_OF)
    patch_holdout(monkeypatch, [hidden], factors)
    hid = handicap_factors(TARGET, as_of=AS_OF)
    write(s.without_day(hidden), tmp_path, monkeypatch, sub="without")
    assert_frame_equal(hid, handicap_factors(TARGET, as_of=AS_OF))
    assert not hid.equals(full)


# ---------------------------------------------------------------------------------------------
# Missing values: an explicit flag per column, no silent imputation, one schema
# ---------------------------------------------------------------------------------------------


def test_every_factor_column_has_an_explicit_missing_flag(tmp_path, monkeypatch):
    write(leak_store(), tmp_path, monkeypatch)
    first = date(2025, 3, 1)
    for race_id, as_of in [(TARGET, AS_OF), (rid(first, "TOK", 1), at_jst(first, 9))]:
        frame = handicap_factors(race_id, as_of=as_of)
        values = [c for c in frame.columns if c not in KEYS and not c.endswith("_missing")]
        assert set(FACTOR_COLUMNS) <= set(values)
        for c in values:
            flag = f"{c}_missing"
            assert flag in frame.columns, f"{c} has no {flag}"
            assert frame.schema[flag] == pl.Boolean, flag
            assert frame[flag].to_list() == frame[c].is_null().to_list(), c
        flags = [c for c in frame.columns if c.endswith("_missing")]
        assert {f.removesuffix("_missing") for f in flags} <= set(values)


HISTORY_NULL = ["speed_best", "speed_last", "speed_mean3", *PACE, "style", "class_change",
                "speed_best_at_class", "speed_slope3", "days_since_last", "runs_since_layoff",
                "wt_carried_change", "jockey_change", *[f"fit_{d}" for d in FIT_DIMS]]  # fmt: skip
HISTORY_ZERO = ["speed_n", "starts_90d", *[f"fit_{d}_{s}" for d in FIT_DIMS for s in ("n", "top3")]]
KNOWN = ["class_today", "field_front_n", "field_front_share", "front_rivals", "jockey_rate",
         "trainer_rate"]  # fmt: skip


def test_a_debutant_gets_nulls_and_zero_counts_not_imputed_values(tmp_path, monkeypatch):
    write(leak_store(), tmp_path, monkeypatch)
    frame = handicap_factors(TARGET, as_of=AS_OF)
    new = row(frame, "h:new")
    assert {c: new[c] for c in HISTORY_NULL} == dict.fromkeys(HISTORY_NULL)
    assert all(new[f"{c}_missing"] for c in HISTORY_NULL)
    assert {c: new[c] for c in HISTORY_ZERO} == dict.fromkeys(HISTORY_ZERO, 0)
    assert all(new[c] is not None for c in KNOWN), {c: new[c] for c in KNOWN}
    # the horses with history do have values
    seasoned = frame.filter(pl.col("horse_id") != "h:new")
    assert seasoned.height >= 5
    assert seasoned["speed_n"].min() > 0
    assert seasoned["days_since_last"].null_count() == 0


def test_the_schema_is_the_same_for_every_race(tmp_path, monkeypatch):
    """A first-day race (every runner a debutant) and the target share one schema, so day
    partitions concatenate."""
    write(leak_store(), tmp_path, monkeypatch)
    target = handicap_factors(TARGET, as_of=AS_OF)
    first, later = date(2025, 3, 1), date(2025, 5, 3)
    for race_id in [rid(first, "TOK", 1), rid(first, "NAK", 2), rid(later, "TOK", 1)]:
        day = date.fromisoformat(race_id.split(":")[1])
        frame = handicap_factors(race_id, as_of=at_jst(day, 9))
        assert frame.schema == target.schema, race_id
