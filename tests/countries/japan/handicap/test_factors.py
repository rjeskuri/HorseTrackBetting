"""Handicap factors per runner as of decision time (ticket #129).

Contract pinned here (`horsetrack.countries.japan.handicap.factors`):

- `handicap_factors(race_id, *, as_of, train_end=None) -> polars.DataFrame`: one row per declared
  (not scratched) runner of the race, keys `race_id`, `horse_no`, `horse_id`. History comes from
  #128's `pp_lines(race_id, as_of=as_of)` (default n = 10 lines). The in-fold pieces (#96's par
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

- Ruling (#219, replaces #129's interim class par): speed figure of a past start = #96's expected
  top-3 time of that start's race - the runner's finish_time_s (seconds faster than expected; higher
  is better). The expected time is `SpeedModel.expected_top3_time` (models/speed.py) of
  `SpeedSpec(PAR_KEYS, VARIANT_KEYS, same_day_variant=True, t_df=5.0)` (#305's primary error model),
  fitted on the race table `speed_run.build_race_table(..., estimand="mean_placed_1_3")` of the
  in-fold days (HISTORY_START .. train_end): the race's top-3 time is the mean of the runners placed
  1-3 (a runner placed 1 or 2 once, those placed 3rd as one slot) and the cell is the six keys
  (venue, surface, distance, course layout, rail, class), not the old four. A start on an in-fold
  day has the variant's prior (0), so its expected time is the par. A start on a day after
  train_end (up to the cut) has its race's day variant: the model observes the results of the days
  after train_end up to the cut, and the figure reads the variant at the race's post time - 2
  minutes (D2's primary decision time), so only races of its (day, venue, surface) resulted before
  then count, never its own race and never a later one. No result after the cut is read. A start
  whose race is not in the table, whose cell has no par, or without a finish time has no figure.
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

import random
from datetime import date, datetime, timedelta

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from handicap_store import R, Store, T, at_jst, patch_holdout, perturb, random_store, rid
from handicap_store import weekend_days
from horsetrack.countries.japan.handicap import factors
from horsetrack.countries.japan.handicap.factors import handicap_factors
from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec
from horsetrack.countries.japan.studies.speed_run import PAR_KEYS, VARIANT_KEYS, build_race_table

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
# 1. Speed: figures against #96's expected top-3 time, fitted on the in-fold days
# ---------------------------------------------------------------------------------------------


def speed_store() -> Store:
    s = Store()
    tok = {"venue": "TOK", "surface": "turf", "distance": 1600, "cls": "C2"}  # K1
    nak = {"venue": "NAK", "surface": "turf", "distance": 2000, "cls": "C1"}  # K2
    dirt = {"venue": "TOK", "surface": "dirt", "distance": 1400, "cls": "C1"}  # K3
    # In every race the placed runners' times are the winner's + 0.2 s per place behind (the
    # fixture's default), so a race with winner time w has top-3 time (w + w+0.2 + w+0.4)/3 = w+0.2.
    # par-only races (no tracked horse); K2 and K3 repeat one top-3 time, so their fitted par is it
    # whatever the error model; K1's top-3 times are symmetric about their centre, so the par is it
    s.race(date(2025, 1, 12), [], winner_time=121.4, fill_to=4, **nak)  # top-3 121.6
    s.race(date(2025, 1, 19), [], winner_time=121.4, fill_to=4, **nak)  # 121.6
    s.race(date(2025, 3, 1), [], winner_time=85.4, fill_to=4, **dirt)  # 85.6
    s.race(date(2025, 3, 8), [], winner_time=85.4, fill_to=4, **dirt)  # 85.6
    s.race(date(2025, 5, 18), [], winner_time=94.8, fill_to=4, **tok)  # 95.0
    # A: four starts, newest first S1 .. S4
    s.race(date(2025, 3, 30), [R("h:a", pos=6)], winner_time=94.0, fill_to=6, **tok)  # 94.2
    s.race(date(2025, 4, 20), [R("h:a", pos=4)], winner_time=85.4, fill_to=6, **dirt)  # 85.6
    s.race(date(2025, 5, 11), [R("h:a", pos=1)], winner_time=121.4, fill_to=6, **nak)  # 121.6
    s.race(date(2025, 6, 1), [R("h:a", pos=3)], winner_time=94.4, fill_to=6, **tok)  # 94.6
    # C: an older start with a figure (alone in its cell), then a newer one it did not finish
    s.race(date(2025, 5, 25), [R("h:c", pos=3)], venue="HSN", distance=1200, winner_time=69.0,
           fill_to=6)  # fmt: skip
    s.race(date(2025, 6, 8), [R("h:c")], venue="KYO", distance=1800, winner_time=108.0,
           fill_to=6)  # fmt: skip
    # bait: a K1 race earlier on T, one the day after (with A in it), and the target's own result
    s.race(T, [], race_no=2, winner_time=99.9, fill_to=4, **tok)
    s.race(T + timedelta(days=1), [R("h:a", pos=1, time=70.0)], fill_to=4, **tok)
    s.race(T, [R("h:a", pos=1, time=80.0), R("h:b", pos=2), R("h:c", pos=3)], race_no=11, **tok)
    return s


# Default (cut 06-14). Pars: K1 top-3 times {94.2 (03-30), 95.0 (05-18), 94.6 (06-01)}, symmetric
# about 94.6 -> 94.6; K2 {121.6 x3} -> 121.6; K3 {85.6 x3} -> 85.6; HSN 1200 {69.2} -> 69.2. A's
# times are the winner's + 0.2 per place behind: 06-01 pos 3 94.8, 05-11 pos 1 121.4, 04-20 pos 4
# 86.0, 03-30 pos 6 95.0.
#   A: S1 94.6-94.8 = -0.2, S2 121.6-121.4 = 0.2, S3 85.6-86.0 = -0.4, S4 94.6-95.0 = -0.4
#      best 0.2, last -0.2, mean3 (-0.2+0.2-0.4)/3, n 4
#   C: newest start unfinished (no figure); pos 3 at 69.4: 69.2-69.4 = -0.2 -> best = last = mean3
# train_end 2025-03-31: K1 {94.2} -> 94.2, so S4 94.2-95.0 = -0.8 and S1 (06-01, a day after
# train_end; no earlier TOK turf race that day, so the variant is its prior) 94.2-94.8 = -0.6;
# K2 {121.6 x2} and K3 {85.6 x2} are unchanged; HSN 1200 has no in-fold race, so C has no figure.
#      A: best 0.2, last -0.6, mean3 (-0.6+0.2-0.4)/3, n 4
SPEED_CASES = {
    None: {
        "h:a": {"speed_best": 0.2, "speed_last": -0.2, "speed_mean3": -0.4 / 3, "speed_n": 4},
        "h:c": {"speed_best": -0.2, "speed_last": -0.2, "speed_mean3": -0.2, "speed_n": 1},
        "h:b": {"speed_best": None, "speed_last": None, "speed_mean3": None, "speed_n": 0},
    },
    date(2025, 3, 31): {
        "h:a": {"speed_best": 0.2, "speed_last": -0.6, "speed_mean3": -0.8 / 3, "speed_n": 4},
        "h:c": {"speed_best": None, "speed_last": None, "speed_mean3": None, "speed_n": 0},
        "h:b": {"speed_best": None, "speed_last": None, "speed_mean3": None, "speed_n": 0},
    },
}


@pytest.mark.parametrize("train_end", list(SPEED_CASES), ids=["as-of-cut", "train-end-0331"])
def test_speed_figures_against_96s_expected_time(tmp_path, monkeypatch, train_end):
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


def cells_store() -> Store:
    """TOK turf 1600 C2 in three #96 cells that differ only in course layout or rail: each cell
    has two par-only races and one tracked start, all of one top-3 time (winner's + 0.2)."""
    s = Store()
    cells = [  # (course, rail, winner time, [par-only days], tracked horse, place, day)
        ("outer", None, 94.0, [date(2025, 1, 4), date(2025, 1, 11)], "h:x1", 1, date(2025, 2, 1)),
        ("inner", None, 95.0, [date(2025, 1, 18), date(2025, 1, 25)], "h:x2", 4, date(2025, 2, 8)),
        ("outer", "B", 96.0, [date(2025, 2, 15), date(2025, 2, 22)], "h:x3", 6, date(2025, 3, 1)),
    ]
    for course, rail, w, days, horse, pos, day in cells:
        for d in days:
            s.race(d, [], course=course, rail=rail, winner_time=w, fill_to=4)
        s.race(day, [R(horse, pos=pos)], course=course, rail=rail, winner_time=w, fill_to=6)
    s.race(T, [R("h:x1", pos=1), R("h:x2", pos=2), R("h:x3", pos=3)], race_no=11)
    return s


# Top-3 times (winner's + 0.2): outer 94.2, inner 95.2, outer on rail B 96.2. A start's time is the
# winner's + 0.2 per place behind: x1 won (94.0), x2 4th (95.6), x3 6th (97.0). Figures against its
# own cell's par: 94.2 - 94.0, 95.2 - 95.6, 96.2 - 97.0. A par pooled over layouts or rails would
# be the median of nine, 95.2, and give 1.2 / -0.4 / -1.8 or similar: not these.
CELL_FIGURES = {"h:x1": 0.2, "h:x2": -0.4, "h:x3": -0.8}


def test_the_par_cell_is_96s_six_keys_so_layout_and_rail_split_it(tmp_path, monkeypatch):
    write(cells_store(), tmp_path, monkeypatch)
    frame = handicap_factors(TARGET, as_of=AS_OF)
    for horse_id, fig in CELL_FIGURES.items():
        want = {"speed_best": fig, "speed_last": fig, "speed_mean3": fig, "speed_n": 1}
        assert_values(frame, horse_id, want)


# The reference: #96's model fitted on the store's in-fold races by hand, as the ruling says.


def reference_figures(
    s: Store, horse_id: str, *, cut: date, train_end: date, variant: bool = True
) -> list[float | None]:
    """The horse's newest 10 starts to the cut, newest first: #96's expected top-3 time of the
    start's race (par from the races up to train_end; the day variant from the races after it up to
    the cut, read 2 minutes before post) minus the finish time; None without one."""
    table, _ = build_race_table(
        pd.DataFrame(s.races), pd.DataFrame(s.runners), estimand="mean_placed_1_3"
    )
    spec = SpeedSpec(PAR_KEYS, VARIANT_KEYS, same_day_variant=variant, t_df=5.0)
    model = SpeedModel(spec).fit(table[table["race_date"] <= train_end])
    later = table[(table["race_date"] > train_end) & (table["race_date"] <= cut)]
    if len(later):
        model.observe(later)
    races = {r["race_id"]: r for r in table.to_dict("records")}
    starts = [
        r
        for r in s.runners
        if r["horse_id"] == horse_id and not r["is_scratched"] and r["race_date"] <= cut
    ]
    starts.sort(key=lambda r: (r["race_date"], r["race_id"]), reverse=True)
    figs: list[float | None] = []
    for r in starts[:10]:
        race = races.get(r["race_id"])
        if race is None or r["finish_time_s"] is None:
            figs.append(None)
            continue
        try:
            expected = model.expected_top3_time(
                race, at_utc=race["post_utc"] - pd.Timedelta(minutes=2)
            )
        except KeyError:  # a cell the fit never saw
            figs.append(None)
            continue
        figs.append(expected - r["finish_time_s"])
    return figs


def speed_aggregates(figs: list[float | None]) -> dict:
    got = [f for f in figs if f is not None]
    return {
        "speed_best": max(got, default=None),
        "speed_last": got[0] if got else None,
        "speed_mean3": sum(got[:3]) / len(got[:3]) if got else None,
        "speed_n": len(got),
    }


NOISY_CELLS = [
    ({"venue": "TOK", "surface": "turf", "distance": 1600, "cls": "C2"}, 95.0),
    ({"venue": "TOK", "surface": "dirt", "distance": 1400, "cls": "C1"}, 83.0),
    ({"venue": "NAK", "surface": "turf", "distance": 2000, "cls": "C1"}, 121.0),
]


def noisy_store() -> Store:
    """Weekend days with six races each over three cells (two races per day, venue and surface),
    a going shift per day and venue, and noisy winner times; 14 horses start seven races each."""
    rng = random.Random(5)
    s = Store()
    horses = [f"h:{i:02d}" for i in range(14)]
    for day in weekend_days(date(2025, 3, 1), date(2025, 6, 8)):
        going = {v: rng.uniform(-1.0, 1.0) for v in ("TOK", "NAK")}
        for k in range(6):
            race, base = NOISY_CELLS[k % 3]
            winner = round(base + going[race["venue"]] + rng.uniform(-0.4, 0.4), 1)
            field, order = rng.sample(horses, 7), rng.sample(range(1, 8), 7)
            s.race(day, [R(h, pos=p) for h, p in zip(field, order, strict=True)],
                   race_no=k + 1, winner_time=winner, **race)  # fmt: skip
    s.race(T, [R(h, pos=i + 1) for i, h in enumerate(horses[:7])], race_no=11)
    return s


@pytest.mark.parametrize(
    "train_end", [None, date(2025, 4, 27)], ids=["as-of-cut", "train-end-0427"]
)
def test_speed_figures_follow_the_fitted_96_model(tmp_path, monkeypatch, train_end):
    """Broad: seven horses with ten starts each over noisy cells and days, every aggregate
    against #96's model fitted by hand. With a train_end before the cut the starts after it carry
    their day's variant; the case is checked to be one where the variant matters."""
    s = noisy_store()
    write(s, tmp_path, monkeypatch)
    cut = T - timedelta(days=1)
    kwargs = {} if train_end is None else {"train_end": train_end}
    frame = handicap_factors(TARGET, as_of=AS_OF, **kwargs)
    horses = sorted({r["horse_id"] for r in s.runners if r["race_id"] == TARGET})
    assert len(horses) == 7
    moved = 0.0
    for horse_id in horses:
        figs = reference_figures(s, horse_id, cut=cut, train_end=train_end or cut)
        want = speed_aggregates(figs)
        assert want["speed_n"] == 10
        assert_values(frame, horse_id, want)
        assert_values(frame, horse_id, {f"{c}_missing": want[c] is None for c in SPEED})
        flat = reference_figures(s, horse_id, cut=cut, train_end=train_end or cut, variant=False)
        moved = max(moved, max(abs(a - b) for a, b in zip(figs, flat, strict=True)))
    if train_end is not None:
        assert moved > 0.05  # the fixture's day variants are not nil


def variant_store(*, early: float = 0.0, own: float = 0.0, late: float = 0.0) -> Store:
    """TOK turf 1600 C2: in-fold races (to 03-23, two a day, a going shift per day) fix the par and
    the variance; then on 05-10 three races of that cell, an early one, the start of h:v (3rd, at
    95.4) and a late one, with winner times 95.0 + early / own / late."""
    rng = random.Random(9)
    s = Store()
    for day in weekend_days(date(2025, 2, 1), date(2025, 3, 23)):
        shift = rng.uniform(-1.0, 1.0)
        for no in (2, 3):
            w = round(95.0 + shift + rng.uniform(-0.3, 0.3), 1)
            s.race(day, [], race_no=no, winner_time=w, fill_to=7)
    day = date(2025, 5, 10)
    s.race(day, [], race_no=1, winner_time=95.0 + early, fill_to=7)  # posts 09:25
    s.race(
        day, [R("h:v", pos=3, time=95.4)], race_no=11, winner_time=95.0 + own, fill_to=7
    )  # 14:35
    s.race(day, [], race_no=12, winner_time=95.0 + late, fill_to=7)  # posts 15:00
    s.race(T, [R("h:v", pos=1)], race_no=11)
    return s


FOLD_END = date(2025, 3, 31)


def v_last(tmp_path, monkeypatch, sub: str, **shifts) -> float:
    s = variant_store(**shifts)
    write(s, tmp_path, monkeypatch, sub=sub)
    frame = handicap_factors(TARGET, as_of=AS_OF, train_end=FOLD_END)
    got = row(frame, "h:v")["speed_last"]
    ref = reference_figures(s, "h:v", cut=T - timedelta(days=1), train_end=FOLD_END)
    assert got == pytest.approx(ref[0], abs=1e-6), (shifts, got, ref)
    return got


def test_a_start_after_train_end_carries_its_days_earlier_results(tmp_path, monkeypatch):
    """The figure of h:v's start on 05-10 is par + the day's variant - 95.4; the variant is read
    from the race 5 hours before it: a slow early race raises the figure, a fast one lowers it,
    each equal to #96's model."""
    fast = v_last(tmp_path, monkeypatch, "fast", early=-1.5)
    flat = v_last(tmp_path, monkeypatch, "flat", early=0.0)
    slow = v_last(tmp_path, monkeypatch, "slow", early=1.5)
    assert fast < flat < slow
    assert slow - fast > 0.3


def test_a_figure_never_reads_its_own_race_or_a_later_one(tmp_path, monkeypatch):
    """Changing the times of the other runners of the start's own race (it resulted after the
    decision time) or of a race 25 minutes after it leaves the figure as it was."""
    base = v_last(tmp_path, monkeypatch, "base")
    own = v_last(tmp_path, monkeypatch, "own", own=4.0)
    late = v_last(tmp_path, monkeypatch, "late", late=-4.0)
    both = v_last(tmp_path, monkeypatch, "both", own=-6.0, late=5.0)
    assert own == pytest.approx(base, abs=1e-9)
    assert late == pytest.approx(base, abs=1e-9)
    assert both == pytest.approx(base, abs=1e-9)


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


# Each start alone in its #96 cell (and its day and track): the par is its race's top-3 time, the
# winner's + 0.2 s, so a figure is 0.2 s less per place behind the winner, 0.2 for the winner.
# (day, class, place, distance), newest first.
CLASS_STARTS = {
    # G: C3 2nd (0.0), C1 1st (0.2), OP 3rd (-0.2), MAIDEN 1st (0.2)
    "h:g": ("TOK", [(date(2025, 6, 1), "C3", 2, 1200), (date(2025, 5, 18), "C1", 1, 1400),
                    (date(2025, 5, 4), "OP", 3, 1800), (date(2025, 4, 20), "MAIDEN", 1, 2000)]),
    # H: C1 2nd (0.0), MAIDEN 1st (0.2)
    "h:h": ("NAK", [(date(2025, 6, 7), "C1", 2, 1200), (date(2025, 5, 24), "MAIDEN", 1, 1400)]),
    # I: G1 5th (-0.6), an unclassed start it did not finish, DEBUT 1st (0.2), C1 1st (0.2)
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
    # Each start is alone in its #96 cell (and its day and track), so its par is its race's top-3
    # time, the winner's + 0.2 s, and the figure of place p is 0.2 - 0.2 (p - 1) = 0.4 - 0.2 p.
    # G: 2 - (3+1+4)/3; best at level >= 2 among C3 (2nd, 0.0) and OP (3rd, -0.2)
    assert_values(frame, "h:g", {"class_today": 2, "class_change": 2 - 8 / 3,
                                 "speed_best_at_class": 0.0})  # fmt: skip
    # H: 2 - (1+0)/2; no start at level >= 2
    assert_values(frame, "h:h", {"class_change": 1.5, "speed_best_at_class": None,
                                 "speed_best_at_class_missing": True})  # fmt: skip
    # I: newest three with a level: G1 8, DEBUT 0, C1 1 -> 2 - 3; best at >= 2: G1 5th, -0.6
    assert_values(frame, "h:i", {"class_change": -1.0, "speed_best_at_class": -0.6})
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
    # newest first: (day, place); each alone in its #96 cell, so the figure is 0.2 - 0.2 per place
    # behind the winner; the slope below is the same as with the figures listed (shift-invariant)
    starts = {
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
    # P, newest first (figure -0.2 per place behind, 0.2 for the winner; each start alone in its #96
    # cell, so every figure is the 0.2 higher than listed, and the shrunk differences are the same):
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


@pytest.mark.parametrize("case", list(LEAK_AS_OF))
def test_speed_figures_ignore_every_result_after_the_cut(tmp_path, monkeypatch, case):
    """Same cells and days before and after the cut (#96's fit and variant): perturbing every
    result after it, and adding a record-time race the day after, moves no speed column."""
    as_of, cut = LEAK_AS_OF[case]
    s = noisy_store()
    write(s, tmp_path, monkeypatch)
    base = handicap_factors(TARGET, as_of=as_of)
    changed = with_a_future_race(perturb(s, lambda d: d > cut, keep_card=TARGET), cut)
    write(changed, tmp_path, monkeypatch, sub="changed")
    cols = [*SPEED, *[f"{c}_missing" for c in SPEED]]
    assert base["speed_n"].min() > 0
    assert_frame_equal(handicap_factors(TARGET, as_of=as_of).select(cols), base.select(cols))


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
