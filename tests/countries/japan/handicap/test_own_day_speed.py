"""The handicap table's speed figure as of each start's own race day (ticket #219, revision).

Contract pinned here (`horsetrack.countries.japan.handicap.factors`):

- `_own_day_expected(race_table) -> polars.DataFrame` (`pp_race_id`, `expected`): #96's expected
  top-3 time of every race of the table (`speed_run.build_race_table`, sorted by day and id) as of
  its own race day E: `SpeedModel` fitted on the races of the days before E, having observed E's
  results, read at post - 2 min (so an earlier race of the meeting counts through the day variant,
  and neither the race itself nor a later one does, nor any later day). No row for a race in the
  first 365 days of the table, on the table's first day, or in a cell with no race on an earlier
  day.
- `build_handicap_table(..., speed_at_start_day=True)`: each past start's `fig` is that expected
  time - the finish time (no figure without a finish time); the default `False` keeps
  `handicap_factors(race, as_of=race day)`.
- The builder's default destination is `handicap_96_dir()` (`features/jra/handicap-96-v2`), never
  #129's `handicap_dir()`. `table_complete(dir, from, to, speed_at_start_day=)` holds only for a
  build that ran through the score pass: it writes `_complete.json` last, naming the range, the
  mode and the day count, and a rebuild starts by removing it.
- `fit_handicap_score` refuses a non-finite objective or Hessian (RuntimeError) rather than spin.

Data: synthetic only, written under tmp_path (tests/countries/japan/handicap/handicap_store.py).
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import contextlib
import random
import shutil
import signal
from datetime import date, timedelta

import pandas as pd
import polars as pl
import pytest

from handicap_store import R, Store, patch_holdout, rid, weekend_days
from horsetrack.countries.japan.handicap import factors
from horsetrack.countries.japan.models.speed import SpeedModel, SpeedSpec
from horsetrack.countries.japan.studies.speed_run import PAR_KEYS, VARIANT_KEYS, build_race_table

D0 = date(2024, 1, 6)  # a Saturday: the first race day of every store here
WARM_END = D0 + timedelta(days=365)  # 2025-01-05, a Sunday: the first day that has a figure
K1 = {"venue": "TOK", "surface": "turf", "distance": 1600, "cls": "C2"}
K2 = {"venue": "NAK", "surface": "dirt", "distance": 1400, "cls": "C1"}
SPEC = SpeedSpec(PAR_KEYS, VARIANT_KEYS, same_day_variant=True, t_df=5.0)


def race_table(s: Store) -> pd.DataFrame:
    table, _ = build_race_table(
        pd.DataFrame(s.races), pd.DataFrame(s.runners), estimand="mean_placed_1_3"
    )
    return table.sort_values(["race_date", "race_id"], kind="stable", ignore_index=True)


def oracle(table: pd.DataFrame, only: set[str] | None = None) -> dict[str, float]:
    """#96's expected top-3 time as of the race's own day, fitted by hand: the model on the days
    before E, E's results observed, read at post - 2 min. Days before WARM_END and the table's
    first day are left out, as is a race in a cell the fit never saw. `only` limits the days."""
    out: dict[str, float] = {}
    for day in sorted(table["race_date"].unique()):
        if day < WARM_END or day == D0:
            continue
        today = table[table["race_date"] == day]
        if only is not None and not today["race_id"].isin(only).any():
            continue
        model = SpeedModel(SPEC).fit(table[table["race_date"] < day])
        model.observe(today)
        for race in today.to_dict("records"):
            try:
                out[race["race_id"]] = model.expected_top3_time(
                    race, at_utc=race["post_utc"] - pd.Timedelta(minutes=2)
                )
            except KeyError:  # a cell with no race on an earlier day
                continue
    return out


def own_day(table: pd.DataFrame) -> dict[str, float]:
    got = factors._own_day_expected(table)
    assert isinstance(got, pl.DataFrame)
    assert got.columns == ["pp_race_id", "expected"]
    assert got["pp_race_id"].n_unique() == got.height
    return dict(got.iter_rows())


# ---------------------------------------------------------------------------------------------
# 1. Broad: a noisy store over three cells, every race against the hand-fitted oracle
# ---------------------------------------------------------------------------------------------


def noisy_store() -> Store:
    """Weekend days from D0 to 2025-03-02: K1 (TOK turf, races 1, 11 and 12) and K2 (NAK dirt,
    races 1 and 11) on every day with a going shift per day and venue and noisy winner times, a
    rare cell K3 on 2025-02-01 and 02-15, and a cell K5 that first appears on 2025-02-08 as two
    races of one meeting."""
    rng = random.Random(3)
    s = Store()
    for day in weekend_days(D0, date(2025, 3, 2)):
        shift = {v: rng.uniform(-1.0, 1.0) for v in ("TOK", "NAK")}
        for cell, nos, base in ((K1, (1, 11, 12), 95.0), (K2, (1, 11), 83.0)):
            for no in nos:
                w = round(base + shift[cell["venue"]] + rng.uniform(-0.3, 0.3), 1)
                s.race(day, [], race_no=no, winner_time=w, fill_to=7, **cell)
    k3 = {"venue": "TOK", "surface": "turf", "distance": 2400, "cls": "C3"}
    for day in (date(2025, 2, 1), date(2025, 2, 15)):
        s.race(day, [], race_no=5, winner_time=148.0, fill_to=7, **k3)
    k5 = {"venue": "NAK", "surface": "turf", "distance": 1200, "cls": "C1"}
    for no in (3, 13):
        s.race(date(2025, 2, 8), [], race_no=no, winner_time=70.0, fill_to=7, **k5)
    s.race(date(2025, 2, 22), [], race_no=3, winner_time=70.5, fill_to=7, **k5)
    return s


def test_every_race_has_its_own_day_figure_or_none_and_the_figure_is_the_hand_fitted_one():
    table = race_table(noisy_store())
    got = own_day(table)
    # who has no figure, worked out from the table alone: the first 365 days (the first day too,
    # which is inside them), and a race whose cell has no race on an earlier day
    cell = table[list(PAR_KEYS)].astype(str).agg("|".join, axis=1)
    first_day = table.groupby(cell)["race_date"].transform("min")
    none = (table["race_date"] < WARM_END) | (first_day == table["race_date"])
    want_ids = set(table.loc[~none, "race_id"])
    assert set(got) == want_ids
    assert len(want_ids) > 50 and none.sum() > 100  # both kinds are well represented
    # the named edge cases
    day = lambda d, v, n: rid(d, v, n)  # noqa: E731
    assert day(WARM_END - timedelta(days=1), "TOK", 1) not in got  # day 364: still warming up
    assert day(WARM_END, "TOK", 1) in got  # day 365: the first with a figure
    assert day(date(2025, 2, 1), "TOK", 5) not in got  # K3's first race
    assert day(date(2025, 2, 15), "TOK", 5) in got  # K3's second
    assert day(date(2025, 2, 8), "NAK", 3) not in got  # K5: an earlier race of the day is not
    assert day(date(2025, 2, 8), "NAK", 13) not in got  # an earlier day
    assert day(date(2025, 2, 22), "NAK", 3) in got
    # the figure of every race that has one
    want = oracle(table)
    assert set(want) == want_ids
    for race_id, value in want.items():
        assert got[race_id] == pytest.approx(value, abs=1e-6), race_id


# ---------------------------------------------------------------------------------------------
# 2. What a figure reads: earlier races of its day, never its own, a later one or a later day
# ---------------------------------------------------------------------------------------------

TARGET_DAY, AFTER = date(2025, 5, 10), date(2025, 5, 17)
EARLY, OWN, LATE = (rid(TARGET_DAY, "TOK", n) for n in (1, 11, 12))  # posts 09:25, 14:35, 15:00


def day_store(
    *, history=0.0, early=0.0, own=0.0, late=0.0, after=0.0, own_runners=(), extra=None
) -> Store:
    """TOK turf 1600 C2 (K1): 2024 weekends with a going shift per day (winner 95.0 + history +
    noise; all before any figure), then three races on TARGET_DAY (early / own / late, winner 95.0
    + each) and two on AFTER (95.0 + after)."""
    rng = random.Random(9)
    s = Store()
    for day in weekend_days(D0, date(2024, 12, 29)):
        shift = rng.uniform(-1.0, 1.0)
        for no in (2, 3):
            w = round(95.0 + history + shift + rng.uniform(-0.3, 0.3), 1)
            s.race(day, [], race_no=no, winner_time=w, fill_to=7, **K1)
    s.race(TARGET_DAY, [], race_no=1, winner_time=95.0 + early, fill_to=7, **K1)
    s.race(TARGET_DAY, list(own_runners), race_no=11, winner_time=95.0 + own, fill_to=7, **K1)
    s.race(TARGET_DAY, [], race_no=12, winner_time=95.0 + late, fill_to=7, **K1)
    for no in (1, 11):
        s.race(AFTER, [], race_no=no, winner_time=95.0 + after, fill_to=7, **K1)
    if extra is not None:
        extra(s)
    return s


def figures(*, check=False, **shifts) -> dict[str, float]:
    table = race_table(day_store(**shifts))
    got = own_day(table)
    if check:  # whatever moves, each figure is the hand-fitted one
        want = oracle(table, {EARLY, OWN, LATE})
        assert {r: got[r] for r in (EARLY, OWN, LATE)} == pytest.approx(
            {r: want[r] for r in (EARLY, OWN, LATE)}, abs=1e-6
        )
    return got


def test_a_figure_moves_with_an_earlier_race_of_its_day_and_with_earlier_days():
    fast, flat, slow = (figures(early=e, check=e == 1.5)[OWN] for e in (-1.5, 0.0, 1.5))
    assert fast < flat < slow  # a slow morning race raises the afternoon's expected time
    assert slow - fast > 0.3
    base = figures()
    # the whole earlier record a second slower: so is the day's first race's figure, which has no
    # earlier race of its day to correct it (a later one reads that race as fast against the par)
    slower = figures(history=1.0)
    assert slower[EARLY] - base[EARLY] == pytest.approx(1.0, abs=0.2)


def test_a_figure_ignores_its_own_race_a_later_race_and_every_later_day():
    base = figures()
    for shift in (
        {"own": 4.0}, {"late": -4.0}, {"after": 3.0}, {"after": -3.0},
        {"own": -6.0, "late": 5.0, "after": 4.0},
    ):  # fmt: skip
        got = figures(check="after" in shift and "own" in shift, **shift)
        for race_id in (EARLY, OWN):
            assert got[race_id] == pytest.approx(base[race_id], abs=1e-9), (shift, race_id)
    # the day's first race has no earlier race to read: it is blind to the rest of its day
    assert figures(early=3.0)[EARLY] == pytest.approx(base[EARLY], abs=1e-9)


# ---------------------------------------------------------------------------------------------
# 3. The builder: speed_at_start_day=True gives each past start that figure
# ---------------------------------------------------------------------------------------------

OLD_START = date(2024, 6, 8)  # a Saturday inside the first 365 days
K3 = {"venue": "TOK", "surface": "dirt", "distance": 2400, "cls": "C3"}  # a cell nobody ran before
TARGET = rid(AFTER, "TOK", 21)
OWN_RUNNERS = (R("h:v", pos=3, time=95.4), R("h:dnf"))


def builder_store() -> Store:
    def extra(s: Store) -> None:
        s.race(OLD_START, [R("h:old", pos=2, time=95.3)], race_no=5, winner_time=95.0, fill_to=7,
               **K1)  # fmt: skip
        s.race(TARGET_DAY, [R("h:nc", pos=1, time=148.0)], race_no=3, winner_time=148.0,
               fill_to=7, **K3)  # fmt: skip
        s.race(AFTER, [R(h, pos=i + 1) for i, h in enumerate(("h:v", "h:dnf", "h:old", "h:nc"))],
               race_no=21, fill_to=8, **K1)  # fmt: skip

    return day_store(own_runners=OWN_RUNNERS, extra=extra)


def built(tmp_path, monkeypatch, *, start_day: bool, out="tbl", **kw):
    data = builder_store().write(tmp_path, monkeypatch)
    out_dir = tmp_path / out
    written = factors.build_handicap_table(
        AFTER, AFTER, out_dir=out_dir, speed_at_start_day=start_day, **kw
    )
    return data, out_dir, written


def horse(written: dict, horse_id: str) -> dict:
    frame = pl.read_parquet(written[AFTER])
    rows = frame.filter(pl.col("horse_id") == horse_id).to_dicts()
    assert len(rows) == 1
    return rows[0]


def test_the_builder_gives_each_past_start_the_figure_of_its_own_day(tmp_path, monkeypatch):
    s = builder_store()
    own_id = rid(TARGET_DAY, "TOK", 11)
    want = oracle(race_table(s), {own_id})[own_id]
    _, _, written = built(tmp_path, monkeypatch, start_day=True)
    v = horse(written, "h:v")
    assert v["speed_n"] == 1
    assert v["speed_last"] == pytest.approx(want - 95.4, abs=1e-6)
    assert v["speed_best"] == pytest.approx(want - 95.4, abs=1e-6)
    # no figure: no finish time; a start in the first 365 days; a cell with no earlier race
    for h in ("h:dnf", "h:old", "h:nc"):
        got = horse(written, h)
        assert got["speed_n"] == 0, h
        assert got["speed_last"] is None and got["speed_last_missing"] is True, h


def test_the_default_builder_keeps_the_fit_to_the_cut_for_every_past_start(tmp_path, monkeypatch):
    """Without the flag every start is in the fit (the same expected time as `handicap_factors`),
    so the horses the day-of-start figure leaves out have one here, and h:v's differs."""
    _, _, flat = built(tmp_path, monkeypatch, start_day=False)
    assert horse(flat, "h:old")["speed_n"] == 1  # inside the first 365 days: a figure here
    assert horse(flat, "h:nc")["speed_n"] == 1  # its own race made its cell's par
    assert horse(flat, "h:dnf")["speed_n"] == 0  # no finish time: none in either mode
    assert horse(flat, "h:v")["speed_n"] == 1


def test_the_two_modes_differ_on_the_same_start(tmp_path, monkeypatch):
    _, _, flat = built(tmp_path, monkeypatch, start_day=False)
    out2 = tmp_path / "tbl-day"
    day = factors.build_handicap_table(AFTER, AFTER, out_dir=out2, speed_at_start_day=True)
    assert horse(flat, "h:v")["speed_last"] != pytest.approx(
        horse(day, "h:v")["speed_last"], abs=1e-4
    )


# ---------------------------------------------------------------------------------------------
# 4. The destination and the completion marker
# ---------------------------------------------------------------------------------------------


def test_the_default_destination_is_the_96_table_and_never_129s(tmp_path, monkeypatch):
    data = builder_store().write(tmp_path, monkeypatch)
    assert factors.handicap_96_dir() == data / "features" / "jra" / "handicap-96-v2"
    assert factors.handicap_dir() == data / "features" / "jra" / "handicap"
    assert factors.handicap_96_dir() != factors.handicap_dir()
    written = factors.build_handicap_table(AFTER, AFTER)
    path = written[AFTER]
    assert path.parent == factors.handicap_96_dir()
    assert path.name == f"{AFTER.isoformat()}.parquet"
    assert not factors.handicap_dir().exists()  # not even created
    assert factors.table_complete(factors.handicap_96_dir(), AFTER, AFTER)


def test_the_marker_is_there_only_for_a_finished_build_of_this_range_and_mode(
    tmp_path, monkeypatch
):
    _, done, written = built(tmp_path, monkeypatch, start_day=True)
    day = AFTER
    assert (done / "_complete.json").is_file()
    assert factors.table_complete(done, day, day, speed_at_start_day=True)
    # another mode or another range is not this build
    assert not factors.table_complete(done, day, day, speed_at_start_day=False)
    assert not factors.table_complete(done, day, day)
    assert not factors.table_complete(done, day - timedelta(days=7), day, speed_at_start_day=True)
    assert not factors.table_complete(done, day, day + timedelta(days=7), speed_at_start_day=True)
    assert not factors.table_complete(tmp_path / "nowhere", day, day, speed_at_start_day=True)
    # a day file gone, or one more than the marker counted
    part = written[day]
    spare = done / "2025-05-24.parquet"
    shutil.copy(part, spare)
    assert not factors.table_complete(done, day, day, speed_at_start_day=True)
    spare.unlink()
    assert factors.table_complete(done, day, day, speed_at_start_day=True)
    kept = part.read_bytes()
    part.unlink()
    assert not factors.table_complete(done, day, day, speed_at_start_day=True)
    part.write_bytes(kept)
    # a marker that is not a marker
    marker = done / "_complete.json"
    good = marker.read_text()
    marker.write_text("{")
    assert not factors.table_complete(done, day, day, speed_at_start_day=True)
    marker.write_text(good)
    assert factors.table_complete(done, day, day, speed_at_start_day=True)


def test_an_interrupted_build_has_partitions_but_is_not_complete(tmp_path, monkeypatch):
    """The score pass fails (a scheme that is no scheme): the day files are written, the marker
    is not, and a marker left by an earlier finished build of the same range does not survive."""
    _, out, written = built(tmp_path, monkeypatch, start_day=True)
    day = AFTER
    assert factors.table_complete(out, day, day, speed_at_start_day=True)
    with pytest.raises(Exception):  # noqa: B017, PT011
        factors.build_handicap_table(
            day, day, out_dir=out, speed_at_start_day=True, scheme="not a scheme"
        )
    assert list(out.glob("*.parquet"))  # partitions are on disk
    assert not (out / "_complete.json").exists()
    assert not factors.table_complete(out, day, day, speed_at_start_day=True)
    assert not factors.table_complete(out, day, day, speed_at_start_day=False)
    # the score pass on the files that are there finishes it
    files = {d: p for d, p in written.items()}
    factors.score_table(files, day, day, speed_at_start_day=True)
    assert factors.table_complete(out, day, day, speed_at_start_day=True)
    assert not factors.table_complete(out, day, day, speed_at_start_day=False)


def test_a_holdout_day_is_not_counted_by_the_marker(tmp_path, monkeypatch):
    """The marker counts the files written: a day marked holdout has none, and the marker of a
    two-day range with one holdout day holds for one file."""
    s = builder_store()
    patch_holdout(monkeypatch, [TARGET_DAY], factors)
    s.write(tmp_path, monkeypatch)
    out = tmp_path / "tbl"
    written = factors.build_handicap_table(TARGET_DAY, AFTER, out_dir=out, speed_at_start_day=True)
    assert sorted(written) == [AFTER]
    assert factors.table_complete(out, TARGET_DAY, AFTER, speed_at_start_day=True)
    assert not factors.table_complete(out, AFTER, AFTER, speed_at_start_day=True)


# ---------------------------------------------------------------------------------------------
# 5. The fit refuses a non-finite objective instead of spinning
# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def no_spin(seconds: int = 30):
    def fire(signum, frame):
        raise AssertionError(f"the fit was still running after {seconds} s")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("bad", [float("inf"), 1e200], ids=["inf", "overflowing"])
def test_a_non_finite_design_raises_instead_of_spinning(bad):
    from test_handicap_score import FEATURES, score_frame

    frame = score_frame(n_days=40)
    poisoned = frame.with_columns(
        pl.when(pl.col("race_date") == date(2025, 1, 3)).then(bad).otherwise(pl.col("x_good"))
        .alias("x_good")
    )  # fmt: skip
    with no_spin(), pytest.raises(RuntimeError):
        factors.fit_handicap_score(poisoned, features=FEATURES)
    # the same frame without the poisoned day fits, and the fit is not just refusing everything
    clean = frame.filter(pl.col("race_date") != date(2025, 1, 3))
    with no_spin():
        assert factors.fit_handicap_score(clean, features=FEATURES).coef["x_good"] > 0
