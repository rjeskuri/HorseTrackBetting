"""Newcomer and maiden members at the final price (ticket #438, epic #73; N7, E34, E37).

Contract pinned here (`horsetrack.countries.japan.studies.debut_members`, imported as `dm`):

- Races. `select_races(frame) -> pd.DataFrame`: the rows of `frame` (one per race or per runner,
  with `race_class` and `race_date`, the local race date as datetime64) that enter the study, with
  their values and index labels unchanged: `race_class` exactly "DEBUT" or "MAIDEN" (the store's
  codes, `jravan/races.py`; `research/pool_scan.py`'s newcomer and maiden groups), on a
  development day (`core.eval.holdout.is_holdout_day` false) on or after 2014-06-07 (D10: the
  final price enters every scored race). Every other class, a null class, a holdout day and an
  earlier day stay out.
- Members. `member_features(frame, *, at_utc_col, members) -> pd.DataFrame`, one row per frame
  row on `frame.index`. A built member's features are the columns named `f"{member}__{name}"`; a
  column without such a prefix is not a feature (the output may carry the frame's own columns).
  A value not known at the row's decision time t = `frame[at_utc_col]` (tz-aware UTC) is NaN in
  every one of its member's columns. The frame carries each requested member's raw inputs:
  - "body_weight": `post_utc` (tz-aware UTC, NaT where the post time is missing), `horse_wt_kg`
    and `horse_wt_change_kg` as stored. Its features `body_weight__horse_wt_kg` and
    `body_weight__horse_wt_change_kg` are those values where post_utc - 60 min <= t (#227,
    `config/availability.yaml`; a gate is open at its own time), NaN otherwise and wherever
    post_utc is NaT.
  - "workouts_jravan" (#229): `race_date`, `workout_date` (datetime64, NaT for none: the newest
    workout the runner's #229 features draw on), `ff_best_vs_par_s` and `ff_trend_s`. Its
    features `workouts_jravan__ff_best_vs_par_s` and `workouts_jravan__ff_trend_s` are those
    values where the workout counts at t by #229's rule (owner, 2026-11-27): dated before the race
    day, and published, at (workout_date + 1 day) 00:00 Asia/Tokyo, strictly before t. NaN
    otherwise, a NaT workout date included.
  - "jvdata_mining" (#406, not landed): a known member that builds no column and needs no input.
  An unknown member name is a ValueError or KeyError.
- Pedigree prior. `pedigree_prior(runs, *, as_of) -> pd.DataFrame`. `runs` holds whole fields:
  race_id, race_date (datetime64), horse_id, sire_id, surface ("turf" or "dirt"), distance_m,
  finish_pos and field_size. One row per (`sire_id`, `surface`, `distance_band`) cell with its
  `prior`, a float: above 0 when the sire's offspring did better than their fields in the cell,
  below 0 when worse, shrunk toward 0 so that it matters (two straight wins give less than 90% of
  the prior forty straight wins give).
  Only runs with race_date strictly before `as_of` count (a past run is known from the next day,
  availability.yaml `horse_runs`); a cell with no such run has no row or prior 0. The banding is
  the coder's, but 1200 m and 2400 m fall in different bands.
- Offset. `offset_probabilities(frame, weights, *, offset_col) -> pd.Series`, P(win) per row on
  `frame.index`: the softmax within `race_id` of u = frame[offset_col] + sum over `weights`
  (column -> weight) of weight * frame[column]. The offset's coefficient is 1 and only
  differences within a race matter, so ln(1 / final odds) gives the normalised market. With every
  weight 0, or no weights, it is the offset's own probabilities whatever the member values, NaN
  included.
- Spec. `build_spec(*, seed) -> dict`: the family's pre-registered spec, plain JSON that
  `Registry.freeze_family(FAMILY_ID, spec)` accepts, built from the repository's config and the
  module's constants alone (it reads no data dir). `spec["members"]` maps every member name to a
  dict with `status` ("built" or "missing") and the member's `ticket` (an int): "body_weight" and
  "workouts_jravan" are built; "jvdata_mining" is missing with ticket 406 (listed, left out, and
  added as a counted variant when it lands).
- Result and public report. `DebutResult` is a dataclass that takes `spec` (as `build_spec` gives
  it) and `study` (`studies.calibration.Study`: counts, tables and variants with their registry
  metrics) by keyword; every other field has a default. `render_public_report(result) -> str`,
  the text of the public `docs/experiments/debut-members-japan.md`, names `FAMILY_ID`, every
  member of `spec["members"]` and every variant (a failed one too), says `withheld (D21)`, and
  shows no figure of the study (D21): no count, rate, span (a fold or data date), sign or bound,
  in any usual format, from its counts, tables, metrics or error messages. It depends on no
  figure at all (nor on a clock): a result whose study differs only in its figures, signs
  flipped, renders the very same text.

Review round 1 (PR #559) pins five more behaviours:

- Late Win boards. `late_win(frame) -> pd.DataFrame` on `frame.index` (the study frame's rows:
  race_id, race_date, horse_id, horse_no, `post_utc` tz-aware UTC from the JRA store, and the
  decision time), with the columns `late_win_move_t20` and `late_win_move_t5`: per runner
  ln s(T-2) - ln s(T-m), s the normalised 1/odds of the race's Win board at T-x = the frame's own
  `post_utc` - x minutes. The board at T-x is each runner's latest WIN row with `snapshot_utc` at
  or before T-x (fetched_utc null) in the race day's jravan odds_timeseries file
  (`processed/jravan/odds_timeseries/<YYYY-MM-DD>.parquet` under the data dir, in
  `logger.compact.SCHEMA`). The file's own `post_utc` and `minutes_to_post` are JRA-VAN's and
  never set a cutoff: with JRA at 12:00 and JRA-VAN at 12:02, the 11:59 snapshot is not on the T-2
  board. The cutoffs are the same when JRA-VAN's post is earlier or missing.
- Win settlement. `settle_win(starters, dividends) -> pd.Series`, the Win return per unit stake
  per row on `starters.index`. `starters` has race_id, horse_no (int), won (bool) and win_odds
  (the final odds on the board). `dividends` holds the store's dividends rows (race_id, pool,
  combination, dividend per 100 yen, is_refund) of every pool. A winner gets its own WIN
  dividend / 100 and a loser 0.0, except that every runner of a race in which any winner (a
  dead heat's too) has no WIN dividend row gets NaN. Such a race is excluded: it is not settled
  at the final odds, nor by a PLACE row, and it is not a loss.
- N5 exclusion. In `compute_study`, a race whose `payout` is NaN (one `settle_win` excludes)
  stays out of its variants' N5 curve (`tables["n5_curve"]`, segment "all": the bets and their
  returns are the scored runners of the other races). The variant still scores, its ranked row's
  `races` still counts every scored race, and that row gives the scored races left out of the
  curve as `n5_excluded_races`.
- Exit status. `horsetrack japan debut-members` exits nonzero when no variant scored (every one
  failed, or none ran) and 0 when one did. Even then it still writes the failure records, one per
  planned variant, and the private report. `run_debut_members` returns its result in that case
  rather than raising. The command calls this module's `run_debut_members`, and the run reads the
  store only through this module's `load_inputs(date_from, date_to, log)`, which returns (frame,
  counts, failed members, builds). These are the seams the tests replace.
- Report identity. Two runs that differ in `date_from`, `date_to` or an input build write their
  private reports (the `.md` at `DebutResult.report_path` and its `.json` beside it, same stem) to
  different paths, so a later run never overwrites an earlier one.

Review round 2 (PR #559) pins one more:

- Aborted read. When the run's input read (`load_inputs` raises, as with the dividends parquet
  missing or no store at all) or its fold split (`walk_forward` on the frame's `race_date`, e.g.
  a race without a date) aborts, `run_debut_members` with the default registry (the command's
  run) still appends one failed record per planned variant to the default registry and writes
  the private report, and the command exits nonzero; but the committed public report is not
  rewritten: an existing one stays byte for byte as it was. A run that reads its inputs and
  scores still rewrites it. Seam: the public report's path is `env.repo_root() / PUBLIC_REPORT`,
  with `repo_root` looked up on `horsetrack.core.env` at call time (as `registry.summary_dir`
  does, tests/core/test_env.py), so `monkeypatch.setattr("horsetrack.core.env.repo_root", ...)`
  sends the public report and the committed summary to a scratch root. The data dir, the default
  registry and the private reports stay under it (tmp_path/data, outside that scratch root).

Not pinned here (review checks them): the members' feature engineering beyond the cases above, the
fit's handling of missing values, the private report's layout, and `run_debut_members`' store
reads.

Every frame is synthetic and every RNG seeded. Expected values come from first principles (the
availability rules written out below, a softmax, hand-built sire records), never from the code
under test; the holdout days come from `is_holdout_day` on the committed seed. An autouse guard
points the data dir at tmp_path and fails a test that changes the committed registry summary or a
holdout log.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.core.eval.folds import FoldScheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import FoldMetric, Registry
from horsetrack.core.eval.tail import ALL as ALL_BETS
from horsetrack.countries.japan.jravan.races import CONDITION_CLASSES, GRADES
from horsetrack.countries.japan.logger.compact import SCHEMA as BOARD_SCHEMA
from horsetrack.countries.japan.research.pool_scan import CLASS_GROUPS
from horsetrack.countries.japan.studies import debut_members as dm
from horsetrack.countries.japan.studies.calibration import Study, Variant

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_SUMMARY = REPO_ROOT / "docs" / "experiments" / "registry"
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"
DATA_DIR_VAR = "HORSETRACK_DATA_DIR"
JST = ZoneInfo("Asia/Tokyo")
AT = "decision_utc"  # the decision-time column: a name no default would pick
PRICE_ERA = date(2014, 6, 7)  # D10
ENTERING = {"DEBUT", "MAIDEN"}
WITHHELD = "withheld (D21)"
BODY, WORKOUTS, MINING = "body_weight", "workouts_jravan", "jvdata_mining"
SEED = 438


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """The data directory is under tmp_path; no test changes the committed summary or a log."""
    monkeypatch.setenv(DATA_DIR_VAR, str(tmp_path / "data"))

    def snapshot() -> dict[Path, bytes]:
        files = [*COMMITTED_SUMMARY.rglob("*"), *REAL_COUNTRIES.glob("*/holdout_log.jsonl")]
        return {p: p.read_bytes() for p in files if p.is_file()}

    before = snapshot()
    yield
    assert snapshot() == before, "a test changed the committed registry summary or a holdout log"


def jst(day: date, clock: time) -> pd.Timestamp:
    """The UTC instant of a wall-clock time on a JST date."""
    return pd.Timestamp(datetime.combine(day, clock, tzinfo=JST)).tz_convert("UTC")


def relabel(frame: pd.DataFrame, rng: np.random.Generator, start: int) -> pd.DataFrame:
    """The rows shuffled, on unique labels that are neither positions nor sorted."""
    order = rng.permutation(len(frame))
    return frame.iloc[order].set_axis(pd.Index(start + 7 * order))


# --------------------------------------------------------------------------- races that enter


def holdout_and_development_days(n: int = 3) -> tuple[list[date], list[date]]:
    """The first n holdout days and n development days from the holdout start (2026-11-28)."""
    held: list[date] = []
    dev: list[date] = []
    day = date(2026, 11, 28)
    while len(held) < n or len(dev) < n:
        (held if is_holdout_day("japan", day) else dev).append(day)
        day += timedelta(days=1)
    return held[:n], dev[:n]


def test_only_newcomer_and_maiden_races_enter():
    """Every class the store writes (and a null one) on days before, at and after D10's
    2014-06-07 and on holdout and development days from 2026-11-28, with 1-3 runner rows per
    race: exactly the DEBUT and MAIDEN rows on development days from 2014-06-07 come back,
    unchanged and on their own labels, every runner of such a race with them."""
    classes = sorted(set(CONDITION_CLASSES.values()) | set(GRADES.values()))
    assert {c for c, g in CLASS_GROUPS.items() if g in ("newcomer", "maiden")} == ENTERING
    assert ENTERING < set(classes)
    held, late_dev = holdout_and_development_days()
    days = [
        date(2011, 2, 5),
        date(2014, 6, 1),
        date(2014, 6, 6),
        PRICE_ERA,
        date(2014, 6, 8),
        date(2019, 9, 14),
        date(2026, 11, 22),
        *held,
        *late_dev,
    ]
    rng = np.random.default_rng(4381)
    rows = []
    for d_i, day in enumerate(days):
        for c_i, race_class in enumerate([*classes, None]):
            race_id = f"jra:{day.isoformat()}:QQV:{c_i:02d}"
            for slot in range(1 + (d_i + c_i) % 3):
                rows.append(
                    {
                        "race_id": race_id,
                        "race_date": pd.Timestamp(day),
                        "race_class": race_class,
                        "horse_id": f"{race_id}:{slot}",
                        "payload": float(rng.normal()),
                    }
                )
    frame = relabel(pd.DataFrame(rows), rng, 9000)

    enters = np.array(
        [
            r.race_class in ENTERING
            and r.race_date.date() >= PRICE_ERA
            and not is_holdout_day("japan", r.race_date.date())
            for r in frame.itertuples()
        ]
    )
    # the fixture holds every way to stay out, for both entering classes
    left_out = frame.loc[~enters & frame["race_class"].isin(list(ENTERING)).to_numpy()]
    assert {d.date() for d in left_out["race_date"]} >= {date(2014, 6, 6), *held}
    assert set(frame.loc[enters, "race_class"]) == ENTERING

    got = dm.select_races(frame)

    assert set(got.index) == set(frame.index[enters])
    pd.testing.assert_frame_equal(
        got.loc[:, list(frame.columns)].sort_index(), frame.loc[enters].sort_index()
    )


# --------------------------------------------------------------------------- members as of t

RACES = (  # race_id, race day, post time (JST)
    ("jra:2021-07-03:QQV:05", date(2021, 7, 3), time(12, 25)),
    ("jra:2022-01-15:QQV:01", date(2022, 1, 15), time(9, 55)),
    ("jra:2023-10-08:QQV:11", date(2023, 10, 8), time(15, 40)),
    ("jra:2024-05-26:QQV:03", date(2024, 5, 26), time(11, 5)),
)
NO_POST = ("jra:2025-03-02:QQV:07", date(2025, 3, 2))  # a race whose post time is missing
BODY_OFFSETS = (  # decision times around the body-weight gate at post - 60 min
    -timedelta(hours=2),
    -timedelta(minutes=61),
    -timedelta(minutes=60, seconds=1),
    -timedelta(minutes=60),
    -timedelta(minutes=59, seconds=59),
    -timedelta(minutes=30),
    -timedelta(minutes=2),
    timedelta(minutes=5),
)
CYCLE = (-6, -1, 1, 0, None, -3)  # workout dates (days from race day) on the body-weight rows


def workout_cases(day: date, post: pd.Timestamp) -> tuple:
    """(workout date in days from race day, decision time) at the edges of #229's rule."""
    later = jst(day + timedelta(days=4), time(9, 0))  # a read days after the race
    return (
        (1, post + timedelta(hours=2)),  # dated after the race, read after it too
        (3, post - timedelta(minutes=30)),  # dated after the race
        (1, later),  # dated after the race and published before t: the date alone keeps it out
        (3, later),
        (0, post - timedelta(minutes=30)),  # race day itself: only a race on a later day counts
        (0, later),  # ... even once it is published
        (-1, jst(day, time(9, 30))),  # the eve's workout, published at 00:00 JST of race day
        (-1, jst(day, time(0, 0))),  # ... which is not strictly before 00:00 itself
        (-1, jst(day - timedelta(days=1), time(23, 0))),  # ... nor known on the eve
        (-6, post - timedelta(minutes=60)),
        (-2, post - timedelta(minutes=90)),
        (-3, later),
        (None, post - timedelta(minutes=30)),  # no workout date: never counts
    )


def runner_frame(seed: int = 4382) -> pd.DataFrame:
    """Runner rows of five races, each with its own decision time, body weight and workout."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def add(race_id: str, day: date, post, at, wd) -> None:
        n = len(rows)
        rows.append(
            {
                "race_id": race_id,
                "horse_id": f"h{n:04d}",
                "race_date": pd.Timestamp(day),
                "post_utc": post,
                AT: at,
                "horse_wt_kg": np.nan if n % 13 == 5 else float(420 + 3 * n),
                "horse_wt_change_kg": float((7 * n) % 21 - 10),
                "workout_date": pd.NaT if wd is None else pd.Timestamp(day + timedelta(days=wd)),
                "ff_best_vs_par_s": float(rng.normal(0.0, 0.6)),
                "ff_trend_s": float(rng.normal(0.0, 0.1)),
            }
        )

    for race_id, day, clock in RACES:
        post = jst(day, clock)
        for k, offset in enumerate(BODY_OFFSETS):
            add(race_id, day, post, post + offset, CYCLE[k % len(CYCLE)])
        for wd, at in workout_cases(day, post):
            add(race_id, day, post, at, wd)
    race_id, day = NO_POST
    for k, clock in enumerate((time(0, 0), time(9, 0), time(12, 0), time(16, 30))):
        add(race_id, day, pd.NaT, jst(day, clock), CYCLE[k])
    add(race_id, day, pd.NaT, jst(day + timedelta(days=1), time(9, 0)), -4)

    frame = pd.DataFrame(rows)
    for col in ("post_utc", AT):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    for col in ("race_date", "workout_date"):
        frame[col] = pd.to_datetime(frame[col])
    return relabel(frame, rng, 300)


def body_weight_known(frame: pd.DataFrame) -> np.ndarray:
    """#227: a body weight is in the views from post_utc - 60 min; a missing post opens nothing."""
    gate = frame["post_utc"] - pd.Timedelta(minutes=60)
    return (frame["post_utc"].notna() & (gate <= frame[AT])).to_numpy(dtype=bool)


def workout_published(frame: pd.DataFrame) -> np.ndarray:
    """#229: a workout is published at (its date + 1 day) 00:00 JST, known strictly after."""
    out = []
    for row in frame.itertuples():
        if pd.isna(row.workout_date):
            out.append(False)
            continue
        published = jst(row.workout_date.date() + timedelta(days=1), time(0, 0))
        out.append(published < getattr(row, AT))
    return np.array(out, dtype=bool)


def workout_counts(frame: pd.DataFrame) -> np.ndarray:
    """#229: dated before race day and published before t."""
    dated_before = (frame["workout_date"] < frame["race_date"]).to_numpy(dtype=bool)
    return dated_before & workout_published(frame)


def member_columns(out: pd.DataFrame, member: str) -> list[str]:
    return [c for c in out.columns if isinstance(c, str) and c.startswith(f"{member}__")]


def test_member_features_use_only_information_at_t():
    """Five races, each runner with its own decision time: a body weight is absent before
    post - 60 min (and wherever the post is missing) and present, as stored, from then on; a
    workout dated after the race (or on race day, or not yet published at t) is absent, and an
    earlier one published before t is present, its values as given."""
    frame = runner_frame()
    body, workout = body_weight_known(frame), workout_counts(frame)
    for known in (body, workout):  # the fixture has both sides of each rule in numbers
        assert known.sum() >= 10 and (~known).sum() >= 10

    out = dm.member_features(frame, at_utc_col=AT, members=[BODY, WORKOUTS])

    assert out.index.equals(frame.index)
    body_cols, workout_cols = member_columns(out, BODY), member_columns(out, WORKOUTS)
    assert {f"{BODY}__horse_wt_kg", f"{BODY}__horse_wt_change_kg"} <= set(body_cols)
    assert {f"{WORKOUTS}__ff_best_vs_par_s", f"{WORKOUTS}__ff_trend_s"} <= set(workout_cols)

    # body weight: nothing before its gate, the stored value from it on
    assert out.loc[~body, body_cols].isna().all().all(), "a body weight leaked before post - 60"
    for col in ("horse_wt_kg", "horse_wt_change_kg"):
        got = out.loc[body, f"{BODY}__{col}"].to_numpy(dtype=float)
        np.testing.assert_array_equal(got, frame.loc[body, col].to_numpy(dtype=float))

    # workouts: nothing that does not count at t, the given values where it does
    assert out.loc[~workout, workout_cols].isna().all().all(), "a workout leaked into t"
    # the date rule on its own: workouts dated on or after race day, published before t
    late = (frame["workout_date"] >= frame["race_date"]).to_numpy(dtype=bool)
    assert (late & workout_published(frame)).sum() >= 12 and not (late & workout).any()
    for col in ("ff_best_vs_par_s", "ff_trend_s"):
        got = out.loc[workout, f"{WORKOUTS}__{col}"].to_numpy(dtype=float)
        np.testing.assert_allclose(got, frame.loc[workout, col].to_numpy(dtype=float))


# --------------------------------------------------------------------------- the pedigree prior

CELLS = (("turf", 1200), ("turf", 2400), ("dirt", 1200), ("dirt", 1800))
FIELD = 8


def saturdays(start: date, n: int) -> list[date]:
    return [start + timedelta(weeks=k) for k in range(n)]


def runs_frame(days: list[date], seed: int, special=None, n_fillers: int = 24) -> pd.DataFrame:
    """One race per cell per day, FIELD runners each (every run a different horse), sires drawn
    from fillers F00.., finishing order random. `special(k, cell)` gives (winner's sire, last's
    sire) for the cell's k-th race, either None to keep the drawn sire."""
    rng = np.random.default_rng(seed)
    fillers = np.array([f"F{i:02d}" for i in range(n_fillers)], dtype=object)
    rows = []
    for k, day in enumerate(days):
        for surface, distance in CELLS:
            race_id = f"jra:{day.isoformat()}:QQV:{surface[0]}{distance}"
            sires = list(rng.choice(fillers, size=FIELD))
            positions = rng.permutation(FIELD) + 1
            winner, last = (None, None) if special is None else special(k, (surface, distance))
            for slot in range(FIELD):
                if winner is not None and positions[slot] == 1:
                    sires[slot] = winner
                if last is not None and positions[slot] == FIELD:
                    sires[slot] = last
                rows.append(
                    {
                        "race_id": race_id,
                        "race_date": pd.Timestamp(day),
                        "horse_id": f"{race_id}:{slot}",
                        "sire_id": sires[slot],
                        "surface": surface,
                        "distance_m": distance,
                        "finish_pos": int(positions[slot]),
                        "field_size": FIELD,
                    }
                )
    return pd.DataFrame(rows)


def priors(out: pd.DataFrame) -> pd.Series:
    """The prior per cell, keyed 'sire|surface|band' and sorted."""
    key = (
        out["sire_id"].astype(str)
        + "|"
        + out["surface"].astype(str)
        + "|"
        + out["distance_band"].astype(str)
    )
    assert key.is_unique, "one row per sire x surface x distance band"
    return pd.Series(out["prior"].to_numpy(dtype=float), index=key.to_numpy()).sort_index()


def assert_same_priors(a: pd.Series, b: pd.Series) -> None:
    pd.testing.assert_series_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


def test_pedigree_prior_uses_earlier_runs_only():
    """As of a race day D, a sire's prior is the same whatever happens on or after D (results
    reversed, races added, a sire seen only from D on) and the same as from the earlier runs
    alone; it moves when an earlier run of its offspring changes, and once D has passed, D's
    runs count."""
    days = saturdays(date(2013, 1, 5), 160)
    runs = runs_frame(days, seed=4383)
    as_of = days[110]
    day = runs["race_date"].dt.date
    future = (day >= as_of).to_numpy()

    base = priors(dm.pedigree_prior(runs, as_of=as_of))
    assert len(base) >= 24

    # on or after D: every finish reversed, a copy of each race, and a sire new from D on
    changed = runs.copy()
    changed.loc[future, "finish_pos"] = FIELD + 1 - changed.loc[future, "finish_pos"]
    changed.loc[future & (changed["finish_pos"] == 1).to_numpy(), "sire_id"] = "NEW"
    copies = runs.loc[future].assign(
        race_id=lambda f: f["race_id"] + ":copy", horse_id=lambda f: f["horse_id"] + ":copy"
    )
    changed = pd.concat([changed, copies], ignore_index=True)
    got = dm.pedigree_prior(changed, as_of=as_of)
    new = got["sire_id"].eq("NEW").to_numpy()
    assert np.all(got.loc[new, "prior"].to_numpy(dtype=float) == 0.0)
    assert_same_priors(priors(got.loc[~new]), base)
    assert_same_priors(priors(dm.pedigree_prior(runs.loc[~future], as_of=as_of)), base)

    # an earlier run changes: the sire with the most earlier turf-1200 lasts (in races another
    # sire won) now wins those races, and its winner finishes last
    winners = runs.loc[runs["finish_pos"].eq(1).to_numpy()].set_index("race_id")["sire_id"]
    early_turf_1200 = (
        ~future & runs["surface"].eq("turf").to_numpy() & runs["distance_m"].eq(1200).to_numpy()
    )
    lasts = runs.loc[early_turf_1200 & runs["finish_pos"].eq(FIELD).to_numpy()]
    lasts = lasts.loc[lasts["sire_id"].to_numpy() != winners.reindex(lasts["race_id"]).to_numpy()]
    sire = lasts["sire_id"].value_counts().idxmax()
    races = lasts.loc[lasts["sire_id"] == sire, "race_id"].head(4)
    assert len(races) >= 3
    better = runs.copy()
    for race_id in races:
        in_race = better["race_id"].eq(race_id).to_numpy()
        lose = in_race & better["sire_id"].eq(sire).to_numpy() & better["finish_pos"].eq(FIELD)
        win = in_race & better["finish_pos"].eq(1).to_numpy()
        better.loc[lose, "finish_pos"], better.loc[win, "finish_pos"] = 1, FIELD
    moved = priors(dm.pedigree_prior(better, as_of=as_of))
    turf = [k for k in base.index if k.startswith(f"{sire}|turf|")]
    assert turf and max(moved[k] - base[k] for k in turf) > 1e-6, "earlier runs must count"

    # a week later, D's (reversed) runs are earlier runs and count
    week = as_of + timedelta(days=7)
    plain = priors(dm.pedigree_prior(runs, as_of=week))
    later = priors(dm.pedigree_prior(changed, as_of=week))
    common = plain.index.intersection(later.index)
    assert len(common) >= 24 and np.abs(later[common] - plain[common]).max() > 1e-6


def test_pedigree_prior_is_shrunk_per_sire_surface_and_band():
    """Hand-built sire records among random fillers: a sire whose offspring always win is above
    0 and one whose offspring always finish last below; 40 straight wins are above 2; a sire
    that wins on turf and finishes last on dirt splits by surface; one that wins at 1200 m and
    finishes last at 2400 m on turf splits by band; before any run nothing is known."""

    def special(k: int, cell: tuple[str, int]):
        if cell == ("turf", 1200):  # wins in races 0-39, 40-41, 42-61 and 62-81
            ends = (("SW40", 40), ("SW2", 42), ("SX", 62), ("SY", 82))
            winner = next((sire for sire, end in ends if k < end), None)
            return winner, "SL40" if k < 40 else None
        if cell == ("dirt", 1200):
            return None, "SX" if k < 20 else None
        if cell == ("turf", 2400):
            return None, "SY" if k < 20 else None
        return None, None

    days = saturdays(date(2013, 1, 5), 100)
    runs = runs_frame(days, seed=4384, special=special)
    out = dm.pedigree_prior(runs, as_of=days[-1] + timedelta(days=7))

    def of(sire: str, surface: str | None = None) -> np.ndarray:
        rows = out["sire_id"].eq(sire)
        if surface is not None:
            rows &= out["surface"].eq(surface)
        return out.loc[rows, "prior"].to_numpy(dtype=float)

    assert len(of("SW40")) and len(of("SW2")) and len(of("SL40"))
    assert np.all(of("SW2") > 0) and np.all(of("SL40") < 0)
    # shrinkage that matters: two straight wins give well under what forty give
    assert 0 < of("SW2").max() < 0.9 * of("SW40").max(), "more runs at one record shrink less"
    assert len(of("SX", "turf")) and np.all(of("SX", "turf") > 0)
    assert len(of("SX", "dirt")) and np.all(of("SX", "dirt") < 0)
    sy_turf = of("SY", "turf")
    assert sy_turf.min() < 0 < sy_turf.max(), "1200 m and 2400 m are different bands"

    before = dm.pedigree_prior(runs, as_of=days[0])
    assert np.all(before["prior"].to_numpy(dtype=float) == 0.0)


# --------------------------------------------------------------------------- the offset


def market_frame(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """200 races of 1 to 18 runners priced at a 20% takeout (odds to 0.1, at least 1.0), the
    offset ln(1 / odds) unnormalised, two member columns with some values missing; and the
    normalised market (1 / odds over the race's sum) per row, from first principles."""
    rng = np.random.default_rng(seed)
    sizes = np.concatenate([[1, 2, 18], rng.integers(3, 19, size=197)])
    codes = np.repeat(np.arange(len(sizes)), sizes)
    g = rng.gamma(0.8, size=len(codes))
    q = g / np.bincount(codes, weights=g)[codes]
    odds = np.maximum(np.round(0.8 / q, 1), 1.0)
    n = len(codes)
    frame = pd.DataFrame(
        {
            "race_id": np.array([f"jra:r{c:04d}" for c in codes], dtype=object),
            "win_odds": odds,
            "ln_price": np.log(1.0 / odds),
            "m1": np.where(rng.random(n) < 0.15, np.nan, rng.normal(size=n)),
            "m2": np.where(rng.random(n) < 0.10, np.nan, rng.normal(size=n)),
        }
    )
    inv = 1.0 / odds
    market = inv / np.bincount(codes, weights=inv)[codes]
    order = rng.permutation(n)
    frame = frame.iloc[order].set_axis(pd.Index([f"row{i:05d}" for i in 3 * order + 11]))
    return frame, market[order]


def softmax_within(u: np.ndarray, race_id: np.ndarray) -> np.ndarray:
    codes = pd.factorize(race_id)[0]
    top = pd.Series(u).groupby(codes).transform("max").to_numpy()
    e = np.exp(u - top)
    return e / np.bincount(codes, weights=e)[codes]


def on_frame(got, frame: pd.DataFrame) -> np.ndarray:
    assert isinstance(got, pd.Series)
    assert got.index.sort_values().equals(frame.index.sort_values())
    return got.reindex(frame.index).to_numpy(dtype=float)


def test_zero_weights_reproduce_the_offset():
    """With the final-price offset and every member weight 0 (or none given), the conditional
    logit gives the market's normalised probabilities exactly, on rows of any race order and
    whatever the member values, missing ones included; with other weights it gives the softmax
    of offset + weights x members, so the market alone is not the answer."""
    frame, market = market_frame(4385)
    races = frame["race_id"].to_numpy(dtype=object)

    for weights in ({"m1": 0.0, "m2": 0.0}, {"m1": 0.0}, {}):
        p = on_frame(dm.offset_probabilities(frame, weights, offset_col="ln_price"), frame)
        np.testing.assert_allclose(p, market, rtol=0, atol=1e-12)
        sums = pd.Series(p).groupby(races).sum().to_numpy()
        np.testing.assert_allclose(sums, 1.0, rtol=0, atol=1e-12)

    filled = frame.fillna({"m1": 0.37, "m2": -0.81})
    offset = filled["ln_price"].to_numpy()
    for w1, w2 in ((0.7, -1.3), (-0.4, 0.0), (2.1, 0.5)):
        want = softmax_within(
            offset + w1 * filled["m1"].to_numpy() + w2 * filled["m2"].to_numpy(), races
        )
        got = dm.offset_probabilities(filled, {"m1": w1, "m2": w2}, offset_col="ln_price")
        p = on_frame(got, filled)
        np.testing.assert_allclose(p, want, rtol=0, atol=1e-12)
        assert np.abs(p - market).max() > 1e-3


# --------------------------------------------------------------------------- a missing member


def test_a_missing_member_is_listed_not_built(tmp_path):
    """#406 has not landed: the spec lists its member as missing (ticket 406) beside the built
    ones and freezes as plain JSON; member_features builds no column for it, alongside a built
    member or alone, and an unknown member name is refused rather than skipped."""
    spec = dm.build_spec(seed=SEED)
    members = spec["members"]
    assert members[MINING]["status"] == "missing" and members[MINING]["ticket"] == 406
    assert members[BODY]["status"] == "built" and members[WORKOUTS]["status"] == "built"
    assert all(m["status"] in ("built", "missing") for m in members.values())
    assert all(isinstance(m["ticket"], int) for m in members.values())
    registry = Registry(tmp_path / "registry.jsonl", holdout_log=tmp_path / "no.jsonl")
    assert registry.freeze_family(dm.FAMILY_ID, spec) == registry.frozen_spec_hash(dm.FAMILY_ID)

    frame = runner_frame().drop(
        columns=["workout_date", "ff_best_vs_par_s", "ff_trend_s"]
    )  # only body weight's inputs: none for the missing member
    out = dm.member_features(frame, at_utc_col=AT, members=[BODY, MINING])
    assert out.index.equals(frame.index)
    assert member_columns(out, BODY)
    assert not [c for c in out.columns if str(c).startswith(MINING)]

    alone = dm.member_features(frame, at_utc_col=AT, members=[MINING])
    assert len(alone) == len(frame)
    assert not [c for c in alone.columns if str(c).startswith(MINING)]
    assert not member_columns(alone, BODY)

    with pytest.raises((ValueError, KeyError)):
        dm.member_features(frame, at_utc_col=AT, members=[BODY, "no_such_member"])


# --------------------------------------------------------------------------- the public report

# Two studies with the same variants (names, hypotheses, statuses) and every figure different:
# counts, weights and their bounds (signs flipped), gains, the tail curve, fold and data dates,
# and the number in a failed variant's error.
FIGURES_A = {
    "races": 48213,
    "starters": 391837,
    "newcomer_races": 20619,
    "maiden_races": 27594,
    "w": 0.61834,
    "w_lo": 0.27346,
    "w_hi": 0.94082,
    "gain": 0.019373,
    "gain_lo": 0.004826,
    "fold_gain": 0.034517,
    "fold_n": 1453,
    "pooled_w": -0.28463,
    "pooled_lo": -0.53917,
    "pooled_hi": -0.03176,
    "enc": 0.47261,
    "enc_lo": 0.13492,
    "enc_hi": 0.81027,
    "tail": 1.18734,
    "tail_lo": 0.76452,
    "tail_hi": 1.52189,
    "tail_n": 3817,
    "tail_m": 0.70519,
    "tail_m_lo": 0.52083,
    "tail_m_hi": 0.88466,
    "tail_m_n": 9127,
    "error_n": 7391,
    "fold": "2017-04-01",
    "error_fold": "2022-07-01",
    "first": "2014-06-14",
    "last": "2026-10-25",
}
FIGURES_B = {
    "races": 51772,
    "starters": 402916,
    "newcomer_races": 19385,
    "maiden_races": 32387,
    "w": -0.41275,
    "w_lo": -0.70131,
    "w_hi": -0.12419,
    "gain": -0.008214,
    "gain_lo": -0.021577,
    "fold_gain": -0.027381,
    "fold_n": 1688,
    "pooled_w": 0.35129,
    "pooled_lo": 0.07724,
    "pooled_hi": 0.62533,
    "enc": -0.16648,
    "enc_lo": -0.43316,
    "enc_hi": 0.11873,
    "tail": 0.83391,
    "tail_lo": 0.61207,
    "tail_hi": 1.05575,
    "tail_n": 2958,
    "tail_m": 1.09442,
    "tail_m_lo": 0.86935,
    "tail_m_hi": 1.31949,
    "tail_m_n": 11204,
    "error_n": 6125,
    "fold": "2019-10-01",
    "error_fold": "2024-01-01",
    "first": "2015-02-21",
    "last": "2025-12-28",
}
FOLD_DATES = ("fold", "error_fold")  # also shown as quarters, e.g. 2017-Q2


def planted_study(f: dict) -> Study:
    """A study whose every count, metric, bound, fold date, table cell and error carries a
    figure of `f`, which the public report must not show."""

    def variant(hypothesis: str, name: str, metrics=(), error=None) -> Variant:
        return Variant(
            hypothesis_id=hypothesis,
            name=name,
            fold_scheme="D9 walk-forward quarters",
            components={"offset": "final Win price"},
            params={"race_class": name.rsplit(":", 1)[-1]},
            metrics=tuple(metrics),
            error=error,
        )

    return Study(
        counts={k: f[k] for k in ("races", "starters", "newcomer_races", "maiden_races")},
        tables={
            "ranked": [
                {
                    "variant": "member:pedigree_prior:DEBUT",
                    "weight": f["w"],
                    "weight_lo": f["w_lo"],
                    "weight_hi": f["w_hi"],
                    "delta_ll": f["gain"],
                    "delta_ll_lo": f["gain_lo"],
                    "encompassing_weight": f["enc"],
                    "n_races": f["newcomer_races"],
                }
            ],
            "n5_curve": [
                {"segment": "DEBUT", "realised": f["tail"], "lo": f["tail_lo"]},
                {"segment": "DEBUT", "hi": f["tail_hi"], "n_bets": f["tail_n"]},
                {"segment": "MAIDEN", "realised": f["tail_m"], "lo": f["tail_m_lo"]},
                {"segment": "MAIDEN", "hi": f["tail_m_hi"], "n_bets": f["tail_m_n"]},
            ],
            "span": [{"first_race_date": f["first"], "last_race_date": f["last"]}],
        },
        variants=[
            variant(
                "E37",
                "member:pedigree_prior:DEBUT",
                (
                    FoldMetric("all_folds", "weight", f["w"], f["w_lo"], f["w_hi"], f["fold_n"]),
                    FoldMetric("all_folds", "delta_ll", f["gain"], f["gain_lo"], None, f["races"]),
                    FoldMetric(f["fold"], "delta_ll", f["fold_gain"], None, None, f["fold_n"]),
                ),
            ),
            variant(
                "E34",
                "member:pooled:MAIDEN",
                (
                    FoldMetric(
                        "all_folds",
                        "weight",
                        f["pooled_w"],
                        f["pooled_lo"],
                        f["pooled_hi"],
                        f["maiden_races"],
                    ),
                    FoldMetric(
                        "all_folds", "encompassing", f["enc"], f["enc_lo"], f["enc_hi"], f["tail_n"]
                    ),
                ),
            ),
            variant(
                "E37",
                "combined:DEBUT",
                (
                    FoldMetric(
                        "all_folds",
                        "tail_realised",
                        f["tail"],
                        f["tail_lo"],
                        f["tail_hi"],
                        f["tail_m_n"],
                    ),
                ),
            ),
            variant(
                "E34",
                "member:body_weight:MAIDEN",
                error=f"ValueError: delta_ll on {f['error_fold']} is not finite over "
                f"{f['error_n']} races",
            ),
        ],
    )


def significant(text: str) -> int:
    """The significant digits of a number's text (its mantissa, without leading zeros)."""
    return len(re.sub(r"\D", "", text.split("e")[0]).lstrip("0"))


def float_forms(v: float) -> set[str]:
    """The usual ways a report prints v (and its magnitude), each with 2+ significant digits."""
    forms = {repr(v), f"{v:.1%}", f"{v:.2%}", f"{v:+.1%}", f"{v:.3g}", f"{v:.4g}", f"{v:.2e}"}
    for digits in (2, 3, 4):
        forms |= {f"{v:.{digits}f}", f"{v:+.{digits}f}"}
    if abs(v) >= 0.1:
        forms.add(f"{100 * v:.1f}")
    if 0 < v < 1:
        forms.add(f"{v:.3f}"[1:])  # ".618"
    if v < 0:
        forms |= float_forms(-v)
    forms |= {f.replace("-", "\N{MINUS SIGN}") for f in forms}
    return {f for f in forms if significant(f) >= 2}


def int_forms(n: int) -> set[str]:
    forms = {str(n), f"{n:,}", f"{n:_}", f"{n:,}".replace(",", " ")}
    if n >= 10_000:
        forms.add(f"{n / 1000:.1f}k")
    return forms


def figure_forms(f: dict) -> set[str]:
    forms: set[str] = set()
    for key, value in f.items():
        if isinstance(value, float):
            forms |= float_forms(value)
        elif isinstance(value, int):
            forms |= int_forms(value)
        else:
            forms.add(value)
            if key in FOLD_DATES:
                day = date.fromisoformat(value)
                forms.add(f"{day.year}-Q{(day.month - 1) // 3 + 1}")
    return forms


def test_public_report_carries_no_figures():
    """The public report of a study full of planted figures names the family, every member
    (the missing one too) and every variant (the failed one too) and says 'withheld (D21)'. It
    shows none of the figures (no count, rate, bound, fold or data date, in any usual format,
    from the counts, tables, metrics or an error message), and no sign either: a study whose
    every figure differs, signs flipped, renders the very same text."""
    spec = dm.build_spec(seed=SEED)
    study = planted_study(FIGURES_A)
    result = dm.DebutResult(spec=spec, study=study)
    text = dm.render_public_report(result)

    assert isinstance(text, str)
    assert WITHHELD in text
    assert dm.FAMILY_ID in text
    for name in spec["members"]:
        assert name in text, f"member {name} is not listed"
    for v in study.variants:
        assert v.name in text, f"variant {v.name} is not listed"

    other = dm.render_public_report(dataclasses.replace(result, study=planted_study(FIGURES_B)))
    for rendered, figures in ((text, FIGURES_A), (other, FIGURES_B)):
        bare = re.sub(r"[0-9a-f]{12,}", " ", rendered)  # a spec hash or git sha is no figure
        shown = sorted(form for form in figure_forms(figures) if form in bare)
        assert not shown, f"the public report shows figures: {shown}"
    assert other == text, "the public report changes with the study's figures"


# --------------------------------------------------------------------------- the late Win boards

# Per race: id, race day, the study's post (JST: the JRA store's post_time_local), the JRA-VAN
# post's offset from it in minutes (None: JRA-VAN has no post time) and the snapshots' times in
# minutes before the study's post (a negative one is after it).
BOARD_RACES = (
    # JRA 12:00, JRA-VAN 12:02: by JRA-VAN's minutes_to_post the 11:59 board is 3 minutes out
    ("jra:2021-07-03:QQV:01", date(2021, 7, 3), time(12, 0), 2,
     (40, 30, 21, 20, 15, 7, 6, 5, 4, 3, 2, 1, -1, -3)),
    ("jra:2021-07-03:QQV:02", date(2021, 7, 3), time(12, 35), 0, (25, 19, 8, 5, 3, 2, 1)),
    ("jra:2021-07-03:QQV:03", date(2021, 7, 3), time(13, 10), -3,
     (30, 22, 18, 9, 6, 4, 3, 2, 1, 0)),
    ("jra:2021-07-04:QQW:05", date(2021, 7, 4), time(11, 20), 7, (27, 20, 14, 13, 5, 2, 1, 0)),
    ("jra:2021-07-04:QQW:06", date(2021, 7, 4), time(14, 45), None, (33, 20, 11, 5, 4, 2, 1)),
)  # fmt: skip
LATE_COLUMNS = {20: "late_win_move_t20", 5: "late_win_move_t5"}
T_NOW = 2  # D2: t = post - 2 min, the late path's latest board


def field_size(k: int) -> int:
    return 6 + k % 4


def write_board_days(data_dir: Path, rng: np.random.Generator) -> dict:
    """Each race day's jravan odds_timeseries file as the builder writes it (logger.compact.SCHEMA):
    every runner's WIN row per snapshot, post_utc the JRA-VAN post and minutes_to_post measured to
    it. Returns the odds per (race, snapshot) as {horse_no: odds}."""
    folder = data_dir / "processed" / "jravan" / "odds_timeseries"
    folder.mkdir(parents=True, exist_ok=True)
    odds_at: dict[tuple[str, int], dict[int, float]] = {}
    by_day: dict[date, list[dict]] = {}
    for k, (race_id, day, clock, shift, schedule) in enumerate(BOARD_RACES):
        post = jst(day, clock)
        jv_post = None if shift is None else post + timedelta(minutes=shift)
        g = rng.gamma(1.2, size=field_size(k))
        base = 0.8 / (g / g.sum())
        for before in schedule:
            snap = post - timedelta(minutes=before)
            odds = np.maximum(np.round(base * np.exp(rng.normal(0.0, 0.3, size=len(g))), 1), 1.0)
            odds_at[(race_id, before)] = {h: float(o) for h, o in enumerate(odds, start=1)}
            for h, o in enumerate(odds, start=1):
                by_day.setdefault(day, []).append(
                    {
                        "race_id": race_id,
                        "venue": race_id.split(":")[2],
                        "race_no": int(race_id.rsplit(":", 1)[1]),
                        "pool": "WIN",
                        "combination": str(h),
                        "snapshot_utc": snap.to_pydatetime(),
                        "fetched_utc": None,
                        "post_utc": None if jv_post is None else jv_post.to_pydatetime(),
                        "minutes_to_post": None
                        if jv_post is None
                        else (jv_post - snap).total_seconds() / 60.0,
                        "odds": float(o),
                        "odds_hi": None,
                        "investment": None,
                        "status_raw": f"{round(10 * o):04d}",
                        "is_final_capture": before == min(schedule),
                        "dividend": None,
                        "is_refund": None,
                    }
                )
    for day, rows in by_day.items():
        rows = [rows[i] for i in rng.permutation(len(rows))]
        table = pa.Table.from_pylist(rows, schema=BOARD_SCHEMA)
        pq.write_table(table, folder / f"{day.isoformat()}.parquet")
    return odds_at


def late_frame(rng: np.random.Generator) -> pd.DataFrame:
    """The study frame's rows of the board races: every runner, the study's post_utc and t."""
    rows = []
    for k, (race_id, day, clock, _, _) in enumerate(BOARD_RACES):
        post = jst(day, clock)
        for h in range(1, field_size(k) + 1):
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": pd.Timestamp(day),
                    "horse_id": f"{race_id}:h{h}",
                    "horse_no": h,
                    "post_utc": post,
                    AT: post - timedelta(minutes=T_NOW),
                }
            )
    frame = pd.DataFrame(rows).astype({"horse_no": "int64"})
    for col in ("post_utc", AT):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return relabel(frame, rng, 500)


def board_pick(schedule: tuple, minutes: int, shift: int = 0) -> int | None:
    """The snapshot the board at `minutes` to post shows: the latest at or before then, measured
    to the study's post (shift 0), or to a post `shift` minutes after it."""
    eligible = [b for b in schedule if b + shift >= minutes]
    return min(eligible) if eligible else None


def shares(odds: dict[int, float]) -> dict[int, float]:
    inv = {h: 1.0 / o for h, o in odds.items()}
    total = sum(inv.values())
    return {h: v / total for h, v in inv.items()}


def expected_moves(frame: pd.DataFrame, odds_at: dict, minutes: int) -> np.ndarray:
    """ln s(T-2) - ln s(T-minutes) per frame row, both boards cut at the study's post."""
    races = {r[0]: r for r in BOARD_RACES}
    out = []
    for row in frame.itertuples():
        schedule = races[row.race_id][4]
        now, then = board_pick(schedule, T_NOW), board_pick(schedule, minutes)
        s_now, s_then = shares(odds_at[(row.race_id, now)]), shares(odds_at[(row.race_id, then)])
        out.append(np.log(s_now[row.horse_no] / s_then[row.horse_no]))
    return np.array(out, dtype=float)


def test_late_win_boards_are_cut_at_the_study_post(tmp_path):
    """The late Win path's T-20, T-5 and T-2 boards are cut at the study's own post_utc (the JRA
    store's), never by the day file's minutes_to_post (JRA-VAN's post): with JRA at 12:00 and
    JRA-VAN at 12:02 the 11:59 snapshot stays off the T-2 board. The same holds when JRA-VAN's post
    is 3 minutes earlier or 7 minutes later, and when JRA-VAN has none. A race whose posts agree is
    the control."""
    rng = np.random.default_rng(4386)
    odds_at = write_board_days(tmp_path / "data", rng)
    frame = late_frame(rng)
    # the fixture tells the posts apart: JRA-VAN's minutes_to_post picks another board in every
    # race but the control
    for _, _, _, shift, schedule in BOARD_RACES:
        moved = shift is None or any(
            board_pick(schedule, m) != board_pick(schedule, m, shift) for m in (T_NOW, 5, 20)
        )
        assert moved == (shift != 0)

    got = dm.late_win(frame)

    assert got.index.sort_values().equals(frame.index.sort_values())
    got = got.reindex(frame.index)
    for minutes, column in LATE_COLUMNS.items():
        want = expected_moves(frame, odds_at, minutes)
        assert np.isfinite(want).all()
        np.testing.assert_allclose(
            got[column].to_numpy(dtype=float), want, rtol=0, atol=1e-9, err_msg=f"T-{minutes}"
        )

    # the reviewer's race: the 11:59 board, a minute after t = 11:58, is not the T-2 board
    race_id, _, _, _, schedule = BOARD_RACES[0]
    runners = frame.loc[frame["race_id"].eq(race_id)]
    s_late, s_then = (
        shares(odds_at[(race_id, 1)]),
        shares(odds_at[(race_id, board_pick(schedule, 5))]),
    )
    leaked = np.log([s_late[h] / s_then[h] for h in runners["horse_no"]])
    shown = got.loc[runners.index, LATE_COLUMNS[5]].to_numpy(dtype=float)
    assert np.abs(shown - leaked).max() > 1e-3, "the 11:59 board entered as the T-2 board"


# --------------------------------------------------------------------------- Win settlement

SETTLE_KINDS = (
    "paid",
    "no WIN row",
    "dead heat, both paid",
    "dead heat, one WIN row",
    "PLACE row only",
    "dead heat, no WIN row",
)


def settlement_case(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, set[str]]:
    """Races of every SETTLE_KINDS kind, 6 of each: (starters, the store's dividends rows of
    every pool, the expected return per unit stake per starter row, the excluded race ids). A
    winner's final odds are near its dividend / 100, never equal; every race has PLACE rows (the
    winners' among them) and a QUINELLA row."""
    rng = np.random.default_rng(seed)
    starters, dividends, want, excluded = [], [], [], set()
    for k in range(6 * len(SETTLE_KINDS)):
        kind = SETTLE_KINDS[k % len(SETTLE_KINDS)]
        race_id = f"jra:2022-05-{1 + k % 28:02d}:QQV:{k:02d}"
        n = int(rng.integers(5, 15))
        heat = kind.startswith("dead heat")
        winners = sorted(int(h) for h in rng.choice(np.arange(1, n + 1), 2 if heat else 1, False))
        div = {h: float(10 * rng.integers(11, 400)) for h in winners}  # yen per 100 yen
        if kind in ("paid", "dead heat, both paid"):
            paid = winners
        elif kind == "dead heat, one WIN row":
            paid = [winners[k % 2]]
        else:
            paid = []
        out = any(h not in paid for h in winners)
        if out:
            excluded.add(race_id)
        for h in range(1, n + 1):
            won = h in winners
            if won:
                odds = max(round(div[h] / 100 * rng.uniform(0.7, 0.95), 1), 1.0)
            else:
                odds = round(float(rng.uniform(2.0, 150.0)), 1)
            starters.append(
                {"race_id": race_id, "horse_id": f"{race_id}:{h}", "horse_no": h, "won": won,
                 "win_odds": odds}
            )  # fmt: skip
            want.append(np.nan if out else (div[h] / 100.0 if won else 0.0))
        for h in paid:
            dividends.append((race_id, "WIN", str(h), div[h]))
        others = [h for h in range(1, n + 1) if h not in winners]
        for h in [*winners, others[0]]:
            dividends.append((race_id, "PLACE", str(h), float(10 * rng.integers(11, 200))))
        pair = sorted([winners[0], others[1]])
        quinella = float(10 * rng.integers(30, 900))
        dividends.append((race_id, "QUINELLA", f"{pair[0]},{pair[1]}", quinella))
    frame = pd.DataFrame(starters).astype({"horse_no": "int64"})
    order = rng.permutation(len(frame))
    frame = frame.iloc[order].set_axis(pd.Index(700 + 3 * order))
    div = pd.DataFrame(dividends, columns=["race_id", "pool", "combination", "dividend"])
    div = div.assign(is_refund=False).iloc[rng.permutation(len(div))]
    return (
        frame,
        div.set_axis(pd.Index(50 + 2 * np.arange(len(div)))),
        np.array(want)[order],
        excluded,
    )


def test_a_winner_is_settled_only_by_its_own_win_dividend():
    """A winner returns its own WIN dividend / 100 and a loser 0. A race any of whose winners (a
    dead heat's too) has no WIN dividend row is excluded: every runner NaN, never the final odds,
    a PLACE dividend or a loss; a dead heat with both WIN rows pays each winner its own."""
    starters, dividends, want, excluded = settlement_case(4387)
    assert 12 <= len(excluded) < starters["race_id"].nunique()

    got = dm.settle_win(starters, dividends)

    assert isinstance(got, pd.Series)
    assert got.index.sort_values().equals(starters.index.sort_values())
    values = got.reindex(starters.index).to_numpy(dtype=float)
    np.testing.assert_allclose(values, want, rtol=0, atol=1e-12)  # NaN exactly where excluded
    nan_races = set(starters.loc[np.isnan(values), "race_id"])
    assert nan_races == excluded


# --------------------------------------------------------------------------- the N5 curve

N5_SCHEME = FoldScheme(
    unit="quarter", test_length=1, step=1, min_train=4, window="expanding", embargo_days=7,
    anchor_month=1,
)  # fmt: skip


def n5_frame(seed: int) -> pd.DataFrame:
    """A study frame (as `load_inputs` gives it) of two years of Saturdays, two meetings a day, a
    DEBUT and a MAIDEN race at each, with the body-weight member's columns carrying signal beside
    the final price; about a fifth of the races have no payout (a winner without a WIN row)."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in saturdays(date(2019, 1, 5), 104):
        for venue in ("QQV", "QQW"):
            for klass in dm.CLASSES:
                race_id = f"jra:{day.isoformat()}:{venue}:{klass[0]}"
                n = int(rng.integers(6, 13))
                g = rng.gamma(1.0, size=n)
                odds = np.maximum(np.round(0.8 / (g / g.sum()), 1), 1.0)
                q = (1.0 / odds) / (1.0 / odds).sum()
                x1, x2 = rng.normal(size=n), rng.normal(size=n)
                u = np.log(q) + 0.9 * x1 - 0.5 * x2
                p = np.exp(u - u.max()) / np.exp(u - u.max()).sum()
                winner = int(rng.choice(n, p=p))
                settled = rng.random() >= 0.2
                dividend = round(odds[winner] * 100 * rng.uniform(0.9, 1.1), -1)
                for h in range(n):
                    won = h == winner
                    payout = (dividend / 100.0 if won else 0.0) if settled else np.nan
                    rows.append(
                        {
                            "race_id": race_id,
                            "race_date": pd.Timestamp(day),
                            "meeting_id": f"{day.isoformat()}:{venue}",
                            "race_class": klass,
                            "horse_no": h + 1,
                            "won": won,
                            "q": q[h],
                            "ln_q": np.log(q[h]),
                            "win_odds": odds[h],
                            "payout": payout,
                            f"{BODY}__horse_wt_kg": np.nan if rng.random() < 0.05 else x1[h],
                            f"{BODY}__horse_wt_change_kg": x2[h],
                        }
                    )
    frame = pd.DataFrame(rows).astype({"horse_no": "int64"})
    return frame.sort_values(["race_date", "race_id", "horse_no"]).reset_index(drop=True)


def test_n5_curve_leaves_out_races_without_a_win_dividend():
    """A race whose payout is NaN (a winner without a WIN dividend row) stays out of the N5
    curve, as no loss and no bet: the curve's bets and returns are the scored runners of the
    other races. The variant still scores, its ΔLL still over every scored race, and its ranked
    row counts the scored races left out of the curve."""
    frame = n5_frame(4388)
    folds = walk_forward(frame["race_date"], scheme=N5_SCHEME)
    spec = dm.build_spec(seed=SEED)
    spec["bootstrap"] = {**spec["bootstrap"], "n_boot": 200}  # a fast bootstrap; same rules
    failed = {m: "FileNotFoundError: synthetic, not loaded" for m in dm.BUILT if m != BODY}

    study = dm.compute_study(frame, folds, spec, failed)

    tested = np.zeros(len(frame), dtype=bool)
    for fold in folds:
        tested[fold.test_idx] = True
    for klass in dm.CLASSES:
        name = f"member:{BODY}:{klass}"
        variant = next(v for v in study.variants if v.name == name)
        assert variant.error is None, f"{name} failed: {variant.error}"
        scored = frame.loc[tested & frame["race_class"].eq(klass).to_numpy()]
        settled = scored["payout"].notna().to_numpy()
        left_out = scored.loc[~settled, "race_id"].nunique()
        assert left_out >= 8 and settled.sum() >= 300  # the fixture has both in numbers

        curve = [
            r
            for r in study.tables.get("n5_curve", [])
            if r["variant"] == name and r["segment"] == ALL_BETS
        ]
        assert sum(r["n_bets"] for r in curve) == settled.sum()
        returned = sum(r["realised"] * r["n_bets"] for r in curve if r["n_bets"])
        assert returned == pytest.approx(scored.loc[settled, "payout"].sum(), rel=1e-9)

        ranked = next(r for r in study.tables["ranked"] if r["variant"] == name)
        assert ranked["races"] == scored["race_id"].nunique()
        assert ranked["n5_excluded_races"] == left_out


# --------------------------------------------------------------------------- the command's exit

BUILDS_A = {
    "jra_races": "processed/jra/races.parquet@1f0e3d2c4b5a6978",
    "jra_dividends": "processed/jra/dividends.parquet@8a7b6c5d4e3f2a10",
}
BUILDS_B = BUILDS_A | {"jra_dividends": "processed/jra/dividends.parquet@0c1d2e3f4a5b6c7d"}


def fake_load_inputs(*, members_load: bool, builds: dict):
    """A stand-in for `load_inputs`, the run's store read: a synthetic study frame of the
    Saturdays in [date_from, date_to] (a DEBUT and a MAIDEN race each, six runners, every built
    member's columns), its counts, the members that failed to load (all, or none) and `builds`."""

    def load_inputs(date_from, date_to, log=print):
        rng = np.random.default_rng(4389)
        first = date_from + timedelta(days=(5 - date_from.weekday()) % 7)
        days = saturdays(first, (date_to - first).days // 7 + 1)
        rows = []
        for day in days:
            for klass in dm.CLASSES:
                race_id = f"jra:{day.isoformat()}:QQV:{klass[0]}"
                odds = np.round(rng.uniform(1.5, 40.0, size=6), 1)
                q = (1.0 / odds) / (1.0 / odds).sum()
                winner = int(rng.integers(6))
                for h in range(6):
                    rows.append(
                        {"race_id": race_id, "race_date": pd.Timestamp(day),
                         "meeting_id": f"{day.isoformat()}:QQV", "race_class": klass,
                         "horse_no": h + 1, "won": h == winner, "q": q[h], "ln_q": np.log(q[h]),
                         "win_odds": odds[h], "payout": odds[h] if h == winner else 0.0}
                    )  # fmt: skip
        frame = pd.DataFrame(rows)
        for member in dm.BUILT:
            for column in dm.feature_columns(member):
                frame[column] = rng.normal(size=len(frame))
        failed = {} if members_load else {m: "FileNotFoundError: no store" for m in dm.BUILT}
        counts = {"races": int(frame["race_id"].nunique()), "starters": len(frame)}
        return frame, counts, failed, dict(builds)

    return load_inputs


def records_registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "registry.jsonl", holdout_log=tmp_path / "holdout_log.jsonl")


def invoke_debut_members(*args: str):
    from horsetrack.cli import app

    result = CliRunner().invoke(app, ["japan", "debut-members", *args])
    assert "Usage:" not in result.output, f"{list(args)} refused: {result.output}"
    return result


@pytest.mark.parametrize(
    ("case", "first", "last", "members_load"),
    [
        ("no member loaded", date(2015, 1, 3), date(2021, 12, 25), False),
        ("no scored fold (22 training quarters)", date(2015, 1, 3), date(2018, 12, 29), True),
    ],
)
def test_the_command_exits_nonzero_when_no_variant_scores(
    case, first, last, members_load, tmp_path, monkeypatch
):
    """Every variant fails (its members did not load, or a range ending in 2018 leaves no test
    fold after D9's 22 training quarters): the command exits nonzero, and still writes one failed
    record per planned variant and the private report."""
    registry, out = records_registry(tmp_path), tmp_path / "reports"
    real = dm.run_debut_members

    def bound(*args, **kwargs):  # the run on a scratch registry: nothing committed is written
        return real(*args, **{**kwargs, "registry": registry, "out_dir": out})

    monkeypatch.setattr(dm, "run_debut_members", bound)
    monkeypatch.setattr(
        dm, "load_inputs", fake_load_inputs(members_load=members_load, builds=BUILDS_A)
    )

    result = invoke_debut_members("--from", first.isoformat(), "--to", last.isoformat())

    records = registry.records(dm.FAMILY_ID)
    assert sorted(r.variant for r in records) == sorted(dm.variant_names()), case
    assert {r.status for r in records} == {"failed"}
    assert list(out.glob("*.md")), "the private report (the run's diagnostics) is not written"
    assert result.exit_code != 0, f"{case}: no variant scored, yet the command exited 0"


def test_the_command_exit_status_says_whether_any_variant_scored(monkeypatch):
    """The exit status follows the result: nonzero when no variant scored (all failed, one
    failed, none ran), 0 when at least one did, whatever the failures beside it."""
    names = dm.variant_names()
    spec = dm.build_spec(seed=SEED)

    def variant(name: str, scored: bool) -> Variant:
        metric = (FoldMetric("all_folds", "delta_ll", 0.25, None, None, 40),) if scored else ()
        return Variant("E34", name, "walk_forward", {"study": "synthetic"}, {}, metric,
                       None if scored else "ValueError: synthetic failure")  # fmt: skip

    cases = [
        ([], [*names], False),
        ([], names[:1], False),
        ([], [], False),
        (names[:1], names[1:], True),
        (names[3:5], names[:3], True),
        ([*names], [], True),
    ]
    for scored, failed, ok in cases:
        study = Study(counts={}, variants=[*(variant(n, True) for n in scored),
                                           *(variant(n, False) for n in failed)])  # fmt: skip
        result = dm.DebutResult(spec=spec, study=study, spec_hash="ab" * 32)
        monkeypatch.setattr(dm, "run_debut_members", lambda *a, _r=result, **k: _r)
        got = invoke_debut_members("--to", "2021-12-25")
        label = f"{len(scored)} scored, {len(failed)} failed"
        assert (got.exit_code == 0) == ok, f"{label}: exit {got.exit_code}"


# --------------------------------------------------------------------------- report identity


def test_private_reports_of_different_runs_do_not_overwrite(tmp_path, monkeypatch):
    """Runs that differ in date_from, date_to or an input build write their private reports (the
    Markdown and its JSON) to different paths: after all of them each report is still the one its
    own run wrote."""
    registry, out = records_registry(tmp_path), tmp_path / "reports"
    runs = (
        (date(2015, 1, 3), date(2021, 12, 25), BUILDS_A),
        (date(2015, 1, 3), date(2021, 6, 26), BUILDS_A),  # another --to
        (date(2016, 1, 2), date(2021, 12, 25), BUILDS_A),  # another --from
        (date(2015, 1, 3), date(2021, 12, 25), BUILDS_B),  # refreshed dividends
    )
    written: dict[Path, tuple[bytes, bytes]] = {}
    for first, last, builds in runs:
        monkeypatch.setattr(dm, "load_inputs", fake_load_inputs(members_load=False, builds=builds))
        result = dm.run_debut_members(
            date_from=first, date_to=last, out_dir=out, registry=registry, log=lambda _: None
        )
        report = Path(result.report_path)
        written[report] = (report.read_bytes(), report.with_suffix(".json").read_bytes())

    assert len(written) == len(runs), f"runs share a private report path: {sorted(written)}"
    for report, (text, tables) in written.items():
        assert report.read_bytes() == text, f"{report.name} was overwritten by a later run"
        assert report.with_suffix(".json").read_bytes() == tables


# --------------------------------------------------------------------------- an aborted read

REAL_PUBLIC_REPORT = REPO_ROOT / "docs" / "experiments" / "debut-members-japan.md"
EARLIER_PUBLIC_REPORT = (
    b"# Experiment report: newcomer and maiden members (an earlier run, kept as written)\n"
    b"\n"
    b"Every variant of that run scored; this text is the committed report a retry must keep.\n"
)
RUN_RANGE = ("--from", "2015-01-03", "--to", "2021-12-25")  # 2020-21: eight scored quarters


@pytest.fixture
def scratch_repo(tmp_path, monkeypatch) -> Path:
    """A scratch repository root holding an earlier public report: `env.repo_root()` is
    tmp_path/repo for the test (the public report and the committed summary go there), while the
    data dir stays tmp_path/data. The repository's own public report is checked unchanged."""
    root = tmp_path / "repo"
    report = root / dm.PUBLIC_REPORT
    report.parent.mkdir(parents=True)
    report.write_bytes(EARLIER_PUBLIC_REPORT)
    real = REAL_PUBLIC_REPORT.read_bytes() if REAL_PUBLIC_REPORT.is_file() else None
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    yield root
    after = REAL_PUBLIC_REPORT.read_bytes() if REAL_PUBLIC_REPORT.is_file() else None
    assert after == real, "a test rewrote the repository's public report"


def missing_dividends(date_from, date_to, log=print):
    """The store read with the dividends parquet gone: what a retry of the command meets."""
    from horsetrack.core import paths

    where = paths.processed_dir("jra") / "dividends.parquet"
    raise FileNotFoundError(2, "No such file or directory", str(where))


def undated_race(date_from, date_to, log=print):
    """The study frame read in full, one race without a date: the fold split aborts on it."""
    frame, counts, failed, builds = fake_load_inputs(members_load=True, builds=BUILDS_A)(
        date_from, date_to, log
    )
    race = frame["race_id"].iloc[len(frame) // 2]
    frame.loc[frame["race_id"].eq(race), "race_date"] = pd.NaT
    return frame, counts, failed, builds


@pytest.mark.parametrize(
    ("case", "load", "error"),
    [
        ("the dividends parquet is missing", missing_dividends, "FileNotFoundError"),
        ("no store at all (the module's own load_inputs)", None, None),
        ("the fold split aborts on a race without a date", undated_race, "ValueError"),
    ],
)
def test_an_aborted_read_keeps_the_public_report(case, load, error, scratch_repo, tmp_path,
                                                 monkeypatch):  # fmt: skip
    """The command's run (default registry) whose input read or fold split aborts: it exits
    nonzero, appends one failed record per planned variant to the default registry and writes
    the private report with the error, but leaves the committed public report byte for byte as
    an earlier run wrote it."""
    if load is not None:
        monkeypatch.setattr(dm, "load_inputs", load)
    public = scratch_repo / dm.PUBLIC_REPORT

    result = invoke_debut_members(*RUN_RANGE)

    from horsetrack.core.eval.registry import default_registry

    records = default_registry("japan").records(dm.FAMILY_ID)
    assert sorted(r.variant for r in records) == sorted(dm.variant_names()), case
    assert {r.status for r in records} == {"failed"}, case
    private = sorted((tmp_path / "data" / "reports").glob("*.md"))
    assert private, f"{case}: the private report (the run's diagnostics) is not written"
    if error is not None:
        assert any(error in p.read_text("utf-8") for p in private), f"{case}: no {error} shown"
    assert result.exit_code != 0, f"{case}: no variant scored, yet the command exited 0"
    assert public.read_bytes() == EARLIER_PUBLIC_REPORT, (
        f"{case}: the aborted run rewrote the committed public report"
    )


def test_a_run_that_reads_its_inputs_rewrites_the_public_report(scratch_repo, monkeypatch):
    """The control: the command's run whose inputs read and whose variants score rewrites the
    committed public report, naming the family and every planned variant."""
    monkeypatch.setattr(dm, "load_inputs", fake_load_inputs(members_load=True, builds=BUILDS_A))
    public = scratch_repo / dm.PUBLIC_REPORT

    result = invoke_debut_members(*RUN_RANGE)

    assert result.exit_code == 0, result.output
    text = public.read_text("utf-8")
    assert public.read_bytes() != EARLIER_PUBLIC_REPORT, "a scored run left the public report"
    assert dm.FAMILY_ID in text
    for name in dm.variant_names():
        assert name in text, f"variant {name} is not in the rewritten public report"
