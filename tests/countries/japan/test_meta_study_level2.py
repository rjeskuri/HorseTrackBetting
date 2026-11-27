"""Ticket #418, round 2 findings A and D: the #87 study's level-2 population and member cutoffs.

Contract pinned here (`horsetrack.countries.japan.studies.meta.run_study`, the real entry point):

A. One level-2 population. The rows `models.stacking.check_level2` validates (its `Level2`'s
   frame and folds) are exactly the rows the study fits and scores: a race that the stacking join
   drops, because one member has no prediction for one of its runners, is in no `fit_meta` or
   `encompassing` input and in no member-scored count (`member_scored_races` in the report), and
   every race the member variants fit or score is one the guard checked.
D. Each member prediction's `member_train_end` (the last day its level-1 model trained on):
   - a `member_train_end` or `train_end` column in the member file is used as given;
   - otherwise it is the last race day before 1 January of the row's `fold_season` year in the
     store's full development race-day calendar: every non-holdout race day of the races table,
     whatever its price or status, not only the days of the priced frame (`store_frame`);
   - when neither gives a cutoff, the run raises `models.stacking.InvalidProvenance` before any
     fit, and (finding F) leaves the registry as it was.

The walk-forward fact the D tests use (`core.eval.folds.walk_forward`): a fold's train_end is the
earlier of its first test day less the 7-day embargo and the day before its quarter. With a priced
race day on 2024-01-05, 2024-Q1 trains to 2023-12-29, so a member trained through 2023-12-30 is
past that fold's train_end for every row it scores, which the guard refuses.

Every value is synthetic, under a tmp HORSETRACK_DATA_DIR (`meta_store_fakes`).
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from meta_store_fakes import (
    FIRST_DAY,
    add_race_day,
    build_store,
    member_rows,
    write_members,
    write_store,
)

from horsetrack.core import paths
from horsetrack.core.eval.folds import load_country_scheme
from horsetrack.core.eval.registry import default_registry
from horsetrack.countries.japan.models import meta as models_meta
from horsetrack.countries.japan.models import stacking as st
from horsetrack.countries.japan.studies import meta

LONG_START = date(2017, 1, 10)  # seven seasons before 2024-Q1: D9 met, no opt-in needed
SCORED_FROM = date(2024, 1, 1)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    real_boot = meta.bootstrap_settings
    monkeypatch.setattr(meta, "bootstrap_settings", lambda: {**real_boot(), "n_boot": 20})

    inputs: list[set[str]] = []  # the race ids of every member-based fit or score input
    real_fit, real_enc = models_meta.fit_meta, models_meta.encompassing

    def spy_fit(frame, *args, **kwargs):
        inputs.append(set(frame["race_id"]))
        return real_fit(frame, *args, **kwargs)

    def spy_enc(frame, *args, **kwargs):
        inputs.append(set(frame["race_id"]))
        return real_enc(frame, *args, **kwargs)

    for module in (meta, models_meta):
        monkeypatch.setattr(module, "fit_meta", spy_fit)
        monkeypatch.setattr(module, "encompassing", spy_enc)

    guarded: list[st.Level2] = []
    real_check = st.check_level2

    def spy_check(level2, *args, **kwargs):
        guarded.append(level2)
        return real_check(level2, *args, **kwargs)

    monkeypatch.setattr(st, "check_level2", spy_check)
    return {"inputs": inputs, "guarded": guarded, "dir": tmp_path}


def registry_bytes() -> bytes:
    path = default_registry("japan").path
    return path.read_bytes() if path.exists() else b""


def report_count(report: str, name: str) -> int:
    match = re.search(rf"^\|\s*{re.escape(name)}\s*\|\s*([\d,]+)\s*\|", report, re.MULTILINE)
    assert match, f"the report has no {name} count"
    return int(match.group(1).replace(",", ""))


def used_races(level2: st.Level2) -> set[str]:
    rows = [i for f in level2.folds for idx in (f.train_idx, f.test_idx) for i in idx]
    return set(level2.frame.iloc[sorted(set(rows))]["race_id"])


def scored_meta_variants(result: dict) -> list[str]:
    names = [v for v in result["record_ids"] if v.startswith(("stage2:", "offset:"))]
    return [v for v in names if v not in result["failed_variants"]]


# --------------------------------------------------------------------------- A


@pytest.mark.parametrize("member_col", ["p_lgbm_no_jrdb_oof", "p_fundamental_oof"])
def test_a_race_the_stacking_join_drops_is_neither_fitted_nor_scored(
    env: dict, member_col: str
) -> None:
    store = build_store()
    write_store(store)
    days = pd.to_datetime(store.races["race_date"])
    scored_races = sorted(store.races.loc[days >= pd.Timestamp(SCORED_FROM), "race_id"])
    bad = scored_races[5]
    rows = member_rows(store, LONG_START)
    runner = rows.index[rows["race_id"] == bad][2]
    rows.loc[runner, member_col] = np.nan  # one member, one runner: every key is still present
    path = write_members(env["dir"] / "members" / "one_null.parquet", rows)

    result = meta.run_study(path, env["dir"] / "out")

    assert env["guarded"], "the level-2 guard never ran"
    level2 = env["guarded"][-1]
    assert bad not in set(level2.frame["race_id"])
    assert env["inputs"], "no member-based fit ran"
    fitted = set().union(*env["inputs"])
    assert bad not in fitted, "a member fits or scores a race the guard never checked"
    assert fitted == used_races(level2)
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert report_count(report, "member_scored_races") == len(scored_races) - 1
    assert scored_meta_variants(result)


# --------------------------------------------------------------------------- D


def late_december_store(kind: str):
    """The fixture store plus a 2023-12-30 race day that the priced frame drops (`kind`: void, or
    run but unpriced) and a priced 2024-01-05 day: 2024-Q1 then trains to 2023-12-29."""
    store = build_store()
    if kind == "void":
        store = add_race_day(store, date(2023, 12, 30), status="abandoned")
    else:
        store = add_race_day(store, date(2023, 12, 30), priced=False)
    return add_race_day(store, date(2024, 1, 5))


def with_train_end(rows: pd.DataFrame, column: str, month_day: str, years_back: int):
    season = rows["fold_season"].str[:4].astype(int) - years_back
    return rows.assign(**{column: pd.to_datetime(season.astype(str) + month_day)})


@pytest.mark.parametrize("kind", ["void", "unpriced"])
def test_the_derived_cutoff_reads_every_race_day_of_the_store(env: dict, kind: str) -> None:
    store = late_december_store(kind)
    write_store(store)
    path = write_members(env["dir"] / "members" / "m.parquet", member_rows(store, LONG_START))
    before = registry_bytes()
    with pytest.raises(st.InvalidProvenance):
        meta.run_study(path, env["dir"] / "out")
    assert env["inputs"] == [], "a meta-model was fitted before the refusal"
    assert registry_bytes() == before


@pytest.mark.parametrize("column", ["member_train_end", "train_end"])
def test_an_explicit_train_end_column_is_used_over_the_derived_cutoff(
    env: dict, column: str
) -> None:
    store = build_store()  # derived, the cutoff is 2023-12-24, within 2024-Q1's 2023-12-31
    write_store(store)
    # the members say they trained to 2 January of their own season: past 2024-Q1's train_end
    rows = with_train_end(member_rows(store, LONG_START), column, "-01-02", 0)
    path = write_members(env["dir"] / "members" / f"{column}.parquet", rows)
    before = registry_bytes()
    with pytest.raises(st.InvalidProvenance):
        meta.run_study(path, env["dir"] / "out")
    assert env["inputs"] == []
    assert registry_bytes() == before


def test_an_explicit_early_train_end_runs_where_the_derived_cutoff_would_refuse(
    env: dict,
) -> None:
    store = late_december_store("void")
    write_store(store)
    rows = with_train_end(member_rows(store, LONG_START), "member_train_end", "-12-20", 1)
    path = write_members(env["dir"] / "members" / "early.parquet", rows)
    result = meta.run_study(path, env["dir"] / "out")
    assert env["inputs"], "the run did not fit"
    assert scored_meta_variants(result)


def test_no_cutoff_from_either_refuses_before_any_fit(env: dict) -> None:
    store = build_store()  # the store starts on 2014-06-10: no race day before season 2014
    write_store(store)
    path = write_members(env["dir"] / "members" / "all.parquet", member_rows(store, FIRST_DAY))
    with pytest.raises(st.InvalidProvenance):
        meta.run_study(path, env["dir"] / "out")
    assert env["inputs"] == []
    assert registry_bytes() == b"", "a refused run wrote to the registry"


# --------------------------------------------------------------------------- fix 1: cutoff types
#
# The owner's restart instructions (2026-11-27), fix 1: `member_predictions` raises `ValueError`
# when the member file's own cutoff column (`train_end`, read as `member_train_end`) is numeric
# (`train_end = 20231230` would otherwise read as nanoseconds after the epoch, January 1970); a
# date, datetime or ISO-string cutoff is still read as that day.

CUTOFF = pd.Timestamp("2023-12-20")
SHORT_SPAN = date(2023, 1, 10)  # a few hundred member rows: enough for the direct calls


def _file_with(env: dict, store, name: str, values) -> Path:
    """A member file with its own `train_end` column of `values(n_rows)`."""
    rows = member_rows(store, SHORT_SPAN)
    rows["train_end"] = values(len(rows))
    return write_members(env["dir"] / "members" / f"cutoff-{name}.parquet", rows)


@pytest.mark.parametrize(
    ("numeric", "accepted"),
    [
        (lambda n: np.full(n, 20231230, dtype=np.int64), lambda n: [CUTOFF.date()] * n),
        (lambda n: np.full(n, 20231230.0), lambda n: ["2023-12-20"] * n),
        (lambda n: np.full(n, 20231230, dtype=np.int32), lambda n: [CUTOFF.to_pydatetime()] * n),
        (  # seconds since the epoch on 2023-12-20: numeric still
            lambda n: np.full(n, int(CUTOFF.timestamp()), dtype=np.int64),
            lambda n: pd.Series([CUTOFF] * n, dtype="datetime64[ns]"),
        ),
    ],
    ids=["int_compact", "float_compact", "int32_compact", "int_epoch_seconds"],
)
def test_a_numeric_member_cutoff_is_refused_and_a_date_cutoff_is_read(
    env: dict, numeric, accepted
) -> None:
    store = build_store()
    write_store(store)
    days = meta.race_calendar()
    numeric_file = _file_with(env, store, "numeric", numeric)
    with pytest.raises(ValueError):  # read_members may refuse it first: either is the refusal
        meta.member_predictions(meta.read_members(numeric_file), days)

    accepted_file = _file_with(env, store, "accepted", accepted)
    out = meta.member_predictions(meta.read_members(accepted_file), days)
    assert len(out) and (pd.to_datetime(out["member_train_end"]) == CUTOFF).all()


def test_python_int_cutoffs_in_an_object_column_are_refused(env: dict) -> None:
    store = build_store()
    write_store(store)
    rows = member_rows(store, SHORT_SPAN)
    members = rows.assign(member_train_end=pd.Series([20231230] * len(rows), dtype=object).values)
    with pytest.raises(ValueError):
        meta.member_predictions(members, meta.race_calendar())


def test_the_run_refuses_an_integer_train_end_before_anything_is_written(env: dict) -> None:
    store = build_store()
    write_store(store)
    rows = member_rows(store, LONG_START)
    # as compact dates, 20 December of the season before: valid cutoffs, were they read as dates
    season = rows["fold_season"].str[:4].astype(int)
    rows["train_end"] = ((season - 1) * 10000 + 1220).astype(np.int64).to_numpy()
    path = write_members(env["dir"] / "members" / "int_train_end.parquet", rows)
    with pytest.raises(ValueError):
        meta.run_study(path, env["dir"] / "out")
    assert env["inputs"] == [], "a meta-model was fitted before the refusal"
    assert registry_bytes() == b"", "a refused run wrote to the registry"


# --------------------------------------------------------------------------- fix 2: normalisation
#
# Fix 2: the checked join carries the file's original probability, and `member_frame` normalises
# it exactly as main did before #418 (`p = col / col.groupby(race_id).transform("sum")`, then
# `np.log`, over the store rows of the member span in store order): bit for bit, so a rerun of the
# same file reproduces the registry's unrounded metrics. The expected values below reproduce
# main's arithmetic on the same synthetic file, independently of the code under test.


def main_normalisation(frame: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Main's `member_frame` (origin/main, before #418): one log-probability column per member,
    index the store positions, races with a starter the file lacks dropped."""
    first, last = members["race_date"].min(), members["race_date"].max()
    span = frame.loc[(frame["race_date"] >= first) & (frame["race_date"] <= last)]
    keyed = members.drop(columns=["race_date", "fold_season"])
    joined = span.reset_index().merge(
        keyed, on=["race_id", "horse_id"], how="left", validate="one_to_one", indicator=True
    )
    whole = joined["_merge"].eq("both").groupby(joined["race_id"]).transform("all")
    out = joined.loc[whole].drop(columns="_merge").set_index("index").rename_axis(None)
    for name, (col, _) in meta.MEMBERS.items():
        p = out[col] / out[col].groupby(out["race_id"]).transform("sum")
        out[name] = np.log(p.to_numpy())
    return out


@pytest.mark.parametrize(("start", "seed"), [(date(2023, 1, 10), 87), (LONG_START, 5)])
def test_member_frame_normalises_the_files_probabilities_as_main_did(
    env: dict, start: date, seed: int
) -> None:
    store = build_store()
    write_store(store)
    rows = member_rows(store, start, seed=seed)
    path = write_members(env["dir"] / "members" / "norm.parquet", rows)
    runners, races = meta.read_development()
    frame, _ = meta.store_frame(runners, races)
    _, scored = meta.study_folds(frame, load_country_scheme("japan"))
    members = meta.read_members(path)
    level2 = meta.check_level2(frame, scored, members, meta.race_calendar(), min_seasons=0)

    out, _ = meta.member_frame(level2, frame, members)
    expected = main_normalisation(frame, members)
    assert len(out) > 0
    assert set(out.index) == set(expected.index)
    for name in meta.MEMBERS:
        got = out[name].to_numpy()
        want = expected[name].reindex(out.index).to_numpy()
        assert np.array_equal(got, want), f"{name}: {int((got != want).sum())} values differ"
