"""The #219 handicap speed study's guards (revision: embargo, directories, marker, failed run).

Contract pinned here (`horsetrack.countries.japan.studies.handicap_speed`, called `hs`):

- `table_dir()` is `features/jra/handicap-96-v2` (the builder default, `factors.handicap_96_dir()`);
  `interim_dir()` is #129's `features/jra/handicap`, pinned by name and not through
  `factors.handicap_dir()`. `run_study` raises ValueError when the two are one directory, before
  any table is built.
- `_meld(source, inputs)`: fold k's (a, b) is fitted on the v0 rows of the folds before k that are
  dated no later than `inputs.v0_train_end[k]` (D9's embargo, from the v0 manifest); the first
  fold only trains. `Inputs` carries `v0_train_end`.
- `run_study` rebuilds the #96 table (`factors.build_handicap_table(FIRST_DAY, LAST_DAY,
  out_dir=table_dir(), speed_at_start_day=True)`) unless `factors.table_complete` holds for that
  range and mode; a run that raises, or passes `max_minutes` (`StudyTimeout`, a BaseException),
  appends one failed record to the registry (variant `RUN_FAILED`, family `FAMILY_ID`) and
  re-raises.

Data: synthetic only, under tmp_path. The registry is the temp data dir's, never the real one.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import json
import signal
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from handicap_store import R, Store
from horsetrack.core.eval.registry import default_registry
from horsetrack.countries.japan.handicap import factors
from horsetrack.countries.japan.models.condlogit import fit_conditional_logit
from horsetrack.countries.japan.studies import handicap_speed as hs

DATA_VAR = "HORSETRACK_DATA_DIR"


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A temp data dir holding a one-race store (the registry's builds name its files)."""
    s = Store()
    s.race(date(2025, 5, 17), [R("h:a", pos=1), R("h:b", pos=2)], fill_to=4)
    return s.write(tmp_path, monkeypatch)


# ---------------------------------------------------------------------------------------------
# The two directories
# ---------------------------------------------------------------------------------------------


def test_the_96_table_and_129s_interim_table_are_different_directories(data):
    jra = data / "features" / "jra"
    assert hs.table_dir() == jra / "handicap-96-v2" == factors.handicap_96_dir()
    assert hs.interim_dir() == jra / "handicap"
    assert hs.table_dir() != hs.interim_dir()


def test_the_study_refuses_one_directory_for_both_tables_before_building(
    data, tmp_path, monkeypatch
):
    built = []
    monkeypatch.setattr(factors, "build_handicap_table", lambda *a, **k: built.append(k) or {})
    monkeypatch.setattr(hs, "TABLE_DIR", "handicap")  # #96's table where #129's is
    assert hs.table_dir() == hs.interim_dir()
    with pytest.raises(ValueError):
        hs.run_study(tmp_path / "reports")
    assert built == []
    assert not (data / "features").exists()


# ---------------------------------------------------------------------------------------------
# V2's meld trains only on rows within the scored fold's embargo
# ---------------------------------------------------------------------------------------------

JAN, FEB, MAR = date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)
CUTOFF = {1: date(2025, 1, 24), 2: date(2025, 2, 21)}  # the v0 folds' training cutoffs


def meld_inputs(*, flip: set[date] = frozenset()) -> tuple[hs.Inputs, pd.DataFrame]:
    """Three v0 folds of 28 race days each (four races of five runners a day); the winner is drawn
    from a mix of the market's and the score's probabilities. The winners of the races on a day
    of `flip` are swapped for the last runner of the race."""
    rng = np.random.default_rng(4)
    rows, scores = [], []
    for fold, first in enumerate((JAN, FEB, MAR)):
        for d in range(28):
            day = first + timedelta(days=d)
            for r in range(4):
                race = f"jra:{day.isoformat()}:TOK:{r + 1:02d}"
                m, h = (rng.dirichlet(np.ones(5) * 2) for _ in range(2))
                mix = 0.5 * m + 0.5 * h
                win = int(rng.choice(5, p=mix / mix.sum()))
                if day in flip:
                    win = 4
                for i in range(5):
                    rows.append((race, i + 1, fold, m[i], day, i == win))
                    scores.append((race, day, i + 1, h[i]))
    v = pd.DataFrame(rows, columns=["race_id", "horse_no", "fold", "p_cal", "race_date", "won"])
    s = pd.DataFrame(scores, columns=["race_id", "race_date", "horse_no", "h"])
    return hs.Inputs(
        scores={hs.V96: s, hs.INTERIM: s},
        outcomes=v[["race_id", "horse_no", "won"]],
        meetings=pd.Series(v["race_id"].unique(), index=v["race_id"].unique()),
        v0=v[["race_id", "horse_no", "fold", "p_cal"]],
        v0_train_end=dict(CUTOFF),
    ), v  # fmt: skip


def meld_coefs(**kw) -> pd.DataFrame:
    inputs, _ = meld_inputs(**kw)
    return hs._meld(hs.V96, inputs)[1].set_index("fold")


def test_the_meld_fits_each_fold_on_earlier_folds_up_to_its_training_cutoff():
    inputs, rows = meld_inputs()
    h = inputs.scores[hs.V96].set_index(["race_id", "horse_no"])["h"]
    rows = rows.assign(h=h.reindex(pd.MultiIndex.from_frame(rows[["race_id", "horse_no"]])).values)
    coefs = hs._meld(hs.V96, inputs)[1].set_index("fold")
    assert list(coefs.index) == [1, 2]  # the first fold only trains
    for k, cutoff in CUTOFF.items():
        train = rows[(rows["fold"] < k) & (rows["race_date"] <= cutoff)]
        fit = fit_conditional_logit(
            np.log(train[["p_cal", "h"]].to_numpy()), train["race_id"], train["won"].to_numpy()
        )
        assert coefs.loc[k, "races fitted"] == train["race_id"].nunique()
        assert (coefs.loc[k, "a"], coefs.loc[k, "b"]) == pytest.approx(tuple(fit.coef), abs=1e-6)
    # the embargo is real: fold 1 trains on January's first 24 days, not all 28 (4 races a day)
    assert coefs.loc[1, "races fitted"] == 24 * 4
    assert coefs.loc[2, "races fitted"] == 28 * 4 + 21 * 4


def test_a_result_inside_the_embargo_changes_no_coefficient_and_one_before_it_does():
    base = meld_coefs()
    after = {
        JAN + timedelta(days=d) for d in range(24, 28)
    }  # 01-25 .. 01-28, after fold 1's cutoff
    again = meld_coefs(flip=after)

    def moved(other: pd.DataFrame, k: int) -> float:
        return float(np.abs(other.loc[k, ["a", "b"]] - base.loc[k, ["a", "b"]]).max())

    # fold 1 never trains on them; fold 2 does (fold 0's rows up to ITS cutoff, 02-21, all of them)
    assert moved(again, 1) < 1e-9
    assert moved(again, 2) > 1e-4
    # fold 2's embargo: February 22-28 (fold 1's rows after its cutoff) are in no fit at all
    later = meld_coefs(flip={FEB + timedelta(days=d) for d in range(21, 28)})
    assert moved(later, 2) < 1e-9
    # and a day before the cutoff does move fold 1
    early = meld_coefs(flip={JAN + timedelta(days=d) for d in range(0, 10)})
    assert moved(early, 1) > 1e-4


# ---------------------------------------------------------------------------------------------
# The run: rebuild unless the marker holds, one failed record for a run that does not finish
# ---------------------------------------------------------------------------------------------


class Stop(Exception):
    pass


def records(family: str | None = None):
    return default_registry("japan").records(family or hs.FAMILY_ID)


def one_failed_record(message: str):
    runs = records()
    assert [r.variant for r in runs] == [hs.RUN_FAILED]  # and no variant of a half-done run
    run = runs[0]
    assert run.status == "failed" and run.ticket == 219 and run.family_id == hs.FAMILY_ID
    assert message in run.params["error"]
    assert list(run.metrics) == []
    return run


def test_a_raise_appends_one_failed_record_and_no_report(data, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise Stop("the table build died")

    monkeypatch.setattr(factors, "build_handicap_table", boom)
    out = tmp_path / "reports"
    with pytest.raises(Stop):
        hs.run_study(out, max_minutes=5)
    run = one_failed_record("the table build died")
    assert "Stop" in run.params["error"]
    assert not list(out.glob("*.md")) if out.exists() else True
    with pytest.raises(Stop):  # the same failure again is the same record, not a second one
        hs.run_study(out, max_minutes=5)
    assert len(records()) == 1


def test_a_run_past_its_deadline_appends_one_failed_record_and_stops(data, tmp_path, monkeypatch):
    assert issubclass(hs.StudyTimeout, BaseException)
    assert not issubclass(hs.StudyTimeout, Exception)  # no per-variant `attempt` swallows it

    def hang(*args, **kwargs):
        for _ in range(400):  # at most 20 s: the deadline is 1.2 s
            time.sleep(0.05)

    monkeypatch.setattr(factors, "build_handicap_table", hang)
    handler = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    with pytest.raises(hs.StudyTimeout):
        hs.run_study(tmp_path / "reports", max_minutes=0.02)
    assert time.monotonic() - started < 10
    run = one_failed_record("StudyTimeout")
    assert run.params["max_minutes"] == 0.02
    # the alarm is gone: no timer left running, the old handler back
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) == handler


def run_and_catch(tmp_path, monkeypatch) -> list[dict]:
    """Run the study with the table build recorded and the rest left to fail (no outcomes, no
    v0 files here); the build's calls are returned."""
    calls: list[dict] = []

    def build(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {}

    monkeypatch.setattr(factors, "build_handicap_table", build)
    with pytest.raises(Exception):  # noqa: B017, PT011
        hs.run_study(tmp_path / "reports", max_minutes=5)
    return calls


def test_the_table_is_built_in_start_day_mode_unless_its_marker_holds(data, tmp_path, monkeypatch):
    new = hs.table_dir()
    new.mkdir(parents=True)
    (new / "2025-05-17.parquet").write_bytes(b"")  # day files with no marker: an interrupted build
    calls = run_and_catch(tmp_path, monkeypatch)
    assert len(calls) == 1
    assert calls[0]["args"][:2] == (hs.FIRST_DAY, hs.LAST_DAY)
    assert calls[0]["kwargs"]["out_dir"] == new
    assert calls[0]["kwargs"]["speed_at_start_day"] is True

    # a marker for this range and mode, with the file it counted: the table is reused
    marker = {
        "date_from": hs.FIRST_DAY.isoformat(), "date_to": hs.LAST_DAY.isoformat(), "days": 1,
        "speed_at_start_day": True,
    }  # fmt: skip
    (new / "_complete.json").write_text(json.dumps(marker))
    assert factors.table_complete(new, hs.FIRST_DAY, hs.LAST_DAY, speed_at_start_day=True)
    assert run_and_catch(tmp_path, monkeypatch) == []

    # the same marker for the other mode (v1's table, no day variant) is not this table
    (new / "_complete.json").write_text(json.dumps(marker | {"speed_at_start_day": False}))
    assert len(run_and_catch(tmp_path, monkeypatch)) == 1


def test_rebuild_builds_even_when_the_table_is_complete(data, tmp_path, monkeypatch):
    new = hs.table_dir()
    new.mkdir(parents=True)
    (new / "2025-05-17.parquet").write_bytes(b"")
    marker = {
        "date_from": hs.FIRST_DAY.isoformat(), "date_to": hs.LAST_DAY.isoformat(), "days": 1,
        "speed_at_start_day": True,
    }  # fmt: skip
    (new / "_complete.json").write_text(json.dumps(marker))
    calls = []
    monkeypatch.setattr(factors, "build_handicap_table", lambda *a, **k: calls.append(k) or {})
    with pytest.raises(Exception):  # noqa: B017, PT011
        hs.run_study(tmp_path / "reports", rebuild=True, max_minutes=5)
    assert len(calls) == 1 and calls[0]["speed_at_start_day"] is True
