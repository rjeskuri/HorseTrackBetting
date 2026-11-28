"""#284: the #131 exacta study's lambda2 refit per walk-forward fold (pure parts).

#131 scored B3 and C at #115's single full-development lambda2 (0.8104), a fit that saw the races
it was scored on. This ticket fits lambda2 again per fold of folds.yaml, on the fold's training
races alone, and scores B3 and C out of sample under new frozen family versions; the
``full_development`` rows stay beside them (the run side is in test_exacta_run.py, section 17).

Contract pinned here (the names are the tests' contract; every input is synthetic):

- ``exacta_run.FoldLambda2`` is a frozen dataclass with the fields, in this order,
  ``test_start, test_end, train_end`` (``datetime.date``), ``n_train_races`` (int), ``lambda2``
  (float, None when the fold has no training race) and ``fit_id`` (str, None likewise).
- ``exacta_run.per_fold_lambda2(fit_frame, scheme, *, calendar=None) -> list[FoldLambda2]``, one
  entry per ``core.eval.folds.walk_forward`` fold, in date order. ``fit_frame`` has the columns of
  ``ordering.COLUMNS`` (race_id, horse_no, p_win, finish_pos, dead_heat) plus ``race_date`` (one
  date per race, ``datetime.date``), rows in any order. ``calendar`` is the iterable of dates
  that defines the folds (every race date of the development calendar); it defaults to the fit
  frame's own race dates. A fold's lambda2 is ``ordering.fit_ordering(<the fit frame's races dated
  on or before the fold's train_end>, family="discounted_harville").lambdas[0]`` (the
  maximum-likelihood stage-1 fit), ``n_train_races`` is that fit's ``n_races`` and ``fit_id`` its
  ``fit_id``. A fold whose training window holds no race of the fit frame has ``n_train_races`` 0
  and ``lambda2`` and ``fit_id`` None (the first scored quarter: #114's out-of-fold file starts
  there, so nothing precedes it).
- ``exacta_study.score_races_per_fold(races, *, holdout_day, fold_lambdas)``: ``score_races``'
  frame with every race scored at the lambda2 of the fold whose [test_start, test_end] window holds
  its date, and a ``lambda2`` column with the value used. ``frame.attrs["excluded"]`` counts a race
  in no window ``no_test_window`` and a race in a fold without a lambda2 ``no_training_races``;
  holdout-day races are dropped first (``holdout_day``), as in ``score_races``.
- ``exacta_study.per_fold_family_specs()``: the two new frozen families, beside (not inside)
  ``family_specs()``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.countries.japan.models.v0 import ordering
from horsetrack.countries.japan.research import exacta_run
from horsetrack.countries.japan.research import exacta_study as study

FAMILY = "discounted_harville"
CATEGORIES = ("fav_first", "fav_second", "no_fav")


# ---------------------------------------------------------------------------------------------
# Synthetic fit frames
# ---------------------------------------------------------------------------------------------


def _fit_frame(dates, per_date, seed, n=7, lam=0.8) -> pd.DataFrame:
    """Plackett-Luce finishes (stage 2 discounted by ``lam``) of ``per_date`` races on each date."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in dates:
        for k in range(per_date):
            p = rng.dirichlet(np.full(n, 1.5))
            p = np.clip(p, 0.02, None)
            p = p / p.sum()
            order, left = [], list(range(n))
            for stage in range(3):
                w = np.array([p[h] ** (lam if stage == 1 else 1.0) for h in left])
                pick = left[int(rng.choice(len(left), p=w / w.sum()))]
                order.append(pick)
                left.remove(pick)
            order += [int(h) for h in rng.permutation(left)]
            pos = {h: i + 1 for i, h in enumerate(order)}
            rid = f"jra:{day.isoformat()}:TOK:{k + 1:02d}"
            rows += [
                {"race_id": rid, "horse_no": h + 1, "p_win": float(p[h]),
                 "finish_pos": float(pos[h]), "dead_heat": False, "race_date": day}
                for h in range(n)
            ]  # fmt: skip
    return pd.DataFrame(rows)


def _perturb(frame: pd.DataFrame, race_ids) -> pd.DataFrame:
    """The same frame with every named race changed: its second finisher swapped with its longest
    shot behind the second, and its Win probabilities reversed across the runners."""
    out = frame.copy()
    for rid in race_ids:
        sub = out[out["race_id"] == rid]
        second = sub.index[sub["finish_pos"] == 2][0]
        behind = sub[sub["finish_pos"] > 2]
        longest = behind["p_win"].idxmin()
        a, b = out.at[second, "finish_pos"], out.at[longest, "finish_pos"]
        out.at[second, "finish_pos"], out.at[longest, "finish_pos"] = b, a
        out.loc[sub.index, "p_win"] = sub["p_win"].to_numpy()[::-1]
    return out


def _races_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.drop_duplicates("race_id")[["race_id", "race_date"]]
        .sort_values(["race_date", "race_id"])
        .reset_index(drop=True)
    )


def _ref_fit(frame: pd.DataFrame, race_ids):
    sub = frame[frame["race_id"].isin(set(race_ids))].drop(columns="race_date")
    return ordering.fit_ordering(sub.reset_index(drop=True), family=FAMILY)


MONTHLY = FoldScheme(
    unit="month", test_length=1, step=1, min_train=3, window="expanding", embargo_days=7,
    anchor_month=1,
)  # fmt: skip
# Every 4th day of 2024 with two races each: seven monthly folds, April to October.
DATES_2024 = [date(2024, 1, 1) + timedelta(days=4 * k) for k in range(76)]


@pytest.fixture(scope="module")
def frame_2024() -> pd.DataFrame:
    return _fit_frame(DATES_2024, per_date=2, seed=284)


@pytest.fixture(scope="module")
def groups_2024(frame_2024):
    """For each reference fold, the race ids of its training window, test window, embargo and
    the later races, from ``walk_forward`` over the race dates (not from the code under test)."""
    races = _races_by_date(frame_2024)
    folds = walk_forward(races["race_date"], scheme=MONTHLY)
    out = []
    for fold in folds:
        train = set(races.loc[fold.train_idx, "race_id"])
        test = set(races.loc[fold.test_idx, "race_id"])
        embargo = {
            r.race_id
            for r in races.itertuples()
            if r.race_date < fold.test_start and r.race_id not in train
        }
        later = {r.race_id for r in races.itertuples() if r.race_date > fold.test_end}
        out.append({"fold": fold, "train": train, "test": test, "embargo": embargo,
                    "later": later})  # fmt: skip
    return out


def test_the_reference_folds_are_the_seven_monthly_windows_with_an_embargo(groups_2024):
    assert [g["fold"].test_start for g in groups_2024] == [date(2024, m, 1) for m in range(4, 11)]
    for g in groups_2024:
        assert g["train"] and g["test"] and g["embargo"], g["fold"].test_start  # not vacuous


# ---------------------------------------------------------------------------------------------
# 1. Each fold's lambda2 is the MLE on the fold's training races alone
# ---------------------------------------------------------------------------------------------


def test_each_folds_lambda2_is_the_ordering_fit_of_its_training_races(frame_2024, groups_2024):
    shuffled = frame_2024.sample(frac=1.0, random_state=7).reset_index(drop=True)  # any row order
    got = exacta_run.per_fold_lambda2(shuffled, MONTHLY)
    assert len(got) == len(groups_2024) == 7
    for entry, g in zip(got, groups_2024, strict=True):
        fold = g["fold"]
        ref = _ref_fit(frame_2024, g["train"])
        assert isinstance(entry, exacta_run.FoldLambda2)
        assert (entry.test_start, entry.test_end, entry.train_end) == (
            fold.test_start, fold.test_end, fold.train_end
        )  # fmt: skip
        assert entry.n_train_races == ref.n_races == len(g["train"])
        # row order moves the optimiser's last digits (xatol 1e-9), not the fit
        assert entry.lambda2 == pytest.approx(ref.lambdas[0], abs=1e-6)
        assert isinstance(entry.fit_id, str) and entry.fit_id  # ordering's id: it hashes the digits
    assert len({e.fit_id for e in got}) == len(got)
    # the folds are expanding windows, and the fits are not one constant
    sizes = [e.n_train_races for e in got]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert max(e.lambda2 for e in got) - min(e.lambda2 for e in got) > 1e-3
    assert all(e.train_end < e.test_start for e in got)


def test_fold_lambda2_is_a_frozen_record_of_its_fields():
    names = [f.name for f in dataclasses.fields(exacta_run.FoldLambda2)]
    assert names == ["test_start", "test_end", "train_end", "n_train_races", "lambda2", "fit_id"]
    entry = exacta_run.FoldLambda2(date(2024, 4, 1), date(2024, 4, 30), date(2024, 3, 26), 5, 0.9,
                                   "abc")  # fmt: skip
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.lambda2 = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------
# 2. The core property: only training races reach a fold's lambda2
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("group", ["test", "embargo", "later", "all_three"])
def test_changing_races_from_the_embargo_on_leaves_that_folds_lambda2_unchanged(
    frame_2024, groups_2024, group
):
    base = exacta_run.per_fold_lambda2(frame_2024, MONTHLY)
    for k, g in enumerate(groups_2024):
        ids = g["test"] | g["embargo"] | g["later"] if group == "all_three" else g[group]
        if not ids:  # no later race after the last window
            assert group == "later" and k == len(groups_2024) - 1
            continue
        changed = exacta_run.per_fold_lambda2(_perturb(frame_2024, ids), MONTHLY)
        assert changed[k].lambda2 == pytest.approx(base[k].lambda2, abs=1e-12), (group, k)
        assert changed[k].fit_id == base[k].fit_id and changed[k].n_train_races == len(g["train"])


def test_changing_a_training_race_moves_exactly_the_folds_that_train_on_it(frame_2024, groups_2024):
    base = exacta_run.per_fold_lambda2(frame_2024, MONTHLY)
    for k, g in enumerate(groups_2024):
        races = _races_by_date(frame_2024)
        in_train = races[races["race_id"].isin(g["train"])]
        rid = in_train.iloc[-1]["race_id"]  # fold k's most recent training race
        changed = exacta_run.per_fold_lambda2(_perturb(frame_2024, [rid]), MONTHLY)
        for j, other in enumerate(groups_2024):
            if rid in other["train"]:
                assert abs(changed[j].lambda2 - base[j].lambda2) > 1e-7, (k, j)
                assert changed[j].fit_id != base[j].fit_id
            else:
                assert changed[j].lambda2 == pytest.approx(base[j].lambda2, abs=1e-12), (k, j)
                assert changed[j].fit_id == base[j].fit_id
        assert rid in g["train"]


# ---------------------------------------------------------------------------------------------
# 3. The Japan scheme (folds.yaml): the first scored quarter has nothing to train on
# ---------------------------------------------------------------------------------------------

# Out-of-fold Win probabilities start with the first scored quarter, 2020-Q1, so nothing precedes
# it in the fit frame; the calendar that defines the folds reaches back to 2014-06-07.
SATURDAYS_2020_21 = [date(2020, 1, 4) + timedelta(days=7 * k) for k in range(78)]
MIDWEEK = [date(2020, 3, 30), date(2020, 3, 31)]  # in 2020-Q1, inside the embargo of 2020-Q2
CALENDAR_BEFORE = [date(2014, 6, 7) + timedelta(days=30 * k) for k in range(60)]


@pytest.fixture(scope="module")
def japan_folds():
    frame = _fit_frame([*SATURDAYS_2020_21, *MIDWEEK], per_date=3, seed=2020)
    scheme = load_country_scheme("japan")
    calendar = [*CALENDAR_BEFORE, *frame["race_date"].unique()]
    assert max(CALENDAR_BEFORE) < date(2020, 1, 1)
    return frame, exacta_run.per_fold_lambda2(frame, scheme, calendar=calendar)


def test_the_japan_scheme_gives_quarterly_folds_from_2020_q1(japan_folds):
    frame, got = japan_folds
    assert [e.test_start for e in got] == [
        date(2020, 1, 1), date(2020, 4, 1), date(2020, 7, 1), date(2020, 10, 1),
        date(2021, 1, 1), date(2021, 4, 1),
    ]  # fmt: skip
    assert [e.test_end for e in got] == [
        date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30), date(2020, 12, 31),
        date(2021, 3, 31), date(2021, 6, 30),
    ]  # fmt: skip


def test_the_first_scored_quarter_has_no_training_race_and_no_lambda2(japan_folds):
    _, got = japan_folds
    first = got[0]
    assert first.n_train_races == 0
    assert first.lambda2 is None and first.fit_id is None
    assert all(e.lambda2 is not None and e.fit_id for e in got[1:])


def test_the_embargo_keeps_the_last_week_before_a_quarter_out_of_its_training(japan_folds):
    frame, got = japan_folds
    q2, q3 = got[1], got[2]
    # 2020-Q2's first race is Saturday 2020-04-04: training ends a week before, 2020-03-28.
    assert q2.train_end == date(2020, 3, 28)
    q1_races = frame[frame["race_date"] <= date(2020, 3, 28)]["race_id"].unique()
    assert len(q1_races) == 13 * 3 == q2.n_train_races  # the 03-30 and 03-31 races are out
    assert q2.lambda2 == pytest.approx(_ref_fit(frame, q1_races).lambdas[0], abs=1e-9)
    # 2020-Q3's first race is Saturday 2020-07-04: 06-27, so those two races now train.
    assert q3.train_end == date(2020, 6, 27)
    upto = frame[frame["race_date"] <= date(2020, 6, 27)]["race_id"].unique()
    assert q3.n_train_races == len(upto) == 26 * 3 + 2 * 3
    assert q3.lambda2 == pytest.approx(_ref_fit(frame, upto).lambdas[0], abs=1e-9)
    assert q2.lambda2 != pytest.approx(q3.lambda2, abs=1e-6)


# ---------------------------------------------------------------------------------------------
# 4. Scoring each race at its fold's lambda2
# ---------------------------------------------------------------------------------------------


def _score_race(rid, day, n, seed, *, winner_kind="no_fav") -> dict:
    rng = np.random.default_rng(seed)
    horses = list(range(1, n + 1))
    steps = 1.0 + np.cumsum(rng.uniform(0.2, 1.4, size=n))
    odds = {
        h: round(float(2.0 * s), 1) for h, s in zip(rng.permutation(horses), steps, strict=True)
    }
    raw = {h: float(rng.uniform(0.7, 1.3) / o) for h, o in odds.items()}
    total = sum(raw.values())
    fav = min(odds, key=lambda h: (odds[h], h))
    others = [h for h in horses if h != fav]
    winner = {
        "fav_first": (fav, others[0]),
        "fav_second": (others[1], fav),
        "no_fav": (others[2], others[3]),
    }[winner_kind]
    return {
        "race_id": rid, "meeting_id": f"jra:{day.isoformat()}:TOK", "race_date": day,
        "race_no": 5, "race_class": "C2", "win_odds": odds,
        "p": {h: v / total for h, v in raw.items()}, "rate": 0.75,
        "boost_may_be_withheld": False, "exacta": [(winner, 840.0, False)],
    }  # fmt: skip


def _folds() -> list:
    make = exacta_run.FoldLambda2
    return [
        make(date(2025, 4, 1), date(2025, 4, 30), date(2025, 3, 25), 40, 0.55, "fa"),
        make(date(2025, 5, 1), date(2025, 5, 31), date(2025, 4, 24), 90, 1.30, "fb"),
        make(date(2025, 6, 1), date(2025, 6, 30), date(2025, 5, 25), 0, None, None),
        make(date(2025, 8, 1), date(2025, 8, 31), date(2025, 7, 25), 200, 0.90, "fd"),
    ]


SCORE_DAYS = {
    "before_first": date(2025, 3, 29),
    "a_start": date(2025, 4, 1),
    "a_mid": date(2025, 4, 12),
    "a_end": date(2025, 4, 30),
    "b_mid": date(2025, 5, 17),
    "b_end": date(2025, 5, 31),
    "no_lambda": date(2025, 6, 14),
    "gap": date(2025, 7, 12),
    "d_mid": date(2025, 8, 9),
    "after_last": date(2025, 9, 6),
}
LAMBDA_OF_DAY = {"a_start": 0.55, "a_mid": 0.55, "a_end": 0.55, "b_mid": 1.30, "b_end": 1.30,
                 "d_mid": 0.90}  # fmt: skip


def _no_holdout(country, day) -> bool:
    return False


def _score_sample():
    kinds = ("fav_first", "fav_second", "no_fav")
    return [
        _score_race(
            f"jra:{day.isoformat()}:TOK:05", day, 7 + 2 * (i % 3), seed=i, winner_kind=kinds[i % 3]
        )  # fmt: skip
        for i, day in enumerate(SCORE_DAYS.values())
    ]


def _single(race, lam) -> pd.DataFrame:
    return study.score_races([race], holdout_day=_no_holdout, lambda2=lam)


def test_every_race_is_scored_at_the_lambda2_of_the_fold_holding_its_date():
    races = _score_sample()
    frame = study.score_races_per_fold(races, holdout_day=_no_holdout, fold_lambdas=_folds())
    scored = {r["race_id"]: LAMBDA_OF_DAY[name] for r, name in zip(races, SCORE_DAYS, strict=True)
              if name in LAMBDA_OF_DAY}  # fmt: skip
    assert set(frame["race_id"]) == set(scored)
    assert len(frame) == 3 * len(scored)  # three category rows per scored race
    for rid, lam in scored.items():
        race = next(r for r in races if r["race_id"] == rid)
        want = _single(race, lam).sort_values("category").reset_index(drop=True)
        got = frame[frame["race_id"] == rid].sort_values("category").reset_index(drop=True)
        assert (got["lambda2"] == lam).all()
        assert list(got["category"]) == list(want["category"]) == sorted(CATEGORIES)
        for column in ("b1", "b2", "b3", "e1_loss", "rate"):
            assert got[column].to_numpy() == pytest.approx(want[column].to_numpy(), rel=1e-12)
    # a different fold's lambda2 would give other B2 and B3 returns: the test can tell them apart
    race = next(r for r in races if r["race_id"].endswith("2025-05-17:TOK:05"))
    other = _single(race, 0.55)
    here = frame[frame["race_id"] == race["race_id"]]
    assert not np.allclose(
        other.sort_values("category")["b3"], here.sort_values("category")["b3"], rtol=1e-3
    )
    assert set(frame.columns) >= set(study.score_races(races, holdout_day=_no_holdout).columns)


def test_races_outside_every_window_or_in_a_fold_without_lambda2_are_counted_not_scored():
    races = _score_sample()
    frame = study.score_races_per_fold(races, holdout_day=_no_holdout, fold_lambdas=_folds())
    excluded = frame.attrs["excluded"]
    assert excluded["no_test_window"] == 3  # 2025-03-29, 2025-07-12 and 2025-09-06
    assert excluded["no_training_races"] == 1  # 2025-06-14, in June's fold, which has no fit
    assert not {"2025-03-29", "2025-06-14", "2025-07-12", "2025-09-06"} & {
        i.split(":")[1] for i in frame["race_id"]
    }
    assert sum(excluded.values()) == len(races) - frame["race_id"].nunique()


def test_a_holdout_day_race_is_dropped_before_its_exacta_is_read():
    class Sealed(list):
        def _boom(self, *a, **k):
            raise AssertionError("a holdout race's exacta was read")

        __iter__ = __len__ = __getitem__ = __bool__ = __contains__ = _boom

    races = _score_sample()
    sealed = copy.copy(races[3])  # 2025-04-30, in April's fold
    sealed["exacta"] = Sealed()
    races[3] = sealed
    held = sealed["race_date"]
    frame = study.score_races_per_fold(
        races, holdout_day=lambda c, d: d == held, fold_lambdas=_folds()
    )
    assert sealed["race_id"] not in set(frame["race_id"])
    assert frame.attrs["excluded"]["holdout_day"] == 1
    assert set(frame["race_id"]) == {
        r["race_id"]
        for r in races
        if r["race_date"] != held and r["race_date"] in {SCORE_DAYS[k] for k in LAMBDA_OF_DAY}
    }


def test_no_races_give_an_empty_frame_with_the_columns():
    frame = study.score_races_per_fold([], holdout_day=_no_holdout, fold_lambdas=_folds())
    assert len(frame) == 0 and frame.attrs["excluded"] == {} and "lambda2" in frame.columns


# ---------------------------------------------------------------------------------------------
# 5. The new frozen family versions
# ---------------------------------------------------------------------------------------------

B3_PF = "japan-131-exacta-category-b3-v3"
C_PF = "japan-131-exacta-logscore-v3"
# What the per-fold version may change against the v2 spec it sits beside.
MAY_DIFFER = {"lambda2", "family_size", "ordering", "role", "revision", "fold_scheme", "relabel",
              "supersedes", "complements", "estimand", "interval", "lambda2_fit"}  # fmt: skip


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_the_per_fold_families_are_two_new_ids_beside_the_v2_families():
    assert study.PER_FOLD == "per_fold"
    assert study.B3_PER_FOLD_ID == B3_PF and study.C_PER_FOLD_ID == C_PF
    assert exacta_run.PER_FOLD_FAMILIES == (B3_PF, C_PF)
    specs = study.per_fold_family_specs()
    assert set(specs) == {B3_PF, C_PF}
    assert set(specs).isdisjoint(study.family_specs())  # the v2 specs and their hashes stand
    assert set(exacta_run.SCORED_FAMILIES) == {
        study.E1_ID,
        study.E1_B1_ID,
        study.E1_B3_ID,
        study.C_ID,
    }
    assert "/" not in B3_PF + C_PF
    assert specs[B3_PF]["estimator"] == "B3" and specs[C_PF]["estimator"] == "C"


def test_the_per_fold_specs_hold_only_the_per_fold_lambda2_and_the_v2_design_otherwise():
    specs, v2 = study.per_fold_family_specs(), study.family_specs()
    b3, c = specs[B3_PF], specs[C_PF]
    assert b3["lambda2"] == "per_fold" and c["lambda2"] == "per_fold"
    assert len(b3["members"]) == b3["family_size"] == 42
    assert c["family_size"] == 1 and c["members"] == [{"category": "all", "cell": "all"}]
    for new, old in ((b3, v2[study.E1_B3_ID]), (c, v2[study.C_ID])):
        changed = {k for k in set(new) | set(old) if new.get(k) != old.get(k)}
        assert changed <= MAY_DIFFER, sorted(changed - MAY_DIFFER)
        assert new["pool"] == old["pool"] and new["sources"] == old["sources"] == ["jra"]
        assert new["ticket"] == old["ticket"] == 131
        assert new["exclusions"] == old["exclusions"]
        assert new["seed"] == old["seed"] and new["n_boot"] == old["n_boot"]
    assert b3["directions"] == v2[study.E1_B3_ID]["directions"]
    # the spec pins the fold scheme it was frozen under: a changed folds.yaml is a new family
    scheme = dataclasses.asdict(load_country_scheme("japan"))
    for spec in specs.values():
        assert spec["fold_scheme"] == scheme
        assert [s for s in _strings(spec) if "in_fold" in s] == []


def test_the_per_fold_specs_are_plain_json_fresh_copies_and_freeze(tmp_path):
    from horsetrack.core.eval.registry import Registry

    first, second = study.per_fold_family_specs(), study.per_fold_family_specs()
    assert first == second
    for spec in first.values():
        assert json.loads(json.dumps(spec, allow_nan=False)) == spec
    first[B3_PF]["members"].clear()
    assert len(study.per_fold_family_specs()[B3_PF]["members"]) == 42
    reg = Registry(tmp_path / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl")
    for family_id, spec in second.items():
        digest = reg.freeze_family(family_id, copy.deepcopy(spec))
        assert reg.frozen_spec_hash(family_id) == digest == exacta_run._spec_hash(spec)
