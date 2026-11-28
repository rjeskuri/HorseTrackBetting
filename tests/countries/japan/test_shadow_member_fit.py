"""The horse member's fit and its place in the v0 pointer (ticket #717).

Contract pinned here (the coder implements against it):

`horsetrack.countries.japan.models.v0.member` (new):
- `fit_member(rows, *, segment, today, features) -> MemberFit`: #438's offset model
  (`studies/debut_members.py`: `fit_offset` on the rows, `predict_offset` to predict) fitted on
  the rows of race class `segment` ("DEBUT" or "MAIDEN", `debut_members.CLASSES`; any other is a
  ValueError) on development days strictly before `today`. Rows of holdout days
  (`core.eval.holdout.is_holdout_day`), of `today` and of later days never shape the fit; with no
  such row left it is a ValueError. `rows` has #438's columns: race_id, race_date (a date),
  race_class, meeting_id, won, ln_q and the `features` columns (#438's `feature_columns` names).
- `MemberFit` (frozen, comparable with ==): `segment`, `fit_id` (a non-empty text, the same for
  the same fit and different for another), `train_end` (its last training day), `n_races` (its
  training races); `to_json()` / `MemberFit.from_json(text)` round-trip it exactly;
  `predict(q, features)` -> P(win) per runner of one race, `q` the market's normalised Win
  probabilities and `features` a DataFrame of the member columns in the same runner order (a
  column left out, or a NaN, is a missing value). The probabilities sum to 1, and are q exactly
  when the member says nothing about the race (every runner with the same values, or none).

`horsetrack.countries.japan.models.v0.loader.load_shadow_models` (extended; test_v0_loader.py
still holds): an optional "member" entry of `current.json` keyed by segment,
{"DEBUT": {"fit": file, "k": number}, "MAIDEN": {...}}, a relative file resolving against the
pointer's directory. `ShadowModels.member` maps each loaded segment to an entry with `.fit` (the
MemberFit) and `.k` (the p/q threshold the leaderboard ranks best: the mark's k lives here); no
"member" entry (absent or null) leaves it empty (falsy), the market only. A named member file
that is missing drops that segment alone, with one warning, and the rest still loads (never None
for the whole models). A member file that exists but does not parse raises; so does a fit whose
segment is not its key, a key other than DEBUT or MAIDEN, and a k that is missing, not a number,
not finite or not above 0. The member is display only: `ids` and the other fits are what they are
without it.

All rows are synthetic (member_synth.py); expected values come from #438's own functions.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels

# isort: split
import member_synth as ms

DAYS = ms.weekend_days(date(2026, 6, 6), date(2026, 12, 20))
HOLDOUT = [d for d in DAYS if is_holdout_day("japan", d)]
TODAYS = (date(2026, 12, 5), date(2026, 12, 12))  # development Saturdays after the holdout start
assert HOLDOUT and all(not is_holdout_day("japan", d) for d in TODAYS)
assert any(d < TODAYS[0] for d in HOLDOUT)  # a holdout day sits inside the training window
# rows of holdout days, of the fit day and after are drawn from the opposite signal: a fit that
# reads them comes out different
ROWS = ms.synthetic_rows(
    DAYS, seed=717, flip_days=[d for d in DAYS if d in HOLDOUT or d >= TODAYS[0]]
)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def member():
    from horsetrack.countries.japan.models.v0 import member as module

    return module


def fit(segment: str, today: date, rows: pd.DataFrame = ROWS):
    return member().fit_member(rows, segment=segment, today=today, features=list(ms.FEATURES))


def unseen_races() -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Races no fit saw: (q, member columns), with a few missing JRDB values."""
    rng = np.random.default_rng(4242)
    out = []
    for n in (6, 9, 12, 16):
        q = ms.market_q(ms.odds_field(rng, n))
        x = {f: rng.normal(0.0, ms.SCALE[f], n) for f in ms.FEATURES}
        x[ms.JRDB_FEATURE][rng.random(n) < 0.2] = np.nan
        out.append((q, x))
    return out


def frame(x: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({f: np.asarray(v, dtype=float) for f, v in x.items()})


# ---------------------------------------------------------------- the fit


@pytest.mark.parametrize("today", TODAYS, ids=str)
@pytest.mark.parametrize("segment", ms.SEGMENTS)
def test_the_member_is_438s_offset_model_on_development_days_before_today(segment, today):
    got = fit(segment, today)
    train = ms.training_rows(ROWS, segment, today)
    want_fit = ms.offset_fit(ROWS, segment, today)

    assert got.segment == segment
    assert got.n_races == train["race_id"].nunique()
    assert got.train_end == max(train["race_date"])
    assert got.train_end < today and not is_holdout_day("japan", got.train_end)
    for q, x in unseen_races():
        p = np.asarray(got.predict(q, frame(x)), dtype=float)
        want = ms.expected_member_p(want_fit, q, x)
        np.testing.assert_allclose(p, want, rtol=1e-6, atol=1e-9, err_msg=f"{segment} {today}")
        assert math.isclose(float(p.sum()), 1.0, rel_tol=1e-9)
        assert np.abs(p - q).max() > 1e-3  # the fixture: the member moves away from the market


def test_holdout_days_today_and_later_days_never_shape_the_fit():
    today = TODAYS[0]
    dev_only = ms.training_rows(ROWS, "DEBUT", today)
    # the same development rows with and without every other row: the same fit
    a, b = fit("DEBUT", today), fit("DEBUT", today, rows=dev_only)
    assert a == b
    assert a.to_json() == b.to_json()
    # the fixture: had the excluded rows been read, the weights would differ
    leaked = ROWS[(ROWS["race_class"] == "DEBUT")]
    design, leak = ms.dm.fit_offset(leaked, list(ms.FEATURES))
    _, clean = ms.offset_fit(ROWS, "DEBUT", today)
    assert np.abs(np.asarray(leak.coef) - np.asarray(clean.coef)).max() > 1e-3

    # nothing left to learn from: only holdout days, the fit day and later days
    excluded = ROWS[[d in HOLDOUT or d >= today for d in ROWS["race_date"]]]
    with pytest.raises(ValueError):
        fit("DEBUT", today, rows=excluded)


@pytest.mark.parametrize("segment", ["C1", "G1", "OPEN", ""])
def test_a_segment_other_than_debut_or_maiden_is_refused(segment):
    with pytest.raises(ValueError):
        fit(segment, TODAYS[0])


def test_the_prediction_is_the_market_when_the_member_says_nothing():
    fits = [fit(s, t) for s in ms.SEGMENTS for t in TODAYS]
    rng = np.random.default_rng(99)
    for n in (5, 8, 14, 18):
        q = ms.market_q(ms.odds_field(rng, n))
        same = {f: np.full(n, rng.normal(0.0, ms.SCALE[f])) for f in ms.FEATURES}
        none = {f: np.full(n, np.nan) for f in ms.FEATURES}
        for member_fit in fits:
            for label, x in (("the same values", frame(same)), ("all missing", frame(none))):
                p = np.asarray(member_fit.predict(q, x), dtype=float)
                np.testing.assert_allclose(p, q, rtol=1e-9, err_msg=(n, label))
            no_columns = pd.DataFrame(index=range(n))
            p = np.asarray(member_fit.predict(q, no_columns), dtype=float)
            np.testing.assert_allclose(p, q, rtol=1e-9, err_msg=(n, "no member columns"))


def test_the_fit_round_trips_through_json():
    seen_ids = set()
    for segment in ms.SEGMENTS:
        for today in TODAYS:
            original = fit(segment, today)
            text = original.to_json()
            json.loads(text)  # a JSON document
            back = member().MemberFit.from_json(text)
            assert back == original
            assert back.to_json() == text
            assert isinstance(back.fit_id, str) and back.fit_id
            for q, x in unseen_races():
                np.testing.assert_allclose(back.predict(q, frame(x)), original.predict(q, frame(x)))
            assert fit(segment, today).fit_id == original.fit_id  # the same fit, the same id
            seen_ids.add(original.fit_id)
    assert len(seen_ids) == len(ms.SEGMENTS) * len(TODAYS)  # another fit, another id


# ---------------------------------------------------------------- the pointer


base_fits, write_pointer, default_folder = ms.base_fits, ms.write_pointer, ms.default_folder


def load(pointer: Path | None = None):
    from horsetrack.countries.japan.models.v0.loader import load_shadow_models

    return load_shadow_models() if pointer is None else load_shadow_models(pointer)


FITS = {(s, t): None for s in ms.SEGMENTS for t in TODAYS}


def fitted(segment: str, today: date = TODAYS[0]):
    if FITS[(segment, today)] is None:
        FITS[(segment, today)] = fit(segment, today)
    return FITS[(segment, today)]


def check_base(got, objects) -> None:
    win, ordering, nowcast = objects
    assert isinstance(got, ShadowModels)
    assert got.win == win and got.ordering == ordering and got.nowcast == nowcast
    assert got.ids == {"win": win.fit_id, "ordering": ordering.fit_id, "nowcast": nowcast.fit_id}


POINTER_CASES = {
    "both segments, default pointer": dict(ks={"DEBUT": 1.25, "MAIDEN": 1.6}, explicit=False),
    "both segments, other ks": dict(ks={"DEBUT": 2.0, "MAIDEN": 1.05}, explicit=False),
    "debut only, given pointer": dict(ks={"DEBUT": 1.4}, explicit=True),
    "maiden only, the later fit": dict(ks={"MAIDEN": 1.33}, explicit=False, today=TODAYS[1]),
}


@pytest.mark.parametrize("case", list(POINTER_CASES.values()), ids=list(POINTER_CASES))
def test_the_pointer_names_a_member_per_segment(tmp_path, case):
    today = case.get("today", TODAYS[0])
    folder = tmp_path / "elsewhere" if case["explicit"] else default_folder()
    folder.mkdir(parents=True, exist_ok=True)
    entry = {}
    for segment, k in case["ks"].items():
        name = f"member-{segment.lower()}.json"
        (folder / name).write_text(fitted(segment, today).to_json(), encoding="utf-8")
        entry[segment] = {"fit": name, "k": k}
    pointer, objects = write_pointer(
        folder, entry, name="pick.json" if case["explicit"] else "current.json"
    )

    got = load(pointer if case["explicit"] else None)

    check_base(got, objects)
    assert set(got.member) == set(case["ks"])
    for segment, k in case["ks"].items():
        assert got.member[segment].fit == fitted(segment, today), segment
        assert got.member[segment].k == pytest.approx(k), segment


def test_an_absolute_member_path_loads(tmp_path):
    outside = tmp_path / "fits" / "debut.json"
    outside.parent.mkdir(parents=True)
    outside.write_text(fitted("DEBUT").to_json(), encoding="utf-8")
    pointer, objects = write_pointer(default_folder(), {"DEBUT": {"fit": str(outside), "k": 1.5}})
    got = load()
    check_base(got, objects)
    assert got.member["DEBUT"].fit == fitted("DEBUT") and got.member["DEBUT"].k == 1.5


@pytest.mark.parametrize("entry", [..., None], ids=["no member key", "member null"])
def test_a_pointer_without_a_member_is_the_market_only(entry, caplog):
    pointer, objects = write_pointer(default_folder(), entry)
    with caplog.at_level(logging.WARNING):
        got = load()
    check_base(got, objects)
    assert not got.member
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_missing_member_file_drops_its_segment_with_one_warning(caplog):
    folder = default_folder()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "maiden.json").write_text(fitted("MAIDEN").to_json(), encoding="utf-8")
    entry = {"DEBUT": {"fit": "gone.json", "k": 1.25}, "MAIDEN": {"fit": "maiden.json", "k": 1.6}}
    pointer, objects = write_pointer(folder, entry)

    with caplog.at_level(logging.WARNING):
        got = load()

    check_base(got, objects)  # the values' fits load as before
    assert set(got.member) == {"MAIDEN"}
    assert got.member["MAIDEN"].fit == fitted("MAIDEN")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]

    # every named file missing: the market only, still one warning per missing file
    caplog.clear()
    pointer.write_text(
        json.dumps(
            {**json.loads(pointer.read_text()), "member": {"DEBUT": {"fit": "x.json", "k": 2}}}
        )
    )
    with caplog.at_level(logging.WARNING):
        got = load()
    check_base(got, objects)
    assert not got.member
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


BROKEN = {
    "not json": lambda: "{ not json",
    "another model's file": lambda: base_fits("zz")[0].to_json(),
    "an empty object": lambda: "{}",
}


@pytest.mark.parametrize("text", list(BROKEN.values()), ids=list(BROKEN))
def test_a_member_file_that_does_not_parse_raises(text):
    folder = default_folder()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "debut.json").write_text(text(), encoding="utf-8")
    write_pointer(folder, {"DEBUT": {"fit": "debut.json", "k": 1.25}})
    with pytest.raises((ValueError, KeyError, TypeError)):
        load()


BAD_ENTRIES = {
    "fit of the other segment": lambda: {"DEBUT": {"fit": "maiden.json", "k": 1.25}},
    "a segment that is not debut or maiden": lambda: {"C1": {"fit": "maiden.json", "k": 1.25}},
    "no k": lambda: {"MAIDEN": {"fit": "maiden.json"}},
    "k zero": lambda: {"MAIDEN": {"fit": "maiden.json", "k": 0}},
    "k negative": lambda: {"MAIDEN": {"fit": "maiden.json", "k": -1.2}},
    "k text": lambda: {"MAIDEN": {"fit": "maiden.json", "k": "high"}},
    "k infinite": lambda: {"MAIDEN": {"fit": "maiden.json", "k": float("inf")}},
}


@pytest.mark.parametrize("entry", list(BAD_ENTRIES.values()), ids=list(BAD_ENTRIES))
def test_a_member_entry_that_does_not_fit_its_segment_raises(entry):
    folder = default_folder()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "maiden.json").write_text(fitted("MAIDEN").to_json(), encoding="utf-8")
    write_pointer(folder, entry())
    with pytest.raises(ValueError):
        load()
