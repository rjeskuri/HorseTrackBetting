"""Loading the current v0 fits for the shadow runner (ticket #120).

Contract pinned here (`horsetrack.countries.japan.models.v0.loader`):
- `load_shadow_models(pointer: Path | None = None) -> ShadowModels | None`. The pointer defaults to
  `data_dir()/models/japan/v0/current.json` (the data dir read on every call), a JSON object
  {"win": file, "ordering": file, "nowcast": file}. A relative file name resolves against the
  pointer's own directory.
- Each file is loaded through its class's `from_json`: WinCalibration (#114), OrderingFit (#115),
  NowcastV0 (#116). `ids` maps "win", "ordering" and "nowcast" to each fit's `fit_id`.
- No "ordering" entry (absent or null) gives `ordering=None`; the other two still load.
- A missing pointer, or a missing win or nowcast file, returns None with exactly one WARNING log
  record and no exception.
- (#125) An optional "residual" entry names a ResidualFit file (its own `to_json`, base embedded).
  When it is present (not absent, not null) `win` is that ResidualFit, in place of the plain
  calibration, and `ids` gains "residual": its fit_id; "win" stays the calibration's fit_id.
- (#411) A nowcast file whose "model" is "nowcast_c1" (#85's `Nowcast.to_json`) loads as
  `models.nowcast.ShadowNowcast`, whose `draws(board, *, pool, tau_min, shocks)` returns `PoolDraws`
  of the structural model. The race's grade is its `race_class` on the stored race card
  (`processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet`, by race), "grade=none" when the card
  leaves it null or there is no card; never None for a graded race.

The fits are tiny, built here and written with each class's own `to_json`. All data is synthetic.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.countries.japan.board.snapshot import DecisionGrid, RaceBoard, RaceContext
from horsetrack.countries.japan.models.v0.calibration import WinCalibration
from horsetrack.countries.japan.models.v0.nowcast import DriftCell, NowcastV0
from horsetrack.countries.japan.models.v0.ordering import OrderingFit
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels

GRID = DecisionGrid((60, 30, 16, 5, 2, 1), 2)
POOLS = ("WIN", "PLACE", "QUINELLA_PLACE")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    return d


def fits(tag: str, slope: float) -> tuple[WinCalibration, OrderingFit, NowcastV0]:
    """Three small fits whose ids and parameters vary with `tag` and `slope`."""
    win = WinCalibration(
        knots=(-4.0, -0.5),
        coef=(slope,),
        fit_id=f"win-{tag}",
        train_start=date(2015, 1, 4),
        train_end=date(2024, 12, 28),
        n_races=120 + len(tag),
    )
    ordering = OrderingFit("discounted_harville", (0.8 + slope / 10, 0.7), f"ord-{tag}", 99)
    nowcast = NowcastV0(
        bands=(2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
        grid=GRID,
        prior_n=50.0,
        days=("2026-11-21", "2026-11-22"),
        sigma0={(pool, tau): 0.05 + 0.01 * tau / 60 for pool in POOLS for tau in GRID.minutes},
        cells={("WIN", 2, 1): DriftCell("WIN", 2, 1, 7, -0.01 * slope, 0.04)},
        fit_id=f"now-{tag}",
    )
    return win, ordering, nowcast


def write_fits(folder: Path, names: dict[str, str], objects, *, pointer_name: str) -> Path:
    """Each fit to `folder/<name>` with its own to_json, and a pointer naming them."""
    folder.mkdir(parents=True, exist_ok=True)
    win, ordering, nowcast = objects
    for key, obj in (("win", win), ("ordering", ordering), ("nowcast", nowcast)):
        if key in names:
            (folder / names[key]).write_text(obj.to_json(), encoding="utf-8")
    pointer = folder / pointer_name
    pointer.write_text(json.dumps(names), encoding="utf-8")
    return pointer


def default_folder() -> Path:
    return paths.data_dir() / "models" / "japan" / "v0"


def load(pointer: Path | None):
    from horsetrack.countries.japan.models.v0.loader import load_shadow_models

    return load_shadow_models() if pointer is None else load_shadow_models(pointer)


CASES = {
    "default pointer": dict(tag="a1b2c3", slope=1.15, explicit=False, ordering=True),
    "explicit pointer": dict(tag="ff00ee", slope=0.9, explicit=True, ordering=True),
    "no ordering fit": dict(tag="0badc0", slope=1.3, explicit=False, ordering=False),
}


@pytest.mark.parametrize("case", list(CASES.values()), ids=list(CASES))
def test_load_shadow_models_reads_the_current_pointer(tmp_path, case):
    objects = fits(case["tag"], case["slope"])
    win, ordering, nowcast = objects
    names = {
        "win": f"win_calibration-{case['tag']}.json",
        "ordering": f"ordering-{case['tag']}.json",
        "nowcast": f"nowcast-{case['tag']}.json",
    }
    if not case["ordering"]:
        del names["ordering"]
    if case["explicit"]:
        pointer = write_fits(tmp_path / "elsewhere", names, objects, pointer_name="pick.json")
        # a decoy at the default place must not be read when a pointer is given
        write_fits(default_folder(), dict(names), fits("decoy", 2.0), pointer_name="current.json")
        got = load(pointer)
    else:
        write_fits(default_folder(), names, objects, pointer_name="current.json")
        got = load(None)

    assert isinstance(got, ShadowModels)
    assert isinstance(got.win, WinCalibration) and got.win == win
    assert isinstance(got.nowcast, NowcastV0) and got.nowcast == nowcast
    assert got.ids["win"] == win.fit_id
    assert got.ids["nowcast"] == nowcast.fit_id
    if case["ordering"]:
        assert isinstance(got.ordering, OrderingFit) and got.ordering == ordering
        assert got.ids == {
            "win": win.fit_id,
            "ordering": ordering.fit_id,
            "nowcast": nowcast.fit_id,
        }
    else:
        assert got.ordering is None


MISSING = {
    "no pointer": dict(write=False, drop=None, explicit=False),
    "no pointer at the given path": dict(write=False, drop=None, explicit=True),
    "win file missing": dict(write=True, drop="win", explicit=False),
    "nowcast file missing": dict(write=True, drop="nowcast", explicit=False),
    "win file missing, given pointer": dict(write=True, drop="win", explicit=True),
}


@pytest.mark.parametrize("case", list(MISSING.values()), ids=list(MISSING))
def test_load_shadow_models_returns_none_when_a_file_is_missing(tmp_path, caplog, case):
    folder = tmp_path / "given" if case["explicit"] else default_folder()
    pointer_name = "pick.json" if case["explicit"] else "current.json"
    names = {"win": "w.json", "ordering": "o.json", "nowcast": "n.json"}
    if case["write"]:
        write_fits(folder, names, fits("x1", 1.1), pointer_name=pointer_name)
        (folder / names[case["drop"]]).unlink()
    pointer = folder / pointer_name

    with caplog.at_level(logging.WARNING):
        got = load(pointer if case["explicit"] else None)

    assert got is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]


RESIDUAL_CASES = {
    "residual named": dict(tag="r1e5d0", gamma=0.2, named=True, ordering=True),
    "residual named, no ordering": dict(tag="r2c4a8", gamma=-0.15, named=True, ordering=False),
    "residual null": dict(tag="r3b7f1", gamma=0.2, named=None, ordering=True),
    "no residual key": dict(tag="r4d9e2", gamma=0.2, named=False, ordering=True),
}


@pytest.mark.parametrize("case", list(RESIDUAL_CASES.values()), ids=list(RESIDUAL_CASES))
def test_load_shadow_models_prefers_the_residual_when_named(case):
    # imported here so the other loader tests still collect before the module exists
    from horsetrack.countries.japan.models.v0.residual import ResidualFit

    objects = fits(case["tag"], 1.1)
    win, ordering, nowcast = objects
    residual = ResidualFit(
        base=win,
        segments=("field_16_plus",),
        gamma=(case["gamma"],),
        ridge=1.0,
        fit_id=f"res-{case['tag']}",
    )
    names = {
        "win": f"win_calibration-{case['tag']}.json",
        "ordering": f"ordering-{case['tag']}.json",
        "nowcast": f"nowcast-{case['tag']}.json",
    }
    if not case["ordering"]:
        del names["ordering"]
    folder = default_folder()
    pointer = write_fits(folder, names, objects, pointer_name="current.json")
    if case["named"] is not None:
        res_name = f"residual-{case['tag']}.json"
        (folder / res_name).write_text(residual.to_json(), encoding="utf-8")
        if case["named"]:
            pointer.write_text(json.dumps({**names, "residual": res_name}), encoding="utf-8")
    else:
        pointer.write_text(json.dumps({**names, "residual": None}), encoding="utf-8")

    got = load(None)

    assert isinstance(got, ShadowModels)
    assert got.nowcast == nowcast
    assert got.ordering == (ordering if case["ordering"] else None)
    want_ids = {"win": win.fit_id, "nowcast": nowcast.fit_id}
    if case["ordering"]:
        want_ids["ordering"] = ordering.fit_id
    race = RaceContext(race_id="jra:2026-11-28:TOK:12", venue="TOK", race_no=12, starters=16)
    q = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
    if case["named"]:
        assert isinstance(got.win, ResidualFit) and got.win == residual
        assert got.win.base == win
        assert got.ids == {**want_ids, "residual": residual.fit_id}
        # the residual is what the shadow values will call: it moves p in a 16-runner race
        assert np.abs(got.win.predict(q, race) - win.predict(q)).max() > 1e-3
    else:
        assert isinstance(got.win, WinCalibration) and got.win == win
        assert got.ids == want_ids


# ---------------------------------------------------------------- the C1 pointer (#411)

C1_GRID = (60, 30, 16, 5, 2, 1)
C1_KNOTS = (0.0, 5.0, 10.0, 20.0, 40.0, 120.0)
C1_PROFILE = (0.0, 0.35, 0.6, 0.9, 1.4, 2.5)  # E ln(V_final / V_m) at each knot
RANK_BETA = 0.15  # the ln-rank feature's coefficient; the velocity features are 0 in the adapter
TAKEOUT = {"WIN": 0.20, "PLACE": 0.20, "QUINELLA_PLACE": 0.225}  # config/jra.yaml
FLOOR = 1.1  # min_dividend 110 for PLACE and Wide
RACE_DAY = "2026-11-22"


def c1_sigma(pool: str, tau: int) -> float:
    return 0.25 + tau / 120 + (0.1 if pool != "WIN" else 0.0)


def c1_fit_json(effects: dict[str, float], fit_id: str) -> str:
    """A #85 structural-nowcast fit file, in the format `Nowcast.to_json` writes ("model":
    "nowcast_c1"), built by hand so every parameter is known here."""
    growth = [
        {
            "pool": pool,
            "knots": list(C1_KNOTS),
            "profile": list(C1_PROFILE),
            "effects": dict(effects),
            "log_v_final": 17.5,
            "n_obs": 400,
        }
        for pool in TAKEOUT
    ]
    cells = [
        {
            "pool": pool,
            "tau_min": tau,
            "n_cases": 50,
            "n_calibration": 50,
            "share_floored": 0.0,
            "beta": [0.0, 0.0, RANK_BETA],
            "sigma": c1_sigma(pool, tau),
            "sigma_rw": 0.1,
            "train_log_score": 0.0,
            "train_log_score_rw": 0.0,
        }
        for pool in TAKEOUT
        for tau in C1_GRID
    ]
    return json.dumps(
        {
            "model": "nowcast_c1",
            "fit_id": fit_id,
            "days": ["2026-10-03", "2026-10-04"],
            "grid": list(C1_GRID),
            "features": True,
            "fixed": None,
            "growth": growth,
            "cells": cells,
        }
    )


def write_c1_pointer(folder: Path, effects: dict[str, float], fit_id: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    win, _, _ = fits("c1win", 1.05)
    (folder / "win.json").write_text(win.to_json(), encoding="utf-8")
    (folder / "c1.json").write_text(c1_fit_json(effects, fit_id), encoding="utf-8")
    pointer = folder / "current.json"
    pointer.write_text(json.dumps({"win": "win.json", "nowcast": "c1.json"}), encoding="utf-8")
    return pointer


def write_card(venue: str, classes: dict[int, str | None], *, day: str = RACE_DAY) -> None:
    """The meeting's stored race card (`processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet`)
    with each race's `race_class`, the grade the C1 fit's effects are keyed by."""
    path = paths.processed_dir("jra") / "upcoming" / f"jra_{day}_{venue}" / "races.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    nos = sorted(classes)
    pq.write_table(
        pa.table(
            {
                "race_id": pa.array([f"jra:{day}:{venue}:{no:02d}" for no in nos], pa.string()),
                "race_no": pa.array(nos, pa.int64()),
                "race_class": pa.array([classes[no] for no in nos], pa.string()),
                "post_time_local": pa.array(
                    [f"{day}T{9 + no:02d}:30:00+09:00" for no in nos], pa.string()
                ),
            }
        ),
        path,
    )


def c1_board(venue, race_no, win, *, minutes: float, place=None, wide=None) -> RaceBoard:
    post = datetime(2026, 11, 22, race_no, 0, tzinfo=timezone.utc)
    capture = post - timedelta(minutes=minutes)
    return RaceBoard(
        race_id=f"jra:{RACE_DAY}:{venue}:{race_no:02d}",
        post_utc=post,
        at_utc=capture,
        capture_utc=capture,
        final=False,
        win=dict(win),
        place=dict(place or {}),
        wide=dict(wide or {}),
        totals={"WIN": 3e7, "PLACE": 1e7, "QUINELLA_PLACE": 8e6},
        scratched=frozenset(),
    )


def bucket(tau: int) -> int:
    above = [m for m in C1_GRID if m >= tau]
    return min(above) if above else max(C1_GRID)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def expected_win_odds(win, *, minutes, effect, tau, shocks) -> np.ndarray:
    """#85's structural draw on a WIN board, from first principles: s = 1/O normalised,
    phi = exp(-h(m) exp(effects)) with h the profile at the board's minutes to post, the late
    money softmax(ln s + beta_rank ln rank + sigma eps), odds keeping the sum of 1/O."""
    keys = sorted(win)
    inverse = 1.0 / np.array([win[h] for h in keys])
    s = inverse / inverse.sum()
    h = max(float(np.interp(minutes, C1_KNOTS, C1_PROFILE)), 0.0)
    phi = min(max(math.exp(-h * math.exp(effect)), 1e-4), 1.0)
    rank = np.empty(len(s))
    rank[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
    eps = np.column_stack([np.asarray(shocks[h], float) for h in keys])
    late = softmax(np.log(s) + RANK_BETA * np.log(rank) + c1_sigma("WIN", bucket(tau)) * eps)
    drawn = phi * s + (1.0 - phi) * late
    return 1.0 / (drawn * inverse.sum())


def shocks_for(runners, n: int, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {h: rng.standard_normal(n) for h in runners}


EFFECTS = {
    "venue=TOK": 0.10,
    "venue=KYO": -0.20,
    "race_no=11": 0.05,
    "race_no=9": -0.05,
    "grade=G1": 0.90,
    "grade=G3": -0.60,
    "grade=none": 0.20,
}
# (venue, race_no, the card's race_class or "no card", minutes to post on the board, tau_min,
#  the summed effect the draw must use)
GRADE_CASES = (
    ("TOK", 11, "G1", 6.5, 5, 0.10 + 0.05 + 0.90),
    ("TOK", 12, "G3", 17.0, 16, 0.10 + 0.00 - 0.60),
    ("TOK", 10, None, 31.0, 30, 0.10 + 0.00 + 0.20),
    ("KYO", 9, "no card", 4.0, 4, -0.20 - 0.05 + 0.20),
    ("TOK", 11, "G1", 2.5, 2, 0.10 + 0.05 + 0.90),
)
WIN_BOARD = {1: 2.4, 2: 4.1, 3: 6.8, 4: 9.5, 5: 15.0, 6: 31.0, 7: 48.0, 8: 77.0}


def test_a_c1_pointer_loads_the_structural_nowcast_and_draws_with_the_race_grade(data_dir):
    """#411 (#85's low findings): a pointer whose nowcast file is a "nowcast_c1" fit loads through
    `load_shadow_models` as `models.nowcast.ShadowNowcast` (ids["nowcast"] its fit_id), and its
    draws are `PoolDraws` of the structural model at the board. The race's grade is the stored
    card's `race_class`, not None: a graded race's phi uses its grade effect; a race the card
    leaves unclassed, or with no card, uses the "grade=none" effect. WIN draws are checked
    against the model computed here from the fit's own parameters, per case of GRADE_CASES."""
    from horsetrack.countries.japan.models.nowcast import ShadowNowcast
    from horsetrack.countries.japan.models.v0.shadow_values import PoolDraws

    write_card("TOK", {10: None, 11: "G1", 12: "G3"})
    pointer = write_c1_pointer(default_folder(), EFFECTS, "c1-411a")

    got = load(None)

    assert isinstance(got, ShadowModels)
    assert isinstance(got.nowcast, ShadowNowcast)
    assert got.nowcast.fit_id == "c1-411a"
    assert got.ids["nowcast"] == "c1-411a"
    assert load(pointer).ids["nowcast"] == "c1-411a"  # the same file through an explicit pointer

    n = 64
    for i, (venue, no, grade, minutes, tau, effect) in enumerate(GRADE_CASES):
        board = c1_board(venue, no, WIN_BOARD, minutes=minutes)
        shocks = shocks_for(WIN_BOARD, n, seed=100 + i)
        draws = got.nowcast.draws(board, pool="WIN", tau_min=tau, shocks=shocks)
        assert isinstance(draws, PoolDraws)
        assert draws.pool == "WIN"
        assert draws.combinations == tuple(str(h) for h in sorted(WIN_BOARD))
        assert draws.odds.shape == (n, len(WIN_BOARD))
        want = expected_win_odds(WIN_BOARD, minutes=minutes, effect=effect, tau=tau, shocks=shocks)
        np.testing.assert_allclose(
            draws.odds, want, rtol=1e-9, err_msg=f"{venue} R{no} (card class {grade!r}), tau {tau}"
        )


def place_board(money: np.ndarray, places: int, takeout: float) -> dict[int, tuple[float, float]]:
    """Low / high figures of the stake-back-share rule for `money` (shares of the pool)."""
    payout = 1.0 - takeout
    out = {}
    for i, own in enumerate(money):
        others = np.sort(np.delete(money, i))
        lo = 1.0 + (payout - own - others[-(places - 1) :].sum()) / (places * own)
        hi = 1.0 + (payout - own - others[: places - 1].sum()) / (places * own)
        out[i + 1] = (max(lo, FLOOR), max(hi, FLOOR))
    return out


def test_the_c1_adapter_uses_the_grade_in_every_pool(data_dir, tmp_path):
    """PLACE and Wide draws of the C1 adapter use the card's grade too. Two fits that differ only
    in where an effect of 0.9 sits, on the grade G1 or on the venue TOK, give the same phi for
    TOK's G1 race 11, so their draws must agree in every pool; with the grade dropped (None) they
    would not. Draws are PoolDraws: n x combinations, Wide pairs "a,b", PLACE and Wide low
    figures at or above the 110 floor and at or below the high figures."""
    from horsetrack.countries.japan.models.v0.loader import load_shadow_models

    write_card("TOK", {11: "G1"})
    by_grade = load_shadow_models(write_c1_pointer(tmp_path / "grade", {"grade=G1": 0.9}, "g"))
    by_venue = load_shadow_models(write_c1_pointer(tmp_path / "venue", {"venue=TOK": 0.9}, "v"))
    assert by_grade is not None and by_venue is not None

    runners = sorted(WIN_BOARD)
    money = 1.0 / np.array([WIN_BOARD[h] for h in runners])
    money = money / money.sum()
    place = place_board(money, 3, TAKEOUT["PLACE"])
    pairs = [(a, b) for a in runners for b in runners if a < b]
    pair_money = np.array([money[a - 1] * money[b - 1] for a, b in pairs])
    pair_money = pair_money / pair_money.sum()
    wide_figures = place_board(pair_money, 3, TAKEOUT["QUINELLA_PLACE"]).values()
    wide = dict(zip(pairs, wide_figures, strict=True))
    board = c1_board("TOK", 11, WIN_BOARD, minutes=9.0, place=place, wide=wide)
    n = 48
    shocks = shocks_for(runners, n, seed=7)

    expected_combinations = {
        "WIN": tuple(str(h) for h in runners),
        "PLACE": tuple(str(h) for h in runners),
        "QUINELLA_PLACE": tuple(f"{a},{b}" for a, b in pairs),
    }
    for pool, combinations in expected_combinations.items():
        a = by_grade.nowcast.draws(board, pool=pool, tau_min=16, shocks=shocks)
        b = by_venue.nowcast.draws(board, pool=pool, tau_min=16, shocks=shocks)
        assert a.pool == pool and a.combinations == combinations
        assert a.odds.shape == (n, len(combinations))
        np.testing.assert_allclose(a.odds, b.odds, rtol=1e-9, err_msg=pool)
        if pool != "WIN":
            assert a.odds_hi is not None and a.odds_hi.shape == a.odds.shape
            np.testing.assert_allclose(a.odds_hi, b.odds_hi, rtol=1e-9, err_msg=pool)
            assert (a.odds >= FLOOR - 1e-12).all(), pool
            assert (a.odds <= a.odds_hi + 1e-9).all(), pool
