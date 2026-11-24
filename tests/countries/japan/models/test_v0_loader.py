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

The fits are tiny, built here and written with each class's own `to_json`. All data is synthetic.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pytest

from horsetrack.core import paths
from horsetrack.countries.japan.board.snapshot import DecisionGrid
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
