"""The drift nowcast refitted on the JRA-VAN odds history (ticket #500, epic #73).

Module under test: `horsetrack.countries.japan.models.v0.nowcast_store` (new), and the command
`horsetrack japan nowcast-fit --from-store`. Contract pinned here:

- `store_examples() -> pd.DataFrame`: #116's (low figure at t, final low figure) pairs from every
  `processed/jravan/odds_timeseries/<YYYY-MM-DD>.parquet` day from `PRICE_ERA_START` (D10), built
  exactly as `studies/nowcast.py` builds its `drift_v0` examples (`read_day`, `models.nowcast.
  extract`, `drift_examples`, one day at a time), with at least the columns `day`, `pool`,
  `tau_min`, `odds_t`, `odds_final`. Holdout days (`is_holdout_day`, D14) are dropped by file name:
  their files are never opened. nowcast_v0.yaml's `max_board_age_s` is not applied (5-minute paths).
- `fit_from_store() -> dict`: fits `NowcastV0.fit` on those pairs with nowcast_v0.yaml's bands and
  prior_n and the shipped decision grid; takes the QUINELLA_PLACE cells and sigma0 unchanged from
  the nowcast fit the current pointer names (the JRA-VAN series has no Wide pool); writes
  `nowcast-<fit_id>.json` and its manifest under `$HORSETRACK_DATA_DIR/models/japan/v0/`; points
  `current.json`'s "nowcast" at the new file (every other entry unchanged, the old fit file kept);
  appends one registry record; returns the manifest.
- The manifest holds `fit_id`, `fit_path`, `manifest_path`, `jravan_days` (the development days
  read, sorted), `pool_sources` ({"WIN": "jravan", "PLACE": "jravan", "QUINELLA_PLACE": "logged"}),
  `wide_fit_id` (the fit the Wide cells came from) and `max_board_age_s_applied` (False).
- The registry record's `params` hold `fit_id`, `pool_sources` and `jravan_days`; its sources name
  "jravan".
- Without a current nowcast fit to take Wide from, the command refuses and writes no fit.

Expected pairs are derived here from the synthetic prices by the board-at-t rule (a row counts
from its snapshot_utc; each combination's latest row at or before post - tau; the final board is
the latest row) and checked against the study's own code path; the expected cells are
`NowcastV0.fit` on those pairs (its maths are #116's and unchanged). All data are synthetic, under
a tmp HORSETRACK_DATA_DIR.
"""

from __future__ import annotations

import importlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from horsetrack.core import config, env, paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import default_registry
from horsetrack.countries.japan.board.snapshot import load_decision_grid
from horsetrack.countries.japan.logger.compact import SCHEMA
from horsetrack.countries.japan.models import nowcast as nc
from horsetrack.countries.japan.models.v0.calibration import WinCalibration
from horsetrack.countries.japan.models.v0.loader import load_shadow_models
from horsetrack.countries.japan.models.v0.nowcast import DriftCell, NowcastV0
from horsetrack.countries.japan.models.v0.ordering import OrderingFit
from horsetrack.countries.japan.models.v0.store import PRICE_ERA_START
from horsetrack.countries.japan.studies import nowcast as study

MODULE = "horsetrack.countries.japan.models.v0.nowcast_store"
COUNTRY = "japan"
UTC = timezone.utc
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
POOL_ROW = "__pool__"

# Live snapshots in minutes to post, oldest first; the final capture is a minute after the post.
# Grid taus (60, 30, 16, 5, 2, 1) read the 75, 31, 17, 5 (a row at exactly post - tau counts), 3
# and 1 minute snapshots. Race 2 stops at 3 minutes, so its T-1 board is 120 s old and every T-2
# board is 60 s old: past nowcast_v0.yaml's 30 s / 45 s limits, kept because no limit applies.
LIVE = (75.0, 44.0, 31.0, 20.0, 17.0, 9.0, 5.0, 3.0, 1.5, 1.0)
FINAL_MIN = -1.0
FINAL_INDEX = len(LIVE)
RACES = {1: (8, LIVE), 2: (6, LIVE[:8]), 3: (10, LIVE)}  # race_no: (runners, live minutes)
SCRATCHED_RUNNER = 10  # of race 3, on the day a variant scratches it at the close
WIN_BASE = (1.6, 2.7, 4.4, 7.5, 13.0, 26.0, 60.0, 130.0, 15.0, 9.0)  # every band of the yaml

HELD = (date(2026, 11, 29), date(2026, 12, 1))  # holdout days by the committed rule
PRE_ERA = date(2014, 6, 1)  # before D10's price era
# Two stores whose development days and drifts differ, so no fixed answer fits both.
VARIANTS = {
    "four-days": {
        "days": {
            date(2026, 11, 21): "KYO",
            date(2026, 11, 22): "KYO",
            date(2026, 11, 28): "TOK",
            date(2026, 11, 30): "TOK",
        },
        "scale": 1.0,
        "scratch_day": date(2026, 11, 22),
    },
    "two-days": {
        "days": {date(2015, 3, 1): "NAK", date(2026, 11, 30): "TOK"},
        "scale": -2.5,
        "scratch_day": date(2015, 3, 1),
    },
}
OLD_ID = "c0ffee116a01"


# ---------------------------------------------------------------- synthetic prices


def win_odds(day: date, race_no: int, runner: int, index: int, scale: float) -> float:
    """The WIN low figure of a runner at snapshot `index` (FINAL_INDEX: the final capture)."""
    tilt = ((runner * 7 + race_no * 3 + day.toordinal()) % 5 - 2) * 0.04 * scale
    base = WIN_BASE[runner - 1] * (1.0 + 0.02 * race_no)
    return round(base * math.exp(tilt * index / FINAL_INDEX), 4)


def place_odds(win: float) -> tuple[float, float]:
    lo = round(1.0 + (win - 1.0) / 3.0, 4)
    return lo, round(lo * 1.35, 4)


def post_of(day: date, race_no: int) -> datetime:
    return datetime(day.year, day.month, day.day, 1, 0, tzinfo=UTC) + timedelta(
        minutes=35 * race_no
    )


def race_id_of(day: date, venue: str, race_no: int) -> str:
    return f"jra:{day.isoformat()}:{venue}:{race_no:02d}"


def row(race_id, venue, race_no, pool, combination, post, minutes, *, odds=None, hi=None,
        investment=None, final=False) -> dict:  # fmt: skip
    return {
        "race_id": race_id,
        "venue": venue,
        "race_no": race_no,
        "pool": pool,
        "combination": combination,
        "snapshot_utc": post - timedelta(minutes=minutes),
        "fetched_utc": None,
        "post_utc": post,
        "minutes_to_post": float(minutes),
        "odds": None if odds is None else float(odds),
        "odds_hi": None if hi is None else float(hi),
        "investment": None if investment is None else float(investment),
        "status_raw": None,
        "is_final_capture": bool(final),
        "dividend": None,
        "is_refund": None,
    }


def day_rows(day: date, venue: str, *, scale: float, scratch: bool) -> list[dict]:
    """A JRA-VAN-style day in logger.compact.SCHEMA: WIN and PLACE paths with pool totals, and a
    QUINELLA pool the drift does not read. With `scratch`, race 3's last runner has no price at
    the close (the store writes a scratch with null odds)."""
    rows = []
    for race_no, (runners, live) in RACES.items():
        race_id, post = race_id_of(day, venue, race_no), post_of(day, race_no)
        snaps = [(i, m, False) for i, m in enumerate(live)] + [(FINAL_INDEX, FINAL_MIN, True)]
        for index, minutes, final in snaps:
            for runner in range(1, runners + 1):
                win = win_odds(day, race_no, runner, index, scale)
                lo, hi = place_odds(win)
                if final and scratch and race_no == 3 and runner == SCRATCHED_RUNNER:
                    win = lo = hi = None
                args = (race_id, venue, race_no)
                rows.append(row(*args, WIN, str(runner), post, minutes, odds=win, final=final))
                rows.append(
                    row(*args, PLACE, str(runner), post, minutes, odds=lo, hi=hi, final=final)
                )
            for pool in (WIN, PLACE):
                rows.append(
                    row(race_id, venue, race_no, pool, POOL_ROW, post, minutes,
                        investment=1e6 * (index + 1), final=final)
                )  # fmt: skip
            rows.append(
                row(race_id, venue, race_no, "QUINELLA", "1,2", post, minutes, odds=9.9,
                    final=final)
            )  # fmt: skip
    return rows


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


def jravan_dir(data_dir: Path) -> Path:
    return data_dir / "processed" / "jravan" / "odds_timeseries"


def write_store(data_dir: Path, variant: dict) -> None:
    """The variant's development days, a valid holdout day and a corrupt one (any read raises),
    a pre-era day, and the jravan races table. The days outside development drift sharply, so a
    leak would move the cells."""
    folder = jravan_dir(data_dir)
    race_ids = []
    for day, venue in variant["days"].items():
        rows = day_rows(day, venue, scale=variant["scale"], scratch=day == variant["scratch_day"])
        write_parquet(folder / f"{day.isoformat()}.parquet", rows)
        race_ids += sorted({r["race_id"] for r in rows})
    for day in (HELD[0], PRE_ERA):
        rows = day_rows(day, "HAN", scale=6.0, scratch=False)
        write_parquet(folder / f"{day.isoformat()}.parquet", rows)
        race_ids += sorted({r["race_id"] for r in rows})
    (folder / f"{HELD[1].isoformat()}.parquet").write_bytes(b"a locked holdout day\n")
    races = pa.table(
        {
            "race_id": race_ids,
            "race_class": [None] * len(race_ids),
            "race_date": [date.fromisoformat(r.split(":")[1]) for r in race_ids],
        }
    )
    pq.write_table(races, data_dir / "processed" / "jravan" / "races.parquet")


def expected_pairs(variant: dict) -> list[tuple]:
    """(day, pool, tau, odds_t, odds_final) by the board-at-t rule, from first principles. A race
    with a runner scratched at the close gives no training pair (the study fits on cases whose
    board at t is the final board)."""
    out = []
    for day, _venue in variant["days"].items():
        for race_no, (runners, live) in RACES.items():
            if day == variant["scratch_day"] and race_no == 3:
                continue
            for tau in load_decision_grid().minutes:
                used = max(i for i, m in enumerate(live) if m >= tau)
                for runner in range(1, runners + 1):
                    w_t = win_odds(day, race_no, runner, used, variant["scale"])
                    w_f = win_odds(day, race_no, runner, FINAL_INDEX, variant["scale"])
                    out.append((day.isoformat(), WIN, int(tau), w_t, w_f))
                    out.append(
                        (day.isoformat(), PLACE, int(tau), place_odds(w_t)[0], place_odds(w_f)[0])
                    )
    return sorted(out)


def study_pairs(data_dir: Path, variant: dict) -> list[tuple]:
    """The pairs the study's drift_v0 path builds on the same development days."""
    out = []
    for day in sorted(variant["days"]):
        frame = study.read_day(jravan_dir(data_dir) / f"{day.isoformat()}.parquet")
        cases, _, _ = nc.extract(frame)
        out += as_pairs(study.drift_examples(cases, [day.isoformat()]))
    return sorted(out)


def as_pairs(examples) -> list[tuple]:
    return sorted(
        (str(d)[:10], str(p), int(t), float(o), float(f))
        for d, p, t, o, f in zip(
            examples["day"], examples["pool"], examples["tau_min"], examples["odds_t"],
            examples["odds_final"], strict=True,
        )
    )  # fmt: skip


def assert_pairs_equal(got: list[tuple], expected: list[tuple]) -> None:
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        assert g[:3] == e[:3], (g, e)
        assert g[3] == pytest.approx(e[3], rel=1e-12), (g, e)
        assert g[4] == pytest.approx(e[4], rel=1e-12), (g, e)


# ---------------------------------------------------------------- the current fits


def v0_config() -> dict:
    cfg = config.load_country_yaml(COUNTRY, "nowcast_v0")
    return {"bands": tuple(float(b) for b in cfg["bands"]), "prior_n": float(cfg["prior_n"])}


def old_nowcast() -> NowcastV0:
    """The logged-days fit the pointer names now: Wide cells at every grid tau, and WIN cells the
    history fit must replace."""
    grid = load_decision_grid()
    cfg = v0_config()
    sigma0 = {
        (pool, tau): 0.03 + 0.002 * tau + 0.01 * i
        for i, pool in enumerate((WIN, PLACE, WIDE))
        for tau in grid.minutes
    }
    cells = {}
    for tau in grid.minutes:
        for band in (1, 3):
            cells[(WIDE, tau, band)] = DriftCell(
                WIDE, tau, band, 10 + tau + band, 0.001 * tau - 0.01 * band, 0.02 + 0.001 * tau
            )
    cells[(WIN, 2, 1)] = DriftCell(WIN, 2, 1, 999, 0.5, 0.5)
    cells[(WIN, 60, 6)] = DriftCell(WIN, 60, 6, 999, -0.5, 0.5)
    return NowcastV0(
        cfg["bands"], grid, cfg["prior_n"], ("2026-11-21", "2026-11-22"), sigma0, cells, OLD_ID
    )


def models_dir(data_dir: Path) -> Path:
    return data_dir / "models" / "japan" / "v0"


def write_current_fits(data_dir: Path) -> dict:
    """The deployed v0 fits and their pointer; returns the pointer's names."""
    folder = models_dir(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    win = WinCalibration(
        knots=(-4.0, -0.5),
        coef=(1.1,),
        fit_id="win-500a",
        train_start=date(2015, 1, 4),
        train_end=date(2024, 12, 28),
        n_races=120,
    )
    ordering = OrderingFit("discounted_harville", (0.85, 0.7), "ord-500a", 99)
    names = {
        "win": "win_calibration-win-500a.json",
        "ordering": "ordering-ord-500a.json",
        "nowcast": f"nowcast-{OLD_ID}.json",
    }
    for key, obj in (("win", win), ("ordering", ordering), ("nowcast", old_nowcast())):
        (folder / names[key]).write_text(obj.to_json(), encoding="utf-8")
    (folder / "current.json").write_text(json.dumps(names), encoding="utf-8")
    return names


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    d = (tmp_path / "data").resolve()
    d.mkdir()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(d))
    summary = tmp_path / "summary"  # a registry summary is never regenerated into the repo here
    monkeypatch.setattr("horsetrack.core.eval.registry.summary_dir", lambda country: summary)
    monkeypatch.setattr(study, "summary_dir", lambda country: summary)
    return d


def store_module(monkeypatch, tmp_path):
    """The module under test, with its registry summary folder (if it names one) under tmp."""
    module = importlib.import_module(MODULE)
    monkeypatch.setattr(module, "summary_dir", lambda c: tmp_path / "summary", raising=False)
    return module


def assert_dates_hold() -> None:
    assert all(is_holdout_day(COUNTRY, d) for d in HELD)
    assert PRE_ERA < PRICE_ERA_START
    for variant in VARIANTS.values():
        assert not any(is_holdout_day(COUNTRY, d) for d in variant["days"])
        assert min(variant["days"]) >= PRICE_ERA_START


def resolve(pointer: Path, name: str) -> Path:
    return Path(name) if Path(name).is_absolute() else pointer.parent / name


# ---------------------------------------------------------------- pairs


@pytest.mark.parametrize("name", list(VARIANTS))
def test_store_examples_are_the_study_drift_pairs(name, data_dir, monkeypatch, tmp_path):
    """The from-store pairs equal the study's drift_v0 pairs on the same days and the pairs the
    board-at-t rule gives by hand: every development day, no holdout or pre-era day, a late
    scratch's race out, and the stale T-2 / T-1 boards kept (no board-age limit)."""
    assert_dates_hold()
    variant = VARIANTS[name]
    write_store(data_dir, variant)
    expected = expected_pairs(variant)
    assert_pairs_equal(study_pairs(data_dir, variant), expected)  # the fixture reads as specified

    examples = store_module(monkeypatch, tmp_path).store_examples()
    assert_pairs_equal(as_pairs(examples), expected)
    assert {str(d)[:10] for d in examples["day"]} == {d.isoformat() for d in variant["days"]}


# ---------------------------------------------------------------- the fit


@pytest.mark.parametrize("name", list(VARIANTS))
def test_fit_from_store_fits_the_study_cells_and_keeps_the_logged_wide_cells(
    name, data_dir, monkeypatch, tmp_path
):
    """WIN and PLACE cells and sigma0 are NowcastV0.fit on the study's pairs with the yaml's bands
    and prior_n; QUINELLA_PLACE's are the current fit's, unchanged; the manifest labels each pool's
    source and says no board-age limit was applied; fit and manifest land under the data dir."""
    variant = VARIANTS[name]
    write_store(data_dir, variant)
    write_current_fits(data_dir)
    old_bytes = (models_dir(data_dir) / f"nowcast-{OLD_ID}.json").read_bytes()
    grid = load_decision_grid()
    cfg = v0_config()
    reference = []
    for day in sorted(variant["days"]):
        frame = study.read_day(jravan_dir(data_dir) / f"{day.isoformat()}.parquet")
        reference.append(study.drift_examples(nc.extract(frame)[0], [day.isoformat()]))
    want = NowcastV0.fit(
        pd.concat(reference, ignore_index=True),
        bands=cfg["bands"],
        grid=grid,
        prior_n=cfg["prior_n"],
    )

    manifest = store_module(monkeypatch, tmp_path).fit_from_store()

    fit_id = manifest["fit_id"]
    fit_path = Path(manifest["fit_path"])
    manifest_path = Path(manifest["manifest_path"])
    assert fit_path == models_dir(data_dir) / f"nowcast-{fit_id}.json"
    assert manifest_path.parent == models_dir(data_dir)
    for path in (fit_path, manifest_path):
        assert path.is_file()
        assert not path.resolve().is_relative_to(env.repo_root().resolve())
    assert json.loads(manifest_path.read_text())["fit_id"] == fit_id
    model = NowcastV0.from_json(fit_path.read_text())
    assert model.fit_id == fit_id != OLD_ID
    assert model.bands == cfg["bands"]
    assert model.prior_n == cfg["prior_n"]
    assert tuple(model.grid.minutes) == tuple(grid.minutes)

    history = {k: c for k, c in model.cells.items() if k[0] != WIDE}
    assert set(history) == set(want.cells)
    for key, cell in want.cells.items():
        got = history[key]
        assert got.n == cell.n, key
        assert got.mu == pytest.approx(cell.mu, abs=1e-12), key
        assert got.sigma == pytest.approx(cell.sigma, rel=1e-9), key
    for key, s0 in want.sigma0.items():
        assert model.sigma0[key] == pytest.approx(s0, rel=1e-9), key

    old = old_nowcast()
    wide = {k: c for k, c in model.cells.items() if k[0] == WIDE}
    assert wide == {k: c for k, c in old.cells.items() if k[0] == WIDE}
    for tau in grid.minutes:
        assert model.sigma0[(WIDE, tau)] == pytest.approx(old.sigma0[(WIDE, tau)], rel=1e-12)

    days = {d.isoformat() for d in variant["days"]}
    assert days <= set(model.days)
    assert not {d.isoformat() for d in (*HELD, PRE_ERA)} & set(model.days)
    assert manifest["jravan_days"] == sorted(days)
    assert manifest["pool_sources"] == {WIN: "jravan", PLACE: "jravan", WIDE: "logged"}
    assert manifest["wide_fit_id"] == OLD_ID
    assert manifest["max_board_age_s_applied"] is False
    assert (models_dir(data_dir) / f"nowcast-{OLD_ID}.json").read_bytes() == old_bytes


def test_holdout_days_are_never_read(data_dir, monkeypatch, tmp_path):
    """D14: one holdout day's file is valid, the other is not parquet; every parquet read is
    watched and none names a holdout day, and the fit still succeeds."""
    assert_dates_hold()
    write_store(data_dir, VARIANTS["four-days"])
    write_current_fits(data_dir)
    module = store_module(monkeypatch, tmp_path)
    opened = []
    targets = (
        (pq, "read_table"),
        (pq, "ParquetFile"),
        (pq, "ParquetDataset"),
        (pq, "read_metadata"),
        (pq, "read_schema"),
        (ds, "dataset"),
        (pd, "read_parquet"),
    )
    for owner, attr in targets:
        real = getattr(owner, attr)

        def spy(source, *args, _real=real, **kwargs):
            opened.append(str(source))
            return _real(source, *args, **kwargs)

        monkeypatch.setattr(owner, attr, spy)

    try:
        manifest = module.fit_from_store()
    except Exception as exc:  # the corrupt holdout file raises on any read
        pytest.fail(f"the from-store fit read a holdout day's file: {exc!r}")
    touched = [p for p in opened if any(d.isoformat() in p for d in HELD)]
    assert not touched, touched
    assert any("odds_timeseries" in p for p in opened)  # the spies saw the day reads
    assert not {d.isoformat() for d in HELD} & set(manifest["jravan_days"])


# ---------------------------------------------------------------- the pointer and registry


def test_the_pointer_names_the_new_fit_and_the_loader_loads_it(data_dir, monkeypatch, tmp_path):
    """Only the pointer's "nowcast" entry changes; load_shadow_models() then loads the new fit,
    the win and ordering fits as before, and the old fit file is kept."""
    write_store(data_dir, VARIANTS["two-days"])
    names = write_current_fits(data_dir)
    before = load_shadow_models()
    assert before is not None and before.ids["nowcast"] == OLD_ID

    manifest = store_module(monkeypatch, tmp_path).fit_from_store()

    pointer = models_dir(data_dir) / "current.json"
    after = json.loads(pointer.read_text())
    assert {k: v for k, v in after.items() if k != "nowcast"} == {
        k: v for k, v in names.items() if k != "nowcast"
    }
    assert resolve(pointer, after["nowcast"]).resolve() == Path(manifest["fit_path"]).resolve()
    assert (models_dir(data_dir) / names["nowcast"]).is_file()
    loaded = load_shadow_models()
    assert loaded is not None
    assert loaded.ids == {**before.ids, "nowcast": manifest["fit_id"]}
    assert isinstance(loaded.nowcast, NowcastV0)
    assert loaded.nowcast.cells == NowcastV0.from_json(Path(manifest["fit_path"]).read_text()).cells


def test_one_registry_record_per_fit(data_dir, monkeypatch, tmp_path):
    """CLAUDE.md: every run is a registry record. One fit appends exactly one, built on jravan,
    naming the fit, its days and each pool's source."""
    variant = VARIANTS["four-days"]
    write_store(data_dir, variant)
    write_current_fits(data_dir)
    registry = default_registry(COUNTRY)
    assert registry.records() == []

    manifest = store_module(monkeypatch, tmp_path).fit_from_store()

    records = registry.records()
    assert len(records) == 1
    record = records[0]
    assert "jravan" in record.sources
    assert record.params["fit_id"] == manifest["fit_id"]
    assert record.params["pool_sources"] == {WIN: "jravan", PLACE: "jravan", WIDE: "logged"}
    assert record.params["jravan_days"] == sorted(d.isoformat() for d in variant["days"])


# ---------------------------------------------------------------- the command


class FakeServer:
    server_address = ("127.0.0.1", 8765)

    def serve_forever(self):
        raise KeyboardInterrupt

    def server_close(self):
        pass


def test_the_command_refits_and_board_serve_and_shadow_day_load_the_new_fit(
    data_dir, monkeypatch, tmp_path
):
    """`horsetrack japan nowcast-fit --from-store` writes and points the fit; `board-serve` and
    `shadow-day` then start with `nowcast=<new id>` (their servers and runs stubbed)."""
    from horsetrack.cli import app
    from horsetrack.countries.japan.board import server
    from horsetrack.countries.japan.shadow import day as shadow_day

    write_store(data_dir, VARIANTS["four-days"])
    write_current_fits(data_dir)
    store_module(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["japan", "nowcast-fit", "--from-store"])
    assert result.exit_code == 0, result.output
    pointer = models_dir(data_dir) / "current.json"
    new_id = NowcastV0.from_json(
        resolve(pointer, json.loads(pointer.read_text())["nowcast"]).read_text()
    ).fit_id
    assert new_id != OLD_ID
    assert new_id in result.output

    monkeypatch.setattr(server, "make_server", lambda *a, **k: FakeServer())
    served = runner.invoke(app, ["japan", "board-serve", "--date", "2026-11-22"])
    assert served.exit_code == 0, served.output
    assert f"nowcast={new_id}" in served.output

    summary = SimpleNamespace(records=0, no_board=0, races=0, sealed=False)
    monkeypatch.setattr(shadow_day, "run_shadow_day", lambda *a, **k: summary)
    shadow = runner.invoke(
        app, ["japan", "shadow-day", "--date", "2026-11-22", "--venues", "KYO", "--once"]
    )
    assert shadow.exit_code == 0, shadow.output
    assert f"nowcast={new_id}" in shadow.output


def test_without_a_current_nowcast_fit_the_command_refuses(data_dir, monkeypatch, tmp_path):
    """Ruling: Wide comes from the fit the pointer names; with no pointer there is no Wide source,
    so the command exits non-zero and writes no fit, manifest or pointer."""
    from horsetrack.cli import app

    write_store(data_dir, VARIANTS["four-days"])
    store_module(monkeypatch, tmp_path)
    result = CliRunner().invoke(app, ["japan", "nowcast-fit", "--from-store"])
    assert result.exit_code != 0, result.output
    folder = models_dir(data_dir)
    written = sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []
    assert written == [], written
