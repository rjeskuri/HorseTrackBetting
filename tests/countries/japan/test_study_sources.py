"""The C2, C3 and C4 study readers take a `source` (ticket #404).

Contract pinned here: each reader below takes a keyword `source: str = "jra"` and passes it to
`models.v0.store.read_store_table(..., source=)`, so it reads
`processed_dir(source)/<table>.parquet`. Without it the read is the `jra` store, as before.
Whatever the source, holdout days (`core.eval.holdout.is_holdout_day('japan', d)`) are never read.

Readers (all under `horsetrack.countries.japan`):
- C2 (#91): `studies.rating.read_results(*, source)`.
- C3 (#96, #97): `models.speed_runs.read_inputs(*, start, end, source)`;
  (#128) `handicap.pp.pp_lines(race_id, *, as_of, n, source)` and
  `handicap.pp.build_pp_table(date_from, date_to, *, source)`.
- C4: #114's calibrated market `models.v0.calibration.run_fit(out_dir, *, source)`, #125's residual
  `models.v0.residual.run_fit(base_path, out_dir, *, source)`, #87's meta-model
  `studies.meta.read_development(*, source)`.

All data is synthetic and written under tmp_path, with HORSETRACK_DATA_DIR pointing there. The
`jra` and `jravan` copies differ in venue, races per day and horses, and jravan starts earlier
(2011, where it "adds years"), so the rows returned show which store was read. Both copies hold
every day from 2026-11-28 for three weeks, so some days are held out and some are open; which is
which comes from `is_holdout_day`, never assumed. Expected rows are derived here from the
synthetic frames, never from the code under test.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import inspect
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core.eval import holdout
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.handicap import pp
from horsetrack.countries.japan.models import speed_runs
from horsetrack.countries.japan.models.v0 import calibration, residual
from horsetrack.countries.japan.models.v0.calibration import WinCalibration
from horsetrack.countries.japan.studies import meta, rating

DATA_VAR = "HORSETRACK_DATA_DIR"
JST = ZoneInfo("Asia/Tokyo")
PRICE_ERA_START = date(2014, 6, 7)  # D10
RATING_FROM = date(2010, 1, 1)  # #91: the rating reads results from the store's first year
FIRST_SCORED = date(2020, 1, 1)  # Japan's fold scheme: min_train 22 quarters from 2014-06-07
N_RUN = 8
N_HORSES = 24
LAST_WEEKLY = date(2021, 6, 26)

# Three weeks from the holdout start: the committed rule holds out some days and not others.
LATE = [date(2026, 11, 28) + timedelta(days=k) for k in range(22)]
HELD = [d for d in LATE if holdout.is_holdout_day("japan", d)]
LATE_OPEN = [d for d in LATE if not holdout.is_holdout_day("japan", d)]

# Distinct copies: a reader that ignores `source` returns the other copy's venue and horses.
PLAN = {
    "jra": {"venue": "TOK", "race_nos": (11,), "first": PRICE_ERA_START, "prefix": "jra-h"},
    "jravan": {"venue": "NAK", "race_nos": (1, 2), "first": date(2011, 1, 8), "prefix": "jv-h"},
}
SEEDS = {"jra": 404, "jravan": 4040}

# The calibrated-market base for #125's residual (a fixed, non-identity spline).
BASE = WinCalibration(
    knots=(-4.0, -2.4, -0.9),
    coef=(1.08, 0.03),
    fit_id="base-404",
    train_start=PRICE_ERA_START,
    train_end=LAST_WEEKLY,
    n_races=1000,
)


# ---------------------------------------------------------------------------------------------
# The synthetic stores
# ---------------------------------------------------------------------------------------------


def race_days(source: str) -> list[date]:
    first = PLAN[source]["first"]
    weekly = [first + timedelta(days=7 * k) for k in range((LAST_WEEKLY - first).days // 7 + 1)]
    return weekly + LATE


def build(source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One copy of the store: races and runners, full-schema columns, every race run."""
    plan, rng = PLAN[source], np.random.default_rng(SEEDS[source])
    venue = plan["venue"]
    horses = np.array([f"{plan['prefix']}{k:02d}" for k in range(N_HORSES)])
    races, runners = [], []
    for day in race_days(source):
        for no in plan["race_nos"]:
            race_id = f"jra:{day.isoformat()}:{venue}:{no:02d}"
            keys = {
                "jurisdiction": "jra",
                "race_id": race_id,
                "meeting_id": f"{day.isoformat()}:{venue}",
                "race_date": day,
                "venue": venue,
                "race_no": no,
                "season": str(day.year),
            }
            q = rng.dirichlet(np.ones(N_RUN))
            for _ in range(3):
                q = np.clip(q, 0.02, 0.6)
                q /= q.sum()
            odds = np.maximum(np.round(0.8 / q, 1), 1.1)
            winner = int(rng.choice(N_RUN, p=q))
            order = [winner, *rng.permutation([h for h in range(N_RUN) if h != winner])]
            finish = np.empty(N_RUN, dtype=int)
            finish[order] = np.arange(1, N_RUN + 1)
            fav = np.argsort(np.argsort(odds, kind="stable"), kind="stable") + 1
            field = rng.choice(horses, size=N_RUN, replace=False)
            races.append(
                {
                    **keys,
                    "race_class": "open",
                    "distance_m": 1600,
                    "surface": "turf",
                    "course_config": "A",
                    "going": "good",
                    "post_time_local": f"{10 + no}:30",
                    "field_size": N_RUN,
                    "finish_time_s": 95.0,
                    "status": "run",
                    "rail_setting": "A",
                }
            )
            for h in range(N_RUN):
                place = int(finish[h])
                runners.append(
                    {
                        **keys,
                        "horse_id": str(field[h]),
                        "horse_no": h + 1,
                        "draw": h + 1,
                        "finish_pos": place,
                        "dead_heat": False,
                        "lbw_lengths": 0.5 * (place - 1),
                        "running_positions": [place, place, place, place],
                        "finish_time_s": 95.0 + 0.2 * (place - 1),
                        "win_odds": float(odds[h]),
                        "is_scratched": False,
                        "actual_wt_kg": 56.0,
                        "horse_wt_kg": 480,
                        "horse_wt_change_kg": 0,
                        "last_3f_s": 34.0 + 0.1 * place,
                        "win_fav_rank": int(fav[h]),
                        "jockey_key": f"j{h}",
                        "jockey_name": f"Jockey {h}",
                        "trainer_key": f"t{h}",
                        "trainer_name": f"Trainer {h}",
                    }
                )
    return pd.DataFrame(races), pd.DataFrame(runners)


WORLD = {source: build(source) for source in PLAN}


def write_stores(data: Path) -> Path:
    """Both copies under `data/processed/<source>/`, as the store writes them."""
    for source, (races, runners) in WORLD.items():
        folder = data / "processed" / source
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in (("races", races), ("runners", runners)):
            spec = schemas.TABLES[name].columns
            schema = pa.schema([(c, tables.ARROW_TYPES[kind]) for c, kind in spec.items()])
            records = [
                {c: row.get(c) for c in spec} for row in frame.to_dict("records")
            ]  # fmt: skip
            pq.write_table(pa.Table.from_pylist(records, schema=schema), folder / f"{name}.parquet")
    return data


def fresh_data(tmp_path: Path, monkeypatch, name: str) -> Path:
    data = (tmp_path / name / "data").resolve()
    write_stores(data)
    monkeypatch.setenv(DATA_VAR, str(data))
    return data


def day_of(race_id: str) -> date:
    return date.fromisoformat(str(race_id).split(":")[1])


def open_rows(frame: pd.DataFrame, start: date = date.min, end: date = date.max) -> pd.DataFrame:
    days = frame["race_date"]
    keep = (days >= start) & (days <= end) & ~days.isin(HELD)
    return frame.loc[keep]


def pairs(frame) -> set[tuple[str, str]]:
    return set(zip(frame["race_id"].astype(str), frame["horse_id"].astype(str), strict=True))


# ---------------------------------------------------------------------------------------------
# Each reader: what it returns for a source, and what it should return, derived from the frames
# ---------------------------------------------------------------------------------------------

SPEED_START = date(2012, 6, 2)  # inside jravan's extra years, before jra's first day
PP_TARGET_DAY = max(d for d in LATE_OPEN if any(h < d for h in HELD))
PP_AS_OF = datetime.combine(PP_TARGET_DAY, time(9, 0), tzinfo=JST)
PP_N = 10_000  # every prior start, so jravan's early years show


def pp_target(source: str) -> str:
    plan = PLAN[source]
    return f"jra:{PP_TARGET_DAY.isoformat()}:{plan['venue']}:{plan['race_nos'][0]:02d}"


def got_rating(kw, tmp_path, monkeypatch, tag):
    fresh_data(tmp_path, monkeypatch, tag)
    return {"starters": pairs(rating.read_results(**kw))}


def want_rating(source):
    _, runners = WORLD[source]
    return {"starters": pairs(open_rows(runners, RATING_FROM))}


def got_speed(kw, tmp_path, monkeypatch, tag):
    fresh_data(tmp_path, monkeypatch, tag)
    races, runners = speed_runs.read_inputs(start=SPEED_START, end=LATE[-1], **kw)
    return {"races": set(races["race_id"].astype(str)), "starters": pairs(runners)}


def want_speed(source):
    races, runners = WORLD[source]
    return {
        "races": set(open_rows(races, SPEED_START, LATE[-1])["race_id"]),
        "starters": pairs(open_rows(runners, SPEED_START, LATE[-1])),
    }


def got_meta(kw, tmp_path, monkeypatch, tag):
    fresh_data(tmp_path, monkeypatch, tag)
    runners, races = meta.read_development(**kw)
    return {"races": set(races["race_id"].astype(str)), "starters": pairs(runners)}


def want_meta(source):
    races, runners = WORLD[source]
    return {
        "races": set(open_rows(races, PRICE_ERA_START)["race_id"]),
        "starters": pairs(open_rows(runners, PRICE_ERA_START)),
    }


def got_pp_lines(kw, tmp_path, monkeypatch, tag):
    fresh_data(tmp_path, monkeypatch, tag)
    source = kw.get("source", "jra")
    frame = pp.pp_lines(pp_target(source), as_of=PP_AS_OF, n=PP_N, **kw)
    lines = frame.filter(pl.col("pp_race_id").is_not_null())
    horse_ids, pp_race_ids = lines.get_column("horse_id"), lines.get_column("pp_race_id")
    return {
        "entries": set(frame.get_column("horse_id").to_list()),
        "lines": set(zip(horse_ids.to_list(), pp_race_ids.to_list(), strict=True)),
    }


def want_pp_lines(source):
    _, runners = WORLD[source]
    entries = set(runners.loc[runners["race_id"] == pp_target(source), "horse_id"])
    prior = open_rows(runners, end=PP_TARGET_DAY - timedelta(days=1))
    prior = prior.loc[prior["horse_id"].isin(entries)]
    return {
        "entries": entries,
        "lines": set(zip(prior["horse_id"], prior["race_id"], strict=True)),
    }


def got_pp_table(kw, tmp_path, monkeypatch, tag):
    fresh_data(tmp_path, monkeypatch, tag)
    written = pp.build_pp_table(LATE[0], LATE[-1], **kw)
    return {
        "days": {
            day: set(pl.read_parquet(path).get_column("race_id").to_list())
            for day, path in written.items()
        }
    }


def want_pp_table(source):
    races, _ = WORLD[source]
    late = open_rows(races, LATE[0], LATE[-1])
    return {"days": {d: set(g["race_id"]) for d, g in late.groupby("race_date")}}


def _fit_keys(manifest: dict) -> dict:
    oof = pq.read_table(manifest["oof_path"]).to_pandas()
    return {"n_races": manifest["n_races"], "scored": set(oof["race_id"].astype(str))}


def got_calibration(kw, tmp_path, monkeypatch, tag):
    data = fresh_data(tmp_path, monkeypatch, tag)
    return _fit_keys(calibration.run_fit(data.parent / "out", **kw))


def got_residual(kw, tmp_path, monkeypatch, tag):
    data = fresh_data(tmp_path, monkeypatch, tag)
    base_path = data.parent / "base.json"
    base_path.write_text(BASE.to_json(), encoding="utf-8")
    return _fit_keys(residual.run_fit(base_path, data.parent / "out", **kw))


def want_fit(source):
    races, _ = WORLD[source]
    developed = open_rows(races, PRICE_ERA_START)
    return {
        "n_races": int(developed["race_id"].nunique()),
        "scored": set(open_rows(races, FIRST_SCORED)["race_id"]),
    }


# name: (read through the reader, expected from the frames, explicit-jra call is cheap)
READERS = {
    "rating.read_results": (got_rating, want_rating, True),
    "speed_runs.read_inputs": (got_speed, want_speed, True),
    "pp.pp_lines": (got_pp_lines, want_pp_lines, True),
    "pp.build_pp_table": (got_pp_table, want_pp_table, True),
    "meta.read_development": (got_meta, want_meta, True),
    "v0.calibration.run_fit": (got_calibration, want_fit, False),
    "v0.residual.run_fit": (got_residual, want_fit, False),
}

PUBLIC = {
    "rating.read_results": rating.read_results,
    "speed_runs.read_inputs": speed_runs.read_inputs,
    "pp.pp_lines": pp.pp_lines,
    "pp.build_pp_table": pp.build_pp_table,
    "meta.read_development": meta.read_development,
    "v0.calibration.run_fit": calibration.run_fit,
    "v0.residual.run_fit": residual.run_fit,
}


# ---------------------------------------------------------------------------------------------
# DoD: each named reader takes `source`, default "jra"
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PUBLIC))
def test_each_reader_takes_a_source_keyword_defaulting_to_jra(name):
    params = inspect.signature(PUBLIC[name]).parameters
    assert "source" in params, f"{name} has no source parameter"
    param = params["source"]
    assert param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
    assert param.default == "jra"


# ---------------------------------------------------------------------------------------------
# DoD: each reader reads the jravan store for source="jravan", the jra store by default, and
# never a holdout day
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(READERS))
def test_each_reader_reads_the_named_store_jra_by_default_and_no_holdout_day(
    name, tmp_path, monkeypatch
):
    got_fn, want_fn, explicit_jra = READERS[name]
    # The fixture: both copies hold held-out and open late days, and the reader's expected rows
    # differ between them, so the read store shows.
    assert HELD and LATE_OPEN and any(h < PP_TARGET_DAY for h in HELD)
    for races, _ in WORLD.values():
        assert set(HELD) <= set(races["race_date"])
    want_jra, want_van = want_fn("jra"), want_fn("jravan")
    assert want_jra != want_van

    default = got_fn({}, tmp_path, monkeypatch, "default")
    assert default == want_jra, f"{name}: the default read is not the jra store"

    if explicit_jra:
        assert got_fn({"source": "jra"}, tmp_path, monkeypatch, "jra") == default

    jravan = got_fn({"source": "jravan"}, tmp_path, monkeypatch, "jravan")
    assert jravan == want_van, f"{name}: source='jravan' did not read the jravan store"

    # Holdout days are never read, whatever the source (the equalities above imply it; this
    # states it on the race ids the readers returned).
    for got in (default, jravan):
        days = {day_of(r) for r in _race_ids(got)}
        assert days, f"{name}: no rows returned"
        assert not days & set(HELD), f"{name}: a holdout day was read"


def _race_ids(got: dict) -> set[str]:
    ids: set[str] = set()
    for key, value in got.items():
        if key == "days":
            for race_ids in value.values():
                ids |= race_ids
        elif key == "races" or key == "scored":
            ids |= set(value)
        elif key == "starters":
            ids |= {race_id for race_id, _ in value}
        elif key == "lines":
            ids |= {race_id for _, race_id in value}
    return ids
