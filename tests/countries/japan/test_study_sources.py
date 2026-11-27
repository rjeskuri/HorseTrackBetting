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
(2009, where it "adds years"), so the rows returned show which store was read. Both copies hold
every day from 2026-11-28 for three weeks, so some days are held out and some are open; which is
which comes from `is_holdout_day`, never assumed. Expected rows are derived here from the
synthetic frames, never from the code under test.

Round 1 of the review adds (all keyword-only, defaults unchanged):
- `studies.speed_run.read_inputs(*, source)` (#305) and `studies.calibration.read_development(
  *, source)` (#77) join the reader list.
- `studies.calibrated_market.run_study(out_dir, *, source)` passes `source` to
  `studies.calibration.read_development`.
- `studies.rating.read_results(*, source, start=RATING_FROM)` and `studies.rating.run_study(
  out_dir, *, source, start=RATING_FROM)`: the run passes `source` to `read_results` and
  `read_development`, `start` to `read_results`, and a jravan run's registry records name
  `jravan` (not `jra`) in `sources` and in every `data_builds` entry.
- `studies.meta.run_study(members_path, out_dir, *, source)`: every record of a jravan run, the
  market-only ones and the member-based ones, names `jravan` (not `jra`) in `sources`.
- `models.v0.residual.run_fit`: a non-jra run's fit, OOF, manifest and report paths name the
  source; the jra paths are unchanged. A base fit carries its store copy: the
  `models.v0.calibration.run_fit(source='jravan')` fit file holds `"sources": ["jravan"]`
  (`WinCalibration.sources`), and a residual on that base names `jravan` in its record, manifest
  and report sources, so the report is withheld.
The jravan copy now starts in 2009 (before RATING_FROM) and both copies run weekly to 2024.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import inspect
import json
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horsetrack.core import paths
from horsetrack.core.eval import holdout, reports
from horsetrack.core.eval.registry import default_registry
from horsetrack.core.store import schemas, tables
from horsetrack.countries.japan.handicap import pp
from horsetrack.countries.japan.models import speed_runs
from horsetrack.countries.japan.models.v0 import calibration, residual
from horsetrack.countries.japan.models.v0.calibration import WinCalibration
from horsetrack.countries.japan.studies import calibrated_market, meta, rating, speed_run
from horsetrack.countries.japan.studies import calibration as study_calibration

DATA_VAR = "HORSETRACK_DATA_DIR"
JST = ZoneInfo("Asia/Tokyo")
PRICE_ERA_START = date(2014, 6, 7)  # D10
RATING_FROM = date(2010, 1, 1)  # #91: the rating reads results from the store's first year
FIRST_SCORED = date(2020, 1, 1)  # Japan's fold scheme: min_train 22 quarters from 2014-06-07
N_RUN = 8
N_HORSES = 24
LAST_WEEKLY = date(2024, 12, 28)  # past #87's member span (2023 on) and scored seasons (2024)
EARLY_START = date(2009, 1, 1)  # an earlier rating start than RATING_FROM (#404 review)

# Three weeks from the holdout start: the committed rule holds out some days and not others.
LATE = [date(2026, 11, 28) + timedelta(days=k) for k in range(22)]
HELD = [d for d in LATE if holdout.is_holdout_day("japan", d)]
LATE_OPEN = [d for d in LATE if not holdout.is_holdout_day("japan", d)]

# Distinct copies: a reader that ignores `source` returns the other copy's venue and horses.
PLAN = {
    "jra": {"venue": "TOK", "race_nos": (11,), "first": PRICE_ERA_START, "prefix": "jra-h"},
    "jravan": {"venue": "NAK", "race_nos": (1, 2), "first": date(2009, 1, 3), "prefix": "jv-h"},
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
    # The joined race and price columns of each line come from the same store copy.
    columns = ("horse_id", "pp_race_id", "venue", "win_odds")
    return {
        "entries": set(frame.get_column("horse_id").to_list()),
        "lines": set(zip(*(lines.get_column(c).to_list() for c in columns), strict=True)),
    }


def want_pp_lines(source):
    _, runners = WORLD[source]
    entries = set(runners.loc[runners["race_id"] == pp_target(source), "horse_id"])
    prior = open_rows(runners, end=PP_TARGET_DAY - timedelta(days=1))
    prior = prior.loc[prior["horse_id"].isin(entries)]
    # Odds are read only inside D10's price era; an older line keeps a null price.
    priced = prior["race_date"] >= PRICE_ERA_START
    odds = [o if p else None for o, p in zip(prior["win_odds"], priced, strict=True)]
    return {
        "entries": entries,
        "lines": set(zip(prior["horse_id"], prior["race_id"], prior["venue"], odds, strict=True)),
    }


def got_pp_table(kw, tmp_path, monkeypatch, tag):
    data = fresh_data(tmp_path, monkeypatch, tag)
    written = pp.build_pp_table(LATE[0], LATE[-1], **kw)
    return {
        "days": {
            day: set(pl.read_parquet(path).get_column("race_id").to_list())
            for day, path in written.items()
        },
        "folders": {Path(path).parent.relative_to(data).as_posix() for path in written.values()},
    }


def want_pp_table(source):
    races, _ = WORLD[source]
    late = open_rows(races, LATE[0], LATE[-1])
    return {
        "days": {d: set(g["race_id"]) for d, g in late.groupby("race_date")},
        "folders": {f"features/{source}/pp_lines"},  # a jravan build never overwrites jra's
    }


def got_speed_study(kw, tmp_path, monkeypatch, tag):
    fresh_data(tmp_path, monkeypatch, tag)
    races, runners = speed_run.read_inputs(**kw)
    return {
        "races": set(races["race_id"].astype(str)),
        "runner_races": Counter(runners["race_id"].astype(str)),
    }


def want_speed_study(source):
    races, runners = WORLD[source]  # the whole store, holdout days dropped
    return {
        "races": set(open_rows(races)["race_id"]),
        "runner_races": Counter(open_rows(runners)["race_id"]),
    }


def got_study_calibration(kw, tmp_path, monkeypatch, tag):
    fresh_data(tmp_path, monkeypatch, tag)
    runners, races = study_calibration.read_development(**kw)
    return {
        "races": set(races["race_id"].astype(str)),
        "runner_races": Counter(runners["race_id"].astype(str)),
    }


def want_study_calibration(source):
    races, runners = WORLD[source]
    return {
        "races": set(open_rows(races, PRICE_ERA_START)["race_id"]),
        "runner_races": Counter(open_rows(runners, PRICE_ERA_START)["race_id"]),
    }


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
    "studies.speed_run.read_inputs": (got_speed_study, want_speed_study, True),
    "studies.calibration.read_development": (got_study_calibration, want_study_calibration, True),
}

PUBLIC = {
    "studies.speed_run.read_inputs": speed_run.read_inputs,
    "studies.calibration.read_development": study_calibration.read_development,
    "studies.calibrated_market.run_study": calibrated_market.run_study,
    "studies.rating.run_study": rating.run_study,
    "studies.meta.run_study": meta.run_study,
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
        elif key == "runner_races":
            ids |= set(value)
        elif key == "lines":
            ids |= {line[1] for line in value}
    return ids


# ---------------------------------------------------------------------------------------------
# Review round 1: the study entry points take the source too
# ---------------------------------------------------------------------------------------------


class _Stop(Exception):
    """Raised by a spy to end a run once the call it watches has been seen."""


def _spy(monkeypatch, module, name, calls: list, *, stop: bool = False):
    """Replace `module.name` with a wrapper that records its keyword arguments (and calls the
    real function, unless `stop`)."""
    real = getattr(module, name)

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        if stop:
            raise _Stop(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)


def test_the_rating_takes_a_start_defaulting_to_rating_from():
    assert rating.RATING_FROM == RATING_FROM
    for fn in (rating.read_results, rating.run_study):
        params = inspect.signature(fn).parameters
        assert "start" in params, f"{fn.__qualname__} has no start parameter"
        assert params["start"].kind == params["start"].KEYWORD_ONLY
        assert params["start"].default == rating.RATING_FROM


@pytest.mark.parametrize("source", list(PLAN))
def test_the_rating_reads_results_from_an_earlier_start(source, tmp_path, monkeypatch):
    fresh_data(tmp_path, monkeypatch, "early")
    _, runners = WORLD[source]
    # The fixture: jravan holds pre-2010 starts the default start leaves out.
    if source == "jravan":
        assert (runners["race_date"] < RATING_FROM).any()
    for start in (EARLY_START, date(2009, 6, 6), date(2012, 3, 3), RATING_FROM):
        got = pairs(rating.read_results(source=source, start=start))
        want = pairs(open_rows(runners, start))
        assert got == want, f"{source} from {start}: not the store's starters from that day"
    if source == "jravan":
        early = pairs(rating.read_results(source=source, start=EARLY_START))
        assert early > pairs(rating.read_results(source=source))


def test_the_calibrated_market_run_passes_its_source_to_the_development_reader(
    tmp_path, monkeypatch
):
    for source, tag in (("jravan", "cm-jravan"), (None, "cm-default")):
        fresh_data(tmp_path, monkeypatch, tag)
        calls: list[dict] = []
        _spy(monkeypatch, study_calibration, "read_development", calls, stop=True)
        kw = {} if source is None else {"source": source}
        with pytest.raises(_Stop):
            calibrated_market.run_study(tmp_path / tag / "out", **kw)
        assert len(calls) == 1
        assert calls[0].get("source", "jra") == (source or "jra")


def _only_jravan(record) -> None:
    assert "jravan" in record.sources, (record.variant, record.sources)
    assert "jra" not in record.sources, (record.variant, record.sources)


def test_a_jravan_rating_run_reads_and_records_the_jravan_store(tmp_path, monkeypatch):
    data = fresh_data(tmp_path, monkeypatch, "rating-run")
    results_calls: list[dict] = []
    development_calls: list[dict] = []
    _spy(monkeypatch, rating, "read_results", results_calls)
    _spy(monkeypatch, rating, "read_development", development_calls)

    rating.run_study(data.parent / "out", source="jravan", start=EARLY_START)

    assert [c.get("source") for c in development_calls] == ["jravan"]
    assert [(c.get("source"), c.get("start")) for c in results_calls] == [("jravan", EARLY_START)]
    records = default_registry("japan").records(rating.FAMILY_ID)
    assert records
    for record in records:
        _only_jravan(record)
        builds = list(record.data_builds.values())
        assert builds and all("jravan" in b for b in builds), builds
        assert not any("processed/jra/" in b for b in builds), builds


def write_members(path: Path, source: str) -> Path:
    """A synthetic #87 member file (HKTrack's OOF layout) over the source's starters from 2023."""
    _, runners = WORLD[source]
    rows = runners.loc[runners["race_date"] >= date(2023, 1, 1)]
    rng = np.random.default_rng(87)
    q = 1.0 / rows["win_odds"].to_numpy()
    table = pa.table(
        {
            "race_id": pa.array(rows["race_id"].astype(str).tolist(), pa.string()),
            "horse_id": pa.array(rows["horse_id"].astype(str).tolist(), pa.string()),
            "race_date": pa.array(pd.to_datetime(rows["race_date"]), pa.timestamp("ns")),
            "fold_season": pa.array([f"{d.year}" for d in rows["race_date"]], pa.string()),
            **{
                col: pa.array(q * rng.uniform(0.6, 1.4, len(rows)), pa.float64())
                for col, _ in meta.MEMBERS.values()
            },
        }
    )
    pq.write_table(table, path)
    return path


def test_a_jravan_meta_run_reads_and_records_the_jravan_store(tmp_path, monkeypatch):
    data = fresh_data(tmp_path, monkeypatch, "meta-run")
    members = write_members(data.parent / "members.parquet", "jravan")
    calls: list[dict] = []
    _spy(monkeypatch, meta, "read_development", calls)

    meta.run_study(members, data.parent / "out", source="jravan")

    assert [c.get("source") for c in calls] == ["jravan"]
    records = default_registry("japan").records(meta.FAMILY_ID)
    market_only = [r for r in records if "jrdb" not in r.sources]
    member_based = [r for r in records if "jrdb" in r.sources]
    assert market_only and member_based, [(r.variant, r.sources) for r in records]
    for record in records:
        _only_jravan(record)


# ---------------------------------------------------------------------------------------------
# Review round 1: the residual's outputs per store copy, and the base fit's provenance
# ---------------------------------------------------------------------------------------------

ARTIFACTS = ("fit_path", "oof_path", "manifest_path", "report_path")


def _latest_record(family_id: str):
    records = default_registry("japan").records(family_id)
    assert records
    return records[-1]


def test_residual_runs_on_two_store_copies_keep_their_own_outputs(tmp_path, monkeypatch):
    data = fresh_data(tmp_path, monkeypatch, "two-copies")
    monkeypatch.setattr(residual, "read_segments", lambda *a, **k: ())  # `segments: []`
    base_path = data.parent / "base.json"
    base_path.write_text(BASE.to_json(), encoding="utf-8")

    jra = residual.run_fit(base_path)
    jra_record = _latest_record(residual.FAMILY_ID)
    van = residual.run_fit(base_path, source="jravan")
    van_record = _latest_record(residual.FAMILY_ID)

    # The jra run writes where it always has.
    v0_dir = data / "models" / "japan" / "v0"
    fit_id = jra["fit_id"]
    assert Path(jra["fit_path"]) == v0_dir / f"residual-{fit_id}.json"
    assert Path(jra["oof_path"]) == v0_dir / f"residual_oof-{fit_id}.parquet"
    assert Path(jra["manifest_path"]) == v0_dir / f"manifest-residual-{fit_id}.json"
    assert Path(jra["report_path"]) == paths.reports_dir() / "japan" / f"residual_v0-{fit_id}.json"

    # The jravan run's outputs name its copy, so neither run overwrites the other.
    for key in ARTIFACTS:
        assert van[key] != jra[key], f"{key}: the jravan run reused the jra path"
        assert "jravan" in Path(van[key]).relative_to(data).as_posix(), van[key]
        assert Path(jra[key]).exists() and Path(van[key]).exists()

    # Each run's files on disk still hold that run's races.
    for manifest, source in ((jra, "jra"), (van, "jravan")):
        oof = pq.read_table(manifest["oof_path"]).to_pandas()
        assert set(oof["race_id"].astype(str)) == want_fit(source)["scored"], source
        on_disk = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
        assert on_disk["n_races"] == want_fit(source)["n_races"], source
        report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
        assert report["n_races"] == want_fit(source)["n_races"], source

    # The jravan run names its copy in the record, its data build and the manifest.
    assert "jravan" in van_record.sources and "jravan" not in jra_record.sources
    assert all("jravan" in b for b in van_record.data_builds.values()), van_record.data_builds
    assert "jravan" in van["sources"]
    assert "sources" not in jra  # the default manifest is unchanged


def test_a_jravan_calibration_fit_file_carries_its_source(tmp_path, monkeypatch):
    data = fresh_data(tmp_path, monkeypatch, "cal-source")
    manifest = calibration.run_fit(data.parent / "out", source="jravan")

    text = Path(manifest["fit_path"]).read_text(encoding="utf-8")
    assert "jravan" in json.loads(text).get("sources", []), "the fit file names no source"
    assert "jravan" in WinCalibration.from_json(text).sources
    # A fit file written before #404 (no `sources`) still loads, as a jra fit.
    assert tuple(WinCalibration.from_json(BASE.to_json()).sources) == ("jra",)


def test_a_residual_on_a_jravan_base_is_labelled_and_withheld_as_jravan(tmp_path, monkeypatch):
    data = fresh_data(tmp_path, monkeypatch, "jravan-base")
    base = calibration.run_fit(data.parent / "base", source="jravan")
    written: list[tuple] = []
    real_write = residual.write_report

    def spy_write(path, text, *, sources, **kwargs):
        written.append(tuple(sources))
        return real_write(path, text, sources=sources, **kwargs)

    monkeypatch.setattr(residual, "write_report", spy_write)

    manifest = residual.run_fit(Path(base["fit_path"]), data.parent / "res")  # the jra store

    record = _latest_record(residual.FAMILY_ID)
    assert {"jra", "jravan"} <= set(record.sources), record.sources
    assert {"jra", "jravan"} <= set(manifest.get("sources", [])), manifest.get("sources")
    report = json.loads(Path(manifest["report_path"]).read_text(encoding="utf-8"))
    assert "jravan" in report.get("sources", [])
    assert written and all(reports.is_withheld(s) for s in written), written
    assert all("jravan" in s for s in written), written
