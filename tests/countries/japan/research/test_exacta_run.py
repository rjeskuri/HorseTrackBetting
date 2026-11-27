"""The #131 exacta run (``exacta_run``): holdout-free provenance, the frozen-spec guard, idempotent
reruns, one fit set per family, and a freeze that leaves the committed summary alone.

Review of 0c39a8d (round 1). Every input is synthetic and lives under ``tmp_path``: a small JRA
store (races, runners, dividends), v0 fit files and a registry. HORSETRACK_DATA_DIR points into
``tmp_path`` and the committed summary folder is redirected there too; an autouse guard fails the
test (and restores the files) if the repository's Japan summary folder changes. The holdout is
the committed Japan definition (A2 from 2026-11-28); HOLDOUT_DAYS and DEV_DAYS are checked
against it from first principles.

Contract pinned here:

- ``load_races() -> (races, meta)``: ``meta["data_builds"]`` (the provenance recorded with every
  run) is a function of development rows only. A store whose holdout-day rows change (prices,
  dividends, class, an added holdout race) gives the same races and meta; a changed development
  dividend or Win price gives other data_builds. No runners or dividends table the run reads
  holds a holdout race's row.
- ``score(registry, races, meta, sha, *, n_boot)`` and ``max_test(registry, frames, meta, sha, *,
  n_boot, note)`` refuse, before writing anything, when a family they score (E1, B1, B3, C for
  score; E1 for max_test) is frozen with a hash other than the hash of
  ``exacta_study.family_specs()[family]``, or not frozen; the message names the family.
- Reruns: the same commit and data reuse identical records (nothing appended, nothing refused);
  a run interrupted after some appends completes on the next run with each record once. The CLI
  ``main(["score"])`` rerun records no failure.
- Fits (finding exacta_run.py:374, behaviour chosen: refuse): when E1 already holds scored
  records under another Win or ordering fit, score and max_test refuse before appending, so a
  family never mixes fits; a data refresh under the same fits proceeds and SPA's family size
  equals its tested rules (84).
- ``main(["freeze"])`` on a fresh data directory records the frozen specs and never drops an
  existing family file or index entry of the committed summary.

Restart of f4c7db6 (Astra review, sections 3-5), sections 6 to 9 below:

- The CLI ``main(["max-test"])`` refuses (``RunRefused`` raised, or a non-zero exit) when E1 is
  not frozen, frozen with another spec, or holds scored records under another fit, before any
  frame is built (``score_frames`` / ``exacta_study.score_races`` are never called); nothing is
  recorded and no report is written.
- ``breakage(registry, races, meta, sha, *, n_boot)`` given development and holdout races never
  reads a holdout race's ``"exacta"`` (its dividends) and returns what the development races
  alone give.
- Unavailable statistics. A member whose cell is empty, whose category has no winning return in
  the cell, or whose bootstrap SE is zero keeps its row (``score()["members"]``, ``breakage()``
  rows) with its count ``n``; a statistic that cannot be computed is unavailable: ``None`` or NaN
  in the returned rows, never 0, and never rendered as nan/inf in a markdown report. A populated
  cell without a winning return has the valid estimate 0.0. B2's ``rho_eb`` is unavailable for a
  member without a finite estimate and a positive SE; the other members of its lambda2 x category
  group are shrunk among themselves (``eb_shrink`` on their estimates and SEs). Every other
  member, C and the max test are still recorded.
- Report artifacts: two data builds at one commit keep separate e1, breakage and max-test
  artifacts (each name holds the 12-digit SHA and more); the first build's files are unchanged
  after the second. RunRecord's fields are unchanged.

Restart of 66a2338 (owner, 2026-11-27), sections 10 to 12 below:

- Family ids (#525): ``Registry.append`` refuses an id with "/", so every family the run writes
  takes the ``japan-131-...`` form: the ``exacta_study`` families (see test_exacta_study),
  ``exacta_run.MAX_TEST_ID`` "japan-131-exacta-category-v2-max-test" and
  ``exacta_run.BREAKAGE_ID`` "japan-131-exacta-breakage". Records already in the registry under
  the legacy "japan/131-..." ids stay as they are.
- Unavailable statistics live in the record's metrics, never in its params (finding
  exacta_run.py:468: params are the variant's identity, so a coverage change made a new variant
  and ``Registry.family_size`` grew past 84 after a backfill). No record has a ``"unavailable"``
  key in ``params``. An unavailable statistic ``name`` of fold ``fold`` is one FoldMetric of
  ``record.metrics`` with that ``fold``, ``metric == f"{name}:unavailable:{reason}"`` (reason
  non-blank), ``lo`` and ``hi`` None and ``n`` the cell's count; its ``value`` is a placeholder
  the registry needs (finite) and carries no statistic. An available statistic's metric name has
  no ":". Two runs under one freeze that differ only in coverage (a cell empty in one run and
  populated in the next, e.g. a backfill of ``field:15-18``) share every variant and its params,
  and ``family_size`` of every family is unchanged by the refresh.
- One-meeting cells (finding exacta_run.py:401): a cell drawn from fewer than two meetings keeps
  its estimate, ``n`` and ``meetings``, but has no bootstrap bounds: ``lo`` and ``hi`` are
  unavailable in the returned rows (score members and breakage rows), no number in the markdown
  reports, and None on the estimate's FoldMetric (``rho``, ``mean_return`` for B3,
  ``rho_unrounded`` for breakage), with the unavailable statistic ``f"{estimate}_bounds"``
  listed on the same fold with its reason. A cell of two or more meetings keeps finite bounds.

GPT-6.2 Sol review round 5 of 2c73ca0, sections 13 to 16 below:

- Corrected Win boards (finding exacta_run.py:324): #114's out-of-fold artifact holds ``q``, the
  normalised implied Win shares (1 / odds over the race's sum) of the board it was built on.
  ``load_races`` compares each development race's current board (the runners' Win odds,
  normalised the same way) with that ``q`` and raises ``RunRefused`` naming the race when they
  differ by more than float noise (a 0.1 tick on the longest price is a difference), even with
  the same horse numbers; so ``main(["score"])`` refuses with nothing recorded and no report. An
  identical board passes, and a holdout race's board is never read, so it never refuses.
- C without a race above 110 yen (finding exacta_run.py:493): ``score`` completes when every
  scored dividend is at or below 110 yen. Both C variants keep their record with the unavailable
  ``log_score_gain`` (value ``UNAVAILABLE_VALUE``, n 0) and their ``log_score`` row (mean, lo, hi
  unavailable, n 0); E1's members, the max test and the e1 reports are all written. A C sample of
  one meeting keeps its mean (log q_w - log p_w at ``implied_share``'s D + 5, discounted Harville
  at each lambda2) without bounds: lo and hi unavailable in the row, None on the FoldMetric, and
  ``log_score_gain_bounds`` unavailable on the same fold.
- Return per 100 yen at the final dividend (finding docs/experiments/131-exacta-overload.md:88):
  each B2 and B1 member (the two boxes that are bets paid at the dividend; B3 is not) carries
  ``return_per_100`` = 100 * sum_r R_r / n, the gross yen returned per 100 yen staked on the box
  in each of the cell's n races, with one-sided 95% meeting-bootstrap bounds
  ``return_per_100_lo`` / ``_hi`` in the member rows and the e1 reports, and the FoldMetric
  ``return_per_100`` (with lo and hi) on the record's development fold. It is a metric, never a
  param: params stay estimator, lambda2, category, cell and direction, so family sizes (84, 42,
  84) and the max test's 84 are unchanged. Bounds follow E1's rules: an empty cell has the
  unavailable ``return_per_100``, a one-meeting cell keeps the value without bounds and lists
  ``return_per_100_bounds``. rho stays the pre-registered estimand.
- Breakage at the minimum payout (finding exacta_run.py:605): a dividend at or below 110 yen (the
  "Plus 10" floor, jra.yaml) may have been raised by the floor, so its unrounded value is unknown.
  Each breakage row carries ``n_floor``, the cell's races at the floor; ``rho_unrounded`` and
  ``breakage_share`` are over the cell's other races only (a 120 yen dividend still gets D + 5),
  unavailable when every race of the cell is at the floor; ``rho`` (paid) is over all the cell's
  races. In the registry a fold with ``n_floor`` > 0 carries one more unavailable marker (the
  floor races' correction), whose FoldMetric ``n`` is ``n_floor``; a fold without one has none.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, fields
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from horsetrack.core import env
from horsetrack.core.eval import registry as registry_module
from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord
from horsetrack.countries.japan.research import exacta_run
from horsetrack.countries.japan.research import exacta_study as study

CATEGORIES = ("fav_first", "fav_second", "no_fav")
SHA = "0123abcd" * 5
N_BOOT = 60
VENUES = ("TOK", "HSN", "FKS")
# Saturdays in spring 2025 plus the two development days in the first A2 weeks.
DEV_DAYS = [date(2025, 4, 5) + timedelta(days=7 * k) for k in range(8)] + [
    date(2026, 11, 28),
    date(2026, 12, 5),
]
HOLDOUT_DAYS = [date(2026, 11, 29), date(2026, 12, 6)]
FAV_ODDS = (1.6, 2.5, 3.8, 6.5)  # one per favourite-odds band
FIELDS = (7, 12, 16)  # one per field-size band
CLASSES = (("MAIDEN", "3歳未勝利"), ("C2", "3歳以上2勝クラス"), ("OP", "3歳以上オープン"))
# Run records of one full score: B2 2 x 3 x 14, B1 3 x 14, B3 2 x 3 x 14, C 2, the max test 1.
EXPECTED_COUNTS = {"B2": 84, "B1": 42, "B3": 84, "C": 2, "max-test": 1}
TOTAL_RECORDS = sum(EXPECTED_COUNTS.values())
# The family ids the run writes (#525 rename, owner 2026-11-27).
MAX_TEST_FAMILY = "japan-131-exacta-category-v2-max-test"
BREAKAGE_FAMILY = "japan-131-exacta-breakage"
REPO_SUMMARY = env.repo_root() / "docs" / "experiments" / "registry" / "japan"


class Interrupted(BaseException):
    """A kill between two appends: not an Exception, so nothing records it as a failure."""


# ---------------------------------------------------------------------------------------------
# Guards and the environment
# ---------------------------------------------------------------------------------------------


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.glob("*")) if p.is_file()}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Data dir and committed summary dir under tmp_path; the repo's summary must not change."""
    before = _snapshot(REPO_SUMMARY)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    summary = tmp_path / "summary" / "japan"
    summary.mkdir(parents=True)
    monkeypatch.setattr(exacta_run, "summary_dir", lambda country: summary)
    monkeypatch.setattr(registry_module, "summary_dir", lambda country: summary)
    monkeypatch.setattr(exacta_run, "_git_sha", lambda: SHA)
    yield {"data": data, "summary": summary}
    after = _snapshot(REPO_SUMMARY)
    if after != before:
        for name in set(after) - set(before):
            (REPO_SUMMARY / name).unlink()
        for name, raw in before.items():
            (REPO_SUMMARY / name).write_bytes(raw)
        pytest.fail("the test wrote the repository's committed Japan summary (restored)")


def _holdout_definition() -> tuple[date, str]:
    path = env.repo_root() / "src/horsetrack/countries/japan/config/holdout.yaml"
    block = yaml.safe_load(path.read_text(encoding="utf-8"))["holdout"]
    return date.fromisoformat(str(block["start"])), str(block["seed"])


def _ref_holdout(day: date) -> bool:
    start, seed = _holdout_definition()
    digest = hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).hexdigest()
    return day >= start and int(digest, 16) % 2 == 0


def test_the_synthetic_days_are_what_the_committed_holdout_says():
    assert all(_ref_holdout(d) for d in HOLDOUT_DAYS)
    assert not any(_ref_holdout(d) for d in DEV_DAYS)
    assert any(d >= _holdout_definition()[0] for d in DEV_DAYS)  # mixed weeks, not just dates


# ---------------------------------------------------------------------------------------------
# The synthetic store
# ---------------------------------------------------------------------------------------------


def _make_race(rng, race_id, day, meeting, race_no, cls, raw, n, fav_odds, category) -> dict:
    fav = int(rng.integers(1, n + 1))
    others = [h for h in range(1, n + 1) if h != fav]
    steps = 1.15 + np.cumsum(rng.uniform(0.3, 1.6, size=n - 1))
    odds = {fav: fav_odds}
    for horse, step in zip(rng.permutation(others), steps, strict=True):
        odds[int(horse)] = round(fav_odds * float(step), 1)
    raw_p = {h: rng.uniform(0.85, 1.15) / o for h, o in odds.items()}
    total = sum(raw_p.values())
    p = {h: v / total for h, v in raw_p.items()}
    if category == "fav_first":
        winner = (fav, int(rng.choice(others)))
    elif category == "fav_second":
        winner = (int(rng.choice(others)), fav)
    else:
        a, b = rng.choice(others, size=2, replace=False)
        winner = (int(a), int(b))
    p_w = p[winner[0]] * p[winner[1]] / (1 - p[winner[0]])
    dividend = max(120.0, math.floor(75 / p_w * rng.uniform(0.6, 1.4) / 10) * 10)
    return {
        "race_id": race_id, "meeting_id": meeting, "race_date": day, "race_no": race_no,
        "race_class": cls, "race_class_raw": raw, "odds": odds, "p": p, "winner": winner,
        "dividend": dividend,
    }  # fmt: skip


def _allocate(days, venues, design, rng, start=0):
    """Races of ``design`` (fav band, field, class, slot, category) on the meetings of ``days``
    x ``venues``: slot 0 takes race numbers 1-4, slot 1 race numbers 5-12."""
    meetings = [(d, v) for d in days for v in venues]
    free = {m: {0: [1, 2, 3, 4], 1: list(range(5, 13))} for m in meetings}
    races = []
    for k, (band, field, klass, slot, category) in enumerate(design):
        for step in range(len(meetings)):
            day, venue = meetings[(start + k + step) % len(meetings)]
            if free[(day, venue)][slot]:
                break
        else:  # pragma: no cover - the design fits the meetings
            raise AssertionError("no free race number")
        race_no = free[(day, venue)][slot].pop(0)
        cls, raw = CLASSES[klass]
        races.append(
            _make_race(
                rng, f"jra:{day.isoformat()}:{venue}:{race_no:02d}", day,
                f"jra:{day.isoformat()}:{venue}", race_no, cls, raw, FIELDS[field],
                FAV_ODDS[band], category,
            )
        )  # fmt: skip
    return races


def _base_races() -> list[dict]:
    """216 development races (every band x field x class x slot x winner category) and 12 races
    on two holdout days."""
    rng = np.random.default_rng(131)
    design = list(itertools.product(range(4), range(3), range(3), range(2), CATEGORIES))
    design = [design[i] for i in rng.permutation(len(design))]
    dev = _allocate(DEV_DAYS, VENUES, design, rng)
    holdout_design = [design[i] for i in range(0, 60, 5)]
    holdout = _allocate(HOLDOUT_DAYS, ("TOK", "HSN"), holdout_design, rng, start=3)
    return dev + holdout


def _write_store(data: Path, races: list[dict]) -> None:
    out = data / "processed" / "jra"
    out.mkdir(parents=True, exist_ok=True)
    race_rows, runner_rows, dividend_rows = [], [], []
    for r in races:
        race_rows.append({k: r[k] for k in exacta_run.RACE_COLUMNS})
        order = [*r["winner"], *sorted(h for h in r["odds"] if h not in r["winner"])]
        for pos, horse in enumerate(order, start=1):
            runner_rows.append({"race_id": r["race_id"], "horse_no": horse,
                                "win_odds": r["odds"][horse], "is_scratched": False,
                                "finish_pos": pos})  # fmt: skip
        dividend_rows.append({"race_id": r["race_id"], "pool": "FORECAST",
                              "combination": f"{r['winner'][0]},{r['winner'][1]}",
                              "dividend": float(r["dividend"]), "is_refund": False})  # fmt: skip
        dividend_rows.append({"race_id": r["race_id"], "pool": "WIN",
                              "combination": str(r["winner"][0]),
                              "dividend": float(round(r["odds"][r["winner"][0]] * 100)),
                              "is_refund": False})  # fmt: skip
    for name, rows in (("races", race_rows), ("runners", runner_rows),
                       ("dividends", dividend_rows)):  # fmt: skip
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False),
                       out / f"{name}.parquet")  # fmt: skip


def _write_fits(data: Path, races: list[dict], *, win_id: str, ordering_id: str, lambda2: float,
                p_shift: float = 0.0) -> None:  # fmt: skip
    """The v0 fit files ``_fits`` reads; the out-of-fold file covers every synthetic race. Like
    #114's artifact it holds ``q``, the normalised implied Win shares of the board ``races`` carry
    (1 / odds over the race's sum of 1 / odds), beside ``p_cal``."""
    model = data / "models" / "japan" / "v0"
    model.mkdir(parents=True, exist_ok=True)
    oof = model / f"win_calibration_oof-{win_id}.parquet"
    rows = []
    for r in races:
        raw = {h: v * (1 + p_shift * ((h % 3) - 1)) for h, v in r["p"].items()}
        total = sum(raw.values())
        inv = sum(1 / o for o in r["odds"].values())
        rows += [{"race_id": r["race_id"], "horse_no": h, "q": (1 / r["odds"][h]) / inv,
                  "p_cal": v / total} for h, v in raw.items()]  # fmt: skip
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), oof)
    (model / f"win-{win_id}.json").write_text(json.dumps({"fit_id": win_id}), encoding="utf-8")
    (model / f"ordering-{ordering_id}.json").write_text(
        json.dumps({"fit_id": ordering_id, "lambdas": [lambda2]}), encoding="utf-8"
    )
    (model / f"manifest-ordering-{ordering_id}.json").write_text(
        json.dumps({"oof": {"path": str(oof)}}), encoding="utf-8"
    )
    (model / "current.json").write_text(
        json.dumps({"win": f"win-{win_id}.json", "ordering": f"ordering-{ordering_id}.json"}),
        encoding="utf-8",
    )


FIT_A = {"win_id": "winfitA0001", "ordering_id": "ordfitA0001", "lambda2": 0.81}


def _setup(data: Path, races: list[dict] | None = None, fits: dict | None = None) -> list[dict]:
    """The store of ``races`` and an out-of-fold file built from the same Win boards (a store
    whose development Win board differs from the artifact's ``q`` is refused, section 13)."""
    races = _base_races() if races is None else races
    _write_store(data, races)
    _write_fits(data, races, **(fits or FIT_A))
    return races


def _holdout_ids(races) -> set[str]:
    return {r["race_id"] for r in races if r["race_date"] in HOLDOUT_DAYS}


def _frozen_registry(path: Path, specs: dict | None = None) -> Registry:
    reg = Registry(path / "japan.jsonl", holdout_log=path / "holdout_log.jsonl")
    for family_id, spec in (specs or study.family_specs()).items():
        reg.freeze_family(family_id, spec)
    return reg


def _family_of(estimator: str) -> str:
    (found,) = [f for f, s in study.family_specs().items() if s["estimator"] == estimator]
    return found


def _counts(reg: Registry) -> dict[str, int]:
    by_family = {_family_of(e): e for e in ("B2", "B1", "B3", "C")}
    out: dict[str, int] = {}
    for r in reg.records():
        key = "max-test" if r.family_id == MAX_TEST_FAMILY else by_family.get(r.family_id)
        out[key] = out.get(key, 0) + 1
    return out


def _canonical_records(reg: Registry) -> list[str]:
    return sorted(json.dumps(asdict(r), sort_keys=True) for r in reg.records())


def _score(reg, races, meta, sha=SHA):
    return exacta_run.score(reg, copy.deepcopy(races), copy.deepcopy(meta), sha, n_boot=N_BOOT)


# ---------------------------------------------------------------------------------------------
# 1. Provenance reads no holdout row (high, exacta_run.py:208)
# ---------------------------------------------------------------------------------------------


def _spy_reads(monkeypatch) -> list[tuple[str, pa.Table]]:
    reads: list[tuple[str, pa.Table]] = []
    real = pq.read_table

    def read_table(source, *args, **kwargs):
        table = real(source, *args, **kwargs)
        reads.append((Path(str(source)).name, table))
        return table

    class SpyFile(pq.ParquetFile):
        def __init__(self, source, *args, **kwargs):
            super().__init__(source, *args, **kwargs)
            self._spy_name = Path(str(source)).name

        def _keep(self, table):
            reads.append((self._spy_name, table))
            return table

        def read(self, *args, **kwargs):
            return self._keep(super().read(*args, **kwargs))

        def read_row_group(self, *args, **kwargs):
            return self._keep(super().read_row_group(*args, **kwargs))

        def read_row_groups(self, *args, **kwargs):
            return self._keep(super().read_row_groups(*args, **kwargs))

        def iter_batches(self, *args, **kwargs):
            for batch in super().iter_batches(*args, **kwargs):
                self._keep(pa.Table.from_batches([batch]))
                yield batch

    monkeypatch.setattr(pq, "read_table", read_table)
    monkeypatch.setattr(pq, "ParquetFile", SpyFile)
    return reads


def _with_holdout_changed(races: list[dict]) -> list[dict]:
    """Every holdout-day value changed (prices, dividends, class) and one holdout race added."""
    out = copy.deepcopy(races)
    for r in out:
        if r["race_date"] in HOLDOUT_DAYS:
            r["odds"] = {h: round(o * 1.37, 1) for h, o in r["odds"].items()}
            r["dividend"] += 230
            r["race_class"], r["race_class_raw"] = CLASSES[2]
    extra = copy.deepcopy(next(r for r in out if r["race_date"] == HOLDOUT_DAYS[1]))
    extra["race_id"] = f"jra:{HOLDOUT_DAYS[1].isoformat()}:FKS:01"
    extra["meeting_id"] = f"jra:{HOLDOUT_DAYS[1].isoformat()}:FKS"
    extra["race_no"] = 1
    return [*out, extra]


def _with_dev_dividend_changed(races):
    out = copy.deepcopy(races)
    out[0]["dividend"] += 10
    return out


def _with_dev_price_changed(races):
    out = copy.deepcopy(races)
    longest = max(out[0]["odds"], key=out[0]["odds"].get)
    out[0]["odds"][longest] = round(out[0]["odds"][longest] + 5.0, 1)
    return out


def test_provenance_depends_on_development_rows_only_and_no_holdout_row_is_read(
    isolated, tmp_path, monkeypatch
):
    base = _base_races()
    assert base[0]["race_date"] not in HOLDOUT_DAYS
    holdout_ids = _holdout_ids(base)
    assert len(holdout_ids) == 12

    def load(variant: list[dict], name: str, spy: bool = False):
        data = tmp_path / name
        _setup(data, races=variant)
        monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
        with monkeypatch.context() as m:
            reads = _spy_reads(m) if spy else None
            races, meta = exacta_run.load_races()
        return races, meta, reads

    races_a, meta_a, reads = load(base, "store-a", spy=True)
    assert any(name == "dividends.parquet" for name, _ in reads)  # the spy saw the reads
    for name, table in reads:
        if name in ("runners.parquet", "dividends.parquet"):
            assert "race_id" in table.column_names, name
            assert not set(table["race_id"].to_pylist()) & holdout_ids, name
    assert {r["race_id"] for r in races_a}.isdisjoint(holdout_ids)
    assert len(races_a) == 216

    # The same store but for its holdout days: the same races and the same recorded provenance.
    races_b, meta_b, _ = load(_with_holdout_changed(base), "store-b")
    assert races_b == races_a
    assert meta_b["data_builds"] == meta_a["data_builds"]
    assert meta_b == meta_a

    # A changed development row is a different build: the provenance is not a constant.
    for name, variant in (("store-c", _with_dev_dividend_changed(base)),
                          ("store-d", _with_dev_price_changed(base))):  # fmt: skip
        _, meta_c, _ = load(variant, name)
        assert meta_c["data_builds"] != meta_a["data_builds"], name


# ---------------------------------------------------------------------------------------------
# 2. The frozen-spec guard compares hashes, in score and in max_test (exacta_run.py:274)
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def loaded(isolated):
    _setup(isolated["data"])
    return exacta_run.load_races()


def _altered(spec: dict) -> dict:
    out = copy.deepcopy(spec)
    out["block_size"] = out["block_size"] + 1  # any change to the design is another spec
    return out


@pytest.mark.parametrize("mode", ["frozen_differs", "spec_in_use_differs"])
@pytest.mark.parametrize("estimator", ["B2", "B1", "B3", "C"])
def test_score_refuses_a_family_frozen_with_another_spec(loaded, tmp_path, monkeypatch, estimator,
                                                         mode):  # fmt: skip
    races, meta = loaded
    family_id = _family_of(estimator)
    specs = study.family_specs()
    if mode == "frozen_differs":
        reg = _frozen_registry(tmp_path, {**specs, family_id: _altered(specs[family_id])})
    else:
        reg = _frozen_registry(tmp_path)
        in_use = {**specs, family_id: _altered(specs[family_id])}
        monkeypatch.setattr(study, "family_specs", lambda: copy.deepcopy(in_use))
    with pytest.raises(Exception, match=re.escape(family_id)):
        _score(reg, races, meta)
    assert reg.records() == []  # refused before any record


def test_max_test_runs_only_against_e1_s_frozen_spec(loaded, tmp_path, monkeypatch):
    races, meta = loaded
    e1 = _family_of("B2")
    reg = _frozen_registry(tmp_path / "r1")
    _score(reg, races, meta)
    frames = exacta_run.score_frames(copy.deepcopy(races), meta)

    # Matching spec: the standalone max test proceeds and records itself.
    n = len(reg.records())
    result = exacta_run.max_test(reg, frames, copy.deepcopy(meta), SHA, n_boot=N_BOOT,
                                 note="rerun, spec unchanged")  # fmt: skip
    assert len(reg.records()) == n + 1
    assert result["spa"]["n_rules"] == result["spa"]["family_size"] == 84

    # The spec in use differs from the frozen one: refused, nothing recorded.
    specs = study.family_specs()
    in_use = {**specs, e1: _altered(specs[e1])}
    with monkeypatch.context() as m:
        m.setattr(study, "family_specs", lambda: copy.deepcopy(in_use))
        with pytest.raises(Exception, match=re.escape(e1)):
            exacta_run.max_test(reg, frames, copy.deepcopy(meta), SHA, n_boot=N_BOOT,
                                note="rerun, spec changed")  # fmt: skip
    assert len(reg.records()) == n + 1

    # E1 frozen with another spec, its members present: refused, no max-test record.
    other = _frozen_registry(tmp_path / "r2", {**specs, e1: _altered(specs[e1])})
    for record in reg.records(e1):
        other.append(record)
    with pytest.raises(Exception, match=re.escape(e1)):
        exacta_run.max_test(other, frames, copy.deepcopy(meta), SHA, n_boot=N_BOOT,
                            note="frozen spec differs")  # fmt: skip
    assert other.records(MAX_TEST_FAMILY) == []


# ---------------------------------------------------------------------------------------------
# 3. Reruns reuse identical records and complete an interrupted run (exacta_run.py:322)
# ---------------------------------------------------------------------------------------------


def test_a_rerun_of_the_same_commit_and_data_reuses_every_record(loaded, tmp_path):
    races, meta = loaded
    reg = _frozen_registry(tmp_path)
    first = _score(reg, races, meta)
    assert _counts(reg) == EXPECTED_COUNTS
    records = _canonical_records(reg)
    second = _score(reg, races, meta)  # no RecordRefused
    assert _canonical_records(reg) == records  # nothing appended, no failure
    assert second["spa"] == first["spa"]
    assert second["stepm"] == first["stepm"]


def _interrupt_after(monkeypatch, k: int) -> None:
    real = Registry.append
    calls = {"n": 0}

    def append(self, record):
        calls["n"] += 1
        if calls["n"] == k:
            raise Interrupted(f"killed at append {k}")
        return real(self, record)

    monkeypatch.setattr(Registry, "append", append)


@pytest.mark.parametrize("k", [2, 120, TOTAL_RECORDS])
def test_a_run_interrupted_after_some_records_completes_on_the_next_run(
    loaded, tmp_path, monkeypatch, k
):
    races, meta = loaded
    clean = _frozen_registry(tmp_path / "clean")
    _score(clean, races, meta)

    reg = _frozen_registry(tmp_path / "run")
    with monkeypatch.context() as m:
        _interrupt_after(m, k)
        with pytest.raises(Interrupted):
            _score(reg, races, meta)
    assert len(reg.records()) == k - 1
    _score(reg, races, meta)  # completes: only the missing records are written
    assert _counts(reg) == EXPECTED_COUNTS
    assert _canonical_records(reg) == _canonical_records(clean)
    assert all(r.status == "scored" for r in reg.records())


def test_the_score_command_completes_an_interrupted_run_and_reruns_without_a_failure(
    isolated, monkeypatch
):
    _setup(isolated["data"])
    monkeypatch.setattr(exacta_run, "score", functools.partial(exacta_run.score, n_boot=N_BOOT))
    reg_path = isolated["data"] / "registry" / "japan.jsonl"
    assert exacta_run.main(["freeze"]) == 0
    with monkeypatch.context() as m:
        _interrupt_after(m, 100)
        with pytest.raises(Interrupted):
            exacta_run.main(["score"])
    for attempt in ("completes", "reruns"):
        assert exacta_run.main(["score"]) == 0, attempt
        reg = Registry(reg_path, holdout_log=reg_path.parent / "none.jsonl")
        assert _counts(reg) == EXPECTED_COUNTS, attempt
        assert [r for r in reg.records() if r.status == "failed"] == [], attempt
    reports = isolated["data"] / "reports" / "japan" / "131-exacta"
    # Named by the commit and more (section 9): one e1 report pair for this commit and data.
    assert len(list(reports.glob(f"e1-{SHA[:12]}*.json"))) == 1
    assert len(list(reports.glob(f"e1-{SHA[:12]}*.md"))) == 1


# ---------------------------------------------------------------------------------------------
# 4. One fit set per family run (exacta_run.py:374). Chosen behaviour: refuse to mix fits.
# ---------------------------------------------------------------------------------------------

REFITS = {
    "win": {"win_id": "winfitB0002", "ordering_id": "ordfitB0002", "lambda2": 0.78,
            "p_shift": 0.08},
    "ordering": {"win_id": FIT_A["win_id"], "ordering_id": "ordfitC0003", "lambda2": 0.70},
}  # fmt: skip


@pytest.mark.parametrize("refit", sorted(REFITS))
def test_a_refreshed_fit_is_refused_and_a_data_refresh_keeps_the_family_size(isolated, tmp_path,
                                                                             refit):  # fmt: skip
    data = isolated["data"]
    base = _setup(data)
    races, meta = exacta_run.load_races()
    reg = _frozen_registry(tmp_path)
    _score(reg, races, meta)
    recorded = _canonical_records(reg)

    _write_fits(data, base, **REFITS[refit])
    races2, meta2 = exacta_run.load_races()
    assert (meta2["win_fit"], meta2["ordering_fit"]) != (meta["win_fit"], meta["ordering_fit"])
    with pytest.raises(Exception):  # noqa: B017 - any refusal, as long as nothing is written
        _score(reg, races2, meta2)
    assert _canonical_records(reg) == recorded
    with pytest.raises(Exception):  # noqa: B017
        exacta_run.max_test(reg, exacta_run.score_frames(copy.deepcopy(races2), meta2),
                            copy.deepcopy(meta2), SHA, n_boot=N_BOOT, note="refit")  # fmt: skip
    assert _canonical_records(reg) == recorded

    # Back on the recorded fits, a refreshed development store is a rerun, not new variants.
    _write_store(data, _with_dev_dividend_changed(base))
    _write_fits(data, base, **FIT_A)
    races3, meta3 = exacta_run.load_races()
    result = _score(reg, races3, meta3)
    assert result["spa"]["n_rules"] == result["spa"]["family_size"] == 84
    assert result["stepm"]["family_size"] == 84


# ---------------------------------------------------------------------------------------------
# 5. freeze leaves the committed summary alone (exacta_run.py:418)
# ---------------------------------------------------------------------------------------------


def test_freeze_on_a_fresh_data_dir_records_the_specs_and_drops_no_summary_entry(isolated):
    summary = isolated["summary"]
    entries = {
        "japan/131-exacta-category-v2": "japan--131-exacta-category-v2.md",
        "japan/999-other-study": "japan--999-other-study.md",
    }
    index = ["# Experiment registry index", ""] + [f"- [{f}]({s})" for f, s in entries.items()]
    (summary / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    for family, stem in entries.items():
        (summary / stem).write_text(f"# synthetic summary of {family}\n", encoding="utf-8")
    family_files = {stem: (summary / stem).read_bytes() for stem in entries.values()}

    assert exacta_run.main(["freeze"]) == 0

    reg_path = isolated["data"] / "registry" / "japan.jsonl"
    reg = Registry(reg_path, holdout_log=reg_path.parent / "none.jsonl")
    for family_id, spec in study.family_specs().items():
        expected = hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert reg.frozen_spec_hash(family_id) == expected, family_id
    assert reg.records() == []  # a freeze is not a run
    for stem, raw in family_files.items():
        assert (summary / stem).read_bytes() == raw, stem
    readme = (summary / "README.md").read_text(encoding="utf-8")
    for family, stem in entries.items():
        assert f"- [{family}]({stem})" in readme, family


# ---------------------------------------------------------------------------------------------
# Shared by sections 6 to 9
# ---------------------------------------------------------------------------------------------

REPORTS = ("reports", "japan", "131-exacta")
LAMBDA_KEYS = (study.FITTED, 1.0)


def _reports(data: Path) -> Path:
    return data.joinpath(*REPORTS)


def _is_unavailable(value) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and math.isnan(value))


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


UNAVAILABLE_TAG = "unavailable"


def _marker(metric: FoldMetric) -> tuple[str, str] | None:
    """``(name, reason)`` of an unavailable-statistic FoldMetric
    (``metric == f"{name}:unavailable:{reason}"``), or None for an available statistic."""
    if ":" not in metric.metric:
        return None
    parts = metric.metric.split(":", 2)
    assert len(parts) == 3 and parts[1] == UNAVAILABLE_TAG, metric
    return parts[0], parts[2]


def _unavailable(record: RunRecord) -> dict[tuple[str, str], str]:
    """The record's unavailable statistics, read from its metrics, as {(fold, name): reason}.
    Never in params: params are the variant's identity."""
    assert "unavailable" not in record.params, record.variant
    out = {}
    for metric in record.metrics:
        found = _marker(metric)
        if found is None:
            continue
        name, reason = found
        assert name and reason.strip(), metric
        assert metric.lo is None and metric.hi is None, metric
        assert (metric.fold, name) not in out, metric  # one entry per statistic
        out[(metric.fold, name)] = reason
    return out


def _present(record: RunRecord) -> dict[tuple[str, str], FoldMetric]:
    """The record's available statistics as {(fold, metric): FoldMetric}."""
    return {(f.fold, f.metric): f for f in record.metrics if _marker(f) is None}


def _cli_n_boot(monkeypatch) -> None:
    for name in ("score", "breakage", "max_test"):
        real = getattr(exacta_run, name)
        monkeypatch.setattr(exacta_run, name, functools.partial(real, n_boot=N_BOOT))


def _cli_registry(data: Path) -> Registry:
    path = data / "registry" / "japan.jsonl"
    return Registry(path, holdout_log=path.parent / "none.jsonl")


def _markdown_cells(text: str) -> tuple[list[str], list[str]]:
    """The table body rows (separator rows dropped) and every cell, lower-cased and stripped."""
    rows = [line for line in text.splitlines() if line.startswith("|")]
    body = [r for r in rows if not set(r.replace("|", "").strip()) <= set("-: ")]
    cells = [c.strip().lower() for r in body for c in r.strip().strip("|").split("|")]
    return body, cells


# ---------------------------------------------------------------------------------------------
# 6. The max-test command refuses before any frame is built (Astra 3, exacta_run.py:590)
# ---------------------------------------------------------------------------------------------


class FramesBuilt(BaseException):
    """The frame builder ran: not an Exception, so no failure record catches it."""


def _guard_frames(monkeypatch, *, refusal_expected: bool) -> None:
    def builder(*args, **kwargs):
        if refusal_expected:
            pytest.fail("frames were built before the max test's freeze and fit checks")
        raise FramesBuilt

    monkeypatch.setattr(exacta_run, "score_frames", builder)
    monkeypatch.setattr(study, "score_races", builder)


def _scored_under_another_fit() -> RunRecord:
    return RunRecord(
        hypothesis_id="E43", family_id=study.E1_ID,
        variant=f"b2|lambda2={study.FITTED}|fav_first|all", ticket=131, git_sha=SHA,
        data_builds={"jra_races": "processed/jra/races.parquet@00000000000000aa"},
        sources=("jra",), fold_scheme="oof:#114 walk-forward test quarters (folds.yaml)",
        components={"win": "winfitZ9999", "ordering": "ordfitZ9999"},
        params={"estimator": "B2"},
        metrics=(FoldMetric("development", "rho", 1.0, 0.9, 1.1, 10),), status="scored",
    )  # fmt: skip


@pytest.mark.parametrize(
    "case", ["not_frozen", "frozen_differs", "spec_in_use_differs", "fit_differs", "matches"]
)
def test_the_max_test_command_checks_the_freeze_and_fits_before_building_frames(
    isolated, monkeypatch, case
):
    data = isolated["data"]
    _setup(data)
    e1 = _family_of("B2")
    specs = study.family_specs()
    (data / "registry").mkdir(parents=True, exist_ok=True)
    if case == "frozen_differs":
        _frozen_registry(data / "registry", {**specs, e1: _altered(specs[e1])})
    elif case != "not_frozen":
        assert exacta_run.main(["freeze"]) == 0
    if case == "spec_in_use_differs":
        in_use = {**specs, e1: _altered(specs[e1])}
        monkeypatch.setattr(study, "family_specs", lambda: copy.deepcopy(in_use))
    if case == "fit_differs":
        _cli_registry(data).append(_scored_under_another_fit())
    before = _canonical_records(_cli_registry(data))

    refusal_expected = case != "matches"
    _guard_frames(monkeypatch, refusal_expected=refusal_expected)
    if not refusal_expected:  # the guard is conditional: a matching freeze builds frames
        with pytest.raises(FramesBuilt):
            exacta_run.main(["max-test", "--note", case])
        return
    try:
        code = exacta_run.main(["max-test", "--note", case])
    except exacta_run.RunRefused:
        code = None
    assert code != 0, "the max test ran on a family it should have refused"
    assert _canonical_records(_cli_registry(data)) == before  # no record, no failure record
    assert list(_reports(data).glob("max-test-*")) == []  # no report written


# ---------------------------------------------------------------------------------------------
# 7. breakage never reads a holdout race's dividends (Astra 3, exacta_run.py:511)
# ---------------------------------------------------------------------------------------------


class HoldoutDividendRead(BaseException):
    """A held-out race's exacta rows were read: not an Exception, so nothing swallows it."""


class _SealedExacta:
    """A held-out race's exacta rows: any read raises."""

    def _read(self, *args, **kwargs):
        raise HoldoutDividendRead("a holdout race's exacta dividend was read")

    __iter__ = __len__ = __getitem__ = __bool__ = __contains__ = __reversed__ = _read


class HeldOutRace(dict):
    """A scoring input of a holdout day whose ``"exacta"`` raises when read."""

    def __getitem__(self, key):
        if key == "exacta":
            raise HoldoutDividendRead(f"{dict.__getitem__(self, 'race_id')}: exacta read")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "exacta":
            raise HoldoutDividendRead(f"{dict.__getitem__(self, 'race_id')}: exacta read")
        return super().get(key, default)


def _holdout_scoring_races() -> list[HeldOutRace]:
    out = []
    for r in _base_races():
        if r["race_date"] not in HOLDOUT_DAYS:
            continue
        out.append(HeldOutRace({
            "race_id": r["race_id"], "race_date": r["race_date"], "meeting_id": r["meeting_id"],
            "race_no": r["race_no"], "race_class": r["race_class"],
            "win_odds": {h: float(o) for h, o in r["odds"].items()}, "p": dict(r["p"]),
            "rate": 0.75, "boost_may_be_withheld": False, "exacta": _SealedExacta(),
        }))  # fmt: skip
    return out


def _direct_breakage(races: list[dict], lambda2: float) -> dict[tuple[str, str], dict]:
    """Per (category, cell): paid rho, breakage share and rho at D + 5, computed here from
    score_races' B2 returns and each race's paid dividend."""
    frame = study.score_races(races, holdout_day=lambda country, day: _ref_holdout(day),
                              lambda2=lambda2)  # fmt: skip
    d = frame["race_id"].map({r["race_id"]: r["exacta"][0][1] for r in races})
    unrounded = frame["b2"] * (d + 5.0) / d
    out = {}
    for category in CATEGORIES:
        for cell in study.SEGMENT_CELLS:
            mask = (frame["category"] == category) & frame[cell]
            paid, up = frame["b2"][mask].sum(), unrounded[mask].sum()
            rate = frame["rate"][mask].sum()
            out[(category, cell)] = {
                "n": int(mask.sum()),
                "rho": paid / rate,
                "breakage_share": 1 - paid / up,
                "rho_unrounded": up / rate,
            }
    return out


def test_breakage_skips_holdout_races_before_reading_their_dividends(loaded, tmp_path):
    races, meta = loaded
    held = _holdout_scoring_races()
    assert len(held) == 12 and all(_ref_holdout(r["race_date"]) for r in held)

    # The development races alone: the populated-sample values are the direct computation.
    reg_dev = _frozen_registry(tmp_path / "dev")
    dev_rows = exacta_run.breakage(reg_dev, copy.deepcopy(races), copy.deepcopy(meta), SHA,
                                   n_boot=N_BOOT)  # fmt: skip
    assert len(dev_rows) == len(LAMBDA_KEYS) * len(CATEGORIES) * len(study.SEGMENT_CELLS)
    for key, value in zip(LAMBDA_KEYS, (meta["lambda2_full_development"], 1.0), strict=True):
        direct = _direct_breakage(races, value)
        for row in (r for r in dev_rows if str(r["lambda2"]) == str(key)):
            want = direct[(row["category"], row["cell"])]
            assert row["n"] == want["n"]
            for stat in ("rho", "breakage_share", "rho_unrounded"):
                assert row[stat] == pytest.approx(want[stat], rel=1e-12), (key, row, stat)
            assert row["lo"] <= row["rho_unrounded"] <= row["hi"]
    records = reg_dev.records(BREAKAGE_FAMILY)
    assert len(records) == len(LAMBDA_KEYS)
    for record in records:
        assert _unavailable(record) == {}  # nothing unavailable in the populated sample

    # Holdout races mixed in: never read, and the result is the development races' result.
    dev = copy.deepcopy(races)
    mixed = [*dev[:100], *held[:6], *dev[100:], *held[6:]]
    rows = exacta_run.breakage(_frozen_registry(tmp_path / "mixed"), mixed, copy.deepcopy(meta),
                               SHA, n_boot=N_BOOT)  # fmt: skip
    pd.testing.assert_frame_equal(pd.DataFrame(rows), pd.DataFrame(dev_rows))


# ---------------------------------------------------------------------------------------------
# 8. Empty, zero-return and zero-variance cells keep their rows (Astra 3 and 5)
# ---------------------------------------------------------------------------------------------

BANDS = ("le1.9", "2.0-2.9", "3.0-4.9", "ge5.0")  # FAV_ODDS 1.6, 2.5, 3.8, 6.5
FIELD_BANDS = ("5-10", "11-14", "15-18")  # FIELDS 7, 12, 16
CLASS_GROUPS = ("maiden", "win1-3", "open")  # CLASSES MAIDEN, C2, OP


def _ref_cells(band: int, field: int, klass: int, slot: int) -> set[str]:
    """The signed cells of a design entry, from the spec's definitions (slot 0 = races 1-4)."""
    less = slot == 0 or klass == 0
    cells = {
        "all",
        f"fav_odds:{BANDS[band]}",
        f"field:{FIELD_BANDS[field]}",
        f"class:{CLASS_GROUPS[klass]}",
        "popularity:less" if less else "popularity:rest",
    }
    if band >= 2 and less:
        cells.add("headline")
    return cells


def _sparse_store() -> tuple[list[dict], list[tuple]]:
    """Development races with no 16-runner field (field:15-18 empty), no favourite-second
    winner at favourite odds >= 5.0 (a populated zero-return cell) and one open race (class:open
    in one meeting: zero bootstrap variance), plus six holdout races."""
    rng = np.random.default_rng(1312)
    full = itertools.product(range(4), range(2), range(2), range(2), CATEGORIES)
    design = [d for d in full if not (d[0] == 3 and d[4] == "fav_second")]
    design.append((0, 0, 2, 1, "fav_first"))
    design = [design[i] for i in rng.permutation(len(design))]
    dev = _allocate(DEV_DAYS, VENUES, design, rng)
    holdout = _allocate(HOLDOUT_DAYS, ("TOK", "HSN"), design[:6], rng, start=3)
    return dev + holdout, design


def _sparse_expectations(dev: list[dict], design: list[tuple]) -> dict:
    n = dict.fromkeys(study.SEGMENT_CELLS, 0)
    meetings: dict[str, set] = {cell: set() for cell in study.SEGMENT_CELLS}
    wins = {(c, cell): 0 for c in CATEGORIES for cell in study.SEGMENT_CELLS}
    for race, entry in zip(dev, design, strict=True):
        for cell in _ref_cells(*entry[:4]):
            n[cell] += 1
            meetings[cell].add(race["meeting_id"])
            wins[(entry[4], cell)] += 1
    empty = {cell for cell, k in n.items() if k == 0}
    zero_return = {m for m, k in wins.items() if k == 0 and m[1] not in empty}
    one_meeting = {cell for cell, ms in meetings.items() if len(ms) == 1}
    # The design is what this section needs: one empty cell and populated zero-return cells
    # (every bootstrap replicate is exactly 0, so the SE is exactly 0), beside ordinary cells.
    # The one-meeting cell's SE is 0 up to rounding (about 1e-16): no expectation is pinned on
    # its SE or shrinkage, only that its row and estimate are kept.
    assert empty == {"field:15-18"}
    assert {("fav_second", "fav_odds:ge5.0"), ("fav_second", "class:open")} <= zero_return
    assert one_meeting == {"class:open"}
    return {"n_races": len(dev), "n": n, "meetings": {c: len(m) for c, m in meetings.items()},
            "empty": empty, "zero_return": zero_return, "one_meeting": one_meeting}  # fmt: skip


@pytest.fixture
def sparse(isolated):
    races, design = _sparse_store()
    _write_store(isolated["data"], races)
    _write_fits(isolated["data"], races, **FIT_A)
    loaded_races, meta = exacta_run.load_races()
    return loaded_races, meta, _sparse_expectations(races[: len(design)], design)


def test_score_keeps_empty_zero_return_and_zero_variance_members(sparse, tmp_path):
    races, meta, exp = sparse
    reg = _frozen_registry(tmp_path)
    result = _score(reg, races, meta)
    assert result["meta"]["scored_races"] == exp["n_races"]
    assert len(result["log_score"]) == 2 and "spa" in result and "stepm" in result

    members = result["members"]
    lambdas_of = {"B2": LAMBDA_KEYS, "B1": (study.FITTED,), "B3": LAMBDA_KEYS}
    want_keys = {(e, str(k), c, cell) for e, ks in lambdas_of.items() for k in ks
                 for c in CATEGORIES for cell in study.SEGMENT_CELLS}  # fmt: skip
    got_keys = [(m["estimator"], str(m["lambda2"]), m["category"], m["cell"]) for m in members]
    assert len(got_keys) == len(want_keys) and set(got_keys) == want_keys  # no member dropped

    for m in members:
        member, b2 = (m["category"], m["cell"]), m["estimator"] == "B2"
        assert m["n"] == exp["n"][m["cell"]], m
        assert m["meetings"] == exp["meetings"][m["cell"]], m
        if m["cell"] in exp["empty"]:  # unavailable, never a zero estimate
            for stat in ("rho", "se", "lo", "hi", *(("rho_eb",) if b2 else ())):
                assert _is_unavailable(m[stat]), (m, stat)
            continue
        assert _finite(m["rho"]), m  # the estimate is kept, one meeting or more
        if m["cell"] in exp["one_meeting"]:  # one meeting: no bootstrap interval, not [x, x]
            assert _is_unavailable(m["lo"]) and _is_unavailable(m["hi"]), m
        else:
            assert _finite(m["lo"]) and _finite(m["hi"]), m
        if member in exp["zero_return"]:  # zero SE: every replicate is 0
            assert m["rho"] == 0.0, m  # a valid zero-return estimate
            assert _is_unavailable(m["se"]) or m["se"] == 0.0, m
            if b2:
                assert _is_unavailable(m["rho_eb"]), m  # no invented precision
            continue
        assert m["rho"] > 0, m
        if m["cell"] not in exp["one_meeting"]:
            assert _finite(m["se"]) and m["se"] > 0, m
            if b2:
                assert _finite(m["rho_eb"]), m

    # rho_eb: the group's members that carry one are shrunk among themselves.
    for key in LAMBDA_KEYS:
        for category in CATEGORIES:
            group = [m for m in members if m["estimator"] == "B2"
                     and str(m["lambda2"]) == str(key) and m["category"] == category]  # fmt: skip
            usable = [m for m in group if not _is_unavailable(m["rho_eb"])]
            assert len(usable) >= 10, (key, category)
            want = study.eb_shrink([m["rho"] for m in usable], [m["se"] for m in usable])
            got = [m["rho_eb"] for m in usable]
            assert got == pytest.approx(list(want), rel=1e-12), (key, category)

    # The registry: every member recorded, unavailable statistics listed with a reason, never
    # written as a metric.
    assert _counts(reg) == EXPECTED_COUNTS
    assert all(r.status == "scored" for r in reg.records())
    by_variant = {(r.family_id, r.variant): r for r in reg.records()}
    for m in members:
        est = m["estimator"]
        record = by_variant[(_family_of(est), f"{est.lower()}|lambda2={m['lambda2']}|"
                                              f"{m['category']}|{m['cell']}")]  # fmt: skip
        present = _present(record)
        missing = _unavailable(record)
        assert not set(present) & set(missing), record.variant
        assert all(f.n == m["n"] for f in record.metrics), record.variant
        estimate = ("development", "mean_return" if est == "B3" else "rho")
        bounds = ("development", f"{estimate[1]}_bounds")
        wanted = {estimate, ("development", "bootstrap_se")}
        if est == "B2":
            wanted.add(("development", "rho_eb"))
        assert wanted <= set(present) | set(missing), record.variant
        member = (m["category"], m["cell"])
        if m["cell"] in exp["empty"]:
            assert wanted <= set(missing), record.variant
            continue
        if m["cell"] in exp["one_meeting"]:  # the estimate without bounds, and why
            assert present[estimate].value == m["rho"], record.variant
            assert (present[estimate].lo, present[estimate].hi) == (None, None), record.variant
            assert bounds in missing, record.variant
        else:
            assert present[estimate].lo is not None and present[estimate].hi is not None
            assert bounds not in missing, record.variant
        if member in exp["zero_return"]:
            assert present[estimate].value == 0.0, record.variant
            if est == "B2":
                assert ("development", "rho_eb") in missing, record.variant
        elif m["cell"] not in exp["one_meeting"]:
            assert wanted <= set(present), record.variant
            assert missing == {}, record.variant


def test_breakage_keeps_empty_and_zero_return_cells(sparse, tmp_path):
    races, meta, exp = sparse
    reg = _frozen_registry(tmp_path)
    rows = exacta_run.breakage(reg, races, meta, SHA, n_boot=N_BOOT)
    got = [(str(r["lambda2"]), r["category"], r["cell"]) for r in rows]
    want = {(str(k), c, cell) for k in LAMBDA_KEYS for c in CATEGORIES
            for cell in study.SEGMENT_CELLS}  # fmt: skip
    assert len(got) == len(want) and set(got) == want

    for r in rows:
        member = (r["category"], r["cell"])
        assert r["n"] == exp["n"][r["cell"]], r
        if r["cell"] in exp["empty"]:
            for stat in ("rho", "breakage_share", "rho_unrounded", "lo", "hi"):
                assert _is_unavailable(r[stat]), (r, stat)
            continue
        if r["cell"] in exp["one_meeting"]:  # one meeting: no bootstrap interval
            assert _is_unavailable(r["lo"]) and _is_unavailable(r["hi"]), r
        else:
            assert _finite(r["lo"]) and _finite(r["hi"]), r
        if member in exp["zero_return"]:
            assert r["rho"] == 0.0 and r["rho_unrounded"] == 0.0, r  # valid zero returns
            assert _is_unavailable(r["breakage_share"]), r  # 0 / 0: no share
        else:
            assert _finite(r["breakage_share"]) and 0 < r["breakage_share"] < 1, r
            assert 0 < r["rho"] < r["rho_unrounded"], r

    records = reg.records(BREAKAGE_FAMILY)
    variants = sorted(f"breakage|lambda2={k}" for k in LAMBDA_KEYS)
    assert sorted(r.variant for r in records) == variants
    for record in records:
        present = _present(record)
        missing = _unavailable(record)
        assert not set(present) & set(missing)
        for category in CATEGORIES:
            for cell in study.SEGMENT_CELLS:
                fold = f"development:{category}:{cell}"
                share, unrounded = (fold, "breakage_share"), (fold, "rho_unrounded")
                bounds = (fold, "rho_unrounded_bounds")
                if cell in exp["empty"]:
                    assert share in missing and unrounded in missing, fold
                    continue
                if cell in exp["one_meeting"]:
                    assert (present[unrounded].lo, present[unrounded].hi) == (None, None), fold
                    assert bounds in missing, fold
                else:
                    assert None not in (present[unrounded].lo, present[unrounded].hi), fold
                    assert bounds not in missing, fold
                if (category, cell) in exp["zero_return"]:
                    assert share in missing, fold
                    assert present[unrounded].value == 0.0, fold
                else:
                    assert share in present and unrounded in present, fold


def test_the_score_and_breakage_commands_report_a_sparse_sample(sparse, isolated, monkeypatch):
    _, _, exp = sparse
    data = isolated["data"]
    _cli_n_boot(monkeypatch)
    assert exacta_run.main(["freeze"]) == 0
    assert exacta_run.main(["score"]) == 0
    assert exacta_run.main(["breakage"]) == 0
    reg = _cli_registry(data)
    assert [r for r in reg.records() if r.status == "failed"] == []
    counts = _counts(reg)
    assert counts.pop(None) == len(LAMBDA_KEYS)  # the breakage records, under no E1 family
    assert counts == EXPECTED_COUNTS
    assert len(reg.records(BREAKAGE_FAMILY)) == len(LAMBDA_KEYS)

    n_members = (2 + 1 + 2) * len(CATEGORIES) * len(study.SEGMENT_CELLS)
    for prefix, n_rows in (("e1-", n_members), ("breakage-", 2 * 3 * len(study.SEGMENT_CELLS))):
        (md,) = _reports(data).glob(f"{prefix}{SHA[:12]}*.md")
        text = md.read_text(encoding="utf-8")
        body, cells = _markdown_cells(text)
        assert len(body) >= n_rows + 1, prefix  # a header and every member
        assert not {"nan", "-nan", "inf", "-inf", "+inf"} & set(cells), prefix
        # A one-meeting cell's row shows no number for its 95% bounds; other rows do.
        bounds_by_cell = _bounds_cells(text)
        assert set(bounds_by_cell) >= {"class:open", "all"}, prefix
        for cell, shown in bounds_by_cell.items():
            if cell in exp["one_meeting"]:
                assert not any(_is_number(c) for c in shown), (prefix, cell, shown)
            elif cell not in exp["empty"]:
                assert all(_is_number(c) for c in shown), (prefix, cell, shown)
        (js,) = _reports(data).glob(f"{prefix}{SHA[:12]}*.json")
        assert js.stat().st_size > 0
    assert exp["empty"] and exp["one_meeting"]  # the sample has both cells to render


def _is_number(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def _bounds_cells(text: str) -> dict[str, list[str]]:
    """Per cell name, the "95% lo" and "95% hi" entries of every table row of that cell, found
    by the header's column names."""
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith("|")
    ]
    header = next(r for r in rows if "cell" in r and "95% lo" in r and "95% hi" in r)
    at = {name: header.index(name) for name in ("cell", "95% lo", "95% hi")}
    out: dict[str, list[str]] = {}
    for row in rows:
        if row == header or set("".join(row)) <= set("-: ") or len(row) != len(header):
            continue
        out.setdefault(row[at["cell"]], []).extend([row[at["95% lo"]], row[at["95% hi"]]])
    return out


# ---------------------------------------------------------------------------------------------
# 9. Two data builds at one commit keep separate report artifacts (Astra 5, exacta_run.py:487)
# ---------------------------------------------------------------------------------------------

ARTIFACT_KINDS = (("e1-", ".json"), ("e1-", ".md"), ("breakage-", ".json"), ("breakage-", ".md"),
                  ("max-test-", ".json"))  # fmt: skip
RUN_RECORD_FIELDS = ["hypothesis_id", "family_id", "variant", "ticket", "git_sha", "data_builds",
                     "sources", "fold_scheme", "components", "params", "metrics", "status",
                     "holdout_open", "owner_signoff"]  # fmt: skip


def test_two_data_builds_at_one_commit_keep_separate_report_artifacts(isolated, monkeypatch):
    data = isolated["data"]
    base = _setup(data)
    _cli_n_boot(monkeypatch)
    assert exacta_run.main(["freeze"]) == 0

    def build() -> dict[str, bytes]:
        for action in (["score"], ["breakage"], ["max-test", "--note", "data refresh"]):
            assert exacta_run.main(action) == 0, action
        return {p.name: p.read_bytes() for p in _reports(data).iterdir() if SHA[:12] in p.name}

    def count(files, kind) -> int:
        return sum(1 for name in files if name.startswith(kind[0]) and name.endswith(kind[1]))

    first = build()
    assert all(count(first, kind) == 1 for kind in ARTIFACT_KINDS), sorted(first)
    _write_store(data, _with_dev_dividend_changed(base))  # same commit, another data build
    second = build()
    for name, raw in first.items():
        assert second.get(name) == raw, f"{name} was overwritten by the second build"
    assert all(count(second, kind) == 2 for kind in ARTIFACT_KINDS), sorted(second)
    builds = {r.data_builds["jra_dividends"] for r in _cli_registry(data).records()}
    assert len(builds) == 2  # the registry tells the builds apart as before
    assert [f.name for f in fields(RunRecord)] == RUN_RECORD_FIELDS  # no schema change


# ---------------------------------------------------------------------------------------------
# 10. Family ids the summary can name (owner, from #525, 2026-11-27)
# ---------------------------------------------------------------------------------------------

RUN_FAMILY_IDS = {
    "japan-131-exacta-category-v2",
    "japan-131-exacta-category-b1-v2",
    "japan-131-exacta-category-b3-v2",
    "japan-131-exacta-logscore-v2",
    "japan-131-exacta-rule-v5",
    MAX_TEST_FAMILY,
    BREAKAGE_FAMILY,
}


def test_every_131_family_id_has_no_slash_and_is_accepted_by_the_registry(tmp_path):
    specs = study.family_specs()
    defined = {*specs, exacta_run.MAX_TEST_ID, exacta_run.BREAKAGE_ID, *exacta_run.SCORED_FAMILIES}
    assert defined == RUN_FAMILY_IDS
    assert exacta_run.MAX_TEST_ID == MAX_TEST_FAMILY
    assert exacta_run.BREAKAGE_ID == BREAKAGE_FAMILY
    assert [f for f in defined if "/" in f] == []

    # #525's rule: append and freeze_family take each id as it is, and each is its own file.
    reg = _frozen_registry(tmp_path / "reg")
    for family_id, spec in specs.items():
        assert (
            reg.frozen_spec_hash(family_id)
            == hashlib.sha256(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        ), family_id
    template = _scored_under_another_fit()
    for i, family_id in enumerate(sorted(defined)):
        reg.append(RunRecord(**{**asdict(template), "family_id": family_id,
                                "variant": f"probe-{i}", "metrics": template.metrics}))  # fmt: skip
    assert {r.family_id for r in reg.records()} == defined
    reg.verify()
    out = tmp_path / "summary-out"
    out.mkdir()
    registry_module.write_summary(reg, out)
    assert {p.name for p in out.iterdir()} == {f"{f}.md" for f in defined} | {"README.md"}


# ---------------------------------------------------------------------------------------------
# 11. Data coverage is not variant identity (finding exacta_run.py:468)
# ---------------------------------------------------------------------------------------------

BACKFILL_DAYS = [date(2024, 4, 6), date(2024, 4, 13)]  # development days before the sparse ones


def _backfill_races() -> list[dict]:
    """Sixteen-runner development races (field:15-18) for every category, on two new days:
    the backfill of the sparse store's empty cell."""
    rng = np.random.default_rng(1313)
    design = list(itertools.product(range(4), (2,), (0, 1), (0, 1), CATEGORIES))
    design = [design[i] for i in rng.permutation(len(design))]
    return _allocate(BACKFILL_DAYS, VENUES, design, rng)


def test_a_backfill_that_fills_an_empty_cell_keeps_every_variant_and_family_size(
    sparse, isolated, tmp_path
):
    races, meta, exp = sparse
    assert "field:15-18" in exp["empty"]
    assert not any(_ref_holdout(d) for d in BACKFILL_DAYS)
    reg = _frozen_registry(tmp_path / "reg")
    first = _score(reg, races, meta)
    exacta_run.breakage(reg, copy.deepcopy(races), copy.deepcopy(meta), SHA, n_boot=N_BOOT)
    families = {_family_of("B2"): 84, _family_of("B1"): 42, _family_of("B3"): 84,
                _family_of("C"): 2, BREAKAGE_FAMILY: len(LAMBDA_KEYS)}  # fmt: skip
    assert {f: reg.family_size(f) for f in families} == families
    assert first["spa"]["family_size"] == 84
    before = {(r.family_id, r.variant): r for r in reg.records()}
    n_before = len(reg.records())
    assert n_before == TOTAL_RECORDS + len(LAMBDA_KEYS)

    # The same freeze and fits, a store backfilled with the empty cell's races.
    sparse_races, _ = _sparse_store()
    backfilled = [*sparse_races, *_backfill_races()]
    _write_store(isolated["data"], backfilled)
    _write_fits(isolated["data"], backfilled, **FIT_A)
    races2, meta2 = exacta_run.load_races()
    assert meta2["data_builds"] != meta["data_builds"]
    assert len(races2) > len(races)
    second = _score(reg, races2, meta2)
    exacta_run.breakage(reg, copy.deepcopy(races2), copy.deepcopy(meta2), SHA, n_boot=N_BOOT)

    assert {f: reg.family_size(f) for f in families} == families  # no new variant
    assert second["spa"]["n_rules"] == second["spa"]["family_size"] == 84
    assert second["stepm"]["family_size"] == 84
    after = reg.records()[n_before:]
    assert len(after) == TOTAL_RECORDS + len(LAMBDA_KEYS)  # each variant run again
    for record in after:
        old = before[(record.family_id, record.variant)]  # an existing variant
        assert record.params == old.params, record.variant
        assert record.data_builds != old.data_builds, record.variant

    # The coverage did change: the empty cell's statistics were unavailable and now are not.
    filled = [r for r in after if r.variant.endswith("|field:15-18")]
    assert len(filled) == (2 + 1 + 2) * len(CATEGORIES)
    for record in filled:
        estimate = "mean_return" if record.params["estimator"] == "B3" else "rho"
        old = before[(record.family_id, record.variant)]
        assert ("development", estimate) in _unavailable(old), record.variant
        assert ("development", estimate) not in _unavailable(record), record.variant
        present = _present(record)[("development", estimate)]
        assert present.n > 0 and None not in (present.lo, present.hi), record.variant
    for record in (r for r in after if r.family_id == BREAKAGE_FAMILY):
        old = before[(record.family_id, record.variant)]
        for category in CATEGORIES:
            fold = (f"development:{category}:field:15-18", "rho_unrounded")
            assert fold in _unavailable(old) and fold in _present(record), (record.variant, fold)


# ---------------------------------------------------------------------------------------------
# Shared by sections 13 to 16: independent references (loops, from the spec's definitions)
# ---------------------------------------------------------------------------------------------

FLOOR_YEN = 110.0  # JRA's minimum payout per 100 yen ("Plus 10", jra.yaml)


def _ref_exacta(p: dict[int, float], lambda2: float) -> dict[tuple[int, int], float]:
    """Discounted Harville by loops: P(i, j) = p_i p_j^l2 / sum_{k != i} p_k^l2, p renormalised."""
    total = sum(p.values())
    q = {h: v / total for h, v in p.items()}
    return {(i, j): q[i] * q[j] ** lambda2 / sum(q[k] ** lambda2 for k in q if k != i)
            for i, j in itertools.permutations(q, 2)}  # fmt: skip


def _ref_favourite(odds: dict[int, float]) -> int:
    return min(odds, key=lambda h: (odds[h], h))


def _ref_category(combo, fav) -> str:
    return "fav_first" if combo[0] == fav else "fav_second" if combo[1] == fav else "no_fav"


def _ref_box(race: dict, lambda2: float) -> dict[str, dict[str, float]]:
    """B1 and B2 return per unit staked on each category's box at the paid dividend."""
    (winner, dividend, _), fav = race["exacta"][0], _ref_favourite(race["win_odds"])
    h = _ref_exacta(race["p"], lambda2)
    cat = _ref_category(tuple(winner), fav)
    box = [c for c in h if _ref_category(c, fav) == cat]
    out = {e: dict.fromkeys(CATEGORIES, 0.0) for e in ("b1", "b2")}
    out["b1"][cat] = dividend / 100 / len(box)
    out["b2"][cat] = h[tuple(winner)] / sum(h[c] for c in box) * dividend / 100
    return out


def _race_cells(race: dict) -> set[str]:
    """The race's signed cells from its race card and final Win odds."""
    odds = race["win_odds"]
    fav_odds, n = odds[_ref_favourite(odds)], len(odds)
    band = ("le1.9" if fav_odds < 2.0 else "2.0-2.9" if fav_odds < 3.0
            else "3.0-4.9" if fav_odds < 5.0 else "ge5.0")  # fmt: skip
    group = {"DEBUT": "maiden", "MAIDEN": "maiden", "C1": "win1-3", "C2": "win1-3",
             "C3": "win1-3"}.get(race["race_class"], "open")  # fmt: skip
    less = race["race_no"] <= 4 or group == "maiden"
    cells = {"all", f"fav_odds:{band}",
             f"field:{'5-10' if n <= 10 else '11-14' if n <= 14 else '15-18'}",
             f"class:{group}", "popularity:less" if less else "popularity:rest"}  # fmt: skip
    if fav_odds >= 3.0 and less:
        cells.add("headline")
    return cells


def _lambda_values(meta) -> dict:
    return {study.FITTED: meta["lambda2_full_development"], 1.0: 1.0}


def test_the_references_on_a_hand_computed_race():
    race = {"win_odds": {1: 1.8, 2: 3.1, 3: 4.6}, "p": {1: 0.5, 2: 0.3, 3: 0.2},
            "exacta": [((1, 2), 480.0, False)]}  # fmt: skip
    plain = _ref_box(race, 1.0)  # fav_first box mass is p_1 = 0.5 at any lambda2
    assert plain["b2"]["fav_first"] == pytest.approx(0.3 / 0.5 * 4.8)  # 2.88
    assert plain["b1"]["fav_first"] == pytest.approx(4.8 / 2)  # 2 combinations, 2.4
    assert plain["b2"]["no_fav"] == plain["b1"]["fav_second"] == 0.0
    half = _ref_box(race, 0.5)
    share = math.sqrt(0.3) / (math.sqrt(0.3) + math.sqrt(0.2))
    assert half["b2"]["fav_first"] == pytest.approx(share * 4.8)  # 2.6425...
    assert half["b1"]["fav_first"] == plain["b1"]["fav_first"]  # B1 does not use p


# ---------------------------------------------------------------------------------------------
# 13. A corrected Win board with the same horses is refused (Sol 5, exacta_run.py:324)
# ---------------------------------------------------------------------------------------------


def _corrected_board(races: list[dict], change: str) -> tuple[list[dict], str]:
    """``races`` with one development race's Win board corrected, its horse numbers unchanged."""
    out = copy.deepcopy(races)
    race = next(r for r in out if r["race_date"] not in HOLDOUT_DAYS)
    odds = race["odds"]
    by_price = sorted(odds, key=lambda h: (odds[h], h))
    if change == "one_tick_on_the_longest_price":
        odds[by_price[-1]] = round(odds[by_price[-1]] + 0.1, 1)
    elif change == "longest_price_up_5":
        odds[by_price[-1]] = round(odds[by_price[-1]] + 5.0, 1)
    else:  # the favourite and the second favourite swap prices: the favourite changes
        a, b = by_price[0], by_price[1]
        odds[a], odds[b] = odds[b], odds[a]
    assert set(odds) == set(races[out.index(race)]["odds"])
    return out, race["race_id"]


CORRECTIONS = ("one_tick_on_the_longest_price", "longest_price_up_5", "favourite_swapped")


@pytest.mark.parametrize("change", CORRECTIONS)
def test_a_corrected_win_board_with_the_same_horses_is_refused(isolated, monkeypatch, change):
    data = isolated["data"]
    base = _base_races()
    corrected, race_id = _corrected_board(base, change)
    if change == "favourite_swapped":
        i = [r["race_id"] for r in base].index(race_id)
        assert _ref_favourite(corrected[i]["odds"]) != _ref_favourite(base[i]["odds"])
    _write_store(data, corrected)
    _write_fits(data, base, **FIT_A)  # the artifact was built on the board before the correction

    with pytest.raises(exacta_run.RunRefused, match=re.escape(race_id)):
        exacta_run.load_races()

    _cli_n_boot(monkeypatch)
    assert exacta_run.main(["freeze"]) == 0
    for action in (["score"], ["breakage"]):
        try:
            code = exacta_run.main(action)
        except exacta_run.RunRefused:
            code = None
        assert code != 0, action
    assert _cli_registry(data).records() == []  # refused: no record, no failure record
    assert not _reports(data).exists() or list(_reports(data).iterdir()) == []  # no report


def test_an_unchanged_board_passes_and_a_holdout_board_is_never_compared(isolated):
    data = isolated["data"]
    base = _base_races()
    _write_store(data, base)
    _write_fits(data, base, **FIT_A)
    races, _ = exacta_run.load_races()
    assert len(races) == 216

    # A holdout race's board corrected after the artifact was built: never read, so no refusal.
    changed = copy.deepcopy(base)
    held = next(r for r in changed if r["race_date"] in HOLDOUT_DAYS)
    by_price = sorted(held["odds"], key=held["odds"].get)
    held["odds"][by_price[0]], held["odds"][by_price[1]] = (held["odds"][by_price[1]],
                                                            held["odds"][by_price[0]])  # fmt: skip
    _write_store(data, changed)
    again, _ = exacta_run.load_races()
    assert again == races


# 13b. A corrected board on a race the scorer always excludes is skipped (Sol 6, exacta_run.py:353).
# Ruling (orchestrator): the race is absent from load_races()'s races and counted in
# meta["dropped_before_scoring"][f"board_corrected_{reason}"], reason exclusion_reason's value.

INELIGIBLE = ("dead_heat", "refund", "no_exacta_row")


def _make_ineligible(data: Path, races: list[dict], race_id: str, reason: str) -> None:
    """Rewrite the store's FORECAST rows of ``race_id`` so the exacta scorer excludes it."""
    path = data / "processed" / "jra" / "dividends.parquet"
    frame = pq.read_table(path).to_pandas()
    mine = (frame["race_id"] == race_id) & (frame["pool"] == "FORECAST")
    assert mine.sum() == 1
    if reason == "dead_heat":  # two horses dead-heat for second: two paying FORECAST rows
        race = next(r for r in races if r["race_id"] == race_id)
        first, second = race["winner"]
        third = min(h for h in race["odds"] if h not in (first, second))
        extra = frame[mine].copy()
        extra["combination"] = f"{first},{third}"
        extra["dividend"] = float(race["dividend"]) + 70.0
        frame = pd.concat([frame, extra], ignore_index=True)
    elif reason == "refund":
        frame.loc[mine, "is_refund"] = True
    else:  # no_exacta_row
        frame = frame[~mine]
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def _check_ineligible(entry: dict, reason: str) -> None:
    """The fixture race is excluded for ``reason`` (structural check, independent of the run)."""
    assert entry["boost_may_be_withheld"] is False and len(entry["win_odds"]) >= 5
    rows = entry["exacta"]
    if reason == "dead_heat":
        assert len(rows) == 2 and not any(refund for _, _, refund in rows)
    elif reason == "refund":
        assert len(rows) == 1 and rows[0][2] is True
    else:
        assert rows == []


@pytest.mark.parametrize("change", CORRECTIONS)
@pytest.mark.parametrize("reason", INELIGIBLE)
def test_a_corrected_board_on_an_excluded_race_is_skipped_and_counted(isolated, reason, change):
    data = isolated["data"]
    base = _base_races()
    corrected, race_id = _corrected_board(base, change)

    # Reference: the same excluded race with the board the artifact was built on is kept.
    _write_store(data, base)
    _make_ineligible(data, base, race_id, reason)
    _write_fits(data, base, **FIT_A)
    kept, kept_meta = exacta_run.load_races()
    assert len(kept) == 216
    (entry,) = [r for r in kept if r["race_id"] == race_id]
    _check_ineligible(entry, reason)
    assert not any(k.startswith("board_corrected_") for k in kept_meta["dropped_before_scoring"])

    # Corrected board on that excluded race: no refusal, the race is dropped and counted.
    _write_store(data, corrected)
    _make_ineligible(data, corrected, race_id, reason)
    races, meta = exacta_run.load_races()
    assert race_id not in {r["race_id"] for r in races}
    assert races == [r for r in kept if r["race_id"] != race_id]
    assert meta["dropped_before_scoring"] == {f"board_corrected_{reason}": 1}


def test_an_eligible_corrected_board_is_still_refused_beside_a_skipped_one(isolated):
    data = isolated["data"]
    base = _base_races()
    corrected, skipped = _corrected_board(base, "longest_price_up_5")
    dev = [r for r in corrected if r["race_date"] not in HOLDOUT_DAYS]
    eligible = dev[-1]  # a second development race, eligible, its board corrected too
    assert eligible["race_id"] != skipped
    by_price = sorted(eligible["odds"], key=lambda h: (eligible["odds"][h], h))
    eligible["odds"][by_price[-1]] = round(eligible["odds"][by_price[-1]] + 5.0, 1)
    _write_store(data, corrected)
    _make_ineligible(data, corrected, skipped, "dead_heat")
    _write_fits(data, base, **FIT_A)

    with pytest.raises(exacta_run.RunRefused, match=re.escape(eligible["race_id"])):
        exacta_run.load_races()


def test_score_and_breakage_complete_with_a_corrected_board_on_a_dead_heat(isolated, monkeypatch):
    data = isolated["data"]
    base = _base_races()
    corrected, race_id = _corrected_board(base, "favourite_swapped")
    _write_store(data, corrected)
    _make_ineligible(data, corrected, race_id, "dead_heat")
    _write_fits(data, base, **FIT_A)
    _cli_n_boot(monkeypatch)

    assert exacta_run.main(["freeze"]) == 0
    assert exacta_run.main(["score"]) == 0
    assert exacta_run.main(["breakage"]) == 0
    reg = _cli_registry(data)
    assert [r for r in reg.records() if r.status == "failed"] == []
    assert {k: v for k, v in _counts(reg).items() if k is not None} == EXPECTED_COUNTS
    assert len(reg.records(BREAKAGE_FAMILY)) == len(LAMBDA_KEYS)
    (js,) = _reports(data).glob(f"e1-{SHA[:12]}*.json")
    report = json.loads(js.read_text(encoding="utf-8"))
    assert report["spa"]["family_size"] == 84


# ---------------------------------------------------------------------------------------------
# 14. C with no race above 110 yen, and C of one meeting (Sol 5, exacta_run.py:493)
# ---------------------------------------------------------------------------------------------


def _at_floor(races: list[dict], keep_meeting: str | None = None) -> list[dict]:
    """Every dividend set to 110 yen, except the races of ``keep_meeting``."""
    out = copy.deepcopy(races)
    for r in out:
        if r["race_date"] in HOLDOUT_DAYS or r["meeting_id"] == keep_meeting:
            continue
        if "dividend" in r:  # a store race
            r["dividend"] = FLOOR_YEN
        else:  # a scoring input
            r["exacta"] = [(combo, FLOOR_YEN, refund) for combo, _, refund in r["exacta"]]
    return out


def _c_records(reg: Registry) -> dict[str, RunRecord]:
    return {r.variant: r for r in reg.records(_family_of("C"))}


def test_score_completes_when_no_dividend_is_above_110_yen(isolated, monkeypatch):
    data = isolated["data"]
    base = _base_races()
    _write_store(data, _at_floor(base))
    _write_fits(data, base, **FIT_A)
    _cli_n_boot(monkeypatch)
    assert exacta_run.main(["freeze"]) == 0
    assert exacta_run.main(["score"]) == 0

    reg = _cli_registry(data)
    assert [r for r in reg.records() if r.status == "failed"] == []
    assert _counts(reg) == EXPECTED_COUNTS  # E1, B1, B3, both C variants and the max test
    c = _c_records(reg)
    assert sorted(c) == sorted(f"c|lambda2={k}" for k in LAMBDA_KEYS)
    for variant, record in c.items():
        assert ("development", "log_score_gain") in _unavailable(record), variant
        assert ("development", "log_score_gain") not in _present(record), variant
        (marker,) = [f for f in record.metrics if f.metric.startswith("log_score_gain:")]
        assert marker.value == exacta_run.UNAVAILABLE_VALUE and marker.n == 0, marker

    (js,) = _reports(data).glob(f"e1-{SHA[:12]}*.json")
    report = json.loads(js.read_text(encoding="utf-8"))
    assert report["spa"]["family_size"] == 84
    assert len(report["members"]) == (2 + 1 + 2) * len(CATEGORIES) * len(study.SEGMENT_CELLS)
    rows = {str(r["lambda2"]): r for r in report["log_score"]}
    assert set(rows) == {str(k) for k in LAMBDA_KEYS}
    for row in rows.values():
        assert row["n"] == 0, row
        for stat in ("mean", "lo", "hi"):
            assert _is_unavailable(row[stat]), (row, stat)
    (md,) = _reports(data).glob(f"e1-{SHA[:12]}*.md")
    _, cells = _markdown_cells(md.read_text(encoding="utf-8"))
    assert not {"nan", "-nan", "inf", "-inf", "+inf"} & set(cells)


def _ref_log_score(races: list[dict], lambda2: float) -> list[float]:
    out = []
    for r in races:
        (winner, dividend, _) = r["exacta"][0]
        if dividend <= FLOOR_YEN:
            continue
        q = r["rate"] / ((dividend + 5.0) / 100)
        out.append(math.log(q) - math.log(_ref_exacta(r["p"], lambda2)[tuple(winner)]))
    return out


def test_a_one_meeting_c_sample_keeps_its_mean_without_bounds(loaded, tmp_path):
    races, meta = loaded
    meeting = races[0]["meeting_id"]
    sample = _at_floor(races, keep_meeting=meeting)
    kept = [r for r in sample if r["meeting_id"] == meeting]
    assert len(kept) >= 2
    reg = _frozen_registry(tmp_path)
    result = _score(reg, sample, meta)
    assert result["spa"]["family_size"] == 84

    rows = {str(r["lambda2"]): r for r in result["log_score"]}
    c = _c_records(reg)
    for key, value in _lambda_values(meta).items():
        want = _ref_log_score(kept, value)
        row = rows[str(key)]
        assert row["n"] == len(want) == len(kept) and row["meetings"] == 1, row
        assert row["mean"] == pytest.approx(sum(want) / len(want), rel=1e-9), row
        assert _is_unavailable(row["lo"]) and _is_unavailable(row["hi"]), row
        record = c[f"c|lambda2={key}"]
        gain = _present(record)[("development", "log_score_gain")]
        assert gain.value == pytest.approx(row["mean"], rel=1e-12)
        assert (gain.lo, gain.hi) == (None, None), gain
        assert ("development", "log_score_gain_bounds") in _unavailable(record), record.variant


# ---------------------------------------------------------------------------------------------
# 15. Return per 100 yen staked at the final dividend (Sol 5, 131-exacta-overload.md:88)
# ---------------------------------------------------------------------------------------------

PARAM_KEYS = {"estimator", "lambda2", "category", "cell", "direction"}


def _ref_return_per_100(races: list[dict], meta: dict) -> dict[tuple, tuple[float, int]]:
    """{(estimator, lambda2 key, category, cell): (100 * sum R / n, n)} for B2 and B1."""
    cells = {r["race_id"]: _race_cells(r) for r in races}
    out = {}
    for key, value in _lambda_values(meta).items():
        boxes = {r["race_id"]: _ref_box(r, value) for r in races}
        for estimator, column in (("B2", "b2"), ("B1", "b1")):
            if estimator == "B1" and key != study.FITTED:
                continue
            for category in CATEGORIES:
                for cell in study.SEGMENT_CELLS:
                    ids = [i for i in boxes if cell in cells[i]]
                    total = sum(boxes[i][column][category] for i in ids)
                    out[(estimator, str(key), category, cell)] = (
                        100 * total / len(ids) if ids else math.nan,
                        len(ids),
                    )
    return out


def test_score_measures_return_per_100_yen_at_the_final_dividend(sparse, tmp_path):
    races, meta, exp = sparse
    assert meta["lambda2_full_development"] != 1.0  # the two lambda2 keys give other returns
    reg = _frozen_registry(tmp_path)
    result = _score(reg, races, meta)
    ref = _ref_return_per_100(races, meta)
    by_variant = {(r.family_id, r.variant): r for r in reg.records()}

    seen_values = set()
    for m in result["members"]:
        est, key = m["estimator"], str(m["lambda2"])
        record = by_variant[(_family_of(est), f"{est.lower()}|lambda2={m['lambda2']}|"
                                              f"{m['category']}|{m['cell']}")]  # fmt: skip
        assert set(record.params) == PARAM_KEYS, record.params  # a metric, never a param
        present, missing = _present(record), _unavailable(record)
        metric, bounds = ("development", "return_per_100"), ("development", "return_per_100_bounds")
        if est == "B3":  # paid at Harville-fair odds, not the dividend: no return per 100 yen
            assert metric not in present, record.variant
            assert _is_unavailable(m.get("return_per_100")), m
            continue
        want, n = ref[(est, key, m["category"], m["cell"])]
        assert m["n"] == n, m
        if m["cell"] in exp["empty"]:
            assert _is_unavailable(m["return_per_100"]), m
            assert metric in missing and metric not in present, record.variant
            continue
        assert m["return_per_100"] == pytest.approx(want, rel=1e-9), m
        assert present[metric].value == pytest.approx(want, rel=1e-9), record.variant
        assert present[metric].n == n, record.variant
        seen_values.add(round(want, 6))
        if m["cell"] in exp["one_meeting"]:
            assert _is_unavailable(m["return_per_100_lo"]), m
            assert _is_unavailable(m["return_per_100_hi"]), m
            assert (present[metric].lo, present[metric].hi) == (None, None), record.variant
            assert bounds in missing, record.variant
        else:
            lo, hi = m["return_per_100_lo"], m["return_per_100_hi"]
            assert _finite(lo) and _finite(hi) and lo <= want <= hi, m
            assert (present[metric].lo, present[metric].hi) == pytest.approx((lo, hi)), m
            assert bounds not in missing, record.variant
            if m["cell"] == "all" and want > 0:
                assert lo < want < hi, m  # a real interval, not [x, x]
    assert len(seen_values) > 20  # varied answers: no constant passes

    # Identity unchanged: the families and the max test keep their sizes.
    sizes = {_family_of("B2"): 84, _family_of("B1"): 42, _family_of("B3"): 84,
             _family_of("C"): 2}  # fmt: skip
    assert {f: reg.family_size(f) for f in sizes} == sizes
    assert result["spa"]["family_size"] == result["stepm"]["family_size"] == 84


def test_the_e1_report_shows_return_per_100_beside_rho(loaded, isolated, tmp_path):
    races, meta = loaded
    result = _score(_frozen_registry(tmp_path), races, meta)
    ref = _ref_return_per_100(races, meta)
    out = isolated["data"] / "reports" / "japan" / "131-exacta"
    exacta_run.write_outputs(result, SHA, out)
    (js,) = out.glob(f"e1-{SHA[:12]}*.json")
    members = json.loads(js.read_text(encoding="utf-8"))["members"]
    checked = 0
    for m in members:
        if m["estimator"] == "B3":
            continue
        want, _ = ref[(m["estimator"], str(m["lambda2"]), m["category"], m["cell"])]
        assert m["return_per_100"] == pytest.approx(want, rel=1e-9), m
        assert m["return_per_100_lo"] <= want <= m["return_per_100_hi"], m
        checked += 1
    assert checked == (2 + 1) * len(CATEGORIES) * len(study.SEGMENT_CELLS)

    (md,) = out.glob(f"e1-{SHA[:12]}*.md")
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in md.read_text(encoding="utf-8").splitlines()
        if line.startswith("|")
    ]
    header = rows[0]
    col = next(i for i, name in enumerate(header) if "100" in name)  # the value, then its bounds
    at = {name: header.index(name) for name in ("estimator", "lambda2", "category", "cell")}
    shown = 0
    for row in rows[2:]:
        if row[at["estimator"]] == "B3" or len(row) != len(header):
            continue
        want, _ = ref[(row[at["estimator"]], row[at["lambda2"]], row[at["category"]],
                       row[at["cell"]])]  # fmt: skip
        assert float(row[col]) == pytest.approx(want, abs=0.51), row
        shown += 1
    assert shown == checked


# ---------------------------------------------------------------------------------------------
# 16. Breakage at the 110-yen floor is counted, not imputed (Sol 5, exacta_run.py:605)
# ---------------------------------------------------------------------------------------------


def _floor_sample(races: list[dict]) -> list[dict]:
    """Every maiden race pays the 110-yen floor; every other win1-3 race pays 120 yen (one step
    above it); the rest keep their dividends (all 120 yen or more)."""
    out = copy.deepcopy(races)
    for k, r in enumerate(out):
        (combo, _, refund) = r["exacta"][0]
        if r["race_class"] == "MAIDEN":
            r["exacta"] = [(combo, FLOOR_YEN, refund)]
        elif r["race_class"] == "C2" and k % 2 == 0:
            r["exacta"] = [(combo, 120.0, refund)]
    return out


def _ref_floor_breakage(races, meta) -> dict[tuple, dict]:
    cells = {r["race_id"]: _race_cells(r) for r in races}
    out = {}
    for key, value in _lambda_values(meta).items():
        rows = [(r, _ref_box(r, value)["b2"], r["exacta"][0][1]) for r in races]
        for category in CATEGORIES:
            for cell in study.SEGMENT_CELLS:
                sub = [(r, b2[category], d) for r, b2, d in rows if cell in cells[r["race_id"]]]
                ok = [(r, ret, d) for r, ret, d in sub if d > FLOOR_YEN]
                up = sum(ret * (d + 5.0) / d for _, ret, d in ok)
                ok_rate = sum(r["rate"] for r, _, _ in ok)
                out[(str(key), category, cell)] = {
                    "n": len(sub),
                    "n_floor": len(sub) - len(ok),
                    "rho": sum(ret for _, ret, _ in sub) / sum(r["rate"] for r, _, _ in sub),
                    "rho_unrounded": up / ok_rate if ok else math.nan,
                    "breakage_share": 1 - sum(ret for _, ret, _ in ok) / up if up > 0
                    else math.nan,
                }  # fmt: skip
    return out


BREAKAGE_STATS = {"breakage_share", "rho_unrounded", "rho_unrounded_bounds"}


def test_breakage_counts_floor_dividends_and_adds_back_only_above_the_floor(loaded, tmp_path):
    races, meta = loaded
    sample = _floor_sample(races)
    assert {r["exacta"][0][1] for r in sample} >= {FLOOR_YEN, 120.0}
    reg = _frozen_registry(tmp_path)
    rows = exacta_run.breakage(reg, copy.deepcopy(sample), copy.deepcopy(meta), SHA,
                               n_boot=N_BOOT)  # fmt: skip
    ref = _ref_floor_breakage(sample, meta)
    assert len(rows) == len(ref)
    kinds = set()
    for row in rows:
        want = ref[(str(row["lambda2"]), row["category"], row["cell"])]
        assert row["n"] == want["n"] and row["n_floor"] == want["n_floor"], row
        assert row["rho"] == pytest.approx(want["rho"], rel=1e-9), row  # paid: every race
        if want["n_floor"] == want["n"]:  # every race at the floor: nothing to add back
            kinds.add("all_floor")
            assert _is_unavailable(row["rho_unrounded"]), row
            assert _is_unavailable(row["breakage_share"]), row
            continue
        kinds.add("some_floor" if want["n_floor"] else "no_floor")
        for stat in ("rho_unrounded", "breakage_share"):
            if math.isnan(want[stat]):
                assert _is_unavailable(row[stat]), (row, stat)
            else:
                assert row[stat] == pytest.approx(want[stat], rel=1e-9), (row, stat)
    assert kinds == {"all_floor", "some_floor", "no_floor"}
    # the 120-yen dividends (win1-3, no floor race) still get half a step added back
    for key in LAMBDA_KEYS:
        win13 = ref[(str(key), "fav_first", "class:win1-3")]
        assert win13["n_floor"] == 0 and win13["rho_unrounded"] > win13["rho"]

    # The registry: the floor races' correction is an unavailable statistic with their count.
    records = reg.records(BREAKAGE_FAMILY)
    assert len(records) == len(LAMBDA_KEYS)
    for record in records:
        key = record.params["lambda2"]
        _unavailable(record)  # markers well formed, none in params
        for category in CATEGORIES:
            for cell in study.SEGMENT_CELLS:
                fold = f"development:{category}:{cell}"
                want = ref[(str(key), category, cell)]
                floor = [f for f in record.metrics if f.fold == fold and _marker(f) is not None
                         and _marker(f)[0] not in BREAKAGE_STATS]  # fmt: skip
                if want["n_floor"]:
                    assert len(floor) == 1, (fold, floor)
                    assert floor[0].n == want["n_floor"], (fold, floor)
                else:
                    assert floor == [], (fold, floor)
