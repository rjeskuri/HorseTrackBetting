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
  cell without a winning return has the valid estimate 0.0. In the registry an unavailable
  statistic is no FoldMetric; the record lists it in ``params["unavailable"]``, a list of
  ``{"fold", "metric", "reason"}`` dicts (reason non-blank). B2's ``rho_eb`` is unavailable for a
  member without a finite estimate and a positive SE; the other members of its lambda2 x category
  group are shrunk among themselves (``eb_shrink`` on their estimates and SEs). Every other
  member, C and the max test are still recorded.
- Report artifacts: two data builds at one commit keep separate e1, breakage and max-test
  artifacts (each name holds the 12-digit SHA and more); the first build's files are unchanged
  after the second. RunRecord's fields are unchanged.
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
    """The v0 fit files ``_fits`` reads; the out-of-fold file covers every synthetic race."""
    model = data / "models" / "japan" / "v0"
    model.mkdir(parents=True, exist_ok=True)
    oof = model / f"win_calibration_oof-{win_id}.parquet"
    rows = []
    for r in races:
        raw = {h: v * (1 + p_shift * ((h % 3) - 1)) for h, v in r["p"].items()}
        total = sum(raw.values())
        rows += [{"race_id": r["race_id"], "horse_no": h, "p_cal": v / total}
                 for h, v in raw.items()]  # fmt: skip
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
    races = _base_races() if races is None else races
    _write_store(data, races)
    _write_fits(data, _base_races(), **(fits or FIT_A))
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
        key = "max-test" if r.family_id.endswith("/max-test") else by_family.get(r.family_id)
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
    assert [r for r in other.records() if r.family_id.endswith("/max-test")] == []


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


def _unavailable(record: RunRecord) -> dict[tuple[str, str], str]:
    """The record's ``params["unavailable"]`` as {(fold, metric): reason}."""
    entries = record.params.get("unavailable", [])
    assert isinstance(entries, list), record.variant
    out = {}
    for entry in entries:
        assert isinstance(entry, dict) and {"fold", "metric", "reason"} <= set(entry), entry
        assert isinstance(entry["reason"], str) and entry["reason"].strip(), entry
        out[(entry["fold"], entry["metric"])] = entry["reason"]
    return out


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
    for record in reg_dev.records(exacta_run.BREAKAGE_ID):
        assert "unavailable" not in record.params  # nothing unavailable: the params as recorded

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
        for stat in ("rho", "lo", "hi"):
            assert _finite(m[stat]), (m, stat)
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
        present = {(f.fold, f.metric): f for f in record.metrics}
        missing = _unavailable(record)
        assert not set(present) & set(missing), record.variant
        assert all(f.n == m["n"] for f in present.values()), record.variant
        estimate = ("development", "mean_return" if est == "B3" else "rho")
        wanted = {estimate, ("development", "bootstrap_se")}
        if est == "B2":
            wanted.add(("development", "rho_eb"))
        assert wanted <= set(present) | set(missing), record.variant
        member = (m["category"], m["cell"])
        if m["cell"] in exp["empty"]:
            assert wanted <= set(missing), record.variant
        elif member in exp["zero_return"]:
            assert present[estimate].value == 0.0, record.variant
            if est == "B2":
                assert ("development", "rho_eb") in missing, record.variant
        elif m["cell"] in exp["one_meeting"]:
            assert present[estimate].value == m["rho"], record.variant
        else:
            assert wanted <= set(present), record.variant
            assert "unavailable" not in record.params, record.variant  # params as recorded


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
        elif member in exp["zero_return"]:
            assert r["rho"] == 0.0 and r["rho_unrounded"] == 0.0, r  # valid zero returns
            assert _is_unavailable(r["breakage_share"]), r  # 0 / 0: no share
            assert _finite(r["lo"]) and _finite(r["hi"]), r
        else:
            assert _finite(r["breakage_share"]) and 0 < r["breakage_share"] < 1, r
            assert 0 < r["rho"] < r["rho_unrounded"], r

    records = reg.records(exacta_run.BREAKAGE_ID)
    variants = sorted(f"breakage|lambda2={k}" for k in LAMBDA_KEYS)
    assert sorted(r.variant for r in records) == variants
    for record in records:
        present = {(f.fold, f.metric): f for f in record.metrics}
        missing = _unavailable(record)
        assert not set(present) & set(missing)
        for category in CATEGORIES:
            for cell in study.SEGMENT_CELLS:
                fold = f"development:{category}:{cell}"
                share, unrounded = (fold, "breakage_share"), (fold, "rho_unrounded")
                if cell in exp["empty"]:
                    assert share in missing and unrounded in missing, fold
                elif (category, cell) in exp["zero_return"]:
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
    assert len(reg.records(exacta_run.BREAKAGE_ID)) == len(LAMBDA_KEYS)

    n_members = (2 + 1 + 2) * len(CATEGORIES) * len(study.SEGMENT_CELLS)
    for prefix, n_rows in (("e1-", n_members), ("breakage-", 2 * 3 * len(study.SEGMENT_CELLS))):
        (md,) = _reports(data).glob(f"{prefix}{SHA[:12]}*.md")
        body, cells = _markdown_cells(md.read_text(encoding="utf-8"))
        assert len(body) >= n_rows + 1, prefix  # a header and every member
        assert not {"nan", "-nan", "inf", "-inf", "+inf"} & set(cells), prefix
        (js,) = _reports(data).glob(f"{prefix}{SHA[:12]}*.json")
        assert js.stat().st_size > 0
    assert exp["empty"]  # the sample has an empty cell to render


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
