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
"""

from __future__ import annotations

import copy
import functools
import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict
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
from horsetrack.core.eval.registry import Registry
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
    assert (reports / f"e1-{SHA[:12]}.json").is_file()
    assert (reports / f"e1-{SHA[:12]}.md").is_file()


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
