"""Ticket #629: score the shadow page's live nowcast projections from the sealed ladder log (#628)
against the observed odds, pre-registered, with a random walk as the null.

Contract pinned here for `horsetrack.countries.japan.research.ladder_nowcast` (the coder
implements what these tests call):

- `FAMILY_ID` (text), `MIN_DAYS` (int >= 1: the minimum number of development race days a scored
  run needs) and `BANDS`, the odds-band cut points, equal to the nowcast's
  (`countries/japan/config/nowcast_v0.yaml` `bands`), so a row's band is the band the projector
  used. `family_spec() -> dict`, the plain-JSON spec the family is frozen with; it holds
  `"min_days": MIN_DAYS` and `"bands": list(BANDS)`. Its prose is
  `docs/specs/ladder-nowcast-prereg.md`.
- `freeze(registry) -> str`: `registry.freeze_family(FAMILY_ID, family_spec())`; returns the hash,
  idempotent.
- `score(records, *, seed, n_boot) -> Scored`: `records` are ladder-log records of any number of
  days (keyed by race_id, pool, combination, checkpoint_min). Every `ok` record at checkpoint k in
  (20, 10, 5, 2) and every later checkpoint j in its `projected` (j in 10, 5, 2 or 0, the final)
  is one pair, scored when the projection is not null and the key has an `ok` record at j. With
  O_k the record's odds, P its projection for j and O_j the observed odds at j:
  - predictor "nowcast": log error ln(P / O_j); predictor "random_walk" (the null, projection =
    O_k): ln(O_k / O_j). Both predictors are scored on the same pairs.
  - a direction hit when sign(ln(P / O_k)) == sign(ln(O_j / O_k)), sign three-valued (-1, 0, +1);
    for the random walk P = O_k, so it hits exactly the keys whose odds did not move.
  - `Scored.rows`: one row per (pool, band, from_min, to_min, predictor) with at least one scored
    pair, `band` the int `band_of(O_k, BANDS)` (the number of cut points at or below O_k), with
    `n`, `median_log_error` (the median, the mean of the middle two for an even n), `mae` (mean
    absolute log error), `hit_rate`, `ci_lo` <= `ci_hi` (the interval of the median log error,
    within the cell's smallest and largest error; a cell of one constant error has both bounds at
    it) and `primary`: True exactly for the page's rule, the previous rung to the next
    (`board.shadow_feed.PREVIOUS_RUNG[to] == from`: 20->10, 10->5, 5->2 and 2->final); every other
    (from, to) pair is a secondary row. Rows whose `band` is not an int (an all-bands summary, say)
    are allowed and not constrained here.
  - `Scored.excluded["null_projection"]`: the pairs whose projection is null;
    `Scored.excluded["no_board"]`: the no_board records. Neither crashes nor makes a row; no row
    has pool "*".
  - The same records and seed give the same rows.
- `FamilyNotFrozen(ValueError)`.
- `run(*, date_from, date_to, out_dir, registry, seed=0, n_boot=200) -> result` with `.status`
  ("scored" or "failed"), `.rows`, `.days` (the development days scored, ascending),
  `.report_path` and `.run_ids`:
  - refuses before reading any ladder file, with nothing appended to `registry`, when FAMILY_ID is
    not frozen in it (`FamilyNotFrozen`) or is frozen with a spec other than `family_spec()`
    (a ValueError);
  - reads `ladder_log_path(day)` for each day from `date_from` to `date_to` that has one, on
    development days only: a holdout day (`core.eval.holdout.is_holdout_day`) is skipped without
    its file being opened, and never counts toward MIN_DAYS;
  - with fewer than MIN_DAYS development days holding a ladder log, is a recorded failure: status
    "failed", at least one 'failed' registry record of FAMILY_ID and no 'scored' one, and a report;
  - otherwise scores the days' records as `score` does (`.rows`), appends at least one 'scored'
    record of FAMILY_ID with `ticket == 629` and sources naming "jravan" (the projections come from
    the nowcast fitted on JRA-VAN history: D21's withheld source), and writes its report through
    `core.eval.reports.write_report`, under `out_dir`, one report per run; an `out_dir` inside the
    repository raises `LicensedOutputInRepo` and leaves no file there.

Every figure expected below is hand-computed or derived here from the synthetic records by a
reference written in this file, never by the code under test. All ladder logs are synthetic and
written under tmp_path; holdout days are picked with `is_holdout_day`, never assumed.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import builtins
import io
import math
import os
import pathlib
import random
import shutil
import statistics
import uuid
from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.countries.japan.board.shadow_feed import PREVIOUS_RUNG
from horsetrack.countries.japan.shadow.ladder_log import LadderLog, ladder_log_path
from horsetrack.countries.japan.research import ladder_nowcast as ln

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_VAR = "HORSETRACK_DATA_DIR"
UTC = timezone.utc
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
CHECKPOINTS = (20, 10, 5, 2)
FINAL = 0
ALL_PAIRS = {(k, j) for k in CHECKPOINTS for j in (*CHECKPOINTS, FINAL) if j < k}
NOWCAST_BANDS = tuple(float(b) for b in load_country_yaml("japan", "nowcast_v0")["bands"])
LAST_DEV = date(2026, 11, 27)  # before the holdout start: every earlier day is development
N_BOOT = 50
TOL = 1e-9


class Forbidden(BaseException):
    """A holdout-day (or, before the freeze, any) ladder file was opened; a BaseException so no
    `except Exception` in the code under test swallows it."""


# ---------------------------------------------------------------- synthetic ladder records


def later(k: int) -> list[int]:
    return [j for j in CHECKPOINTS if j < k] + [FINAL]


def iso(at: datetime) -> str:
    return at.astimezone(UTC).isoformat()


def race_id(day: date, no: int) -> str:
    return f"jra:{day.isoformat()}:TOK:{no:02d}"


def post_of(day: date, no: int) -> datetime:
    return datetime.combine(day, time(1, 0), UTC) + timedelta(minutes=30 * no)


def rec(rid: str, post: datetime, pool: str, comb: str, k: int, odds, projected=None) -> dict:
    """One ok record at checkpoint k (k == 0: a final record)."""
    at = post - timedelta(minutes=k)
    return {
        "race_id": rid,
        "pool": pool,
        "combination": comb,
        "checkpoint_min": k,
        "status": "ok",
        "odds": odds,
        "fetched_utc": iso(at - timedelta(seconds=20)),
        "board_age_s": None if k == FINAL else 20.0,
        "projected": {} if projected is None else projected,
        "nowcast_id": "nowcast-synthetic",
        "mode": "replay",
        "post_utc": iso(post),
        "written_utc": iso(at + timedelta(seconds=15)),
        "git_sha": "0" * 40,
    }


def no_board(rid: str, post: datetime, k: int) -> dict:
    at = post - timedelta(minutes=k)
    return {
        "race_id": rid,
        "pool": "*",
        "combination": "*",
        "checkpoint_min": k,
        "status": "no_board",
        "odds": None,
        "fetched_utc": None,
        "board_age_s": None,
        "projected": {},
        "nowcast_id": "nowcast-synthetic",
        "mode": "replay",
        "post_utc": iso(post),
        "written_utc": iso(at + timedelta(seconds=15)),
        "git_sha": "0" * 40,
    }


START_ODDS = (
    1.4,
    2.6,
    4.1,
    7.5,
    13.0,
    28.0,
    64.0,
    140.0,
)  # one in each of the seven bands and more


def day_records(day: date, salt: int = 0, *, races: int = 2, scale: float = 1.0) -> list[dict]:
    """A synthetic day: each race has four WIN runners, three PLACE runners and three Wide pairs,
    the four checkpoints and a final; odds take a random walk (a quarter of the moves are none),
    projections are the from-odds times noise (some equal to it, some null), and a race may lose
    one checkpoint to a no_board record. A projection is null only when its target exists, and is
    never null toward a no_board checkpoint, so the null count is unambiguous."""
    rng = random.Random(f"{day.isoformat()}:{salt}")
    out: list[dict] = []
    for no in range(1, races + 1):
        rid, post = race_id(day, no), post_of(day, no)
        missing = rng.choice((None, None, 20, 5, 10))
        keys = (
            [(WIN, str(h)) for h in range(1, 5)]
            + [(PLACE, str(h)) for h in range(1, 4)]
            + [(WIDE, f"{a},{b}") for a, b in ((1, 2), (1, 3), (2, 3))]
        )
        path: dict[tuple[str, str], dict[int, float]] = {}
        for key in keys:
            odds = rng.choice(START_ODDS) * scale
            path[key] = {}
            for k in (*CHECKPOINTS, FINAL):
                if rng.random() >= 0.25:
                    odds = max(1.1, odds * math.exp(rng.gauss(0.0, 0.18)))
                path[key][k] = round(odds, 1)
        for k in CHECKPOINTS:
            if k == missing:
                out.append(no_board(rid, post, k))
                continue
            for pool, comb in keys:
                o = path[(pool, comb)][k]
                projected: dict[str, float | None] = {}
                for j in later(k):
                    draw = rng.random()
                    if j != missing and draw < 0.12:
                        projected[str(j)] = None
                    elif draw < 0.22:
                        projected[str(j)] = o
                    else:
                        projected[str(j)] = round(max(1.0, o * math.exp(rng.gauss(0, 0.12))), 2)
                out.append(rec(rid, post, pool, comb, k, o, projected))
        for pool, comb in keys:
            out.append(rec(rid, post, pool, comb, FINAL, path[(pool, comb)][FINAL]))
    return out


# ---------------------------------------------------------------- the reference


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def reference(records: list[dict]) -> tuple[dict, dict]:
    """(cell -> {"nowcast": [(error, hit)], "random_walk": [...]}, excluded counts), from the
    ticket's definitions alone."""
    observed = {
        (r["race_id"], r["pool"], r["combination"], r["checkpoint_min"]): r["odds"]
        for r in records
        if r["status"] == "ok"
    }
    cells: dict = defaultdict(lambda: {"nowcast": [], "random_walk": []})
    excluded = {"null_projection": 0, "no_board": 0}
    for r in records:
        if r["status"] != "ok":
            excluded["no_board"] += 1
            continue
        k = r["checkpoint_min"]
        if k == FINAL:
            continue
        o = r["odds"]
        for j_text, p in r["projected"].items():
            j = int(j_text)
            if p is None:
                excluded["null_projection"] += 1
                continue
            to = observed.get((r["race_id"], r["pool"], r["combination"], j))
            if to is None:
                continue
            move = sign(math.log(to / o))
            cell = (r["pool"], bisect_right(NOWCAST_BANDS, o), k, j)
            cells[cell]["nowcast"].append((math.log(p / to), sign(math.log(p / o)) == move))
            cells[cell]["random_walk"].append((math.log(o / to), move == 0))
    return dict(cells), excluded


def int_band_rows(rows: list[dict]) -> dict[tuple, dict]:
    out = {}
    for row in rows:
        band = row["band"]
        if isinstance(band, int) and not isinstance(band, bool):
            key = (row["pool"], band, row["from_min"], row["to_min"], row["predictor"])
            assert key not in out, f"two rows for {key}"
            out[key] = row
    return out


def assert_rows_match(rows: list[dict], records: list[dict]) -> None:
    cells, _ = reference(records)
    got = int_band_rows(rows)
    expected_keys = {
        (*cell, predictor) for cell in cells for predictor in ("nowcast", "random_walk")
    }
    assert set(got) == expected_keys
    for (pool, band, k, j, predictor), row in got.items():
        pairs = cells[(pool, band, k, j)][predictor]
        errors = [e for e, _ in pairs]
        assert row["n"] == len(pairs), (pool, band, k, j, predictor)
        assert row["median_log_error"] == pytest.approx(statistics.median(errors), abs=TOL)
        assert row["mae"] == pytest.approx(sum(abs(e) for e in errors) / len(errors), abs=TOL)
        assert row["hit_rate"] == pytest.approx(sum(h for _, h in pairs) / len(pairs), abs=TOL)
        assert row["primary"] is (PREVIOUS_RUNG[j] == k)
        if len(errors) >= 2:
            assert min(errors) - TOL <= row["ci_lo"] <= row["ci_hi"] <= max(errors) + TOL


# ---------------------------------------------------------------- score: hand-computed cells


def literal_records() -> list[dict]:
    """Race 1: three WIN runners at 4.0 at T-10 (band 1, 2-5) projected for T-5 only; every other
    projection null. Race 2: one WIN runner at 12.0 at T-10 (band 3, 10-20) whose T-5 is a
    no_board checkpoint, so only its T-10 -> T-2 projection scores; its T-20 is no_board too."""
    day = date(2026, 11, 20)
    r1, p1 = race_id(day, 1), post_of(day, 1)
    r2, p2 = race_id(day, 2), post_of(day, 2)
    out = []
    t10 = {"1": 3.6, "2": 4.4, "3": 3.8}
    t5 = {"1": 3.0, "2": 5.0, "3": 4.6}
    t2 = {"1": 3.1, "2": 4.8, "3": 4.5}
    t0 = {"1": 3.2, "2": 4.7, "3": 4.4}
    for h in ("1", "2", "3"):
        null20 = {"10": None, "5": None, "2": None, "0": None}
        out.append(rec(r1, p1, WIN, h, 20, 4.2, null20))
        out.append(rec(r1, p1, WIN, h, 10, 4.0, {"5": t10[h], "2": None, "0": None}))
        out.append(rec(r1, p1, WIN, h, 5, t5[h], {"2": None, "0": None}))
        out.append(rec(r1, p1, WIN, h, 2, t2[h], {"0": None}))
        out.append(rec(r1, p1, WIN, h, FINAL, t0[h]))
    out.append(no_board(r2, p2, 20))
    out.append(rec(r2, p2, WIN, "1", 10, 12.0, {"5": 11.0, "2": 10.5, "0": None}))
    out.append(no_board(r2, p2, 5))
    out.append(rec(r2, p2, WIN, "1", 2, 10.0, {"0": None}))
    out.append(rec(r2, p2, WIN, "1", FINAL, 9.0))
    return out


def test_bands_are_the_nowcasts_and_the_spec_holds_them():
    assert tuple(float(b) for b in ln.BANDS) == NOWCAST_BANDS
    assert isinstance(ln.MIN_DAYS, int) and not isinstance(ln.MIN_DAYS, bool) and ln.MIN_DAYS >= 1
    spec = ln.family_spec()
    assert spec["min_days"] == ln.MIN_DAYS
    assert [float(b) for b in spec["bands"]] == list(NOWCAST_BANDS)
    assert (REPO_ROOT / "docs" / "specs" / "ladder-nowcast-prereg.md").is_file()


def test_hand_computed_cells_and_exclusions():
    scored = ln.score(literal_records(), seed=0, n_boot=N_BOOT)
    rows = int_band_rows(scored.rows)
    assert set(rows) == {
        (WIN, 1, 10, 5, "nowcast"),
        (WIN, 1, 10, 5, "random_walk"),
        (WIN, 3, 10, 2, "nowcast"),
        (WIN, 3, 10, 2, "random_walk"),
    }
    assert all(row["pool"] != "*" for row in scored.rows)

    # Race 1, T-10 -> T-5: projections 3.6, 4.4, 3.8 against 3.0, 5.0, 4.6 from 4.0.
    nowcast = rows[(WIN, 1, 10, 5, "nowcast")]
    errors = [math.log(3.6 / 3.0), math.log(4.4 / 5.0), math.log(3.8 / 4.6)]
    assert nowcast["n"] == 3
    assert nowcast["median_log_error"] == pytest.approx(math.log(4.4 / 5.0), abs=TOL)
    assert nowcast["mae"] == pytest.approx(sum(map(abs, errors)) / 3, abs=TOL)
    assert nowcast["hit_rate"] == pytest.approx(2 / 3, abs=TOL)  # down/down, up/up, down/up
    assert nowcast["primary"] is True
    assert min(errors) - TOL <= nowcast["ci_lo"] <= nowcast["ci_hi"] <= max(errors) + TOL

    walk = rows[(WIN, 1, 10, 5, "random_walk")]
    errors = [math.log(4.0 / 3.0), math.log(4.0 / 5.0), math.log(4.0 / 4.6)]
    assert walk["n"] == 3
    assert walk["median_log_error"] == pytest.approx(math.log(4.0 / 4.6), abs=TOL)
    assert walk["mae"] == pytest.approx(sum(map(abs, errors)) / 3, abs=TOL)
    assert walk["hit_rate"] == pytest.approx(0.0, abs=TOL)  # it never calls a move; all three moved
    assert walk["primary"] is True

    # Race 2, T-10 -> T-2 (its T-5 target is no_board): 10.5 against 10.0 from 12.0.
    nowcast = rows[(WIN, 3, 10, 2, "nowcast")]
    assert nowcast["n"] == 1
    assert nowcast["median_log_error"] == pytest.approx(math.log(1.05), abs=TOL)
    assert nowcast["mae"] == pytest.approx(math.log(1.05), abs=TOL)
    assert nowcast["hit_rate"] == pytest.approx(1.0, abs=TOL)
    assert nowcast["primary"] is False
    walk = rows[(WIN, 3, 10, 2, "random_walk")]
    assert walk["median_log_error"] == pytest.approx(math.log(1.2), abs=TOL)
    assert walk["mae"] == pytest.approx(math.log(1.2), abs=TOL)
    assert walk["hit_rate"] == pytest.approx(0.0, abs=TOL)
    assert walk["primary"] is False

    # Nulls: race 1 has 3 x (4 + 2 + 2 + 1) null projections, race 2 one at T-10 and one at T-2.
    assert scored.excluded["null_projection"] == 27 + 2
    assert scored.excluded["no_board"] == 2


@pytest.mark.parametrize("salt", [0, 1, 2, 3])
def test_metrics_match_the_reference_over_varied_days(salt):
    records = [
        r
        for i in range(3)
        for r in day_records(date(2026, 10, 5) + timedelta(days=7 * i), salt, races=3)
    ]
    scored = ln.score(records, seed=salt, n_boot=N_BOOT)
    assert_rows_match(scored.rows, records)
    _, excluded = reference(records)
    assert scored.excluded["null_projection"] == excluded["null_projection"]
    assert scored.excluded["no_board"] == excluded["no_board"]
    assert all(row["pool"] != "*" for row in scored.rows)


def test_primary_pairs_are_the_pages_rule_and_every_other_pair_is_secondary():
    records = [
        r for i in range(4) for r in day_records(date(2026, 9, 1) + timedelta(days=i), 9, races=4)
    ]
    rows = int_band_rows(ln.score(records, seed=0, n_boot=N_BOOT).rows)
    pairs = {(row["from_min"], row["to_min"]) for row in rows.values()}
    assert pairs == ALL_PAIRS  # secondary rows are kept, not only the page's pairs
    primary = {(row["from_min"], row["to_min"]) for row in rows.values() if row["primary"]}
    assert primary == {(20, 10), (10, 5), (5, 2), (2, 0)}


def test_the_random_walk_scores_exactly_zero_drift_on_flat_odds():
    """Every key's odds are the same at every checkpoint and the final: the null's projection is
    the from-odds, so each of its errors is exactly zero and it calls every (absent) move; the
    nowcast, projecting moves that never come, does not."""
    day = date(2026, 11, 2)
    rng = random.Random(629)
    records = []
    for no in (1, 2):
        rid, post = race_id(day, no), post_of(day, no)
        for pool, comb in [(WIN, "1"), (WIN, "2"), (PLACE, "1"), (WIDE, "1,2"), (WIN, "3")]:
            o = rng.choice(START_ODDS)
            for k in CHECKPOINTS:
                projected = {
                    str(j): round(o * rng.choice((0.8, 0.9, 1.1, 1.25)), 2) for j in later(k)
                }
                records.append(rec(rid, post, pool, comb, k, o, projected))
            records.append(rec(rid, post, pool, comb, FINAL, o))
    rows = int_band_rows(ln.score(records, seed=3, n_boot=N_BOOT).rows)
    walks = [row for key, row in rows.items() if key[4] == "random_walk"]
    assert walks
    for row in walks:
        assert row["median_log_error"] == 0.0
        assert row["mae"] == 0.0
        assert row["hit_rate"] == 1.0
        if row["n"] >= 2:
            assert row["ci_lo"] == 0.0 and row["ci_hi"] == 0.0
    nowcasts = [row for key, row in rows.items() if key[4] == "nowcast"]
    assert {(r["pool"], r["band"], r["from_min"], r["to_min"]) for r in nowcasts} == {
        (r["pool"], r["band"], r["from_min"], r["to_min"]) for r in walks
    }
    assert all(row["mae"] > 0.0 and row["hit_rate"] == 0.0 for row in nowcasts)
    assert_rows_match(list(rows.values()), records)


def test_same_records_and_seed_give_the_same_rows():
    records = [r for i in range(2) for r in day_records(date(2026, 8, 3) + timedelta(days=i), 4)]
    first = ln.score(records, seed=11, n_boot=N_BOOT).rows
    again = ln.score(list(records), seed=11, n_boot=N_BOOT).rows
    assert first == again


# ---------------------------------------------------------------- run: files, registry, report


def registry_at(root: Path) -> Registry:
    return Registry(root / "registry" / "japan.jsonl", holdout_log=root / "holdout_log.jsonl")


def write_day(day: date, records: list[dict]) -> Path:
    path = ladder_log_path(day)
    log = LadderLog(path, day=day)
    log.append(records)
    log.seal(sealed_utc=datetime.combine(day, time(9, 0), UTC))
    return path


def dev_days(n: int) -> list[date]:
    return [LAST_DEV - timedelta(days=i) for i in range(n)][::-1]


def first_holdout_day() -> date:
    day = date(2026, 11, 28)
    while not is_holdout_day("japan", day):
        day += timedelta(days=1)
    return day


def forbid_opening(monkeypatch, token: str) -> list[str]:
    """Make every way of opening a file raise Forbidden for a path containing `token`."""
    opened: list[str] = []
    targets = [
        (builtins, "open"),
        (io, "open"),
        (os, "open"),
        (pathlib.Path, "open"),
        (pathlib.Path, "read_bytes"),
        (pathlib.Path, "read_text"),
    ]
    for owner, name in targets:
        real = getattr(owner, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            for value in (*args, *kwargs.values()):
                if isinstance(value, (str, os.PathLike)) and token in os.fspath(value):
                    opened.append(f"{_name}({os.fspath(value)})")
                    raise Forbidden(f"{_name} opened {os.fspath(value)}")
            return _real(*args, **kwargs)

        monkeypatch.setattr(owner, name, spy)
    return opened


@pytest.fixture
def data(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv(DATA_VAR, str(root))
    return root


def test_freeze_freezes_the_family_spec_once(data):
    registry = registry_at(data)
    digest = ln.freeze(registry)
    assert digest == registry.frozen_spec_hash(ln.FAMILY_ID)
    assert registry.frozen_spec(ln.FAMILY_ID) == ln.family_spec()
    assert ln.freeze(registry) == digest


def test_a_run_without_the_frozen_spec_refuses_before_any_read(data, tmp_path, monkeypatch):
    for i, day in enumerate(dev_days(ln.MIN_DAYS)):
        write_day(day, day_records(day, i))
    out = tmp_path / "out"
    registry = registry_at(data)
    opened = forbid_opening(monkeypatch, ".ladder.jsonl")
    with pytest.raises(ln.FamilyNotFrozen):
        ln.run(
            date_from=dev_days(ln.MIN_DAYS)[0],
            date_to=LAST_DEV,
            out_dir=out,
            registry=registry,
            seed=0,
            n_boot=N_BOOT,
        )
    assert opened == []
    assert registry.records() == []
    assert not out.exists() or not any(out.iterdir())

    other = registry_at(tmp_path / "other")
    other.freeze_family(ln.FAMILY_ID, {"min_days": 1, "bands": [3.0]})
    with pytest.raises(ValueError):
        ln.run(
            date_from=dev_days(ln.MIN_DAYS)[0],
            date_to=LAST_DEV,
            out_dir=out,
            registry=other,
            seed=0,
            n_boot=N_BOOT,
        )
    assert opened == []
    assert other.records() == []


def test_a_run_scores_development_days_and_never_opens_a_holdout_day(data, tmp_path, monkeypatch):
    days = dev_days(ln.MIN_DAYS)
    records: list[dict] = []
    for i, day in enumerate(days):
        day_recs = day_records(day, 100 + i)
        write_day(day, day_recs)
        records += day_recs
    held = first_holdout_day()
    held_path = write_day(held, day_records(held, 7, scale=3.0))
    before = days[0] - timedelta(days=1)  # a development day outside the range
    write_day(before, day_records(before, 8, scale=2.0))
    assert held_path.is_file()

    registry = registry_at(data)
    ln.freeze(registry)
    out = data / "reports"
    opened = forbid_opening(monkeypatch, held.isoformat())
    result = ln.run(
        date_from=days[0], date_to=held, out_dir=out, registry=registry, seed=5, n_boot=N_BOOT
    )
    assert opened == []
    assert result.status == "scored"
    assert tuple(result.days) == tuple(days)
    assert_rows_match(result.rows, records)

    runs = registry.records(ln.FAMILY_ID)
    assert runs and all(r.status == "scored" for r in runs)
    assert all(r.ticket == 629 and "jravan" in r.sources for r in runs)
    assert len(result.run_ids) == len(runs)
    report = Path(result.report_path)
    assert report.is_file() and report.stat().st_size > 0
    assert report.resolve().is_relative_to(out.resolve())


def test_too_few_development_days_is_a_recorded_failure(data, tmp_path, monkeypatch):
    days = dev_days(ln.MIN_DAYS - 1)
    for i, day in enumerate(days):
        write_day(day, day_records(day, 200 + i))
    held = first_holdout_day()  # a holdout day's file never makes up the count
    write_day(held, day_records(held, 3))
    registry = registry_at(data)
    ln.freeze(registry)
    out = data / "reports"
    opened = forbid_opening(monkeypatch, held.isoformat())
    result = ln.run(
        date_from=LAST_DEV - timedelta(days=ln.MIN_DAYS + 5),
        date_to=held,
        out_dir=out,
        registry=registry,
        seed=0,
        n_boot=N_BOOT,
    )
    assert opened == []
    assert result.status == "failed"
    runs = registry.records(ln.FAMILY_ID)
    assert runs and all(r.status == "failed" for r in runs)
    assert all(r.ticket == 629 for r in runs)
    assert Path(result.report_path).is_file()


def test_each_run_appends_its_own_record_and_report(data, tmp_path):
    days = dev_days(ln.MIN_DAYS)
    for i, day in enumerate(days):
        write_day(day, day_records(day, 300 + i))
    registry = registry_at(data)
    ln.freeze(registry)
    out = data / "reports"
    common = {"date_from": days[0], "date_to": days[-1], "out_dir": out, "registry": registry}
    first = ln.run(**common, seed=1, n_boot=N_BOOT)
    n_first = len(registry.records(ln.FAMILY_ID))
    second = ln.run(**common, seed=2, n_boot=N_BOOT)
    n_second = len(registry.records(ln.FAMILY_ID))
    assert 0 < n_first < n_second
    assert Path(first.report_path) != Path(second.report_path)
    assert Path(first.report_path).is_file() and Path(second.report_path).is_file()


def test_a_report_inside_the_repository_is_refused(data, tmp_path):
    days = dev_days(ln.MIN_DAYS)
    for i, day in enumerate(days):
        write_day(day, day_records(day, 400 + i))
    registry = registry_at(data)
    ln.freeze(registry)
    inside = REPO_ROOT / f".ladder-nowcast-test-{uuid.uuid4().hex}"
    try:
        with pytest.raises(LicensedOutputInRepo):
            ln.run(
                date_from=days[0],
                date_to=days[-1],
                out_dir=inside,
                registry=registry,
                seed=0,
                n_boot=N_BOOT,
            )
        assert not inside.exists() or not any(p.is_file() for p in inside.rglob("*"))
    finally:
        shutil.rmtree(inside, ignore_errors=True)
