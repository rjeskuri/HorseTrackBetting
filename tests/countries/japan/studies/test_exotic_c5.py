"""The C5 exotic layer run against the pools' own dividends (ticket #459, epic #73; 08 E43, E44,
E16; owner answers 1-7 on #99 of 2026-11-26, the reframe of 2026-11-27, the 'drop'/'bound' comment
on #459 of 2026-11-27).

Contract pinned here for ``horsetrack.countries.japan.studies.exotic_c5`` (the coder implements
what these tests call):

Constants. ``COUNTRY = "japan"``, ``TICKET = 459``, ``FAMILY_ID`` (one new family, a safe summary
stem; every record of the run is in it). ``POOLS`` holds every pool of ``exotic.POOLS``;
``ORDERINGS`` every family of ``exotic.FAMILIES``. ``NORMALISED = "normalised"`` (1/odds normalised
per race), ``CALIBRATED = "calibrated"`` (D4's spline, ``models.v0.calibration.
fit_win_calibration`` at its default knots, refitted on each fold's training rows), ``FORM =
"stage2:q:form"`` (#87's member, JRDB-fed); ``INPUTS = (NORMALISED, CALIBRATED, FORM)``.
``E44_POOLS = (TRIO, TIERCE)``; ``FLOOR_TREATMENTS = ("drop", "bound")``; ``PLACE_WIDE_DEVIGS`` a
non-empty tuple of de-vig names; ``BLENDS = ("none", "linear", "log_linear")``.

Spec. ``build_spec(scheme: FoldScheme, bootstrap: Mapping) -> dict``, plain JSON, carrying
``family_id``, ``ticket``, ``fold_scheme`` (``dataclasses.asdict(scheme)``), ``bootstrap``
(``n_boot``, ``level``, ``seed`` read by every bound) and the variant grid as lists, read by
``compute_study``: ``pools``, ``orderings``, ``inputs``, ``blends``, ``e44_pools``,
``place_wide_devigs``, ``floor_treatments`` (a test may shorten a list to run less).
``default_spec()`` is ``build_spec`` on ``load_country_scheme("japan")`` and D7's bootstrap
(config/scoring.yaml: cluster unit, level, n_boot; an integer seed), the full grid: what
``run_study`` freezes.

The pure study. ``compute_study(races, runners, dividends, spec, *, form=None) ->
studies.calibration.Study``: store-shaped tables (races: race_id, race_date (datetime.date),
meeting_id, venue, race_no, race_class_raw, field_size, status; runners: race_id, race_date,
meeting_id, venue, race_no, horse_id, horse_no, win_odds, finish_pos, dead_heat, is_scratched;
dividends: race_id, pool, combination, dividend, is_refund) and ``form`` (race_id, horse_no,
p_form) or None. Folds are ``walk_forward`` on the race dates under ``spec["fold_scheme"]``; a
scored race is one in a fold's test window whose pool pays. Each variant has ``hypothesis_id``,
``params`` (plain JSON), ``metrics`` (``FoldMetric``) and ``error`` (None, or why it failed):

- E43 (``hypothesis_id == "E43"``), one variant per grid point: params ``pool``, ``input``,
  ``ordering`` (WIN may carry None: its Win marginal is the input itself), ``blend`` ("none" or
  "linear"), and for PLACE and QUINELLA_PLACE also ``devig`` and ``floor``. Per scored race the
  log loss is -(1/k) sum ln p(c) over its k winning combinations. The model p is P_C5: the input's
  Win probabilities (renormalised per race) through the ordering (``Ordering.fit`` on the fold's
  training rows; fold k's parameters on fold k's test races) by horse number, Place's places by
  ``field_size``. The baseline is ``exotic.dividend_shares`` on the race-pool's dividends at the
  race's own payout rate (boosts and ``race_class_raw`` in). Metric ``delta_ll_vs_dividends`` on
  fold ``all_folds``: the mean over scored races of (baseline log loss - model log loss), its lo
  and hi ``core.eval.bootstrap.paired_bootstrap`` over meeting_id clusters at
  ``spec["bootstrap"]``, n the scored races. ``blend == "linear"``: p = alpha q + (1 - alpha) P_C5
  on each winning combination, alpha in [0, 1] minimising the mean log loss over the fold's
  training races (dates up to its ``train_end``) of that pool, input and ordering; one ``alpha``
  metric per fold, fold label ``test_start.isoformat()``. PLACE and QUINELLA_PLACE: 'drop' leaves
  out the race-pools ``exotic.dividend_floor_bound`` flags; 'bound' scores every race-pool.
- ``blend == "log_linear"`` (``exotic.pool_blend``) is recorded and fails: not runnable yet.
- E16 (``"E16"``): the E43 grid again (same pool, input, ordering, blend, devig, floor) on the
  race-pools that paid the super_premium rate (``super_premium_subset``) alone.
- E44 (``"E44"``): for each ordering and each input but NORMALISED, params ``ordering``,
  ``input`` (the model) and ``market == NORMALISED``; metrics on ``all_folds``: ``win_gain``
  (per race the market's Win log loss minus the model's), ``joint_gain[<pool>]`` for each pool of
  ``spec["e44_pools"]`` (the market's pool log loss minus the model's, both through the same
  ordering: one fit per fold serving both inputs), each with D7 bounds and n (the same races), and
  ``gain_ratio[<pool>]`` = joint_gain / win_gain, a point estimate (lo and hi None).

``super_premium_subset(frame, pool) -> DataFrame``: the rows of a frame with race_id (and
race_class_raw, which an age-restricted boost needs) whose race-pool paid exactly the
super_premium rate (``takeout.race_takeout(...).boost == "super_premium"``): ultra_premium,
premium and unboosted race-pools are left out; order and index kept.

The run. Seams the tests replace: ``load_store(days) -> (races, runners, dividends, builds)``
and ``load_form(days) -> (form, builds)``, given ``datetime.date`` days; either may be called once
or per chunk. ``run_study(*, date_from, date_to, out_dir, registry=None) -> result``: refuses an
``out_dir`` inside the repository (``LicensedOutputInRepo`` or ``DataDirInsideRepo``) before any
read; freezes ``default_spec()`` under ``FAMILY_ID`` before either seam is called
(``FamilyAlreadyFrozen`` on another spec stops it before any read, with nothing appended); asks
the seams for development days only (``is_holdout_day`` False, in [date_from, date_to]), every
such day; appends one record per variant (ticket 459, status "failed" for a variant with an
error, else "scored"); when ``load_form`` raises, every FORM variant is a failed record and the
rest run. Records of a FORM variant name ``jrdb`` among their sources (licensed, withheld by the
summary); no other record names a withheld source. The private report goes through
``write_report`` under ``out_dir``. ``render_public_report(result) -> str`` names FORM as
withheld and carries no figure built on it. The run writes nothing inside the repository and
never opens the holdout. CLI: ``horsetrack japan exotic-c5``.

Place and Wide's second de-vig (review of PR #668; DoD box 2, outline 7, the reframe on #99 of
2026-11-27): ``PER_TICKET = "per_ticket"``, the chance a paid ticket c pays taken as its own
implied share min(1, R * 100 / (D_c + 5)) (R the race's payout rate, boosts in; D_c its dividend
per 100 yen), beside the statute's min(1, P x_c). It is its own family, ``PER_TICKET_FAMILY_ID``
(not ``FAMILY_ID``, a safe summary stem), because ``FAMILY_ID`` is frozen with ``default_spec()``,
which stays byte-identical (its sha256 is pinned below). ``build_per_ticket_spec(scheme,
bootstrap) -> dict`` (the keys ``build_spec`` has, read by ``compute_study``) and
``default_per_ticket_spec()`` (Japan's scheme, ``bootstrap_settings()``): pools PLACE and
QUINELLA_PLACE only, ``place_wide_devigs == ["per_ticket"]``, both floor treatments, every
ordering, input and blend, no E44 pool; E43 and E16 as in the first family.
``run_study(..., family=PER_TICKET_FAMILY_ID)`` (default ``FAMILY_ID``; an unknown family is a
``ValueError`` before any read) freezes that spec under that family before either seam reads a
row and appends its records there; with the default registry it writes that family's own
committed summary file and index line, never another family's file or the first family's public
report. CLI: ``horsetrack japan exotic-c5 --family <PER_TICKET_FAMILY_ID>``, whose JSON names the
family it ran.

Every table is synthetic (``exotic_c5_world``); expected values come from first principles there
and here (Harville by enumeration, the implied share by formula, alpha by a bounded scalar
minimisation), or from the dependencies the ticket names (``core.eval.bootstrap``,
``core.eval.folds``, ``models.v0.calibration``), never from the module under test.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
import hashlib
import itertools
import json
import math
import re
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from typer.testing import CliRunner

from horsetrack.cli import app
from horsetrack.core import env
from horsetrack.core.config import load_country_yaml
from horsetrack.core.eval import holdout
from horsetrack.core.eval import registry as registry_module
from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.folds import FoldScheme, load_country_scheme, walk_forward
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.leaderboard import rows_from_registry
from horsetrack.core.eval.registry import (
    FamilyAlreadyFrozen,
    Registry,
    RunRecord,
    is_licensed_record,
    write_summary,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.core.paths import DataDirInsideRepo
from horsetrack.countries.japan.models import exotic
from horsetrack.countries.japan.models.v0.calibration import fit_win_calibration
from horsetrack.countries.japan.studies import exotic_c5

# isort: split
from exotic_c5_world import (
    PLACE,
    POOLS,
    SINGLE,
    TRIFECTA,
    TRIO,
    WIDE,
    WIN,
    Meeting,
    World,
    floor_bound,
    implied_share,
    longshots_win,
    make_world,
    member_store,
    normalised,
    pool_probabilities,
    replace_races,
    write_member_store,
)

REPO = env.repo_root()
GUARDED = [
    REPO / "docs" / "ledger.md",
    REPO / "docs" / "experiments",
    REPO / "docs" / "leaderboard",
    *sorted((REPO / "src" / "horsetrack" / "countries").glob("*/holdout_log.jsonl")),
]
FIRST_DAY = date(2014, 6, 7)
LAST_DAY = date(2026, 12, 31)
STORE_BUILDS = {
    "jra_races": "processed/jra/races.parquet@" + "a" * 16,
    "jra_runners": "processed/jra/runners.parquet@" + "b" * 16,
    "jra_dividends": "processed/jra/dividends.parquet@" + "c" * 16,
}
FORM_BUILDS = {"meta_form_oof": "c4/stage2-q-form.parquet@" + "d" * 16}
GRID_KEYS = ("pool", "input", "ordering", "blend", "devig", "floor")
DELTA = "delta_ll_vs_dividends"
NO_BLEND, LINEAR, LOG_LINEAR = "none", "linear", "log_linear"
SUPER_DAYS = (date(2020, 9, 21), date(2020, 12, 13))  # super_premium on every pool, every venue
ULTRA_DAY = date(2024, 9, 16)  # ultra_premium: 80% + 5%, not super_premium
SMALL_SCHEME = FoldScheme(
    unit="quarter",
    test_length=1,
    step=1,
    min_train=2,
    window="expanding",
    embargo_days=7,
    anchor_month=1,
)
SMALL_BOOT = {"cluster_unit": "meeting", "sided": "one-sided", "level": 0.95, "n_boot": 400,
              "seed": 4590}  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------------------------


def _second_saturday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(5 - first.weekday()) % 7 + 7)


def _late_2026_days() -> tuple[list[date], list[date]]:
    """The first weekend days from 2026-11-28 that are development days, and holdout days."""
    days = [date(2026, 11, 28) + timedelta(days=k) for k in range(35)]
    weekend = [d for d in days if d.weekday() >= 5]
    dev = [d for d in weekend if not is_holdout_day("japan", d)][:2]
    held = [d for d in weekend if is_holdout_day("japan", d)][:2]
    assert len(dev) == 2 and len(held) == 2, (dev, held)
    return dev, held


DEV_LATE, HELD = _late_2026_days()
VENUES = ("TOK", "NAK", "KYO", "HSN", "CKY", "NII")
TRAIN_MEETINGS = [
    Meeting(_second_saturday(2014 + (q + 2) // 4, 3 * ((q + 2) % 4) + 2), VENUES[q % len(VENUES)])
    for q in range(22)  # one meeting a quarter, 2014-Q3 to 2019-Q4: training only
]
TEST_MEETINGS = [
    Meeting(date(2020, 9, 12), "NAK", hot=(1,)),
    Meeting(SUPER_DAYS[0], "NAK", hot=(2,)),
    Meeting(date(2020, 12, 5), "HSN"),
    Meeting(SUPER_DAYS[1], "HSN", hot=(3,)),
    Meeting(date(2024, 8, 3), "NII", hot=(1,)),
    Meeting(ULTRA_DAY, "NAK", hot=(2,)),
    Meeting(date(2026, 11, 21), "TOK"),
    *(Meeting(d, "TOK", hot=(1,)) for d in DEV_LATE),
]
HOLDOUT_MEETINGS = [Meeting(d, "TOK", hot=(2,)) for d in HELD]  # in the store, never to be read
RUN_MEETINGS = TRAIN_MEETINGS + TEST_MEETINGS + HOLDOUT_MEETINGS
RUN_SEED = 45901


def _dev_races(world: World) -> pd.DataFrame:
    races = world.races
    held = races["race_date"].map(lambda d: is_holdout_day("japan", d)).astype(bool)
    return races.loc[~held.to_numpy()].reset_index(drop=True)


@dataclass
class FoldInfo:
    train_end: date
    test_start: date
    train_ids: list[str]
    test_ids: list[str]


def fold_info(races: pd.DataFrame, scheme: FoldScheme) -> list[FoldInfo]:
    """The walk-forward folds of the races (one row each), with their race ids."""
    races = races.reset_index(drop=True)
    ids = races["race_id"].to_numpy(dtype=object)
    return [
        FoldInfo(f.train_end, f.test_start, list(ids[f.train_idx]), list(ids[f.test_idx]))
        for f in walk_forward(races["race_date"], scheme=scheme)
    ]


def clusters_of(world: World) -> pd.Series:
    return pd.Series({r: t.meeting_id for r, t in world.truth.items()})


# ---------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------


def model_ll(p: dict[int, float], pool: str, winners: Sequence[str], places: int,
             lambdas=(1.0, 1.0)) -> float:  # fmt: skip
    probs = pool_probabilities(p, lambdas, places=places)[pool]
    return -float(np.mean([math.log(probs[c]) for c in winners]))


def dividend_ll(truth, pool: str) -> float:
    """The single-winner baseline: -ln of the implied share R * 100 / (D + 5)."""
    (combo,) = truth.winners[pool]
    return -math.log(implied_share(truth.dividends[pool][combo], truth.rates[pool]))


def e43_deltas(world: World, race_ids, pool: str, p_of) -> pd.Series:
    """Per race: the dividend baseline's log loss minus the Harville C5 model's."""
    out = {}
    for r in race_ids:
        t = world.truth[r]
        out[r] = dividend_ll(t, pool) - model_ll(p_of(r, t), pool, t.winners[pool], t.places)
    return pd.Series(out, dtype=float)


def bounds(per_race: pd.Series, world: World, boot: dict):
    return paired_bootstrap(
        per_race,
        clusters_of(world),
        n_boot=boot["n_boot"],
        level=boot["level"],
        seed=boot["seed"],
    )


def assert_metric_matches(m, per_race: pd.Series, world: World, boot: dict, *, tol=1e-9):
    iv = bounds(per_race, world, boot)
    assert m.n == len(per_race), (m, len(per_race))
    assert m.value == pytest.approx(per_race.mean(), abs=tol)
    assert m.lo is not None and m.hi is not None
    assert m.lo == pytest.approx(iv.lo, abs=tol)
    assert m.hi == pytest.approx(iv.hi, abs=tol)


def reference_alpha(q: np.ndarray, p: np.ndarray) -> float:
    res = minimize_scalar(
        lambda a: -np.mean(np.log(a * q + (1.0 - a) * p)),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(res.x)


def blend_objective(alpha: float, q: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(np.log(alpha * q + (1.0 - alpha) * p)))


def single_q_p(world: World, race_ids, pool: str) -> tuple[np.ndarray, np.ndarray]:
    """The implied share and the normalised-market Harville probability of each race's winning
    combination in a single-winner pool."""
    q, p = [], []
    for r in race_ids:
        t = world.truth[r]
        (c,) = t.winners[pool]
        q.append(implied_share(t.dividends[pool][c], t.rates[pool]))
        p.append(pool_probabilities(t.p_norm, places=t.places)[pool][c])
    return np.array(q), np.array(p)


def calibrated_p(world: World, folds: list[FoldInfo]) -> dict[str, dict[int, float]]:
    """D4's spline fitted on each fold's training rows, applied to its test races."""
    runners = world.runners
    out = {}
    for f in folds:
        train = runners[runners["race_id"].isin(f.train_ids)]
        rows = pd.DataFrame(
            {
                "race_id": train["race_id"].to_numpy(dtype=object),
                "race_date": train["race_date"].to_numpy(dtype=object),
                "win_odds": train["win_odds"].to_numpy(dtype=float),
                "won": (train["finish_pos"].to_numpy(dtype=float) == 1.0),
            }
        )
        model = fit_win_calibration(rows)
        for r in f.test_ids:
            t = world.truth[r]
            q = np.array([t.p_norm[h] for h in t.horses])
            p = model.predict_many(q, np.array([r] * len(q), dtype=object))
            out[r] = dict(zip(t.horses, (float(x) for x in p), strict=True))
    return out


# ---------------------------------------------------------------------------------------------
# Record and variant lookups
# ---------------------------------------------------------------------------------------------


def select(items, hypothesis: str, **want) -> list:
    found = [
        x
        for x in items
        if x.hypothesis_id == hypothesis and all(x.params.get(k) == v for k, v in want.items())
    ]
    if want.get("pool") == WIN and "ordering" not in want:
        found = [x for x in found if x.params.get("ordering") in (None, "harville")]
    return found


def one(items, hypothesis: str, **want):
    found = select(items, hypothesis, **want)
    assert len(found) == 1, (hypothesis, want, [x.params for x in found])
    return found[0]


def metric(item, name: str, fold: str = "all_folds"):
    found = [m for m in item.metrics if m.metric == name and m.fold == fold]
    assert len(found) == 1, (name, fold, [(m.fold, m.metric) for m in item.metrics])
    return found[0]


def grid_key(params: dict) -> tuple:
    return tuple(params.get(k) for k in GRID_KEYS)


# ---------------------------------------------------------------------------------------------
# The run's fakes and guards
# ---------------------------------------------------------------------------------------------


class FakeStore:
    """The store seam: the world's tables on the days asked for (only those), and every day
    asked for."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.asked: list[date] = []
        self.calls = 0

    def __call__(self, days):
        self.calls += 1
        asked = list(days)
        for day in asked:
            assert isinstance(day, date) and not isinstance(day, datetime), day
        self.asked += asked
        wanted = set(asked)
        races = self.world.races[self.world.races["race_date"].isin(wanted)]
        ids = set(races["race_id"])
        runners = self.world.runners[self.world.runners["race_id"].isin(ids)]
        dividends = self.world.dividends[self.world.dividends["race_id"].isin(ids)]
        return (
            races.reset_index(drop=True),
            runners.reset_index(drop=True),
            dividends.reset_index(drop=True),
            dict(STORE_BUILDS),
        )


class FakeForm:
    """The stage2:q:form seam (#87's member probabilities, JRDB-fed)."""

    def __init__(self, world: World, *, fails: bool = False) -> None:
        self.world = world
        self.fails = fails
        self.asked: list[date] = []
        self.calls = 0

    def __call__(self, days):
        self.calls += 1
        asked = list(days)
        for day in asked:
            assert isinstance(day, date) and not isinstance(day, datetime), day
        self.asked += asked
        if self.fails:
            raise FileNotFoundError("planted: the stage2:q:form member file is missing")
        wanted = set(asked)
        ids = set(self.world.races.loc[self.world.races["race_date"].isin(wanted), "race_id"])
        form = self.world.form[self.world.form["race_id"].isin(ids)]
        return form.reset_index(drop=True), dict(FORM_BUILDS)


def _snapshot() -> dict[Path, bytes]:
    found = {}
    for target in GUARDED:
        if not target.exists():
            continue
        files = [target] if target.is_file() else sorted(target.rglob("*"))
        found.update({f: f.read_bytes() for f in files if f.is_file()})
    return found


def _restore(before: dict[Path, bytes]) -> bool:
    """Put the guarded files back; True if anything had changed."""
    after = _snapshot()
    if after == before:
        return False
    for path in set(after) - set(before):
        path.unlink()
    for path, raw in before.items():
        path.write_bytes(raw)
    return True


@pytest.fixture(autouse=True)
def guarded(tmp_path, monkeypatch):
    """The data directory is under tmp_path; a test fails (and the files are restored) if it
    changed the ledger, the committed experiments or leaderboard, or a holdout log."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path / "data"))
    before = _snapshot()
    yield
    if _restore(before):
        pytest.fail("the test wrote the repository's ledger, docs or a holdout log (restored)")


HOLDOUT_OPENED: list[tuple] = []


def _refuse_holdout(*args, **kwargs):
    HOLDOUT_OPENED.append((args, kwargs))
    raise AssertionError("the holdout was opened")


def _planted_henery_failure(p):
    raise ValueError("planted: the Henery quadrature failed")


@dataclass
class Run:
    world: World
    result: object
    registry: Registry
    records: list[RunRecord]
    out_dir: Path
    summary: Path
    store: FakeStore
    form: FakeForm


def _registry(root: Path) -> Registry:
    return Registry(root / "registry" / "japan.jsonl", holdout_log=root / "holdout_log.jsonl")


def _run(root: Path, world: World, *, form_fails=False, henery_fails=False) -> Run:
    root.mkdir(parents=True, exist_ok=True)
    data, summary, out = root / "data", root / "summary" / "japan", root / "reports"
    data.mkdir()
    summary.mkdir(parents=True)
    store, form = FakeStore(world), FakeForm(world, fails=form_fails)
    registry = _registry(root)
    before = _snapshot()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HORSETRACK_DATA_DIR", str(data))
        mp.setattr(registry_module, "summary_dir", lambda country: summary)
        mp.setattr(exotic_c5, "summary_dir", lambda country: summary, raising=False)
        mp.setattr(holdout.LockedHoldout, "open", _refuse_holdout)
        mp.setattr(exotic_c5, "load_store", store)
        mp.setattr(exotic_c5, "load_form", form)
        if henery_fails:
            mp.setattr(exotic, "_henery_trifecta", _planted_henery_failure)
        try:
            result = exotic_c5.run_study(
                date_from=FIRST_DAY, date_to=LAST_DAY, out_dir=out, registry=registry
            )
        finally:
            changed = _restore(before)
    assert not changed, "the run wrote the repository's ledger, docs or a holdout log"
    return Run(world, result, registry, registry.records(), out, summary, store, form)


@pytest.fixture(scope="module")
def world() -> World:
    return make_world(RUN_MEETINGS, seed=RUN_SEED)


@pytest.fixture(scope="module")
def run_a(world, tmp_path_factory) -> Run:
    return _run(tmp_path_factory.mktemp("run-a"), world)


@pytest.fixture(scope="module")
def run_b(tmp_path_factory) -> Run:
    """The same store as run_a; only the JRDB-fed member's probabilities differ."""
    other = make_world(RUN_MEETINGS, seed=RUN_SEED, form_seed=RUN_SEED + 17)
    return _run(tmp_path_factory.mktemp("run-b"), other)


@pytest.fixture(scope="module")
def run_broken(world, tmp_path_factory) -> Run:
    """The Henery quadrature fails on every race and the member file is missing."""
    return _run(tmp_path_factory.mktemp("run-broken"), world, form_fails=True, henery_fails=True)


@pytest.fixture(scope="module")
def scored(world) -> list[FoldInfo]:
    """The run's folds under the country's scheme, development races only."""
    return fold_info(_dev_races(world), load_country_scheme("japan"))


def _scored_ids(folds: list[FoldInfo]) -> list[str]:
    return [r for f in folds for r in f.test_ids]


# ---------------------------------------------------------------------------------------------
# 1. The spec is frozen before any outcome is read
# ---------------------------------------------------------------------------------------------


class _StopHere(BaseException):  # not an Exception: a run that records failures cannot swallow it
    pass


def test_spec_is_frozen_before_any_outcome_is_read(tmp_path, monkeypatch, world):
    spec = exotic_c5.default_spec()
    json.dumps(spec)  # plain JSON, as the registry requires
    assert spec == exotic_c5.default_spec()
    assert spec["family_id"] == exotic_c5.FAMILY_ID and spec["ticket"] == exotic_c5.TICKET == 459
    assert spec["fold_scheme"] == dataclasses.asdict(load_country_scheme("japan"))
    d7 = load_country_yaml("japan", "scoring")["bootstrap"]
    assert spec["bootstrap"]["level"] == d7["level"] and spec["bootstrap"]["n_boot"] == d7["n_boot"]
    assert isinstance(spec["bootstrap"]["seed"], int)
    for key, values in {
        "pools": exotic_c5.POOLS,
        "orderings": exotic_c5.ORDERINGS,
        "inputs": exotic_c5.INPUTS,
        "blends": exotic_c5.BLENDS,
        "e44_pools": exotic_c5.E44_POOLS,
        "place_wide_devigs": exotic_c5.PLACE_WIDE_DEVIGS,
        "floor_treatments": exotic_c5.FLOOR_TREATMENTS,
    }.items():
        assert spec[key] == list(values), key
    other = exotic_c5.build_spec(SMALL_SCHEME, SMALL_BOOT)
    assert other["fold_scheme"] == dataclasses.asdict(SMALL_SCHEME) and other != spec

    # A fresh registry: the seams see the family frozen with the default spec, and no record.
    seen = []
    registry = _registry(tmp_path)

    def seam(name):
        def call(days):
            seen.append(
                (
                    name,
                    registry.frozen_spec(exotic_c5.FAMILY_ID),
                    list(registry.records(exotic_c5.FAMILY_ID)),
                )
            )
            raise _StopHere

        return call

    monkeypatch.setattr(holdout.LockedHoldout, "open", _refuse_holdout)
    monkeypatch.setattr(exotic_c5, "load_store", seam("store"))
    monkeypatch.setattr(exotic_c5, "load_form", seam("form"))
    with pytest.raises(_StopHere):
        exotic_c5.run_study(
            date_from=FIRST_DAY, date_to=LAST_DAY, out_dir=tmp_path / "out", registry=registry
        )
    assert seen, "neither seam was called"
    for name, frozen, records in seen:
        assert frozen == json.loads(json.dumps(spec)), f"{name} was read before the freeze"
        assert records == [], f"{name} was read after a record was appended"

    # Frozen with another spec: refused before any read, nothing appended or written.
    refused = _registry(tmp_path / "refused")
    refused.freeze_family(exotic_c5.FAMILY_ID, {**spec, "changed_after_the_freeze": True})
    store, form = FakeStore(world), FakeForm(world)
    monkeypatch.setattr(exotic_c5, "load_store", store)
    monkeypatch.setattr(exotic_c5, "load_form", form)
    out = tmp_path / "refused-out"
    with pytest.raises(FamilyAlreadyFrozen):
        exotic_c5.run_study(date_from=FIRST_DAY, date_to=LAST_DAY, out_dir=out, registry=refused)
    assert store.calls == 0 and form.calls == 0
    assert refused.records() == []
    assert not out.exists() or not any(p.is_file() for p in out.rglob("*"))


# ---------------------------------------------------------------------------------------------
# 2. E43: paired log loss per pool against the dividends, with D7 bounds
# ---------------------------------------------------------------------------------------------


def test_paired_log_loss_per_pool_matches_the_dividend_baseline_reference(run_a, scored):
    """Every single-winner pool, the publishable inputs and the JRDB-fed one, through Harville."""
    world, records = run_a.world, run_a.records
    boot = exotic_c5.default_spec()["bootstrap"]
    ids = _scored_ids(scored)
    assert len(ids) == 27 and len(scored) == 4
    cal = calibrated_p(world, scored)
    inputs = {
        exotic_c5.NORMALISED: lambda r, t: t.p_norm,
        exotic_c5.FORM: lambda r, t: normalised(t.p_form),
        exotic_c5.CALIBRATED: lambda r, t: cal[r],
    }
    values = {}
    for pool in SINGLE:
        for name, p_of in inputs.items():
            if name == exotic_c5.CALIBRATED and pool not in (WIN, TRIO):
                continue
            want = {"pool": pool, "input": name, "blend": NO_BLEND}
            if pool != WIN:
                want["ordering"] = "harville"
            rec = one(records, "E43", **want)
            assert rec.status == "scored", (pool, name)
            per_race = e43_deltas(world, ids, pool, p_of)
            tol = 1e-6 if name == exotic_c5.CALIBRATED else 1e-9
            assert_metric_matches(metric(rec, DELTA), per_race, world, boot, tol=tol)
            values[(pool, name)] = per_race.mean()
    # the inputs and the pools are told apart: no constant passes
    assert len({round(v, 9) for v in values.values()}) == len(values)


def test_every_scored_e43_and_e16_record_carries_bounds(run_a):
    for rec in run_a.records:
        if rec.hypothesis_id in ("E43", "E16") and rec.status == "scored":
            m = metric(rec, DELTA)
            assert m.lo is not None and m.hi is not None and m.lo <= m.hi, rec.params
            assert m.n > 0


def test_place_wide_drop_leaves_out_floor_bound_race_pools_and_bound_keeps_them(run_a, scored):
    world, records = run_a.world, run_a.records
    ids = _scored_ids(scored)
    devigs = exotic_c5.default_spec()["place_wide_devigs"]
    assert devigs
    for pool in (PLACE, WIDE):
        flagged = sum(floor_bound(world.truth[r].dividends[pool]) for r in ids)
        assert 0 < flagged < len(ids), (pool, flagged)
        for devig in devigs:
            got = {}
            for floor in ("drop", "bound"):
                want = {"pool": pool, "input": exotic_c5.NORMALISED, "ordering": "harville"}
                want |= {"blend": NO_BLEND, "devig": devig, "floor": floor}
                rec = one(records, "E43", **want)
                assert rec.status == "scored", (pool, devig, floor)
                got[floor] = metric(rec, DELTA)
            assert got["bound"].n == len(ids), (pool, devig)
            assert got["drop"].n == len(ids) - flagged, (pool, devig)


# ---------------------------------------------------------------------------------------------
# 3. The linear blend: alpha fitted per pool on each fold's training races
# ---------------------------------------------------------------------------------------------

BLEND_MEETINGS = [
    Meeting(date(2021 + q // 4, 3 * (q % 4) + 2, 6 + 7 * k), venue, races=4)
    for q in range(6)
    for k, venue in enumerate(("TOK", "KYO"))
]
BLEND_MU = {TRIO: (0.6, -0.5, 0.4, -0.3, 0.5, 0.0), WIN: (-0.4, 0.3, -0.2, 0.5, -0.5, 0.1)}


def _quarter_ratio(mus, *, sigma=1.0):
    """Planted share over the market's Harville probability, lognormal around a mean that moves
    by quarter (8 races a quarter), so each fold's training optimum differs."""

    def ratio(index, pool, rng):
        mu = mus.get(pool, (0.0,) * 6)[min(index // 8, 5)]
        return float(np.exp(rng.normal(mu, sigma)))

    return ratio


def _constant_ratio(values):
    def ratio(index, pool, rng):
        rng.normal()  # keep the stream in step with the other worlds
        return values.get(pool, 1.0)

    return ratio


def _blend_spec(pools=(WIN, TRIO)) -> dict:
    spec = exotic_c5.build_spec(SMALL_SCHEME, SMALL_BOOT)
    spec.update(pools=list(pools), orderings=["harville"], inputs=[exotic_c5.NORMALISED],
                blends=[NO_BLEND, LINEAR], e44_pools=[])  # fmt: skip
    return spec


def _alphas(study, pool) -> dict[str, float]:
    var = one(study.variants, "E43", pool=pool, input=exotic_c5.NORMALISED, blend=LINEAR,
              **({} if pool == WIN else {"ordering": "harville"}))  # fmt: skip
    assert var.error is None, var.error
    return {m.fold: m.value for m in var.metrics if m.metric == "alpha"}


def _blend_variant(study, pool):
    return one(study.variants, "E43", pool=pool, input=exotic_c5.NORMALISED, blend=LINEAR,
               **({} if pool == WIN else {"ordering": "harville"}))  # fmt: skip


def test_blend_alpha_is_fitted_on_training_races_only():
    base = make_world(BLEND_MEETINGS, seed=4591, ratio=_quarter_ratio(BLEND_MU))
    folds = fold_info(base.races, SMALL_SCHEME)
    assert len(folds) == 3
    labels = [f.test_start.isoformat() for f in folds]
    spec = _blend_spec()
    study = exotic_c5.compute_study(*base.store(), spec)

    got = {pool: _alphas(study, pool) for pool in (WIN, TRIO)}
    for pool in (WIN, TRIO):
        assert sorted(got[pool]) == sorted(labels), (pool, got[pool])
        for f, label in zip(folds, labels, strict=True):
            alpha = got[pool][label]
            assert 0.0 <= alpha <= 1.0
            q, p = single_q_p(base, f.train_ids, pool)
            ref = reference_alpha(q, p)
            assert 0.02 < ref < 0.98, (pool, label, ref)  # the planted optimum is interior
            assert alpha == pytest.approx(ref, abs=2e-3), (pool, label)
            assert blend_objective(alpha, q, p) <= blend_objective(ref, q, p) + 1e-9
        # the blend's own score uses each fold's alpha on that fold's test races
        per_race = {}
        for f, label in zip(folds, labels, strict=True):
            a = got[pool][label]
            q, p = single_q_p(base, f.test_ids, pool)
            for r, qi, pi in zip(f.test_ids, q, p, strict=True):
                per_race[r] = dividend_ll(base.truth[r], pool) + math.log(a * qi + (1 - a) * pi)
        assert_metric_matches(
            metric(_blend_variant(study, pool), DELTA),
            pd.Series(per_race, dtype=float),
            base,
            SMALL_BOOT,
        )
    assert len({round(a, 3) for a in got[TRIO].values()}) > 1  # one alpha per fold
    assert any(abs(got[WIN][k] - got[TRIO][k]) > 1e-3 for k in labels)  # one alpha per pool

    # New outcomes and dividends on the last fold's test races: no fold trains on them, so every
    # alpha stays, while the blend's score (which tests them) moves.
    other = make_world(BLEND_MEETINGS, seed=4592, ratio=_quarter_ratio(BLEND_MU))
    late = replace_races(base, other, folds[-1].test_ids)
    late_study = exotic_c5.compute_study(*late.store(), spec)
    for pool in (WIN, TRIO):
        moved = _alphas(late_study, pool)
        for label in labels:
            assert moved[label] == pytest.approx(got[pool][label], abs=1e-9), (pool, label)
        before = metric(_blend_variant(study, pool), DELTA).value
        after = metric(_blend_variant(late_study, pool), DELTA).value
        assert abs(after - before) > 1e-6, pool

    # New outcomes on the first fold's test races: that fold's alpha stays; the later folds,
    # which train on those races, move.
    early = replace_races(base, other, folds[0].test_ids)
    early_study = exotic_c5.compute_study(*early.store(), spec)
    for pool in (WIN, TRIO):
        moved = _alphas(early_study, pool)
        assert moved[labels[0]] == pytest.approx(got[pool][labels[0]], abs=1e-9), pool
        for f, label in zip(folds[1:], labels[1:], strict=True):
            q, p = single_q_p(early, f.train_ids, pool)
            assert moved[label] == pytest.approx(reference_alpha(q, p), abs=2e-3), (pool, label)
        assert any(abs(moved[k] - got[pool][k]) > 1e-3 for k in labels[1:]), pool

    # alpha is held to [0, 1]: a pool whose dividends always beat the model sits at 1, one whose
    # dividends always lose sits at 0.
    edge = make_world(BLEND_MEETINGS, seed=4591, ratio=_constant_ratio({TRIO: 3.0, WIN: 0.4}))
    edge_study = exotic_c5.compute_study(*edge.store(), spec)
    for pool, bound in ((TRIO, 1.0), (WIN, 0.0)):
        for label, alpha in _alphas(edge_study, pool).items():
            assert 0.0 <= alpha <= 1.0
            assert alpha == pytest.approx(bound, abs=1e-4), (pool, label, alpha)


# ---------------------------------------------------------------------------------------------
# 4. E44: one ordering fitted once, serving both inputs
# ---------------------------------------------------------------------------------------------

REAL_ORDERING = exotic.Ordering


@dataclass
class _Fit:
    windows: list[tuple[date, date]]
    params: list[dict[str, float]]


FITS: list[_Fit] = []


def _frac(x: float) -> float:
    return x - math.floor(x)


class PlantedDH(REAL_ORDERING):
    """discounted_harville whose `fit` plants lambdas from the training rows' own p_win (so a
    fit on another input or another fold gives other lambdas) and logs every fit."""

    def fit(self, frame, folds):
        if self.family != "discounted_harville":
            return super().fit(frame, folds)
        params = []
        for fold in folds:
            rows = frame.iloc[np.asarray(fold.train_idx, dtype=int)]
            p = rows["p_win"].to_numpy(dtype=float)
            s = math.fsum(p * p) / len(p)
            params.append({"lambda2": 0.55 + 0.9 * _frac(s * 9173.0),
                           "lambda3": 0.5 + 0.9 * _frac(s * 6151.0)})  # fmt: skip
        fitted = REAL_ORDERING(self.family, self.params)
        fitted.fold_params = params
        FITS.append(_Fit([(f.test_start, f.test_end) for f in folds], params))
        return fitted


def _e44_reference(world: World, fit: _Fit, race_ids, pools) -> dict[str, pd.Series]:
    """Per race: the market's (normalised) log loss minus the model's (form), in Win and in each
    pool through the fit's lambdas of the race's fold, the same lambdas for both."""
    win, joint = {}, {pool: {} for pool in pools}
    for r in race_ids:
        t = world.truth[r]
        k = next(i for i, (lo, hi) in enumerate(fit.windows) if lo <= t.race_date <= hi)
        lam = (fit.params[k]["lambda2"], fit.params[k]["lambda3"])
        form = normalised(t.p_form)
        win[r] = math.log(form[t.order[0]]) - math.log(t.p_norm[t.order[0]])
        for pool in pools:
            joint[pool][r] = model_ll(t.p_norm, pool, t.winners[pool], t.places, lam) - model_ll(
                form, pool, t.winners[pool], t.places, lam
            )
    return {"win": pd.Series(win, dtype=float)} | {
        pool: pd.Series(v, dtype=float) for pool, v in joint.items()
    }


def _matches(var, ref: dict[str, pd.Series], world: World, boot: dict, pools) -> bool:
    try:
        assert_metric_matches(metric(var, "win_gain"), ref["win"], world, boot)
        for pool in pools:
            assert_metric_matches(metric(var, f"joint_gain[{pool}]"), ref[pool], world, boot)
    except AssertionError:
        return False
    return True


def test_e44_uses_one_ordering_for_both_inputs(monkeypatch):
    world = make_world(BLEND_MEETINGS, seed=4593)
    folds = fold_info(world.races, SMALL_SCHEME)
    pools = [TRIO, TRIFECTA]
    spec = exotic_c5.build_spec(SMALL_SCHEME, SMALL_BOOT)
    spec.update(pools=pools, orderings=["discounted_harville"],
                inputs=[exotic_c5.NORMALISED, exotic_c5.FORM], blends=[NO_BLEND],
                e44_pools=pools)  # fmt: skip
    FITS.clear()
    monkeypatch.setattr(exotic, "Ordering", PlantedDH)
    monkeypatch.setattr(exotic_c5, "Ordering", PlantedDH, raising=False)

    study = exotic_c5.compute_study(*world.store(), spec, form=world.form)

    var = one(study.variants, "E44", ordering="discounted_harville", input=exotic_c5.FORM)
    assert var.error is None, var.error
    assert var.params.get("market") == exotic_c5.NORMALISED
    assert FITS, "the ordering was never fitted through Ordering.fit"
    ids = _scored_ids(folds)
    candidates = [f for f in FITS if len(f.windows) == len(folds)]
    served = [f for f in candidates if _matches(var, _e44_reference(world, f, ids, pools),
                                                world, SMALL_BOOT, pools)]  # fmt: skip
    assert served, "E44 is not one fit's lambdas applied to both inputs, fold by fold"
    fit = served[0]
    assert len({tuple(p.values()) for p in fit.params}) == len(folds)  # per fold, not one set
    win = metric(var, "win_gain")
    for pool in pools:
        joint = metric(var, f"joint_gain[{pool}]")
        ratio = metric(var, f"gain_ratio[{pool}]")
        assert joint.n == win.n == len(ids)
        assert ratio.value == pytest.approx(joint.value / win.value, rel=1e-12)
        assert ratio.lo is None and ratio.hi is None  # a point estimate (owner answer 1(b))
    # Each input's own fit plants other lambdas (they are a function of the rows' p_win), so a
    # run that fitted the ordering on each input in turn matches no single fit above.


def test_e44_through_harville_matches_the_reference_on_the_run(run_a, scored):
    world, records = run_a.world, run_a.records
    boot = exotic_c5.default_spec()["bootstrap"]
    ids = _scored_ids(scored)
    rec = one(records, "E44", ordering="harville", input=exotic_c5.FORM)
    assert rec.status == "scored" and rec.params.get("market") == exotic_c5.NORMALISED
    harville = _Fit([(date(2000, 1, 1), date(2100, 1, 1))], [{"lambda2": 1.0, "lambda3": 1.0}])
    ref = _e44_reference(world, harville, ids, list(exotic_c5.E44_POOLS))
    assert _matches(rec, ref, world, boot, exotic_c5.E44_POOLS)
    win = metric(rec, "win_gain")
    for pool in exotic_c5.E44_POOLS:
        ratio = metric(rec, f"gain_ratio[{pool}]")
        assert ratio.value == pytest.approx(metric(rec, f"joint_gain[{pool}]").value / win.value)
        assert ratio.lo is None and ratio.hi is None


# ---------------------------------------------------------------------------------------------
# 5. E16: the E43 table on super_premium race-pools only
# ---------------------------------------------------------------------------------------------

BOOSTED = [  # race_id, race_class_raw, pools that paid exactly super_premium (hand-labelled from
    # config/takeout_boosts.yaml)
    ("jra:2020-09-21:NAK:11", "3歳以上1勝クラス", set(POOLS)),
    ("jra:2020-12-13:HSN:03", "3歳以上2勝クラス", set(POOLS)),
    ("jra:2024-09-16:NAK:02", "3歳以上1勝クラス", set()),  # ultra_premium, every pool
    ("jra:2024-11-24:TOK:11", "3歳以上オープン", set()),  # the GI race: ultra_premium
    ("jra:2024-11-24:TOK:10", "3歳以上2勝クラス", set(POOLS)),  # same day, super_premium
    ("jra:2022-12-28:NAK:05", "3歳以上1勝クラス", set(POOLS) - {WIN, WIDE}),  # premium stacks
    ("jra:2019-12-28:NAK:04", "2歳未勝利", set(POOLS) - {WIN}),  # the 2-year-old Win premium
    ("jra:2019-12-28:NAK:09", "3歳以上2勝クラス", set(POOLS)),  # not 2-year-olds: super only
    ("jra:2015-05-30:TOK:03", "3歳以上1勝クラス", set()),  # premium on QUINELLA only
    ("jra:2020-09-12:NAK:01", "3歳以上1勝クラス", set()),  # no boost
]


def test_super_premium_subset_keeps_super_premium_races_only(run_a, scored):
    rows = [
        {"race_id": race_id, "race_class_raw": raw, "horse_no": h, "tag": f"{race_id}#{h}"}
        for race_id, raw, _ in BOOSTED
        for h in (1, 2)
    ]
    frame = pd.DataFrame(rows, index=pd.RangeIndex(100, 100 + len(rows)))
    paid_super = {race_id: pools for race_id, _, pools in BOOSTED}
    for pool in POOLS:
        kept = exotic_c5.super_premium_subset(frame, pool)
        mask = np.array([pool in paid_super[r] for r in frame["race_id"]], dtype=bool)
        want = frame.loc[mask]
        assert 0 < len(want) < len(frame), pool
        assert list(kept["tag"]) == list(want["tag"]), pool
        assert list(kept.index) == list(want.index), pool
        assert list(kept.columns) == list(frame.columns)

    # E16 on the run: the E43 deltas of the super_premium race-pools alone (the ultra_premium day
    # and every unboosted day left out).
    world, records = run_a.world, run_a.records
    boot = exotic_c5.default_spec()["bootstrap"]
    ids = _scored_ids(scored)
    supers = [r for r in ids if world.truth[r].race_date in SUPER_DAYS]
    assert len(supers) == 6 and any(world.truth[r].race_date == ULTRA_DAY for r in ids)
    for pool in SINGLE:
        for name, p_of in {
            exotic_c5.NORMALISED: lambda r, t: t.p_norm,
            exotic_c5.FORM: lambda r, t: normalised(t.p_form),
        }.items():
            want = {"pool": pool, "input": name, "blend": NO_BLEND}
            if pool != WIN:
                want["ordering"] = "harville"
            rec = one(records, "E16", **want)
            assert rec.status == "scored", want
            per_race = e43_deltas(world, supers, pool, p_of)
            assert_metric_matches(metric(rec, DELTA), per_race, world, boot)
            full = metric(one(records, "E43", **want), DELTA)
            assert full.n == len(ids) > metric(rec, DELTA).n


# ---------------------------------------------------------------------------------------------
# 6. Every variant is a registry record, failures included
# ---------------------------------------------------------------------------------------------


def _expected_e43_keys(spec: dict) -> set[tuple]:
    keys = set()
    for pool in spec["pools"]:
        for name in spec["inputs"]:
            orderings = [None] if pool == WIN else spec["orderings"]
            for ordering in orderings:
                for blend in (NO_BLEND, LINEAR):
                    if pool in (PLACE, WIDE):
                        for devig in spec["place_wide_devigs"]:
                            for floor in spec["floor_treatments"]:
                                keys.add((pool, name, ordering, blend, devig, floor))
                    else:
                        keys.add((pool, name, ordering, blend, None, None))
    return keys


def _keys_of(records, hypothesis: str) -> set[tuple]:
    out = set()
    for r in records:
        if r.hypothesis_id == hypothesis and r.params.get("blend") != LOG_LINEAR:
            key = grid_key(r.params)
            if key[0] == WIN:
                key = (key[0], key[1], None, *key[3:])  # WIN: the ordering may be recorded or not
            out.add(key)
    return out


def _expected_failed(r: RunRecord) -> bool:
    p = r.params
    if p.get("blend") == LOG_LINEAR or p.get("input") == exotic_c5.FORM:
        return True
    return p.get("ordering") == "henery" and p.get("pool") != WIN


def _assert_complete(run: Run) -> None:
    spec = exotic_c5.default_spec()
    records = run.records
    assert records, "no record was appended"
    assert {r.family_id for r in records} == {exotic_c5.FAMILY_ID}
    assert {r.ticket for r in records} == {459}
    assert {r.hypothesis_id for r in records} == {"E43", "E16", "E44"}
    assert run.registry.frozen_spec(exotic_c5.FAMILY_ID) == json.loads(json.dumps(spec))
    assert run.registry.family_size(exotic_c5.FAMILY_ID) == len(records)  # one record a variant
    for r in records:
        json.dumps(r.params)
    want = _expected_e43_keys(spec)
    assert want <= _keys_of(records, "E43"), sorted(want - _keys_of(records, "E43"), key=str)[:5]
    assert want <= _keys_of(records, "E16"), sorted(want - _keys_of(records, "E16"), key=str)[:5]
    e44 = {(r.params.get("ordering"), r.params.get("input")) for r in records
           if r.hypothesis_id == "E44"}  # fmt: skip
    for ordering in spec["orderings"]:
        for name in (exotic_c5.CALIBRATED, exotic_c5.FORM):
            assert (ordering, name) in e44, (ordering, name)
    log_linear = [r for r in records if r.params.get("blend") == LOG_LINEAR]
    assert log_linear, "pool_blend's log-linear form is not recorded"
    for r in log_linear:
        assert r.status == "failed", r.params  # not runnable yet: no full final boards
        assert not any(m.metric == DELTA for m in r.metrics)


def test_every_variant_appends_a_registry_record(run_a, run_broken):
    _assert_complete(run_a)
    for r in run_a.records:
        if r.params.get("blend") != LOG_LINEAR:
            assert r.status == "scored", (r.hypothesis_id, r.params)
        if r.status == "scored" and r.hypothesis_id == "E44":
            for pool in exotic_c5.E44_POOLS:
                for name in ("win_gain", f"joint_gain[{pool}]", f"gain_ratio[{pool}]"):
                    metric(r, name)

    # A failing family and a missing member file: the grid is still complete, each of those
    # variants a failed record, every other variant scored.
    _assert_complete(run_broken)
    failed = 0
    for r in run_broken.records:
        if r.params.get("pool") == WIN and r.params.get("ordering") == "henery":
            if r.params.get("input") == exotic_c5.FORM:
                assert r.status == "failed"
            continue
        want = "failed" if _expected_failed(r) else "scored"
        assert r.status == want, (r.hypothesis_id, r.params, r.status)
        failed += r.status == "failed"
    assert failed > len(run_broken.records) // 3
    # the freeze line comes before every record line
    lines = [json.loads(x) for x in run_broken.registry.path.read_text().splitlines()]
    kinds = [x["kind"] for x in lines]
    assert kinds[0] == "freeze" and kinds.count("freeze") == 1


# ---------------------------------------------------------------------------------------------
# 7. No holdout day is read; JRDB-fed results stay out of the repository (D21)
# ---------------------------------------------------------------------------------------------


def test_no_holdout_day_reaches_the_seams_or_the_scores(run_a, scored):
    world = run_a.world
    held_races = [r for r, t in world.truth.items() if is_holdout_day("japan", t.race_date)]
    assert len(held_races) == 6, "the store holds races on holdout days"
    dev_days = {t.race_date for t in world.truth.values()} - set(HELD)
    for seam in (run_a.store, run_a.form):
        asked = set(seam.asked)
        assert asked, "the seam was never called"
        assert not any(is_holdout_day("japan", d) for d in asked), "a holdout day was asked for"
        assert all(FIRST_DAY <= d <= LAST_DAY for d in asked), "a day outside the span"
        assert dev_days <= asked, "a development day with races was not read"
    assert HOLDOUT_OPENED == []
    rec = one(run_a.records, "E43", pool=TRIO, input=exotic_c5.NORMALISED, ordering="harville",
              blend=NO_BLEND)  # fmt: skip
    assert metric(rec, DELTA).n == len(_scored_ids(scored)) == 27


def test_jrdb_fed_results_are_written_outside_the_tree(run_a, run_b, tmp_path, monkeypatch, world):
    form_records = [r for r in run_a.records if r.params.get("input") == exotic_c5.FORM]
    public_records = [r for r in run_a.records if r.params.get("input") != exotic_c5.FORM]
    assert form_records and public_records
    for r in form_records:
        assert "jrdb" in r.sources and is_licensed_record(r), r.params
    for r in public_records:
        assert not is_licensed_record(r), (r.params, r.sources, r.data_builds)

    # Only the JRDB-fed member differs between the two runs: its figures move, nothing public does.
    def identity(r):
        return (r.hypothesis_id, json.dumps(r.params, sort_keys=True), r.status)

    def public(run):
        return sorted(
            (identity(r), tuple(dataclasses.astuple(m) for m in r.metrics))
            for r in run.records
            if not is_licensed_record(r)
        )

    assert public(run_a) == public(run_b)
    form_a = {identity(r): r.metrics for r in form_records}
    form_b = {identity(r): r.metrics for r in run_b.records if is_licensed_record(r)}
    assert set(form_a) == set(form_b)
    assert any(form_a[k] != form_b[k] for k in form_a), "the member change moved no figure"

    # The committed aggregate (the registry summary) and the public report do not change with it.
    texts = []
    for run in (run_a, run_b):
        out = tmp_path / f"summary-{len(texts)}"
        write_summary(run.registry, out)
        texts.append(
            {p.name: p.read_text(encoding="utf-8") for p in sorted(out.glob("*.md"))}
            | {
                f"own/{p.name}": p.read_text(encoding="utf-8")
                for p in sorted(run.summary.rglob("*"))
                if p.is_file()
            }
        )
    assert texts[0] == texts[1]
    page = texts[0][f"{exotic_c5.FAMILY_ID}.md"]
    assert registry_module.WITHHELD_LABEL in page
    report_a = exotic_c5.render_public_report(run_a.result)
    report_b = exotic_c5.render_public_report(run_b.result)
    assert isinstance(report_a, str) and report_a == report_b
    assert "withheld" in report_a.lower() and exotic_c5.FORM in report_a

    # The private report holds the member's results, outside the repository.
    files = [p for p in run_a.out_dir.rglob("*") if p.is_file()]
    assert files, "no private report"
    assert not any(p.resolve().is_relative_to(REPO.resolve()) for p in files)
    assert any(exotic_c5.FORM in p.read_text(encoding="utf-8", errors="ignore") for p in files)

    # An out_dir inside the repository is refused before anything is read.
    inside = REPO / f".exotic-c5-test-{uuid.uuid4().hex[:8]}"
    store, form = FakeStore(world), FakeForm(world)
    registry = _registry(tmp_path / "inside")
    monkeypatch.setattr(exotic_c5, "load_store", store)
    monkeypatch.setattr(exotic_c5, "load_form", form)
    monkeypatch.setattr(holdout.LockedHoldout, "open", _refuse_holdout)
    try:
        with pytest.raises((LicensedOutputInRepo, DataDirInsideRepo)):
            exotic_c5.run_study(
                date_from=FIRST_DAY, date_to=LAST_DAY, out_dir=inside, registry=registry
            )
        assert store.calls == 0 and form.calls == 0
        assert registry.records() == []
        assert not inside.exists(), "a file was created inside the repository"
    finally:
        if inside.exists():
            shutil.rmtree(inside)


# ---------------------------------------------------------------------------------------------
# 8. The command
# ---------------------------------------------------------------------------------------------


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")  # as tests/test_cli.py


def _help(*command: str):
    """`--help` of a command and its text without ANSI escapes: typer forces colour under CI
    (GITHUB_ACTIONS, FORCE_COLOR), and Rich styles `-` and `-family` apart; a wide console keeps
    Rich from wrapping or eliding an option name at the runner's width."""
    result = CliRunner().invoke(app, [*command, "--help"], env={"COLUMNS": "200", "NO_COLOR": "1"})
    return result, _ANSI.sub("", result.output)


def test_the_japan_command_is_registered():
    result = CliRunner().invoke(app, ["japan", "exotic-c5", "--help"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------------------------
# 9. Place and Wide's per-ticket de-vig: a second family of its own (review of PR #668)
# ---------------------------------------------------------------------------------------------

# The first family's frozen spec: docs/specs/exotic-c5-prereg.md and docs/experiments/
# exotic-c5-japan.md name it, and the private registry holds it. A changed default_spec() would
# make every rerun of the first family a FamilyAlreadyFrozen.
FIRST_SPEC_SHA256 = "cdaa71d90e9888d9e9498d3e91dbd4c99a37c0b64d73ff658f6f6758dfbef76e"
PER_TICKET = "per_ticket"
OTHER_FAMILY = "japan-stand-in-family"  # another ticket's committed family
SUMMARY = "docs/experiments/registry/japan"
INDEX_LINE = re.compile(r"- \[([^\]]+)\]\(([^)]+)\.md\)")


def statute_chances(dividends: dict[str, float], rate: float, *, cap: bool = True) -> dict:
    """P x_c per paid combination (04 §1.2): a ticket on c pays (W_c + L / P) R / W_c per yen, so
    with r_c = (D_c + 5) / (100 R), W_c / L = 1 / (P (r_c - 1)) and x_c = (W_c / L) / (1 + sum
    W / L); capped at 1."""
    n = len(dividends)
    on_vs_lost = {c: 1.0 / (n * ((d + 5.0) / (100.0 * rate) - 1.0)) for c, d in dividends.items()}
    total = math.fsum(on_vs_lost.values())
    out = {c: n * w / (1.0 + total) for c, w in on_vs_lost.items()}
    return {c: min(1.0, v) for c, v in out.items()} if cap else out


def per_ticket_chances(dividends: dict[str, float], rate: float) -> dict:
    """Each paid combination's own implied share, R * 100 / (D_c + 5), capped at 1."""
    return {c: min(1.0, rate * 100.0 / (d + 5.0)) for c, d in dividends.items()}


CHANCES = {"statute": statute_chances, PER_TICKET: per_ticket_chances}


def multi_base_ll(t, pool: str, devig: str) -> float:
    chance = CHANCES[devig](t.dividends[pool], t.rates[pool])
    return -float(np.mean([math.log(chance[c]) for c in t.winners[pool]]))


def multi_deltas(world: World, race_ids, pool: str, p_of, devig: str) -> pd.Series:
    """Per race: the de-vig baseline's log loss minus the Harville C5 model's, at 1/k over the
    race-pool's k paid combinations."""
    out = {}
    for r in race_ids:
        t = world.truth[r]
        out[r] = multi_base_ll(t, pool, devig) - model_ll(
            p_of(r, t), pool, t.winners[pool], t.places
        )
    return pd.Series(out, dtype=float)


def multi_q_p(world: World, race_ids, pool: str, devig: str) -> list[tuple]:
    """Per race: the de-vig's chance and the normalised market's Harville probability of each
    paid combination."""
    out = []
    for r in race_ids:
        t = world.truth[r]
        chance = CHANCES[devig](t.dividends[pool], t.rates[pool])
        probs = pool_probabilities(t.p_norm, places=t.places)[pool]
        out.append(
            (
                np.array([chance[c] for c in t.winners[pool]]),
                np.array([probs[c] for c in t.winners[pool]]),
            )
        )
    return out


def multi_blend_loss(alpha: float, races: list[tuple]) -> float:
    return float(np.mean([-np.mean(np.log(alpha * q + (1.0 - alpha) * p)) for q, p in races]))


def multi_reference_alpha(races: list[tuple]) -> float:
    res = minimize_scalar(lambda a: multi_blend_loss(a, races), bounds=(0.0, 1.0),
                          method="bounded", options={"xatol": 1e-12})  # fmt: skip
    return float(res.x)


def top_pick_returns(t, pool: str) -> list[float]:
    """The return of ¥100 on the normalised market's most probable combination (Harville), at
    its paid dividend (0 when it lost): one value, or one per combination tied with it within
    rounding (two runners at the same odds), since a tie-break on floats is not pinned here."""
    probs = pool_probabilities(t.p_norm, places=t.places)[pool]
    best = max(probs.values())
    return sorted({t.dividends[pool].get(c, 0.0) for c, v in probs.items() if v > best - 1e-9})


def assert_return_matches(row, options: dict[str, list[float]], world: World, boot: dict):
    """The row's mean, bounds and n are those of one reading of the tied picks' returns."""
    assert row.n == len(options)
    assert sum(len(v) > 1 for v in options.values()) <= 2, "too many tied top picks"
    readings = [
        pd.Series(dict(zip(options, values, strict=True)), dtype=float)
        for values in itertools.product(*options.values())
    ]
    for per_race in readings:
        if abs(per_race.mean() - row.return_per_100) > 1e-9:
            continue
        iv = bounds(per_race, world, boot)
        if abs(iv.lo - row.ci_lo) <= 1e-9 and abs(iv.hi - row.ci_hi) <= 1e-9:
            return
    pytest.fail(f"{row.option_id}: {row.return_per_100} [{row.ci_lo}, {row.ci_hi}] matches no "
                f"reading of {[round(s.mean(), 6) for s in readings]}")  # fmt: skip


def _stand_in_repo(repo: Path) -> None:
    """The committed tree a per-ticket run meets: the first family's summary file and public
    report, another family's file, and the index of both."""
    summary = repo / SUMMARY
    summary.mkdir(parents=True)
    first = exotic_c5.FAMILY_ID
    (summary / f"{first}.md").write_bytes(b"# stand-in: the first family's committed summary\n")
    (summary / f"{OTHER_FAMILY}.md").write_bytes(b"# stand-in: another ticket's family\n")
    index = [
        "# Experiment registry index",
        "",
        "Written by `horsetrack.core.eval.registry.write_summary`; do not edit by hand. One file",
        "per family, each holding that family's runs.",
        "",
        f"- [{first}]({first}.md)",
        f"- [{OTHER_FAMILY}]({OTHER_FAMILY}.md)",
    ]
    (summary / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    report = repo / exotic_c5.PUBLIC_REPORT
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(b"# stand-in: the first family's public report\n")


def _files(root: Path) -> dict[str, bytes]:
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in files}


@dataclass
class TicketRun:
    world: World
    cli: object
    registry: Registry
    records: list[RunRecord]  # the per-ticket family's
    planted: RunRecord  # the first family's record the registry held before the run
    repo: Path
    stand_in: dict[str, bytes]  # the committed tree before the run
    committed: dict[str, bytes]  # and after it
    out_dir: Path
    store: FakeStore
    form: FakeForm


def _ticket_run(root: Path, world: World, planted: RunRecord) -> TicketRun:
    """`horsetrack japan exotic-c5 --family <per-ticket>` with the default registry (the first
    family already frozen there with its spec and holding a record) and the repository root a
    stand-in under tmp, so the committed outputs are written where the test reads them."""
    family = exotic_c5.PER_TICKET_FAMILY_ID
    root.mkdir(parents=True, exist_ok=True)
    data, repo, out = root / "data", root / "repo", root / "reports"
    data.mkdir()
    _stand_in_repo(repo)
    stand_in = _files(repo)
    store, form = FakeStore(world), FakeForm(world)

    def folder(country):
        return repo / "docs" / "experiments" / "registry" / country

    before = _snapshot()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HORSETRACK_DATA_DIR", str(data))
        mp.setattr(env, "repo_root", lambda: repo)
        mp.setattr(registry_module, "summary_dir", folder)
        mp.setattr(exotic_c5, "summary_dir", folder, raising=False)
        mp.setattr(holdout.LockedHoldout, "open", _refuse_holdout)
        mp.setattr(exotic_c5, "load_store", store)
        mp.setattr(exotic_c5, "load_form", form)
        try:
            registry = registry_module.default_registry("japan")
            registry.freeze_family(exotic_c5.FAMILY_ID, exotic_c5.default_spec())
            registry.append(planted)
            args = ["japan", "exotic-c5", "--family", family, "--from", FIRST_DAY.isoformat(),
                    "--to", LAST_DAY.isoformat(), "--out", str(out)]  # fmt: skip
            cli = CliRunner().invoke(app, args)
        finally:
            changed = _restore(before)
    assert not changed, "the run wrote the repository's ledger, docs or a holdout log"
    assert cli.exit_code == 0, (cli.output, cli.exception)
    return TicketRun(world, cli, registry, registry.records(family), planted, repo, stand_in,
                     _files(repo), out, store, form)  # fmt: skip


def _planted(run: Run) -> RunRecord:
    return next(r for r in run.records if r.status == "scored" and not is_licensed_record(r))


@pytest.fixture(scope="module")
def pt_a(world, run_a, tmp_path_factory) -> TicketRun:
    return _ticket_run(tmp_path_factory.mktemp("per-ticket-a"), world, _planted(run_a))


@pytest.fixture(scope="module")
def pt_b(run_a, tmp_path_factory) -> TicketRun:
    """pt_a's store; only the JRDB-fed member's probabilities differ."""
    other = make_world(RUN_MEETINGS, seed=RUN_SEED, form_seed=RUN_SEED + 17)
    return _ticket_run(tmp_path_factory.mktemp("per-ticket-b"), other, _planted(run_a))


def test_per_ticket_family_has_its_own_spec_and_the_first_stays_frozen(tmp_path):
    family = exotic_c5.PER_TICKET_FAMILY_ID
    assert exotic_c5.PER_TICKET == PER_TICKET
    assert isinstance(family, str) and registry_module.summary_stem(family) == family
    assert family.casefold() != exotic_c5.FAMILY_ID.casefold()

    spec = exotic_c5.default_per_ticket_spec()
    json.dumps(spec)  # plain JSON, as the registry requires
    assert spec == exotic_c5.default_per_ticket_spec()
    scheme = load_country_scheme("japan")
    assert spec == exotic_c5.build_per_ticket_spec(scheme, exotic_c5.bootstrap_settings())
    assert spec["family_id"] == family and spec["ticket"] == 459
    assert spec["fold_scheme"] == dataclasses.asdict(scheme)
    d7 = load_country_yaml("japan", "scoring")["bootstrap"]
    assert spec["bootstrap"]["level"] == d7["level"] and spec["bootstrap"]["n_boot"] == d7["n_boot"]
    assert isinstance(spec["bootstrap"]["seed"], int)
    assert sorted(spec["pools"]) == sorted([PLACE, WIDE])  # Place and Wide only
    for key, values in {
        "place_wide_devigs": [PER_TICKET],
        "floor_treatments": ["drop", "bound"],
        "orderings": list(exotic_c5.ORDERINGS),
        "inputs": list(exotic_c5.INPUTS),
        "blends": list(exotic_c5.BLENDS),
        "e44_pools": [],  # E44 is Trio and Trifecta: the first family's
    }.items():
        assert spec[key] == values, key
    small = exotic_c5.build_per_ticket_spec(SMALL_SCHEME, SMALL_BOOT)
    assert small["fold_scheme"] == dataclasses.asdict(SMALL_SCHEME)
    assert small["bootstrap"] == SMALL_BOOT and small != spec

    # The first family's spec is the one frozen: byte-identical, so its rerun freezes as a no-op;
    # the second family freezes beside it in the same registry.
    first = exotic_c5.default_spec()
    assert first["place_wide_devigs"] == ["statute"]
    registry = _registry(tmp_path)
    assert registry.freeze_family(exotic_c5.FAMILY_ID, first) == FIRST_SPEC_SHA256
    second = registry.freeze_family(family, spec)
    assert second != FIRST_SPEC_SHA256
    rerun = registry.freeze_family(exotic_c5.FAMILY_ID, exotic_c5.default_spec())  # no raise
    assert rerun == FIRST_SPEC_SHA256
    assert registry.frozen_spec_hash(family) == second
    assert registry.frozen_spec(family) == json.loads(json.dumps(spec))


def test_per_ticket_spec_is_frozen_before_any_outcome_is_read(tmp_path, monkeypatch, world):
    family = exotic_c5.PER_TICKET_FAMILY_ID
    spec = exotic_c5.default_per_ticket_spec()
    registry = _registry(tmp_path)
    # The first family is frozen here with another spec: the per-ticket run neither reads nor
    # refreezes it.
    registry.freeze_family(exotic_c5.FAMILY_ID, {**exotic_c5.default_spec(), "planted": True})
    first = registry.frozen_spec_hash(exotic_c5.FAMILY_ID)
    seen = []

    def seam(name):
        def call(days):
            seen.append((name, registry.frozen_spec(family), list(registry.records(family))))
            raise _StopHere

        return call

    monkeypatch.setattr(holdout.LockedHoldout, "open", _refuse_holdout)
    monkeypatch.setattr(exotic_c5, "load_store", seam("store"))
    monkeypatch.setattr(exotic_c5, "load_form", seam("form"))
    with pytest.raises(_StopHere):
        exotic_c5.run_study(date_from=FIRST_DAY, date_to=LAST_DAY, out_dir=tmp_path / "out",
                            registry=registry, family=family)  # fmt: skip
    assert seen, "neither seam was called"
    for name, frozen, records in seen:
        assert frozen == json.loads(json.dumps(spec)), f"{name} was read before the freeze"
        assert records == [], f"{name} was read after a record was appended"
    assert registry.frozen_spec_hash(exotic_c5.FAMILY_ID) == first

    # Frozen with another spec, or an unknown family: refused before any read, nothing appended.
    refused = _registry(tmp_path / "refused")
    refused.freeze_family(family, {**spec, "changed_after_the_freeze": True})
    store, form = FakeStore(world), FakeForm(world)
    monkeypatch.setattr(exotic_c5, "load_store", store)
    monkeypatch.setattr(exotic_c5, "load_form", form)
    out = tmp_path / "refused-out"
    with pytest.raises(FamilyAlreadyFrozen):
        exotic_c5.run_study(date_from=FIRST_DAY, date_to=LAST_DAY, out_dir=out, registry=refused,
                            family=family)  # fmt: skip
    unknown = _registry(tmp_path / "unknown")
    with pytest.raises(ValueError):
        exotic_c5.run_study(date_from=FIRST_DAY, date_to=LAST_DAY, out_dir=out, registry=unknown,
                            family="japan-c5-no-such-family")  # fmt: skip
    assert store.calls == 0 and form.calls == 0
    assert refused.records() == [] and unknown.records() == []
    assert unknown.frozen_spec("japan-c5-no-such-family") is None
    assert not out.exists() or not any(p.is_file() for p in out.rglob("*"))


def test_place_wide_baselines_match_the_first_principles_reference(run_a, pt_a, scored):
    """Both de-vigs (the first family's statute, the second's per-ticket) against the reference:
    per paid combination min(1, P x_c) or min(1, R * 100 / (D_c + 5)), scored at 1/k; Place's
    places by the field at sale; 'drop' leaves out the floor-bound race-pools."""
    world = pt_a.world
    ids = _scored_ids(scored)
    supers = [r for r in ids if world.truth[r].race_date in SUPER_DAYS]
    capped = 0
    for r in ids:
        t = world.truth[r]
        finish = {h: k + 1 for k, h in enumerate(t.order)}
        for pool in (PLACE, WIDE):
            paid = exotic.winning_combinations(finish, pool, field_at_sale=len(t.horses))
            assert sorted(paid) == sorted(t.winners[pool]) == sorted(t.dividends[pool]), (r, pool)
            raw = statute_chances(t.dividends[pool], t.rates[pool], cap=False)
            capped += any(v > 1.0 for v in raw.values())
    assert capped > 0, "the statute's cap at 1 binds on no race-pool"
    assert len({len(world.truth[r].winners[PLACE]) for r in ids}) == 2  # two and three places

    inputs = {
        exotic_c5.NORMALISED: lambda r, t: t.p_norm,
        exotic_c5.FORM: lambda r, t: normalised(t.p_form),
    }
    families = {
        "statute": (run_a.records, exotic_c5.default_spec()["bootstrap"]),
        PER_TICKET: (pt_a.records, exotic_c5.default_per_ticket_spec()["bootstrap"]),
    }
    values = {}
    for devig, (records, boot) in families.items():
        for pool in (PLACE, WIDE):
            kept = {
                "bound": ids,
                "drop": [r for r in ids if not floor_bound(world.truth[r].dividends[pool])],
            }
            for name, p_of in inputs.items():
                per_race = multi_deltas(world, ids, pool, p_of, devig)
                for floor, races in kept.items():
                    want = {"pool": pool, "input": name, "ordering": "harville"}
                    want |= {"blend": NO_BLEND, "devig": devig, "floor": floor}
                    rec = one(records, "E43", **want)
                    assert rec.status == "scored", want
                    assert_metric_matches(metric(rec, DELTA), per_race[races], world, boot)
                    values[(devig, pool, name, floor)] = per_race[races].mean()
                    sp = one(records, "E16", **want)
                    sp_races = [r for r in races if r in supers]
                    assert_metric_matches(metric(sp, DELTA), per_race[sp_races], world, boot)
    # de-vig, pool, input and floor each move the figure: no constant passes
    assert len({round(v, 9) for v in values.values()}) == len(values)

    # The per-ticket linear blend: alpha per fold minimises the training races' log loss of
    # alpha q + (1 - alpha) P_C5, q the per-ticket chance; each fold's alpha on its test races.
    boot = families[PER_TICKET][1]
    labels = [f.test_start.isoformat() for f in scored]
    for pool in (PLACE, WIDE):
        rec = one(pt_a.records, "E43", pool=pool, input=exotic_c5.NORMALISED, ordering="harville",
                  blend=LINEAR, devig=PER_TICKET, floor="bound")  # fmt: skip
        assert rec.status == "scored", pool
        alphas = {m.fold: m.value for m in rec.metrics if m.metric == "alpha"}
        assert sorted(alphas) == sorted(labels), (pool, alphas)
        per_race = {}
        for f, label in zip(scored, labels, strict=True):
            a = alphas[label]
            train = multi_q_p(world, f.train_ids, pool, PER_TICKET)
            ref = multi_reference_alpha(train)
            assert 0.0 <= a <= 1.0 and a == pytest.approx(ref, abs=2e-3), (pool, label, a, ref)
            assert multi_blend_loss(a, train) <= multi_blend_loss(ref, train) + 1e-9
            test = multi_q_p(world, f.test_ids, pool, PER_TICKET)
            for r, (q, p) in zip(f.test_ids, test, strict=True):
                blended = -float(np.mean(np.log(a * q + (1.0 - a) * p)))
                per_race[r] = multi_base_ll(world.truth[r], pool, PER_TICKET) - blended
        assert_metric_matches(metric(rec, DELTA), pd.Series(per_race, dtype=float), world, boot)


def test_every_per_ticket_variant_is_a_registry_record_and_a_ranked_row(pt_a, scored):
    family = exotic_c5.PER_TICKET_FAMILY_ID
    spec = exotic_c5.default_per_ticket_spec()
    registry, records = pt_a.registry, pt_a.records
    assert records, "no per-ticket record was appended"
    # Its own family, frozen with its own spec; the first family's freeze and records untouched.
    assert registry.frozen_spec(family) == json.loads(json.dumps(spec))
    assert registry.frozen_spec_hash(exotic_c5.FAMILY_ID) == FIRST_SPEC_SHA256
    assert registry.records(exotic_c5.FAMILY_ID) == [pt_a.planted]
    assert {r.family_id for r in registry.records()} == {exotic_c5.FAMILY_ID, family}
    assert registry.family_size(family) == len(records)  # one record a variant
    assert {r.ticket for r in records} == {459}
    assert {r.hypothesis_id for r in records} == {"E43", "E16"}
    want = _expected_e43_keys(spec)
    assert want and {k[0] for k in want} == {PLACE, WIDE} and {k[4] for k in want} == {PER_TICKET}
    assert _keys_of(records, "E43") == want
    assert _keys_of(records, "E16") == want
    log_linear = [r for r in records if r.params.get("blend") == LOG_LINEAR]
    assert log_linear, "pool_blend's log-linear form is not recorded"
    for r in records:
        json.dumps(r.params)
        assert r.params.get("pool") in (PLACE, WIDE), r.params
        if r.params.get("blend") == LOG_LINEAR:
            assert r.status == "failed", r.params  # not runnable yet: no full final boards
            assert not any(m.metric == DELTA for m in r.metrics)
        else:
            assert r.status == "scored", (r.hypothesis_id, r.params)
        if r.params.get("input") == exotic_c5.FORM:
            assert "jrdb" in r.sources and is_licensed_record(r), r.params
        else:
            assert not is_licensed_record(r), (r.params, r.sources, r.data_builds)

    # Each no-blend variant is a measured leaderboard row (E16 its super_premium segment).
    rows = {row.option_id: row for row in rows_from_registry(registry)}
    for r in records:
        if r.params.get("blend") != NO_BLEND:
            continue
        option = f"{family}/{r.variant}" + ("[super_premium]" if r.hypothesis_id == "E16" else "")
        row = rows.get(option)
        assert row is not None and row.return_per_100 is not None, option
        assert row.ci_lo is not None and row.ci_hi is not None and row.ci_lo <= row.ci_hi, option
        assert row.pool == r.params["pool"], option
        assert row.licensed == (r.params.get("input") == exotic_c5.FORM), option
    # Its value: ¥100 on the most probable combination, at the paid dividend (0 when it lost).
    world, ids = pt_a.world, _scored_ids(scored)
    boot = spec["bootstrap"]
    got = {}
    for pool in (PLACE, WIDE):
        kept = {
            "bound": ids,
            "drop": [r for r in ids if not floor_bound(world.truth[r].dividends[pool])],
        }
        for floor, races in kept.items():
            rec = one(records, "E43", pool=pool, input=exotic_c5.NORMALISED, ordering="harville",
                      blend=NO_BLEND, devig=PER_TICKET, floor=floor)  # fmt: skip
            row = rows[f"{family}/{rec.variant}"]
            options = {r: top_pick_returns(world.truth[r], pool) for r in races}
            assert_return_matches(row, options, world, boot)
            got[(pool, floor)] = row.return_per_100
    assert len({round(v, 9) for v in got.values()}) == len(got)

    # Development days only, every one with races; the holdout is never opened.
    dev_days = {t.race_date for t in world.truth.values()} - set(HELD)
    for seam in (pt_a.store, pt_a.form):
        asked = set(seam.asked)
        assert asked, "the seam was never called"
        assert not any(is_holdout_day("japan", d) for d in asked), "a holdout day was asked for"
        assert all(FIRST_DAY <= d <= LAST_DAY for d in asked), "a day outside the span"
        assert dev_days <= asked, "a development day with races was not read"
    assert HOLDOUT_OPENED == []


def test_per_ticket_jrdb_fed_results_stay_out_of_the_tree(pt_a, pt_b):
    family = exotic_c5.PER_TICKET_FAMILY_ID

    def identity(r):
        return (r.hypothesis_id, json.dumps(r.params, sort_keys=True), r.status)

    def public(run):
        return sorted(
            (identity(r), tuple(dataclasses.astuple(m) for m in r.metrics))
            for r in run.records
            if not is_licensed_record(r)
        )

    # Only the JRDB-fed member differs between the two runs: its figures move, nothing public does.
    assert public(pt_a) and public(pt_a) == public(pt_b)
    form_a = {identity(r): r.metrics for r in pt_a.records if is_licensed_record(r)}
    form_b = {identity(r): r.metrics for r in pt_b.records if is_licensed_record(r)}
    assert form_a and set(form_a) == set(form_b)
    assert any(form_a[k] != form_b[k] for k in form_a), "the member change moved no figure"

    # The committed outputs: the family's own summary file and index line, the same bytes in both
    # runs; the first family's summary and public report, and another family's file, untouched.
    assert pt_a.committed == pt_b.committed
    own = pt_a.committed.get(f"{SUMMARY}/{registry_module.summary_stem(family)}.md")
    assert own is not None, "the family's committed summary file was not written"
    text = own.decode("utf-8")
    assert family in text and registry_module.WITHHELD_LABEL in text
    for r in pt_a.records:
        assert f"| {r.hypothesis_id} | {r.variant} |" in text, r.variant
    for path, raw in pt_a.stand_in.items():
        if path != f"{SUMMARY}/README.md":
            assert pt_a.committed.get(path) == raw, f"{path} was rewritten"
    index = pt_a.committed[f"{SUMMARY}/README.md"].decode("utf-8")
    listed = {m[1] for line in index.splitlines() if (m := INDEX_LINE.fullmatch(line))}
    assert listed == {exotic_c5.FAMILY_ID, OTHER_FAMILY, family}

    # The private report holds the member's results, outside the repository and its stand-in.
    files = [p for p in pt_a.out_dir.rglob("*") if p.is_file()]
    assert files, "no private report"
    for p in files:
        assert not p.resolve().is_relative_to(REPO.resolve())
        assert not p.resolve().is_relative_to(pt_a.repo.resolve())
    assert any(exotic_c5.FORM in p.read_text(encoding="utf-8", errors="ignore") for p in files)


def test_the_command_runs_the_per_ticket_family(pt_a):
    shown, text = _help("japan", "exotic-c5")
    assert shown.exit_code == 0, shown.output
    assert re.search(r"(?m)^\W*--family\s", text), text  # an option of the command, by its name
    stdout = pt_a.cli.stdout
    printed = json.loads(stdout[stdout.index("{") : stdout.rindex("}") + 1])
    assert printed["family_id"] == exotic_c5.PER_TICKET_FAMILY_ID
    assert printed["spec_hash"] == pt_a.registry.frozen_spec_hash(exotic_c5.PER_TICKET_FAMILY_ID)
    assert printed["variants"] == len(pt_a.records)


# ---------------------------------------------------------------------------------------------
# 10. The real stage2:q:form seam on a member file (review of PR #668 at 59d5eb3)
# ---------------------------------------------------------------------------------------------

# Every other test replaces `load_form`; this one runs it, so a change in #87's or #418's API
# (`studies.meta`, `models.stacking`) that breaks the member's rebuild fails here, not as a run
# that records every FORM variant failed.
MEMBER_STORE_FROM, MEMBER_STORE_TO = date(2016, 1, 10), date(2024, 6, 24)
MEMBER_FROM = date(2017, 1, 10)  # seven member seasons before 2024: D9's five met at level 2
FORM_Q1 = (date(2024, 1, 1), date(2024, 3, 31))  # the first scored quarter (from 2024-01-01)
FORM_Q2 = (date(2024, 4, 1), date(2024, 6, 30))  # trains on 2024-Q1's outcomes


def test_load_form_rebuilds_the_form_member_out_of_fold_from_a_member_file(tmp_path):
    """The real `load_form` on a synthetic store (the tmp data dir) and member file (outside the
    repository): p_form for every starter of the asked days, keyed by race and horse number, a
    probability per race that follows the member; and out of fold: other 2024-Q1 outcomes move no
    2024-Q1 p_form but move 2024-Q2's, whose fold trains on them."""
    natural = member_store(MEMBER_STORE_FROM, MEMBER_STORE_TO, MEMBER_FROM, seed=459)
    members = tmp_path / "hktrack" / "outputs" / "oof" / "members.parquet"
    scored_days = sorted({d for d in natural.runners["race_date"] if d >= FORM_Q1[0]})
    asked = [d for d in scored_days if d != date(2024, 5, 10)]  # one scored day not asked for
    write_member_store(natural, members)
    form, builds = exotic_c5.load_form(asked, members)

    assert list(form.columns) == ["race_id", "horse_no", "p_form"]
    runners = natural.runners
    on = runners.loc[np.array([d in set(asked) for d in runners["race_date"]], dtype=bool)]
    want = set(zip(on["race_id"], on["horse_no"].astype(int), strict=True))
    got = list(zip(form["race_id"], form["horse_no"].astype(int), strict=True))
    assert len(got) == len(set(got)), "a starter has two p_form rows"
    assert set(got) == want, "a starter of an asked day has no p_form, or one of another day has"
    day_of = dict(zip(natural.races["race_id"], natural.races["race_date"], strict=True))
    assert not any(is_holdout_day("japan", day_of[r]) for r in set(form["race_id"]))

    p = form["p_form"].to_numpy(dtype=float)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    np.testing.assert_allclose(form.groupby("race_id")["p_form"].sum().to_numpy(), 1.0, atol=1e-9)
    # keyed by horse number: the member follows the normalised price, so p_form does too
    keyed = on.assign(horse_no=on["horse_no"].astype(int)).merge(
        form.assign(horse_no=form["horse_no"].astype(int)), on=["race_id", "horse_no"]
    )
    inv = 1.0 / keyed["win_odds"]
    q = (inv / inv.groupby(keyed["race_id"]).transform("sum")).to_numpy(dtype=float)
    assert np.corrcoef(q, keyed["p_form"].to_numpy(dtype=float))[0, 1] > 0.8
    digest = hashlib.sha256(members.read_bytes()).hexdigest()[:16]
    assert any(digest in str(v) for v in builds.values()), "the build does not name the file"

    write_member_store(longshots_win(natural, *FORM_Q1), members)  # the same member file
    again, _ = exotic_c5.load_form(asked, members)
    both = form.merge(again, on=["race_id", "horse_no"], suffixes=("", "_after"), validate="1:1")
    assert len(both) == len(form) == len(again)
    when = [day_of[r] for r in both["race_id"]]
    q1 = np.array([FORM_Q1[0] <= d <= FORM_Q1[1] for d in when], dtype=bool)
    q2 = np.array([FORM_Q2[0] <= d <= FORM_Q2[1] for d in when], dtype=bool)
    assert q1.any() and q2.any()
    before, after = both["p_form"].to_numpy(), both["p_form_after"].to_numpy()
    np.testing.assert_allclose(after[q1], before[q1], rtol=0, atol=1e-12)
    moved = after[q2] - before[q2]
    assert np.abs(moved).max() > 1e-6, "2024-Q2's p_form ignores the outcomes its fold trains on"
