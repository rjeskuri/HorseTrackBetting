"""The pre-registered Family P selection study (ticket #88; 08 §5.2 end to end, §5.3, §8 A6).

Contract pinned here (`horsetrack.countries.japan.studies.selection`):

- `RuleSpec(rule_id, claim, segment, selection_cols, thresholds)` and `FamilySpec(family_id, rules,
  decision_time, cluster_unit, benchmark, max_stat)` are dataclasses. `claim` is 'beta', 'alpha'
  or 'omega'. `max_stat` is the one field these tests add to the ticket's signature: the
  multiple-testing settings, `{"level": StepM's family-wise level, "block_size": the stationary
  bootstrap's mean block length in clusters, "n_boot": SPA's and StepM's replicates}`, passed to
  #75's `spa` and `stepm`. They are owner decisions with no default (#75), so they are frozen
  with the family rather than read from code. The values below are test inputs, not the owner's.
- Frozen means `registry.freeze_family(spec.family_id, dataclasses.asdict(spec))`. `run_study`
  raises `FamilyNotFrozen` before it scores or writes anything unless
  `registry.frozen_spec_hash(spec.family_id)` is the hash `freeze_family` gives
  `dataclasses.asdict(spec)`: an unfrozen family, or any edit after the freeze, is refused, and
  the study never freezes the family itself.
- `data` is one row per runner, development rows only, every row a starter, one winner per race:
    race_id          the race
    meeting_id       the race's meeting (venue x day); `cluster_unit` names this column
    race_date        the race day (pd.Timestamp): orders the clusters for SPA and StepM
    horse_id         the runner within its race
    odds_t           decimal Win odds on the board at the decision time t (selection may use it)
    odds_final       final decimal Win odds: the price the log-loss stage conditions on
    win_dividend     the Win dividend per unit stake the runner pays if it wins; it differs from
                     odds_final by a little (a late pool move), so ROI provably reads the dividend
    won              1 for the race's winner, else 0
    takeout          the Win pool's takeout t (0.2)
    final_odds_band  odds band of odds_final (1 to 4): the ω model's FLB fixed effects
    final_rank       rank of odds_final in the race (1 = favourite)
    final_entropy    normalised entropy of the race's final-odds probabilities
    seg, jockey      segment columns: `seg` is constant within a race, `jockey` varies within it
- A rule's segment `{col: level}` holds the runners with `data[col] == level` for every pair. A
  'beta' claim's segment columns are race-level (#77's `segment_beta`). `thresholds` maps each
  selection column to `[lo, hi]`, both ends inclusive, `None` an open end; the rule bets one unit
  to Win on each runner of its segment whose selection columns are all in range.
- `benchmark="final_price"` is the normalised final Win price.
- Look-ahead: `odds_final` and every `final_*` column are final-price columns. They may condition
  a probability claim (the log-loss stage at the final price, the ω model's odds bands) but never
  select a bet: a rule naming one in `selection_cols` is refused with `LookAheadSelection` at the
  ROI stage. Board-at-t columns (`odds_t`) are allowed.
- `run_study(...) -> StudyReport(family_size, stages, survivors, p_beats_takeout, rungs)`:
  - `stages` holds "log_loss", "max_stat", "omega" and "roi" in that order (other keys allowed):
    `stages["log_loss"][rule_id]["passed"]` (bool) for every rule (its probability claim on all
    runners of its segment against the benchmark, #69's paired bootstrap over `cluster_unit`);
    `stages["max_stat"]["superior"]`, the rule ids StepM finds superior; `stages["roi"]`, keyed
    by the survivors only, `{"roi": float, "n_bets": int}` where roi = sum(won * win_dividend) /
    n_bets - 1 over every row of `data` the rule selects.
  - `survivors`: the rules that passed the log-loss stage and that StepM finds superior. The
    log-loss verdict comes from #69's `paired_bootstrap` and the max-statistic one from #75's
    `stepm` (ticket outline); one test replaces each, in every loaded horsetrack module that
    holds it, to force its verdict.
  - `p_beats_takeout[rule_id]`: P(ω_s > 1/(1 - t) | data) for the rule's segment (#78), in [0, 1].
  - `rungs[rule_id]`: "S0" or "S1" for every rule; nothing is adopted.
  - `family_size`: `registry.family_size(family_id)` after the run: the rules (one registry record
    each, variant = rule_id) plus the variants already in the family, failed (dropped) included.
  - The holdout passed in is never opened: the study completes with one whose `open` raises.

Not pinned: the log-loss bootstrap's level and replicates (D7, config/scoring.yaml), whether
claims are fitted in-sample or walk-forward, the ω model's draws, and which rule is S0 or S1.

All data is synthetic and seeded. Expected values come from the generator's own arrays (selected
rows, realised wins, a conditional-logit Newton written here), never from the code under test.
Each test first asserts the premise it relies on. Rule ids start with the Family P candidate they
belong to (P3, P4, P6), the registry's hypothesis-id form. The families are test-only and decide
nothing.
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import dataclasses
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval.bootstrap import paired_bootstrap
from horsetrack.core.eval.holdout import DEFAULT_COUNTRIES_DIR, LOG_NAME, LockedHoldout
from horsetrack.core.eval.registry import Registry, RunRecord
from horsetrack.core.eval.snooping import stepm
from horsetrack.countries.japan.studies.selection import (
    FamilyNotFrozen,
    FamilySpec,
    LookAheadSelection,
    RuleSpec,
    run_study,
)

TAKEOUT = 0.2
N_MEETINGS = 240
RACES_PER_MEETING = 6
FIRST_DAY = date(2014, 6, 7)  # D10: no price before this day
LAST_DAY = date(2025, 12, 31)  # before the holdout (2026-11-28); 2020 on is in D9's scored folds
SEGMENTS = ("A", "B", "C", "D")
# J1 rides only in segment A's races (a quarter of the runners there) and J2 and J3 never do, so
# J1's edge, paid for by the other runners of its races, leaves J2's and J3's prices exact.
J1_SHARE_IN_A = 0.25
A_JOCKEYS = ("J4", "J5", "J6", "J7", "J8")
OTHER_JOCKEYS = ("J2", "J3", "J4", "J5", "J6", "J7", "J8")
BOARD_NOISE = 0.10  # sd of ln(odds at t / final odds)
DIVIDEND_NOISE = 0.02  # sd of ln(dividend / final odds)

REAL_BETA = 1.8  # segment A's favourites are underbet: P(win) proportional to q^1.8
NULL_BETA = dict.fromkeys(SEGMENTS, 1.0)
J1_BOOST = math.log(4.0)  # J1's runners win 4 times as often as their price says, before
# normalising within the race (about twice as often after it)

MAIN_SEED = 8802
OMEGA_SEED = 8803
STUDY_SEED = 88
MAX_STAT = {"level": 0.05, "block_size": 4, "n_boot": 500}  # test inputs, not the owner's
LOOK_AHEAD = ("odds_final", "final_odds_band", "final_rank", "final_entropy")
HOLDOUT_LOG = DEFAULT_COUNTRIES_DIR / "japan" / LOG_NAME

# (rule_id, claim, segment, selection_cols, thresholds); only segment A's claim is real.
MAIN_RULES = [
    ("P4-seg-a", "beta", {"seg": "A"}, ["odds_t"], {"odds_t": [1.5, 4.0]}),
    ("P4-seg-b", "beta", {"seg": "B"}, ["odds_t"], {"odds_t": [1.5, 4.0]}),
    ("P6-seg-c", "beta", {"seg": "C"}, ["odds_t"], {"odds_t": [None, 3.0]}),
    ("P3-seg-d", "beta", {"seg": "D"}, ["odds_t"], {"odds_t": [5.0, 50.0]}),
]
MAIN_REAL = "P4-seg-a"
MAIN_DROPPED = ["P4-seg-a-wide", "P5-overlay"]

# Only J1's claim is real; J1 has a genuine positive edge at the final dividend.
OMEGA_RULES = [
    ("P4-j1", "omega", {"jockey": "J1"}, ["odds_t"], {"odds_t": [3.0, None]}),
    ("P4-j2", "omega", {"jockey": "J2"}, ["odds_t"], {"odds_t": [3.0, None]}),
    ("P4-j3", "omega", {"jockey": "J3"}, ["odds_t"], {"odds_t": [None, 10.0]}),
]
OMEGA_REAL = "P4-j1"
OMEGA_DROPPED = ["P4-j1-late"]


# ---------------------------------------------------------------------------------------------
# Synthetic runners
# ---------------------------------------------------------------------------------------------


def make_runners(seed: int, *, beta: dict[str, float], jockey_boost: float = 0.0) -> pd.DataFrame:
    """N_MEETINGS meetings of RACES_PER_MEETING races, 8 to 14 runners each (module docstring).

    Each race's final odds are (1 - t) / share for a clipped Dirichlet(1) share, rounded to 0.1.
    With q the normalised inverse final odds, the winner is drawn from softmax(beta[seg] * ln q +
    jockey_boost * [jockey == J1]), so beta 1 and no boost is a race the final price gets right.
    Segments cycle over the races; J1 rides only in segment A's races, J2 and J3 only elsewhere.
    """
    rng = np.random.default_rng(seed)
    span = (LAST_DAY - FIRST_DAY).days
    offsets = np.sort(rng.choice(span + 1, size=N_MEETINGS, replace=False))
    frames = []
    for m, offset in enumerate(offsets):
        day = pd.Timestamp(FIRST_DAY + timedelta(days=int(offset)))
        for r in range(RACES_PER_MEETING):
            n = int(rng.integers(8, 15))
            share = rng.gamma(1.0, size=n)
            share /= share.sum()
            for _ in range(3):
                share = np.clip(share, 0.004, 0.7)
                share /= share.sum()
            odds_final = np.maximum(np.round((1 - TAKEOUT) / share, 1), 1.1)
            q = (1 / odds_final) / (1 / odds_final).sum()
            segment = SEGMENTS[(m * RACES_PER_MEETING + r) % len(SEGMENTS)]
            if segment == "A":
                others = rng.choice(A_JOCKEYS, size=n)
                jockey = np.where(rng.random(n) < J1_SHARE_IN_A, "J1", others)
            else:
                jockey = rng.choice(OTHER_JOCKEYS, size=n)
            utility = beta[segment] * np.log(q) + jockey_boost * (jockey == "J1")
            won = np.zeros(n, dtype=int)
            won[np.argmax(utility + rng.gumbel(size=n))] = 1
            board = odds_final * np.exp(rng.normal(0.0, BOARD_NOISE, size=n))
            paid = odds_final * np.exp(rng.normal(0.0, DIVIDEND_NOISE, size=n))
            frames.append(
                pd.DataFrame(
                    {
                        "race_id": f"m{m:03d}-r{r + 1:02d}",
                        "meeting_id": f"m{m:03d}",
                        "race_date": day,
                        "horse_id": np.arange(1, n + 1),
                        "seg": segment,
                        "jockey": jockey,
                        "odds_t": np.maximum(np.round(board, 1), 1.1),
                        "odds_final": odds_final,
                        "win_dividend": np.maximum(np.round(paid, 1), 1.0),
                        "won": won,
                        "takeout": TAKEOUT,
                        "final_odds_band": np.digitize(odds_final, [3.0, 8.0, 30.0]) + 1,
                        "final_rank": pd.Series(odds_final).rank(method="min").to_numpy(int),
                        "final_entropy": float(-(q * np.log(q)).sum() / np.log(n)),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def implied(frame: pd.DataFrame) -> pd.Series:
    """q: the normalised inverse final odds within each race."""
    inverse = 1.0 / frame["odds_final"]
    return inverse / inverse.groupby(frame["race_id"]).transform("sum")


def beta_z(frame: pd.DataFrame) -> float:
    """(beta_hat - 1) / se of P(win) proportional to q^beta over frame's races (one winner each),
    by Newton on the conditional-logit score and information written out here."""
    x = np.log(implied(frame).to_numpy())
    codes = pd.factorize(frame["race_id"])[0]
    won = frame["won"].to_numpy(dtype=bool)
    beta, info = 1.0, 0.0
    for _ in range(50):
        u = beta * x
        top = pd.Series(u).groupby(codes).transform("max").to_numpy()
        w = np.exp(u - top)
        total = np.bincount(codes, weights=w)
        mean = np.bincount(codes, weights=w * x) / total
        info = float((np.bincount(codes, weights=w * x * x) / total - mean**2).sum())
        step = (x[won].sum() - mean.sum()) / info
        beta += step
        if abs(step) < 1e-12:
            break
    return (beta - 1.0) * math.sqrt(info)


def omega_premise(frame: pd.DataFrame, mask: pd.Series) -> tuple[float, float]:
    """Realised wins over the wins the final price expects, and that excess as a z-score."""
    q = implied(frame)[mask]
    wins = float(frame.loc[mask, "won"].sum())
    return wins / q.sum(), (wins - q.sum()) / math.sqrt((q * (1 - q)).sum())


def selected(data: pd.DataFrame, rule: tuple) -> pd.DataFrame:
    """The rows a rule bets on: its segment, every selection column in [lo, hi] inclusive."""
    _, _, segment, selection_cols, thresholds = rule
    keep = pd.Series(True, index=data.index)
    for col, level in segment.items():
        keep &= data[col] == level
    for col in selection_cols:
        lo, hi = thresholds[col]
        if lo is not None:
            keep &= data[col] >= lo
        if hi is not None:
            keep &= data[col] <= hi
    return data[keep]


def realised_roi(rows: pd.DataFrame) -> float:
    return float((rows["won"] * rows["win_dividend"]).sum() / len(rows) - 1.0)


# ---------------------------------------------------------------------------------------------
# Families, registry and holdout doubles
# ---------------------------------------------------------------------------------------------


def make_rule(rule_id, claim, segment, selection_cols, thresholds) -> RuleSpec:
    return RuleSpec(
        rule_id=rule_id,
        claim=claim,
        segment=dict(segment),
        selection_cols=list(selection_cols),
        thresholds={col: list(bounds) for col, bounds in thresholds.items()},
    )


def make_family(family_id: str, rules: list[tuple]) -> FamilySpec:
    return FamilySpec(
        family_id=family_id,
        rules=[make_rule(*rule) for rule in rules],
        decision_time="T-5",  # five minutes before the off: a test value
        cluster_unit="meeting_id",
        benchmark="final_price",
        max_stat=dict(MAX_STAT),
    )


def new_registry(root: Path) -> Registry:
    log = root / "countries" / "japan" / LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    return Registry(root / "data" / "registry" / "japan.jsonl", holdout_log=log)


def freeze(registry: Registry, spec: FamilySpec) -> str:
    return registry.freeze_family(spec.family_id, dataclasses.asdict(spec))


def add_dropped(registry: Registry, family_id: str, variants: list[str]) -> None:
    """Variants tried and dropped before the family was frozen: failed records of the family."""
    for variant in variants:
        registry.append(
            RunRecord(
                hypothesis_id=variant.split("-")[0],
                family_id=family_id,
                variant=variant,
                ticket=88,
                git_sha="0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c",
                data_builds={"runners": "synthetic"},
                sources=("synthetic",),
                fold_scheme="synthetic",
                components={"rule": variant},
                params={"dropped": True},
                metrics=(),
                status="failed",
            )
        )


class HoldoutOpened(BaseException):
    """Raised by the double's open; a BaseException so `except Exception` cannot swallow it."""


class TrippedHoldout(LockedHoldout):
    """A LockedHoldout whose open records the call and raises."""

    def __init__(self) -> None:  # no rows: there is nothing behind this holdout
        self.calls: list[dict] = []

    def open(self, **kwargs) -> pd.DataFrame:
        self.calls.append(kwargs)
        raise HoldoutOpened(f"the study opened the holdout: {kwargs}")

    def __repr__(self) -> str:
        return "TrippedHoldout()"


def read_holdout_log() -> bytes | None:
    return HOLDOUT_LOG.read_bytes() if HOLDOUT_LOG.exists() else None


@dataclass
class Run:
    rules: list[tuple]
    spec: FamilySpec
    data: pd.DataFrame
    registry: Registry
    holdout: TrippedHoldout
    report: object
    log_before: bytes | None
    log_after: bytes | None


def run_frozen(root: Path, family_id: str, rules: list[tuple], data, dropped=()) -> Run:
    registry = new_registry(root)
    add_dropped(registry, family_id, list(dropped))
    spec = make_family(family_id, rules)
    freeze(registry, spec)
    holdout = TrippedHoldout()
    before = read_holdout_log()
    with pytest.MonkeyPatch.context() as patch:
        (root / "data").mkdir(parents=True, exist_ok=True)
        patch.setenv("HORSETRACK_DATA_DIR", str(root / "data"))
        report = run_study(spec, data.copy(), registry=registry, holdout=holdout, seed=STUDY_SEED)
    return Run(rules, spec, data, registry, holdout, report, before, read_holdout_log())


@pytest.fixture(scope="module")
def main_runners() -> pd.DataFrame:
    return make_runners(MAIN_SEED, beta={**NULL_BETA, "A": REAL_BETA})


@pytest.fixture(scope="module")
def omega_runners() -> pd.DataFrame:
    return make_runners(OMEGA_SEED, beta=NULL_BETA, jockey_boost=J1_BOOST)


@pytest.fixture(scope="module")
def main_run(tmp_path_factory, main_runners) -> Run:
    root = tmp_path_factory.mktemp("main")
    return run_frozen(root, "test-88-segments", MAIN_RULES, main_runners, MAIN_DROPPED)


@pytest.fixture(scope="module")
def omega_run(tmp_path_factory, omega_runners) -> Run:
    root = tmp_path_factory.mktemp("omega")
    return run_frozen(root, "test-88-jockeys", OMEGA_RULES, omega_runners, OMEGA_DROPPED)


def check_main_premise(data: pd.DataFrame) -> None:
    """Segment A's beta is far above 1; B, C and D's are within 1 se of 1."""
    z = {s: beta_z(data[data["seg"] == s]) for s in SEGMENTS}
    assert z["A"] > 5, z
    assert all(abs(z[s]) < 1.0 for s in ("B", "C", "D")), z


def check_omega_premise(data: pd.DataFrame) -> None:
    """J1 wins far more often than its price says; J2 and J3 within 1 se of their price."""
    ratio, z = omega_premise(data, data["jockey"] == "J1")
    assert ratio > 1.6 and z > 5, (ratio, z)
    for jockey in ("J2", "J3"):
        _, z = omega_premise(data, data["jockey"] == jockey)
        assert abs(z) < 1.0, (jockey, z)


# ---------------------------------------------------------------------------------------------
# DoD: the study refuses to score before the family is frozen
# ---------------------------------------------------------------------------------------------


def _edit_rule(spec: FamilySpec, k: int, **changes) -> FamilySpec:
    rules = list(spec.rules)
    rules[k] = dataclasses.replace(rules[k], **changes)
    return dataclasses.replace(spec, rules=rules)


def _edits(spec: FamilySpec) -> list[tuple[str, FamilySpec]]:
    """Each a single, self-consistent change to a frozen spec, every one runnable on the data."""
    extra = dataclasses.replace(spec.rules[1], rule_id="P4-seg-b-late")
    return [
        ("a threshold", _edit_rule(spec, 0, thresholds={"odds_t": [1.5, 4.5]})),
        ("an open end", _edit_rule(spec, 2, thresholds={"odds_t": [1.0, 3.0]})),
        ("a segment level", _edit_rule(spec, 1, segment={"seg": "C"})),
        ("a claim", _edit_rule(spec, 3, claim="omega")),
        ("the selection", _edit_rule(spec, 0, selection_cols=[], thresholds={})),
        ("a rule id", _edit_rule(spec, 1, rule_id="P4-seg-b2")),
        ("a rule added", dataclasses.replace(spec, rules=[*spec.rules, extra])),
        ("a rule dropped", dataclasses.replace(spec, rules=spec.rules[:-1])),
        ("the decision time", dataclasses.replace(spec, decision_time="T-10")),
        ("the cluster unit", dataclasses.replace(spec, cluster_unit="race_date")),
        ("the StepM level", dataclasses.replace(spec, max_stat={**MAX_STAT, "level": 0.10})),
        ("the block length", dataclasses.replace(spec, max_stat={**MAX_STAT, "block_size": 1})),
    ]


def test_family_is_frozen_before_scoring(tmp_path, monkeypatch, main_runners):
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    spec = make_family("test-88-segments", MAIN_RULES)

    # Never frozen: refused, nothing scored, and the study does not freeze it itself.
    registry = new_registry(tmp_path / "unfrozen")
    with pytest.raises(FamilyNotFrozen):
        run_study(spec, main_runners.copy(), registry=registry, holdout=TrippedHoldout(), seed=1)
    assert registry.records() == []
    assert registry.frozen_spec_hash(spec.family_id) is None

    # The same content frozen under another family id does not freeze this one.
    registry.freeze_family("test-88-other", dataclasses.asdict(spec))
    with pytest.raises(FamilyNotFrozen):
        run_study(spec, main_runners.copy(), registry=registry, holdout=TrippedHoldout(), seed=1)
    assert registry.records() == []

    # Frozen, then edited: every edit is refused and the frozen hash stays the original.
    registry = new_registry(tmp_path / "edited")
    frozen = freeze(registry, spec)
    for what, edited in _edits(spec):
        assert dataclasses.asdict(edited) != dataclasses.asdict(spec), what
        with pytest.raises(FamilyNotFrozen):
            run_study(
                edited, main_runners.copy(), registry=registry, holdout=TrippedHoldout(), seed=1
            )
        assert registry.records() == [], what
        assert registry.frozen_spec_hash(spec.family_id) == frozen, what


# ---------------------------------------------------------------------------------------------
# DoD: ROI only for the rules that survived the log-loss and max-statistic stages
# ---------------------------------------------------------------------------------------------


def test_roi_stage_runs_only_on_log_loss_survivors(main_run, omega_run):
    """Observed through `report.stages`: in each family only the rule whose claim is real
    reaches the ROI stage, and its ROI is the realised return of the rows it selects at t, paid
    at the final dividend."""
    check_main_premise(main_run.data)
    check_omega_premise(omega_run.data)

    for run, real in ((main_run, MAIN_REAL), (omega_run, OMEGA_REAL)):
        report, ids = run.report, [rule[0] for rule in run.rules]
        names = list(report.stages)
        assert {"log_loss", "max_stat", "omega", "roi"} <= set(names), names
        order = [name for name in names if name in ("log_loss", "max_stat", "omega", "roi")]
        assert order == ["log_loss", "max_stat", "omega", "roi"], names

        log_loss, superior = report.stages["log_loss"], report.stages["max_stat"]["superior"]
        assert set(log_loss) == set(ids)  # every rule's claim is tested, not only some
        assert set(superior) <= set(ids)
        passed_both = {r for r in ids if log_loss[r]["passed"] is True and r in superior}
        assert set(report.survivors) == passed_both
        assert list(report.survivors) == [real]
        assert set(report.stages["roi"]) == {real}

        rule = next(rule for rule in run.rules if rule[0] == real)
        rows = selected(run.data, rule)
        assert len(rows) > 50
        assert realised_roi(rows) != pytest.approx(  # the dividend, not the final odds
            float((rows["won"] * rows["odds_final"]).sum() / len(rows) - 1.0), abs=1e-6
        )
        roi = report.stages["roi"][real]
        assert roi["n_bets"] == len(rows)
        assert roi["roi"] == pytest.approx(realised_roi(rows), rel=1e-9, abs=1e-12)

    # Both ends of a range are inclusive: rows on the bounds are among the main rule's bets.
    rows = selected(main_run.data, MAIN_RULES[0])
    assert (rows["odds_t"] == 1.5).any() and (rows["odds_t"] == 4.0).any()


def _stepm_finding_nothing(*args, **kwargs):
    """#75's StepM, except that it finds no rule superior."""
    return dataclasses.replace(stepm(*args, **kwargs), superior=[])


def _bootstrap_of_the_reversed_difference(diff, clusters, **kwargs):
    """#69's bootstrap of the negated difference: every claim's verdict flips, whichever sign
    the study gives the difference."""
    return paired_bootstrap(-diff, clusters, **kwargs)


def _replace_everywhere(patch: pytest.MonkeyPatch, original, fake) -> None:
    """Point every loaded horsetrack module's name for `original` at `fake`."""
    for name, module in list(sys.modules.items()):
        if name.startswith("horsetrack") and getattr(module, original.__name__, None) is original:
            patch.setattr(module, original.__name__, fake)


def test_roi_needs_both_the_log_loss_and_the_max_statistic_stage(tmp_path, main_runners):
    """Either gate alone stops a rule. With StepM finding nothing, segment A still passes the
    log-loss stage and gets no ROI; with the log-loss bootstrap's verdict reversed, StepM's view
    of A alone does not let it through. The stages' tools are #75's stepm and #69's
    paired_bootstrap (ticket outline), replaced here to force their verdicts."""
    check_main_premise(main_runners)
    gates = [
        ("max_stat", stepm, _stepm_finding_nothing),
        ("log_loss", paired_bootstrap, _bootstrap_of_the_reversed_difference),
    ]
    for gate, original, fake in gates:
        with pytest.MonkeyPatch.context() as patch:
            _replace_everywhere(patch, original, fake)
            run = run_frozen(tmp_path / gate, f"test-88-{gate}-gate", MAIN_RULES[:2], main_runners)
        report = run.report
        assert list(report.survivors) == [], gate
        assert report.stages["roi"] == {}, gate
        if gate == "max_stat":
            assert report.stages["max_stat"]["superior"] == []
            assert report.stages["log_loss"][MAIN_REAL]["passed"] is True
        else:
            assert report.stages["log_loss"][MAIN_REAL]["passed"] is False


# ---------------------------------------------------------------------------------------------
# Selection hygiene: a final-odds selection variable is refused at the ROI stage
# ---------------------------------------------------------------------------------------------


def test_a_final_odds_selection_variable_is_refused_at_the_roi_stage(
    tmp_path, monkeypatch, main_runners
):
    """The real rule (segment A, which survives whatever it selects on) is re-pointed at each
    final-price column in turn, alone or beside a board-at-t column; every run is refused."""
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(tmp_path))
    check_main_premise(main_runners)
    cases = [[col] for col in LOOK_AHEAD] + [["odds_t", "final_rank"]]
    ranges = {
        "odds_t": [1.5, 4.0],
        "odds_final": [1.5, 4.0],
        "final_odds_band": [1, 2],
        "final_rank": [1, 3],
        "final_entropy": [0.0, 0.8],
    }
    for k, cols in enumerate(cases):
        real = ("P4-seg-a", "beta", {"seg": "A"}, cols, {col: ranges[col] for col in cols})
        rules = [real, MAIN_RULES[1]]
        registry = new_registry(tmp_path / f"case{k}")
        spec = make_family(f"test-88-look-ahead-{k}", rules)
        freeze(registry, spec)
        holdout = TrippedHoldout()
        with pytest.raises(LookAheadSelection):
            run_study(spec, main_runners.copy(), registry=registry, holdout=holdout, seed=1)
        assert holdout.calls == [], cols


# ---------------------------------------------------------------------------------------------
# DoD: the holdout is not opened
# ---------------------------------------------------------------------------------------------


def test_the_holdout_is_never_opened(main_run, omega_run):
    """Both studies were handed a LockedHoldout whose open raises, and both completed."""
    for run in (main_run, omega_run):
        assert run.report.family_size > 0  # the study completed and reported
        assert run.holdout.calls == []
        assert run.log_after == run.log_before  # the country's holdout log is untouched
        records = run.registry.records(run.spec.family_id)
        assert records
        assert all(record.holdout_open is None for record in records)


# ---------------------------------------------------------------------------------------------
# DoD: the report states family size (dropped variants included), survivors and P(ω_s > 1/(1-t))
# ---------------------------------------------------------------------------------------------


def test_report_family_size_includes_dropped_variants(main_run, omega_run):
    for run, dropped in ((main_run, MAIN_DROPPED), (omega_run, OMEGA_DROPPED)):
        ids = [rule[0] for rule in run.rules]
        family_id = run.spec.family_id
        assert run.report.family_size == len(ids) + len(dropped)
        assert run.report.family_size == run.registry.family_size(family_id)
        variants = {record.variant for record in run.registry.records(family_id)}
        assert set(ids) | set(dropped) == variants  # each rule recorded, the dropped kept


def test_report_states_p_beats_takeout_per_segment(main_run, omega_run):
    """One probability per rule's segment; J1, whose runners return about 2 * (1 - t) per unit
    at the final price, beats the takeout and J2 and J3 do not."""
    check_omega_premise(omega_run.data)
    for run in (main_run, omega_run):
        p = run.report.p_beats_takeout
        assert set(p) == {rule[0] for rule in run.rules}
        for value in p.values():
            assert isinstance(value, float) and 0.0 <= value <= 1.0
    p = omega_run.report.p_beats_takeout
    assert p["P4-j1"] > 0.5
    assert p["P4-j2"] < 0.5 and p["P4-j3"] < 0.5


# ---------------------------------------------------------------------------------------------
# DoD: nothing is promoted beyond stake rung S1
# ---------------------------------------------------------------------------------------------


def test_no_rule_is_placed_above_rung_s1(main_run, omega_run):
    """Even J1, which passes every stage with a positive realised return at the final dividend,
    stops at shadow; nothing in the registry is adopted."""
    j1 = selected(omega_run.data, OMEGA_RULES[0])
    assert realised_roi(j1) > 0.2  # premise: an edge a naive ladder would stake on
    for run in (main_run, omega_run):
        assert set(run.report.rungs) == {rule[0] for rule in run.rules}
        assert set(run.report.rungs.values()) <= {"S0", "S1"}
        records = run.registry.records(run.spec.family_id)
        assert all(record.status != "adopted" for record in records)
        assert all(record.owner_signoff is None for record in records)
