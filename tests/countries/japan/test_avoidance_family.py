"""The L1 thresholds as owner-registered grids, and the family they make (#81; owner answer of
2026-11-25: `l1_threshold_trio` [20, 50, 100], `l1_threshold_quinella` [5, 10, 20], each grid member
counted in the D17 family size).

Contract pinned here (`horsetrack.countries.japan.studies.avoidance`), beside test_avoidance.py:
- config/avoidance.yaml registers each L1 threshold as a grid: `l1_threshold_trio` and
  `l1_threshold_quinella` are non-empty lists of distinct decimal odds, each a finite real above
  1.0. `owner_inputs()` returns each grid ascending, whatever order the owner wrote it in. An empty
  list, a member at or under 1.0, a non-real or non-finite member, a duplicate, or a bare number (a
  one-member grid is written `[x]`) is a ValueError. A missing file or key is still
  `MissingOwnerDecision`, raised before any value is checked.
- `variant_names(owner)`: one L1 variant per pool, per grid member, per selection (the final price,
  look-ahead, and the board at t). Each L1 name starts with "L1", names its pool and carries its
  threshold as a number; every name is unique. The family is those L1 variants, the L4 variants
  (the same-bracket pair, and the market-order trifecta against each comparator at both
  selections) and the L5 calendar report.
- `build_spec(owner, bootstrap, minutes)`: `rules.L1.thresholds` holds both grids in full,
  ascending; `variants` is `variant_names(owner)` and `family_size` its length (D17: every grid
  member counts). The same grid written in another order, or as ints, freezes the same spec.
- `compute_study(data, spec)`: every variant of the spec, once. Each L1 variant's params carry its
  pool and its own threshold, and its returns follow that threshold alone.
- `run_study()`: the spec is frozen before any row is read, and every variant leaves one registry
  record, so the registry's family size is the spec's.

Expected values come from first principles: the family size from the product of pools, grid
members and selections; returns from the stakes and dividends of the synthetic races below. Every
frame is synthetic; no date is a Japan holdout day (on or after 2026-11-28).
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from datetime import date
from statistics import fmean

import pandas as pd
import pytest

from horsetrack.core.eval.registry import FamilyAlreadyFrozen, Registry
from horsetrack.countries.japan.studies import avoidance
from horsetrack.countries.japan.studies.avoidance import (
    MissingOwnerDecision,
    RaceView,
    StudyData,
    build_spec,
    compute_study,
    owner_inputs,
    variant_names,
)

TRIO, QUINELLA = "l1_threshold_trio", "l1_threshold_quinella"
GRID_KEYS = {"TRIO": TRIO, "QUINELLA": QUINELLA}
OWNER = {  # the owner's answers on #81 (2026-11-25), as config/avoidance.yaml carries them
    TRIO: [20, 50, 100],
    QUINELLA: [5, 10, 20],
    "sets": "every_set",
    "l4_market_order_ties": "lower_horse_no",
    "l4_trifecta_comparators": ["trio", "box"],
    "l5_mode": "report",
    "owner_signoff": 81,
}
BOOT = {"n_boot": 200, "level": 0.95, "seed": 81}
MINUTES = 2
LOOK_AHEAD = "final (look-ahead)"
NAN, INF = float("nan"), float("inf")


def use_config(monkeypatch, raw: dict | None) -> None:
    """config/avoidance.yaml reads as `raw` (None: the file does not exist); every other config as
    shipped."""
    shipped = avoidance.load_country_yaml

    def load(country, name):
        if name != "avoidance":
            return shipped(country, name)
        if raw is None:
            raise FileNotFoundError(f"countries/{country}/config/{name}.yaml")
        return copy.deepcopy(raw)

    monkeypatch.setattr(avoidance, "load_country_yaml", load)


def owner_with(monkeypatch, **changes) -> dict:
    use_config(monkeypatch, {**OWNER, **changes})
    return owner_inputs()


def family_size(trio: list, quinella: list, comparators: list) -> int:
    """D17's count with l5_mode 'report': per pool, per grid member, two selections; L4's
    same-bracket pair plus the market-order trifecta against each comparator at two selections;
    L5's calendar report."""
    return 2 * (len(trio) + len(quinella)) + 1 + 2 * len(comparators) + 1


FAMILIES = [  # (trio grid, quinella grid, L4 trifecta comparators)
    ([20, 50, 100], [5, 10, 20], ["trio", "box"]),  # the owner's answer
    ([40], [8], ["trio"]),
    ([12.5, 40, 300, 1000], [3, 8.5], ["box"]),
    ([20, 50], [5, 10, 20, 30, 50], ["trio", "box"]),
]

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def l1_key(name: str, grids: dict[str, list[float]]) -> tuple[str, float]:
    """The (pool, threshold) an L1 variant name carries: exactly one pool named, and exactly one
    member of that pool's grid among the numbers in the name."""
    pools = [pool for pool in grids if pool.lower() in name.lower()]
    assert len(pools) == 1, name
    numbers = {float(n) for n in NUMBER.findall(name)}
    hits = numbers & {float(x) for x in grids[pools[0]]}
    assert len(hits) == 1, (name, sorted(hits))
    return pools[0], hits.pop()


# ---------------------------------------------------------------------------------------------
# owner_inputs: each L1 threshold is a grid
# ---------------------------------------------------------------------------------------------


GRIDS = [  # (trio as written, quinella as written)
    ([20, 50, 100], [5, 10, 20]),
    ([100, 20, 50], [20, 5, 10]),
    ([20.0, 50.0, 100.0], [5.0, 10.0, 20.0]),
    ([1.5], [2]),
    ([12.5, 300, 40.0, 7.25], [99.9, 1.01]),
]


def test_owner_inputs_reads_each_l1_threshold_as_an_ascending_grid(monkeypatch):
    for trio, quinella in GRIDS:
        got = owner_with(monkeypatch, **{TRIO: trio, QUINELLA: quinella})
        assert list(got[TRIO]) == sorted(float(x) for x in trio), trio
        assert list(got[QUINELLA]) == sorted(float(x) for x in quinella), quinella

    got = owner_with(monkeypatch)
    assert list(got[TRIO]) == [20.0, 50.0, 100.0]
    assert list(got[QUINELLA]) == [5.0, 10.0, 20.0]
    assert got["sets"] == "every_set"
    assert got["l4_market_order_ties"] == "lower_horse_no"
    assert list(got["l4_trifecta_comparators"]) == ["trio", "box"]
    assert got["l5_mode"] == "report"
    assert got["owner_signoff"] == 81


def test_a_bare_threshold_is_not_a_grid(monkeypatch):
    """Ruling: a one-member grid is written [x]; a bare number (the pre-grid form) is refused, so
    a stale config cannot run a smaller family than the owner registered."""
    use_config(monkeypatch, OWNER)
    assert list(owner_inputs()[TRIO]) == [20.0, 50.0, 100.0]  # the other grid alone is read

    for key in GRID_KEYS.values():
        for scalar in (20, 20.0, 1.5, 300):
            use_config(monkeypatch, {**OWNER, key: scalar})
            with pytest.raises(ValueError):
                owner_inputs()


MALFORMED = [
    [],
    [1.0],
    [0],
    [20, 1.0],
    [20, 0.5],
    [20, -5],
    [20, "50"],
    ["20"],
    [True, 20],
    [20, None],
    [20, NAN],
    [20, INF],
    [20, 20],
    [50, 20, 50.0],
    [20, [50]],
    "20, 50",
    {"x": 20},
    None,
]


def test_owner_inputs_rejects_a_malformed_grid(monkeypatch):
    use_config(monkeypatch, OWNER)
    assert list(owner_inputs()[TRIO]) == [20.0, 50.0, 100.0]  # the well-formed grid is read

    for key in GRID_KEYS.values():
        for bad in MALFORMED:
            use_config(monkeypatch, {**OWNER, key: bad})
            with pytest.raises(ValueError):
                owner_inputs()


def test_a_missing_decision_is_raised_before_a_grid_is_checked(monkeypatch):
    use_config(monkeypatch, OWNER)
    assert list(owner_inputs()[QUINELLA]) == [5.0, 10.0, 20.0]

    for missing in OWNER:
        raw = {k: v for k, v in OWNER.items() if k != missing}
        for key in GRID_KEYS.values():  # the grids still present are malformed
            if key in raw:
                raw[key] = []
        use_config(monkeypatch, raw)
        with pytest.raises(MissingOwnerDecision):
            owner_inputs()
    for raw in (None, {}):
        use_config(monkeypatch, raw)
        with pytest.raises(MissingOwnerDecision):
            owner_inputs()


def test_the_shipped_config_is_the_owners_answer():
    got = owner_inputs()  # config/avoidance.yaml as committed

    assert list(got[TRIO]) == [20.0, 50.0, 100.0]
    assert list(got[QUINELLA]) == [5.0, 10.0, 20.0]
    assert got["sets"] == "every_set"
    assert got["l4_market_order_ties"] == "lower_horse_no"
    assert list(got["l4_trifecta_comparators"]) == ["trio", "box"]
    assert got["l5_mode"] == "report"
    assert got["owner_signoff"] == 81


# ---------------------------------------------------------------------------------------------
# variant_names and build_spec: every grid member is a variant of the frozen family
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("trio", "quinella", "comparators"), FAMILIES)
def test_variant_names_count_every_grid_member(monkeypatch, trio, quinella, comparators):
    owner = owner_with(
        monkeypatch, **{TRIO: trio, QUINELLA: quinella, "l4_trifecta_comparators": comparators}
    )

    names = variant_names(owner)

    assert len(names) == family_size(trio, quinella, comparators)
    assert len(set(names)) == len(names)
    l1 = [name for name in names if name.startswith("L1")]
    assert len(l1) == 2 * (len(trio) + len(quinella))
    grids = {"TRIO": trio, "QUINELLA": quinella}
    carried = Counter(l1_key(name, grids) for name in l1)
    want = {(pool, float(x)): 2 for pool, grid in grids.items() for x in grid}
    assert dict(carried) == want  # each (pool, threshold): its two selections


@pytest.mark.parametrize(("trio", "quinella", "comparators"), FAMILIES)
def test_build_spec_records_both_grids_and_the_family_size(
    monkeypatch, trio, quinella, comparators
):
    owner = owner_with(
        monkeypatch, **{TRIO: trio, QUINELLA: quinella, "l4_trifecta_comparators": comparators}
    )

    spec = build_spec(owner, BOOT, MINUTES)

    thresholds = spec["rules"]["L1"]["thresholds"]
    assert {pool: list(grid) for pool, grid in thresholds.items()} == {
        "TRIO": sorted(float(x) for x in trio),
        "QUINELLA": sorted(float(x) for x in quinella),
    }
    assert list(spec["variants"]) == variant_names(owner)
    assert spec["family_size"] == len(variant_names(owner))
    assert spec["family_size"] == family_size(trio, quinella, comparators)
    json.dumps(spec, allow_nan=False)  # plain JSON, as freeze_family stores it


def test_the_same_grid_freezes_the_same_family(monkeypatch, tmp_path):
    registry = Registry(tmp_path / "japan.jsonl", holdout_log=tmp_path / "holdout_log.jsonl")
    family = "test-family-l1-grid"

    def spec(trio, quinella):
        return build_spec(
            owner_with(monkeypatch, **{TRIO: trio, QUINELLA: quinella}), BOOT, MINUTES
        )

    frozen = registry.freeze_family(family, spec([20.0, 50.0, 100.0], [5.0, 10.0, 20.0]))
    # the owner's grid written in another order, or as ints, is the same family
    assert registry.freeze_family(family, spec([100, 20, 50], [20, 5, 10])) == frozen
    assert registry.freeze_family(family, spec([20, 50, 100], [5, 10, 20])) == frozen
    # a grid with a member dropped or added is another family
    for trio, quinella in (
        ([20, 50], [5, 10, 20]),
        ([20, 50, 100], [5, 10]),
        ([20, 50, 100, 200], [5, 10, 20]),
        ([20, 50, 100], [5, 10, 20, 40]),
    ):
        with pytest.raises(FamilyAlreadyFrozen):
            registry.freeze_family(family, spec(trio, quinella))


# ---------------------------------------------------------------------------------------------
# compute_study and run_study: each grid threshold is scored as its own variant
# ---------------------------------------------------------------------------------------------

STARTERS = range(1, 9)  # eight starters in every race, one per bracket
RACES = (
    "jra:2026-06-06:TOK:01",
    "jra:2026-06-06:TOK:02",
    "jra:2026-06-07:KYO:01",
    "jra:2026-06-07:KYO:02",
)
A, B, C, D = RACES
FINISH = {A: (3, 1, 2), B: (5, 7, 2), C: (8, 4, 6), D: (2, 6, 1)}
LEGS = {"TRIO": 3, "QUINELLA": 2}
ORDERED = {"TRIO": "TIERCE", "QUINELLA": "FORECAST"}
# final dividends per 100 yen: (the direct pool's, the winning order's in the ordered pool)
FINAL = {
    "TRIO": {A: (3500, 24000), B: (1500, 12000), C: (7500, 30000), D: (15000, 120000)},
    "QUINELLA": {A: (700, 1600), B: (1500, 2400), C: (2500, 6000), D: (380, 900)},
}
# the board at t: the winning set's odds, and a losing set with its odds
AT_T = {
    "TRIO": {
        A: (35.0, "4,5,6", 80.0),
        B: (22.0, "1,3,8", 120.0),
        C: (75.0, "1,2,3", 10.0),
        D: (150.0, "3,4,5", 45.0),
    },
    "QUINELLA": {
        A: (7.0, "2,4", 3.0),
        B: (12.0, "1,8", 40.0),
        C: (25.0, "5,6", 6.0),
        D: (6.0, "7,8", 9.0),
    },
}
AT_T_LABEL = f"board at t (T-{MINUTES})"


def joined(horses) -> str:
    return ",".join(str(int(h)) for h in horses)


def winning_set(pool: str, race: str) -> str:
    return joined(sorted(FINISH[race][: LEGS[pool]]))


def study_data() -> StudyData:
    """Four run races in two meetings, eight starters each, the four exotic pools' paying rows,
    and a TRIO and QUINELLA board at t for every race."""
    day = {race: date.fromisoformat(race.split(":")[1]) for race in RACES}
    races = pd.DataFrame(
        [(r, day[r], ":".join(r.split(":")[1:3]), "run", None) for r in RACES],
        columns=["race_id", "race_date", "meeting_id", "status", "race_class_raw"],
    )
    runners = pd.DataFrame(
        [(r, day[r], h, h, False, 1.5 + 2.0 * h) for r in RACES for h in STARTERS],
        columns=["race_id", "race_date", "horse_no", "bracket", "is_scratched", "win_odds"],
    )
    rows = []
    for pool, by_race in FINAL.items():
        for race, (direct, ordered) in by_race.items():
            top = FINISH[race][: LEGS[pool]]
            rows.append((race, pool, winning_set(pool, race), float(direct), False))
            rows.append((race, ORDERED[pool], joined(top), float(ordered), False))
    dividends = pd.DataFrame(
        rows, columns=["race_id", "pool", "combination", "dividend", "is_refund"]
    )
    boards = {
        race: pd.DataFrame(
            [
                row
                for pool, by_race in AT_T.items()
                for row in (
                    (pool, winning_set(pool, race), by_race[race][0]),
                    (pool, by_race[race][1], by_race[race][2]),
                )
            ],
            columns=["pool", "combination", "odds"],
        )
        for race in RACES
    }
    return StudyData(
        races=races,
        runners=runners,
        dividends=dividends,
        logged={"TRIO": list(RACES), "QUINELLA": list(RACES)},
        view_at_t=lambda race: RaceView(race, boards[race], AT_T_LABEL),
        board_files=[],
    )


def expected_returns(pool: str, threshold: float, at_t: bool) -> tuple[float, float]:
    """Mean rule and comparator return (payout / stake) over the four races. Each set is staked
    k orders x 100 yen: in the direct pool when its price is under the threshold, else as its k
    ordered tickets of 100 yen; the comparator always in the direct pool. On history every
    combination of the 8 starters is bet and the price is the winning set's final dividend / 100;
    at t the two sets priced on the board are bet, by their price at t. Losing sets pay nothing
    on either side."""
    legs = LEGS[pool]
    k = math.factorial(legs)
    stake = 100 * k
    rule, comparator = [], []
    for race in RACES:
        direct, ordered = FINAL[pool][race]
        if at_t:
            price, sets = AT_T[pool][race][0], 2
        else:
            price, sets = direct / 100, math.comb(len(STARTERS), legs)
        in_direct = stake * direct / 100
        in_ordered = (stake // k) * ordered / 100  # the one winning order
        rule.append((in_direct if price < threshold else in_ordered) / (sets * stake))
        comparator.append(in_direct / (sets * stake))
    return fmean(rule), fmean(comparator)


@pytest.mark.parametrize(
    ("trio", "quinella", "comparators"),
    [FAMILIES[0], ([12.5, 40], [6.5, 20], ["box"])],
)
def test_compute_study_scores_each_grid_threshold_as_its_own_variant(
    monkeypatch, trio, quinella, comparators
):
    owner = owner_with(
        monkeypatch, **{TRIO: trio, QUINELLA: quinella, "l4_trifecta_comparators": comparators}
    )
    spec = build_spec(owner, BOOT, MINUTES)
    grids = {"TRIO": [float(x) for x in trio], "QUINELLA": [float(x) for x in quinella]}
    # The fixture tells the thresholds apart: a set priced between two grid members is in the
    # direct pool under the larger one and in the ordered synthetic under the smaller one, so
    # each threshold has its own expected difference, at the final price and at t.
    for pool, grid in grids.items():
        for at_t in (False, True):
            gaps = {round(r - c, 12) for r, c in (expected_returns(pool, x, at_t) for x in grid)}
            assert len(gaps) == len(grid), (pool, at_t)

    study = compute_study(study_data(), spec)

    names = [v.name for v in study.variants]
    assert Counter(names) == Counter(spec["variants"])
    assert len(set(names)) == len(names) == spec["family_size"]

    l1 = {}
    for v in study.variants:
        if v.hypothesis_id != "L1":
            continue
        pool, threshold = v.params["pool"], float(v.params["threshold"])
        assert l1_key(v.name, grids) == (pool, threshold)
        at_t = v.components["selection"] != LOOK_AHEAD
        assert (pool, threshold, at_t) not in l1, v.name
        l1[(pool, threshold, at_t)] = v
    assert set(l1) == {
        (pool, x, at_t) for pool, grid in grids.items() for x in grid for at_t in (False, True)
    }

    for (pool, threshold, at_t), v in l1.items():
        assert v.error is None, (v.name, v.error)
        if at_t:
            assert "look-ahead" not in v.components["selection"]
        metrics = {m.metric: m for m in v.metrics}
        rule, comparator = expected_returns(pool, threshold, at_t)
        got = metrics["rule_mean_return"].value, metrics["comparator_mean_return"].value
        assert got == (pytest.approx(rule, rel=1e-9), pytest.approx(comparator, rel=1e-9)), v.name
        difference = metrics["paired_return_difference"]
        assert difference.value == pytest.approx(rule - comparator, rel=1e-9, abs=1e-12), v.name
        assert difference.n == len(RACES)


def test_run_study_freezes_the_grid_family_first_and_records_every_variant(monkeypatch, tmp_path):
    use_config(monkeypatch, OWNER)
    registry = Registry(tmp_path / "registry" / "japan.jsonl", holdout_log=tmp_path / "h.jsonl")
    seen_at_read = []

    def read_development(minutes):
        lines = registry.path.read_text().splitlines() if registry.path.exists() else []
        seen_at_read.append([json.loads(line)["kind"] for line in lines])
        return study_data()

    monkeypatch.setattr(avoidance, "default_registry", lambda country: registry)
    monkeypatch.setattr(avoidance, "read_development", read_development)
    monkeypatch.setattr(
        avoidance, "store_builds", lambda files: {"jra_races": "processed/jra/races@synthetic"}
    )

    result = avoidance.run_study(tmp_path / "out")

    assert seen_at_read == [["freeze"]]  # frozen before any row was read, nothing scored yet
    entries = [json.loads(line) for line in registry.path.read_text().splitlines()]
    frozen = entries[0]["spec"]
    assert entries[0]["kind"] == "freeze"
    assert registry.frozen_spec_hash(result["family_id"]) == result["spec_hash"]
    assert frozen["rules"]["L1"]["thresholds"] == {
        "TRIO": [20.0, 50.0, 100.0],
        "QUINELLA": [5.0, 10.0, 20.0],
    }
    assert frozen["family_size"] == 18

    records = registry.records(result["family_id"])
    assert sorted(r.variant for r in records) == sorted(frozen["variants"])
    assert len(records) == result["records_appended"] == 18
    assert registry.family_size(result["family_id"]) == frozen["family_size"]
    l1 = Counter(
        (r.params["pool"], float(r.params["threshold"])) for r in records if r.hypothesis_id == "L1"
    )
    assert dict(l1) == {
        (pool, float(x)): 2
        for pool, grid in (("TRIO", [20, 50, 100]), ("QUINELLA", [5, 10, 20]))
        for x in grid
    }
