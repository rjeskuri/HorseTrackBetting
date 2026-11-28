"""The C2 member with every interaction block, as a ranked member row in a new frozen spec (ticket
#571, after #93; `studies.interactions`).

#93's family `japan-c2-interactions-final-win-v2` is frozen in the registry (spec sha256
2dde5583...) and holds its records; v1 `japan-c2-interactions-final-win` is superseded. The
all-blocks member is pre-registered in a spec of its own family, frozen before any outcome is
read, so neither of #93's frozen specs changes.

Contract pinned here:

- `planned()` (every variant a completed run records, and each of a stopped run's failed records)
  names the all-blocks member's ranked member row: its tau, its stage-2 meta (and the meeting-
  clustered bounds of it), its standalone ΔLL against the base, its ΔLL at the final price
  against the base's stage-2 meta and against the market-only meta, and its tail at the final
  price (`tail:stage2:<member>`, the leaderboard row). `option_id(member)` is its own option, no
  other member's.
- `run_study` freezes the launch's spec before it calls `compute_study`, under a family that is
  neither #93's v2 nor v1; the spec it passes to the compute is the frozen one, it names the
  member and its option id, and the launch returns that family and hash. A registry where v2 is
  already frozen keeps v2's frozen hash and gets no record in v2 (the launch neither refreezes
  nor appends to it). The launch's records are in its own family, each naming the frozen hash,
  and the member's tail and ΔLL records among them, scored; the report lists the run id of each.

Not pinned: the new family's name and what its spec says beyond naming the member and its option
id (its sections, whether it supersedes v2 or extends it), the hypothesis id of the member's
variants, how its level-1 fits are checkpointed or reused, and the committed report's text (the
reviewer reads docs/experiments/interactions-japan.md for the row of the real run, which is
local). The member is found by its content (`interactions_synth.all_blocks_member`), never its
name. `load_inputs` and `compute_study` are replaced (the store is external; the compute's
level-1 member is pinned in test_interactions.py); the store files are synthetic bytes; every
path is under `tmp_path`, and an autouse guard fails a test that changes the committed registry
summary or a holdout log.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from interactions_synth import all_blocks_member

from horsetrack.core.eval.registry import FamilyAlreadyFrozen, FoldMetric, default_registry
from horsetrack.countries.japan.studies import interactions
from horsetrack.countries.japan.studies.calibration import Study, Variant
from horsetrack.countries.japan.studies.meta import CLUSTERED

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_SUMMARY = REPO_ROOT / "docs" / "experiments" / "registry"
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"
DATA_DIR_VAR = "HORSETRACK_DATA_DIR"

V1 = "japan-c2-interactions-final-win"
V2 = "japan-c2-interactions-final-win-v2"
# A stand-in for v2's frozen spec in the temporary registry (the real one is local).
V2_STAND_IN = {"family_id": V2, "ticket": 93, "stand_in": "#93's frozen v2 spec"}
SHA = "57" * 20
FOLDS = "walk_forward:synthetic"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """The data directory is under tmp_path; no test changes the committed summary or a log."""
    monkeypatch.setenv(DATA_DIR_VAR, str(tmp_path / "data"))

    def snapshot() -> dict[Path, bytes]:
        files = [*COMMITTED_SUMMARY.rglob("*"), *REAL_COUNTRIES.glob("*/holdout_log.jsonl")]
        return {p: p.read_bytes() for p in files if p.is_file()}

    before = snapshot()
    yield
    assert snapshot() == before, "a test changed the committed registry summary or a holdout log"


def spec_hash(spec: dict) -> str:
    """The registry's spec hash, by hand: sha256 of the sorted, compact JSON."""
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def ranked_row(member: str) -> set[str]:
    """The variants of a member's ranked row, as #93's members have them."""
    return {
        f"tau:{member}",
        f"stage2:q:{member}",
        f"{CLUSTERED}:stage2:q:{member}",
        f"level1:{member}_vs_base",
        f"marginal:stage2:{member}_vs_base",
        f"marginal:stage2:{member}_vs_market_only",
        f"tail:stage2:{member}",
    }


def test_the_all_blocks_member_is_planned_as_a_ranked_member_row():
    """planned() names the all-blocks member's row in full, as it names each of #93's block
    members' (horse_x_jockey's checks the expectation's form), and its leaderboard option is its
    own."""
    member = all_blocks_member(interactions.MEMBERS)
    names = {name for _, name in interactions.planned()}
    assert ranked_row("horse_x_jockey") <= names, sorted(ranked_row("horse_x_jockey") - names)
    missing = ranked_row(member) - names
    assert not missing, sorted(missing)
    options = {m: interactions.option_id(m) for m in interactions.MEMBERS}
    assert list(options.values()).count(options[member]) == 1, options


# --------------------------------------------------------------------------- a launch


def fake_store(data: Path) -> None:
    for name in ("runners", "races", "dividends"):
        path = data / "processed" / "jra" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {name}, one build".encode())


def member_results(member: str) -> tuple[Study, pd.DataFrame]:
    """What a compute returns for the member's row: its ΔLLs with bounds and its tail."""
    comps = {"study": "studies/interactions.py", "member": member}
    variants = [
        Variant(
            "E93",
            f"level1:{member}_vs_base",
            FOLDS,
            comps,
            {},
            metrics=(FoldMetric("all_folds", "delta_ll", 0.0061, 0.0049, 0.0073, 480),),
        ),
        Variant(
            "E93",
            f"marginal:stage2:{member}_vs_base",
            FOLDS,
            comps,
            {},
            metrics=(FoldMetric("all_folds", "delta_ll", -0.00002, -0.00005, 0.00001, 480),),
        ),
        Variant(
            "E93",
            f"tail:stage2:{member}",
            FOLDS,
            comps | {"option": interactions.option_id(member)},
            {},
            metrics=(
                FoldMetric("all_folds", "return_per_100[claim_q[0.99,1]]", 82.4, 77.1, 87.9, 33),
            ),
        ),
    ]
    tables = {"fit_revisions": [{"member": member, "revision": SHA, "n": 3}]}
    export = pd.DataFrame({"race_id": ["jra:m00001"], "horse_id": ["h:01"], f"u_{member}": [0.25]})
    return Study(counts={"level1_races": 400}, tables=tables, variants=variants), export


def test_the_all_blocks_member_is_preregistered_in_a_new_frozen_family(tmp_path, monkeypatch):
    """A launch on a registry where #93's v2 is frozen: the spec the compute receives is frozen
    first, under a family other than v1 and v2, and names the member and its option id; v2's
    frozen hash is unchanged and v2 gets no record; the member's row is in the launch's family,
    scored, each record naming the frozen hash, and the report lists each record's run id."""
    member = all_blocks_member(interactions.MEMBERS)
    root = tmp_path.resolve()
    fake_store(root / "data")
    registry = default_registry("japan")
    v2_hash = registry.freeze_family(V2, V2_STAND_IN)
    seen: dict = {}

    def compute(inputs, spec, scheme, **kwargs):
        seen["hash"] = spec_hash(spec)
        seen["frozen"] = {
            e["family_id"]: e["spec_hash"]
            for e in map(json.loads, registry.path.read_text("utf-8").splitlines())
            if e.get("kind") == "freeze"
        }
        seen["spec"] = spec
        return member_results(member)

    monkeypatch.setattr(interactions, "load_inputs", lambda log=print: None)
    monkeypatch.setattr(interactions, "compute_study", compute)
    monkeypatch.setattr(interactions, "write_committed_summary", lambda registry, log=print: [])
    monkeypatch.setattr(interactions, "_git_sha", lambda: SHA)
    try:
        result = interactions.run_study(out_dir=root / "reports", log=lambda message: None)
    except FamilyAlreadyFrozen as exc:
        pytest.fail(f"the launch froze another spec under #93's family: {exc}")

    family = result["family_id"]
    assert family not in (V1, V2), family
    assert seen["frozen"].get(family) == seen["hash"], "the spec was not frozen before the compute"
    assert result["spec_hash"] == seen["hash"] == registry.frozen_spec_hash(family)
    text = json.dumps(seen["spec"], sort_keys=True)
    assert member in text and interactions.option_id(member) in text

    assert registry.frozen_spec_hash(V2) == v2_hash, "#93's frozen spec changed"
    assert registry.records(V2) == [], "the launch appended to #93's family"

    held = {r.variant: r for r in registry.records(family)}
    want = {f"level1:{member}_vs_base", f"marginal:stage2:{member}_vs_base"}
    want.add(f"tail:stage2:{member}")
    assert want <= set(held), sorted(held)
    for name in want:
        assert held[name].status == "scored", name
        assert held[name].params.get("spec_sha256") == seen["hash"], name
        assert held[name].family_id == family
    lines = registry.path.read_text("utf-8").splitlines()
    run_ids = {
        e["record"]["variant"]: e["run_id"]
        for e in map(json.loads, lines)
        if e.get("kind") == "run" and e["record"]["family_id"] == family
    }
    report = Path(result["report_path"]).read_text("utf-8")
    for name in want:
        assert run_ids[name][:16] in report, name
