"""Ticket #716: the member-filter study's run, on synthetic races only: every #713 rule x member x
k x checkpoint scored on the predicted final board and settled at the final dividends, its gain
over the rule alone, the registry records and the leaderboard rows.

Contract pinned here for `horsetrack.countries.japan.research.member_filter` (with the names of
`test_member_filter.py`):

- `run_study(*, races, predictions, provenance, nowcast, registry, out_dir, seed, n_boot)`:
  - `races` are #713's `predicted_final.RaceInput`s; `predictions` maps member id -> race id ->
    horse -> p_member (the member's win probability, taken as given). A race a member has no entry
    for is not covered by it. A member id that `MEMBERS` does not register raises ValueError before
    anything is appended or written (only pre-registered members are scored);
  - `provenance` maps member id -> its declaration, a plain dict {"fit_id": non-empty str,
    "out_of_fold": bool, "final_board_inputs": bool}. Every member with predictions needs one with
    out_of_fold True and final_board_inputs False (p built only from what is known at post - k);
    otherwise ValueError before a race is read, a run appended or a file written. Each record of a
    member's families carries its declaration as `params["member_provenance"]`;
  - refuses with `FamilyNotFrozen`, before it reads a race, unless every family of `FAMILIES` is
    frozen in `registry` with `family_spec(family_id)`; an `out_dir` inside the repository raises
    `LicensedOutputInRepo`; a refusal appends no run and writes no file;
  - drops every race on a holdout day (`core.eval.holdout.is_holdout_day`) and never calls
    `LockedHoldout.open`;
  - per race, checkpoint k and rule: projects `race.boards[k]` with #713's `project_board`, selects
    the rule's tickets on the projection (#713's `select`), and keeps the tickets that pass
    `qualifying` with p_member from `predictions` and q = `crowd_q` of the projected board; each
    ticket stakes 100 yen, settled at the race's final dividends (`paid`), as #713 settles them;
  - `.rows`: one dict per combination x checkpoint, every one (an unmeasured #713 rule, a member
    registered unavailable and a member with no predictions included, as unmeasured rows with a
    `note`), with combination_id, rule_id, member_id, k, checkpoint_min, family_id, selection
    (#713's `selection_label`), n_bets (races with a qualifying ticket), n_tickets, n_meetings,
    staked_yen, returned_yen, return_per_100 (None without a bet), ci_lo and ci_hi (the meeting
    bootstrap, as #713), and gain, gain_lo, gain_hi: `paired_gain` against the rule alone on the
    races the member covers where the rule bought (None without a qualifying ticket);
  - appends registry records (ticket 716, the families of `FAMILIES`, sources naming jravan, the
    nowcast's fit id recorded, each gain a metric) and writes the private report through
    `write_report` under `out_dir` (`.report_path`);
  - every combination x checkpoint is a row of the Japan leaderboard (`leaderboard_build.
    build_rows`) with rule the combination id, segment 'all', pool the rule's pool and selection
    `predicted_final@T-k`, measured from the run's return per 100 and meetings, withheld (D21),
    ranked with the rest (`leaderboard.rank`).

The synthetic nowcast shortens every Win price in [5, 10) by a fifth and leaves the rest. The four
development races and one holdout race are hand-built; each member's p is a hand-set ratio times
the crowd's q on the projected T-2 board, so every match, total and gain at T-2 below is a hand
value (checked against the plain reference here at import, which also shows that the projection,
the normalisation of q, the every-horse rule and the covered-race base each change the answer).
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import json
import math
import shutil
import uuid
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from horsetrack.core.eval import holdout, leaderboard
from horsetrack.core.eval.registry import Registry, _spec_hash
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.countries.japan.board.snapshot import DecisionGrid, RaceBoard
from horsetrack.countries.japan.models.v0.nowcast import DriftCell, NowcastV0
from horsetrack.countries.japan.research import predicted_final as pf
from horsetrack.countries.japan.research import member_filter as mf

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_VAR = "HORSETRACK_DATA_DIR"
CHECKPOINTS = (10, 5, 2)
K_GRID = (1.0, 1.25, 1.5, 2.0)
SEED, N_BOOT = 20261129, 100
FIT_ID = "synthetic-nowcast-716"

FAVW = "x435:offset:normalised:WIN:top1"  # Win on the projected favourite
BAND = "n1:WIN:band:5-10"  # Win on every runner projected in [5, 10)
WIDE = "n8:wide:1/2-3"  # Wide on the favourite with each of ranks 2 and 3
RULES = (FAVW, BAND, WIDE)
POOL_OF = {FAVW: "WIN", BAND: "WIN", WIDE: "QUINELLA_PLACE"}

# --------------------------------------------------------------------------- the nowcast

GRID = DecisionGrid(minutes=(60, 30, 16, 5, 2, 1), primary=2)
BANDS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
SHORTEN = math.log(0.8)  # WIN band 2 ([5, 10)) drifts by ln 0.8 at every tau; every other cell 0
POOLS = ("WIN", "PLACE", "QUINELLA_PLACE")


def make_nowcast() -> NowcastV0:
    cells = {
        (pool, tau, band): DriftCell(
            pool, tau, band, 40, SHORTEN if (pool, band) == ("WIN", 2) else 0.0, 0.3
        )
        for pool in POOLS
        for tau in GRID.minutes
        for band in range(len(BANDS) + 1)
    }
    sigma0 = {(pool, tau): 0.3 for pool in POOLS for tau in GRID.minutes}
    return NowcastV0(BANDS, GRID, 50.0, ("2025-05-31",), sigma0, cells, FIT_ID)


def ref_project(odds: float) -> float:
    shift = SHORTEN if bisect_right(BANDS, odds) == 2 else 0.0
    return round(max(1.0, math.exp(math.log(odds) + shift)), 2)


# --------------------------------------------------------------------------- the races

D1 = date(2025, 6, 7)  # before the holdout start: a development day
_AFTER = [date(2026, 11, 28) + timedelta(days=i) for i in range(60)]
D2 = next(d for d in _AFTER if not holdout.is_holdout_day("japan", d))
HELD = next(d for d in _AFTER if holdout.is_holdout_day("japan", d))

CARD = {"R1": (D1, "TOK", 1), "R2": (D1, "TOK", 2), "R3": (D1, "NAK", 5), "R4": (D2, "TOK", 3),
        "H": (HELD, "TOK", 1)}  # fmt: skip
DEVELOPMENT = ("R1", "R2", "R3", "R4")

OBSERVED = {
    "R1": {1: 3.0, 2: 6.0, 3: 7.5, 4: 12.0, 5: 20.0},
    "R2": {1: 2.5, 2: 4.0, 3: 8.0, 4: 9.0, 5: 30.0},
    "R3": {1: 1.8, 2: 5.5, 3: 6.5, 4: 15.0, 5: 40.0},
    "R4": {1: 2.2, 2: 3.5, 3: 9.5, 4: 11.0, 5: 25.0},
}
OBSERVED["H"] = OBSERVED["R1"]
R2_T10 = {1: 4.5, 2: 4.0, 3: 8.0, 4: 9.0, 5: 30.0}  # R2 at T-10 only: 2 is the favourite


def observed(name: str, k: int) -> dict[int, float]:
    return R2_T10 if (name, k) == ("R2", 10) else OBSERVED[name]


# The projected T-2 boards by hand (every price in [5, 10) times 0.8).
PROJECTED_T2 = {
    "R1": {1: 3.0, 2: 4.8, 3: 6.0, 4: 12.0, 5: 20.0},
    "R2": {1: 2.5, 2: 4.0, 3: 6.4, 4: 7.2, 5: 30.0},
    "R3": {1: 1.8, 2: 4.4, 3: 5.2, 4: 15.0, 5: 40.0},
    "R4": {1: 2.2, 2: 3.5, 3: 7.6, 4: 11.0, 5: 25.0},
}
PROJECTED_T2["H"] = PROJECTED_T2["R1"]

# (pool, combination, dividend per 100 yen): finishes R1 2-1-3, R2 4-1-2, R3 1-3-2, R4 3-2-1.
WIDE_POOL = "QUINELLA_PLACE"
PAID = {
    "R1": [("WIN", "2", 480), (WIDE_POOL, "1,2", 250), (WIDE_POOL, "1,3", 300),
           (WIDE_POOL, "2,3", 400)],
    "R2": [("WIN", "4", 900), (WIDE_POOL, "1,4", 350), (WIDE_POOL, "2,4", 600),
           (WIDE_POOL, "1,2", 150)],
    "R3": [("WIN", "1", 180), (WIDE_POOL, "1,3", 220), (WIDE_POOL, "1,2", 160),
           (WIDE_POOL, "2,3", 500)],
    "R4": [("WIN", "3", 950), (WIDE_POOL, "2,3", 700), (WIDE_POOL, "1,3", 330),
           (WIDE_POOL, "1,2", 200)],
}  # fmt: skip
# the holdout race: R1's boards, every R1 combination and the favourite's Win paying 99,990 yen
PAID["H"] = [(p, c, 99_990) for p, c, _ in PAID["R1"]] + [("WIN", "1", 99_990)]

# Each member's p_member / q at T-2 (q the crowd's on the projected T-2 board), by hand.
# A is #438's member and covers every race; B is #87's and covers R1 and R3 only, with no p for
# R3's horse 3. Both rate every runner of the holdout race 2.4 times the crowd.
RATIOS = {
    "A": {
        "R1": {1: 0.9, 2: 1.6, 3: 1.2, 4: 1.1, 5: 0.8},
        "R2": {1: 1.1, 2: 0.7, 3: 0.9, 4: 2.2, 5: 0.95},
        "R3": {1: 1.3, 2: 0.8, 3: 1.4, 4: 0.6, 5: 1.2},
        "R4": {1: 0.8, 2: 1.2, 3: 1.9, 4: 0.9, 5: 0.7},
        "H": {h: 2.4 for h in range(1, 6)},
    },
    "B": {
        "R1": {1: 1.4, 2: 1.1, 3: 2.1, 4: 0.5, 5: 0.5},
        "R3": {1: 1.6, 2: 1.3, 4: 0.9, 5: 1.1},
        "H": {h: 2.4 for h in range(1, 6)},
    },
}
MEMBER_TICKET = {"A": 438, "B": 87, "C": 92, "J": 715}  # C: no predictions; J: unavailable


def member(label: str) -> str:
    return sorted(m for m, v in mf.MEMBERS.items() if v.ticket == MEMBER_TICKET[label])[0]


def race_id(name: str) -> str:
    day, venue, no = CARD[name]
    return f"jra:{day.isoformat()}:{venue}:{no:02d}"


def meeting_id(name: str) -> str:
    day, venue, _ = CARD[name]
    return f"{day.isoformat()}:{venue}"


def post_of(name: str) -> datetime:
    day, _, no = CARD[name]
    return datetime(day.year, day.month, day.day, 1, 0, tzinfo=timezone.utc) + timedelta(
        minutes=30 * no
    )


def make_board(name: str, k: int) -> RaceBoard:
    post = post_of(name)
    at = post - timedelta(minutes=k)
    return RaceBoard(
        race_id=race_id(name),
        post_utc=post,
        at_utc=at,
        capture_utc=at,
        final=k == 0,
        win=dict(observed(name, k)),
        place={},
        wide={},
        totals={"WIN": 12_000_000.0},
        scratched=frozenset(),
    )


def make_race(name: str) -> pf.RaceInput:
    day, _, _ = CARD[name]
    rows = PAID[name]
    paid = pd.DataFrame(
        {
            "race_id": [race_id(name)] * len(rows),
            "pool": [p for p, _, _ in rows],
            "combination": [c for _, c, _ in rows],
            "dividend": [float(d) for _, _, d in rows],
            "is_refund": [False] * len(rows),
        }
    )
    return pf.RaceInput(
        race_id=race_id(name),
        race_date=day,
        meeting_id=meeting_id(name),
        race_class="MAIDEN",
        surface="turf",
        boards={k: make_board(name, k) for k in CHECKPOINTS},
        final=make_board(name, 2),
        paid=paid,
        brackets={h: h for h in OBSERVED[name]},
    )


def all_races() -> list:
    return [make_race(name) for name in ("R1", "R2", "H", "R3", "R4")]


# --------------------------------------------------------------------------- the reference


def ref_q(win: dict[int, float], *, normalise: bool = True) -> dict[int, float]:
    inv = {h: 1.0 / o for h, o in win.items()}
    total = sum(inv.values()) if normalise else 1.0
    return {h: v / total for h, v in inv.items()}


P_MEMBER = {
    label: {
        name: {h: r * ref_q(PROJECTED_T2[name])[h] for h, r in ratios.items()}
        for name, ratios in by_race.items()
    }
    for label, by_race in RATIOS.items()
}


def predictions() -> dict:
    return {
        member(label): {race_id(name): dict(p) for name, p in by_race.items()}
        for label, by_race in P_MEMBER.items()
    }


def ref_select(rule: str, win: dict[int, float]) -> set[tuple[int, ...]]:
    ranked = sorted(win, key=lambda h: (win[h], h))
    if rule == FAVW:
        return {(ranked[0],)}
    if rule == BAND:
        return {(h,) for h, o in win.items() if 5.0 <= o < 10.0}
    if rule == WIDE:
        return {tuple(sorted((ranked[0], h))) for h in ranked[1:3]}
    raise KeyError(rule)


def ref_return(name: str, pool: str, tickets) -> float:
    paying = {(p, c): d for p, c, d in PAID[name]}
    return float(sum(paying.get((pool, ",".join(map(str, t))), 0.0) for t in tickets))


def ref_totals(rule, label, k, ck, *, board="projected", normalise=True, legs=all, base="covered"):
    """(n_bets, n_tickets, n_meetings, staked, returned, gain) of rule x member x k at T-ck."""
    pool = POOL_OF[rule]
    bets = tickets = 0
    staked = returned = base_staked = base_returned = 0.0
    meetings = set()
    for name in DEVELOPMENT:
        projected = {h: ref_project(o) for h, o in observed(name, ck).items()}
        picks = ref_select(rule, projected)
        p = P_MEMBER[label].get(name)
        if p is None and base == "covered":
            continue
        base_staked += 100.0 * len(picks)
        base_returned += ref_return(name, pool, picks)
        if p is None:
            continue
        q = ref_q(projected if board == "projected" else observed(name, ck), normalise=normalise)
        kept = {t for t in picks if legs(h in p and p[h] / q[h] >= k for h in t)}
        if kept:
            bets, tickets = bets + 1, tickets + len(kept)
            staked += 100.0 * len(kept)
            returned += ref_return(name, pool, kept)
            meetings.add(meeting_id(name))
    gain = None
    if staked > 0:
        gain = 100.0 * returned / staked - 100.0 * base_returned / base_staked
    return (bets, tickets, len(meetings), staked, returned, gain)


# --------------------------------------------------------------------------- hand values (T-2)

# (rule, member, k) -> (n_bets, n_tickets, n_meetings, staked, returned, gain) at T-2.
# A's rule bases at T-2: FAVW 180 / 400 = 45; BAND 1850 / 500 = 370; WIDE 1610 / 800 = 201.25.
# B's (R1 and R3 only): FAVW 180 / 200 = 90; BAND 0 / 200 = 0; WIDE 930 / 400 = 232.5.
NONE = (0, 0, 0, 0.0, 0.0, None)
HAND = {
    (FAVW, "A", 1.0): (2, 2, 2, 200.0, 180.0, 90.0 - 45.0),  # R2 (1.1) and R3 (1.3)
    (FAVW, "A", 1.25): (1, 1, 1, 100.0, 180.0, 180.0 - 45.0),
    (FAVW, "A", 1.5): NONE,
    (FAVW, "A", 2.0): NONE,
    (FAVW, "B", 1.0): (2, 2, 2, 200.0, 180.0, 0.0),
    (FAVW, "B", 1.25): (2, 2, 2, 200.0, 180.0, 0.0),
    (FAVW, "B", 1.5): (1, 1, 1, 100.0, 180.0, 180.0 - 90.0),
    (FAVW, "B", 2.0): NONE,
    (BAND, "A", 1.0): (4, 4, 3, 400.0, 1850.0, 462.5 - 370.0),
    (BAND, "A", 1.25): (3, 3, 3, 300.0, 1850.0, 185000.0 / 300 - 370.0),
    (BAND, "A", 1.5): (2, 2, 2, 200.0, 1850.0, 925.0 - 370.0),
    (BAND, "A", 2.0): (1, 1, 1, 100.0, 900.0, 900.0 - 370.0),
    **{(BAND, "B", k): (1, 1, 1, 100.0, 0.0, 0.0) for k in K_GRID},  # R3's horse 3 has no p
    (WIDE, "A", 1.0): (1, 1, 1, 100.0, 220.0, 220.0 - 201.25),  # R3's 1-3 only
    (WIDE, "A", 1.25): (1, 1, 1, 100.0, 220.0, 220.0 - 201.25),
    (WIDE, "A", 1.5): NONE,
    (WIDE, "A", 2.0): NONE,
    (WIDE, "B", 1.0): (2, 3, 2, 300.0, 710.0, 71000.0 / 300 - 232.5),
    (WIDE, "B", 1.25): (2, 2, 2, 200.0, 460.0, 230.0 - 232.5),
    (WIDE, "B", 1.5): NONE,
    (WIDE, "B", 2.0): NONE,
}


def _same(a, b) -> bool:
    return all(
        (x is None and y is None)
        or (x is not None and y is not None and math.isclose(x, y, abs_tol=1e-9))
        for x, y in zip(a, b, strict=True)
    )


def _check_tables() -> None:
    for name, win in PROJECTED_T2.items():
        assert {h: ref_project(o) for h, o in OBSERVED[name].items()} == win, name
    for key, expected in HAND.items():
        assert _same(ref_totals(*key, 2), expected), (key, ref_totals(*key, 2))
    # each reading the study pins changes some T-2 value
    variants = {
        "q on the observed board": {"board": "observed"},
        "q not normalised": {"normalise": False},
        "any horse of the ticket": {"legs": any},
        "the rule's base over every race": {"base": "all"},
    }
    for what, options in variants.items():
        assert any(not _same(ref_totals(*key, 2, **options), v) for key, v in HAND.items()), what
    # the checkpoint matters: R2's T-10 board moves the favourite
    assert any(
        not _same(ref_totals(r, m, k, 10), ref_totals(r, m, k, 2))
        for r in RULES
        for m in ("A", "B")
        for k in K_GRID
    )
    # k and the member each move a value
    assert len({ref_totals(BAND, "A", k, 2) for k in K_GRID}) == len(K_GRID)
    assert ref_totals(FAVW, "A", 1.0, 2) != ref_totals(FAVW, "B", 1.0, 2)


_check_tables()


def _check_projection() -> None:
    """The fixture's projection is #713's (the existing projector), not only this file's."""
    for name in DEVELOPMENT:
        for k in CHECKPOINTS:
            got = pf.project_board(make_board(name, k), nowcast=make_nowcast(), checkpoint_min=k)
            assert got.win == pytest.approx(
                {h: ref_project(o) for h, o in observed(name, k).items()}
            )


_check_projection()


# --------------------------------------------------------------------------- the run


class Forbidden(AssertionError):
    pass


def temp_registry(root: Path) -> Registry:
    return Registry(root / "registry" / "japan.jsonl", holdout_log=root / "holdout_log.jsonl")


def frozen_registry(root: Path) -> Registry:
    registry = temp_registry(root)
    for family_id in mf.FAMILIES:
        registry.freeze_family(family_id, mf.family_spec(family_id))
    return registry


FIT_OF = {"A": "oof-438-synthetic-a1", "B": "oof-87-synthetic-b7"}
DROP = object()  # a declaration field left out


def declaration(label: str, **changes) -> dict:
    out = {"fit_id": FIT_OF[label], "out_of_fold": True, "final_board_inputs": False, **changes}
    return {k: v for k, v in out.items() if v is not DROP}


def provenances() -> dict:
    return {member(label): declaration(label) for label in P_MEMBER}


def run(mp, root: Path, registry: Registry, out_dir: Path, races=None, preds=None, prov=None):
    mp.setenv(DATA_VAR, str(root / "data"))
    return mf.run_study(
        races=all_races() if races is None else races,
        predictions=predictions() if preds is None else preds,
        provenance=provenances() if prov is None else prov,
        nowcast=make_nowcast(),
        registry=registry,
        out_dir=out_dir,
        seed=SEED,
        n_boot=N_BOOT,
    )


@dataclass
class StudyRun:
    result: object
    registry: Registry
    root: Path
    out_dir: Path
    opened: list


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> StudyRun:
    root = tmp_path_factory.mktemp("member-filter").resolve()
    registry = frozen_registry(root)
    out_dir = root / "data" / "reports"
    opened: list = []

    def no_open(*args, **kwargs):
        opened.append(args)
        raise Forbidden("LockedHoldout.open was called: the study reads development days only")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(holdout.LockedHoldout, "open", no_open)
        result = run(mp, root, registry, out_dir)
    return StudyRun(result, registry, root, out_dir, opened)


@pytest.fixture(scope="module")
def rows(study) -> dict:
    out = {}
    for row in study.result.rows:
        key = (row["rule_id"], row["member_id"], float(row["k"]), int(row["checkpoint_min"]))
        assert key not in out, f"{key} is two rows"
        out[key] = row
    return out


def test_every_combination_and_checkpoint_is_one_row(rows):
    expected = {
        (c.rule_id, c.member_id, float(c.k), ck)
        for c in mf.COMBINATIONS.values()
        for ck in CHECKPOINTS
    }
    assert set(rows) == expected
    family_of = {c: f for f, members in mf.FAMILIES.items() for c in members}
    for (rule, m, k, ck), row in rows.items():
        cid = mf.combination_id(rule, m, k)
        assert row["combination_id"] == cid
        assert row["family_id"] == family_of[cid]
        assert row["selection"] == f"predicted_final@T-{ck}"


@pytest.mark.parametrize(("rule", "label", "k"), sorted(HAND))
def test_the_filtered_rule_at_t2_equals_the_hand_values(rows, rule, label, k):
    """The hand totals hold the development races only: the holdout race (every runner 2.4 times
    the crowd, every dividend 99,990 yen) would move each measured one."""
    row = rows[(rule, member(label), k, 2)]
    n_bets, n_tickets, n_meetings, staked, returned, gain = HAND[(rule, label, k)]
    assert row["n_bets"] == n_bets
    assert row["n_tickets"] == n_tickets
    assert row["n_meetings"] == n_meetings
    assert row["staked_yen"] == pytest.approx(staked)
    assert row["returned_yen"] == pytest.approx(returned)
    if n_bets:
        assert row["return_per_100"] == pytest.approx(100.0 * returned / staked, rel=1e-9)
        assert row["gain"] == pytest.approx(gain, abs=1e-9)
    else:
        assert row["return_per_100"] is None
        assert row["gain"] is None
        assert row["ci_lo"] is None and row["gain_lo"] is None and row["gain_hi"] is None


@pytest.mark.parametrize("ck", CHECKPOINTS)
def test_every_checkpoint_matches_the_reference(rows, ck):
    for rule in RULES:
        for label in ("A", "B"):
            for k in K_GRID:
                row = rows[(rule, member(label), k, ck)]
                expected = ref_totals(rule, label, k, ck)
                got = (
                    row["n_bets"],
                    row["n_tickets"],
                    row["n_meetings"],
                    row["staked_yen"],
                    row["returned_yen"],
                    row["gain"],
                )
                assert _same(got, expected), (rule, label, k, ck, got, expected)


def test_the_bounds_follow_the_meetings(rows):
    checked = 0
    for row in rows.values():
        if row["return_per_100"] is None:
            continue
        if row["n_meetings"] >= 2:
            lo, hi, value = row["ci_lo"], row["ci_hi"], row["return_per_100"]
            assert lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi)
            assert lo <= value + 1e-9 <= hi + 2e-9, (row["combination_id"], lo, value, hi)
            assert row["gain_lo"] is not None and row["gain_hi"] is not None
            assert math.isfinite(row["gain_lo"]) and row["gain_lo"] <= row["gain_hi"]
            checked += 1
        else:
            assert row["gain_lo"] is None and row["gain_hi"] is None, row["combination_id"]
    assert checked >= 10


def test_unavailable_members_and_unmeasured_rules_are_unmeasured_rows(rows):
    """#715's member is registered unavailable; #92's has no predictions in this run; a #713 rule
    no projected board reads stays unmeasured under every member."""
    unmeasured_rule = next(r for r, rule in pf.RULES.items() if rule.unmeasured)
    picked = [key for key in rows if key[1] in (member("C"), member("J"))]
    picked += [key for key in rows if key[0] == unmeasured_rule]
    assert len(picked) > 2 * len(pf.RULES)
    for key in picked:
        row = rows[key]
        assert row["return_per_100"] is None and row["gain"] is None, key
        assert row["n_bets"] == 0, key
        assert isinstance(row["note"], str) and row["note"].strip(), key


def test_holdout_days_are_never_scored_nor_opened(study, rows):
    assert study.opened == []
    assert holdout.is_holdout_day("japan", HELD) and not holdout.is_holdout_day("japan", D2)
    # the holdout race would qualify every A ticket of the favourite rule at k = 2.0
    assert rows[(FAVW, member("A"), 2.0, 2)]["n_bets"] == 0
    assert rows[(BAND, member("A"), 2.0, 2)]["returned_yen"] == pytest.approx(900.0)


def test_an_unregistered_member_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    registry = frozen_registry(tmp_path)
    out_dir = tmp_path / "data" / "reports"
    preds = {**predictions(), "not-a-registered-member": {race_id("R1"): {1: 0.5, 2: 0.5}}}
    with pytest.raises(ValueError):
        run(monkeypatch, tmp_path, registry, out_dir, preds=preds)
    assert registry.records() == []
    assert not out_dir.exists() or not any(p.is_file() for p in out_dir.rglob("*"))


# --------------------------------------------------------------------------- records and rows


def test_the_run_is_recorded_in_the_registry(study):
    records = study.registry.records()
    assert records, "the run appended no record"
    assert {r.family_id for r in records} == set(mf.FAMILIES), "every family is recorded"
    for r in records:
        assert r.ticket == 716
        assert r.holdout_open is None
        assert "jravan" in {s.lower().replace("-", "") for s in r.sources}
        provenance = json.dumps(
            {"c": r.components, "p": r.params, "b": r.data_builds}, default=str, sort_keys=True
        )
        assert FIT_ID in provenance, "the nowcast's fit id is on the record"
    for family_id in mf.FAMILIES:
        assert study.registry.frozen_spec_hash(family_id) == _spec_hash(mf.family_spec(family_id))
    assert study.result.run_ids
    # the paired gain is recorded, not only reported
    cid = mf.combination_id(BAND, member("A"), 1.0)
    gains = [
        m.value
        for r in records
        for m in r.metrics
        if cid in m.metric and "gain" in m.metric.lower() and m.value is not None
    ]
    assert any(math.isclose(v, 462.5 - 370.0, abs_tol=1e-9) for v in gains), gains


def test_every_combination_and_checkpoint_is_a_ranked_leaderboard_row(study):
    from horsetrack.countries.japan.research import leaderboard_build

    candidate_dir = study.out_dir / "candidate-cells"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = study.root / "depth"
    depth_dir.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_VAR, str(study.root / "data"))
        built = leaderboard_build.build_rows(
            registry=study.registry, candidate_dir=candidate_dir, depth_dir=depth_dir
        )
    index: dict = {}
    for r in built:
        if r.family_id in mf.FAMILIES and r.segment == "all":
            key = (r.rule, r.selection)
            assert key not in index, f"{key} is two rows"
            index[key] = r
    expected = {(cid, f"predicted_final@T-{ck}") for cid in mf.COMBINATIONS for ck in CHECKPOINTS}
    assert set(index) == expected
    ranked = {rr.row.option_id: rr.rank for rr in leaderboard.rank(leaderboard.shrink(built))}
    for (rule, label, k), (n_bets, _, n_meetings, staked, returned, _) in HAND.items():
        row = index[(mf.combination_id(rule, member(label), k), "predicted_final@T-2")]
        assert row.pool == POOL_OF[rule]
        assert row.withheld, "built on JRA-VAN: withheld from the public leaderboard (D21)"
        if n_bets:
            assert row.return_per_100 == pytest.approx(100.0 * returned / staked, rel=1e-9)
            assert row.n_meetings == n_meetings
            assert ranked[row.option_id] is not None, "a measured combination is ranked"
        else:
            assert row.return_per_100 is None, "a combination with no bet is an unmeasured row"
    for k in K_GRID:
        row = index[(mf.combination_id(BAND, member("J"), k), "predicted_final@T-2")]
        assert row.return_per_100 is None and ranked[row.option_id] is None


def test_the_report_is_private_and_under_out_dir(study):
    path = Path(study.result.report_path).resolve()
    assert path.is_file() and path.stat().st_size > 0
    assert path.is_relative_to(study.out_dir.resolve())
    assert not path.is_relative_to(REPO_ROOT)


# --------------------------------------------------------------------------- refusals


def test_the_run_refuses_before_reading_a_race_when_a_family_is_not_frozen(tmp_path, monkeypatch):
    registry = temp_registry(tmp_path)
    families = list(mf.FAMILIES)
    for family_id in families[1:]:  # every family but the first
        registry.freeze_family(family_id, mf.family_spec(family_id))
    read = []

    def races():
        read.append(True)
        yield from all_races()

    out_dir = tmp_path / "data" / "reports"
    with pytest.raises(mf.FamilyNotFrozen):
        run(monkeypatch, tmp_path, registry, out_dir, races=races())
    assert read == [], "a race was read before the freeze was checked"
    assert registry.records() == []
    assert not out_dir.exists() or not any(p.is_file() for p in out_dir.rglob("*"))


def test_the_run_refuses_a_family_frozen_with_another_spec(tmp_path, monkeypatch):
    registry = temp_registry(tmp_path)
    families = list(mf.FAMILIES)
    registry.freeze_family(families[0], {"not": "the spec", "ticket": 716})
    for family_id in families[1:]:
        registry.freeze_family(family_id, mf.family_spec(family_id))
    read = []

    def races():
        read.append(True)
        yield from all_races()

    with pytest.raises(mf.FamilyNotFrozen):
        run(monkeypatch, tmp_path, registry, tmp_path / "data" / "reports", races=races())
    assert read == []
    assert registry.records() == []


def test_an_out_dir_inside_the_repository_is_refused(tmp_path, monkeypatch):
    inside = REPO_ROOT / f".member-filter-716-{uuid.uuid4().hex[:8]}"
    registry = frozen_registry(tmp_path)
    try:
        with pytest.raises(LicensedOutputInRepo):
            run(monkeypatch, tmp_path, registry, inside)
        assert not inside.exists() or not any(p.is_file() for p in inside.rglob("*"))
        assert registry.records() == []
    finally:
        shutil.rmtree(inside, ignore_errors=True)


# --------------------------------------------------------------------------- member provenance


def test_each_record_carries_its_members_provenance(study):
    """A's and B's declarations differ in their fit ids: each family's records carry their own
    member's, so the registry says which out-of-fold fit each scored p came from."""
    declared = provenances()
    seen = set()
    for r in study.registry.records():
        member_id = r.params["member_id"]
        if member_id in declared:
            assert r.params.get("member_provenance") == declared[member_id], r.family_id
            seen.add(member_id)
    assert seen == set(declared)


BAD_DECLARATIONS = {
    "in-sample": {"out_of_fold": False},
    "final-board inputs": {"final_board_inputs": True},
    "out_of_fold not declared": {"out_of_fold": DROP},
    "final_board_inputs not declared": {"final_board_inputs": DROP},
    "out_of_fold not a bool": {"out_of_fold": None},
    "final_board_inputs not a bool": {"final_board_inputs": None},
    "no fit id": {"fit_id": DROP},
    "an empty fit id": {"fit_id": ""},
}


def _refused(monkeypatch, tmp_path, prov) -> None:
    registry = frozen_registry(tmp_path)
    out_dir = tmp_path / "data" / "reports"
    read = []

    def races():
        read.append(True)
        yield from all_races()

    with pytest.raises(ValueError):
        run(monkeypatch, tmp_path, registry, out_dir, races=races(), prov=prov)
    assert read == [], "a race was read before the provenance was checked"
    assert registry.records() == []
    assert not out_dir.exists() or not any(p.is_file() for p in out_dir.rglob("*"))


@pytest.mark.parametrize("label", ["A", "B"])
def test_a_member_with_predictions_and_no_provenance_is_refused(tmp_path, monkeypatch, label):
    prov = {m: d for m, d in provenances().items() if m != member(label)}
    _refused(monkeypatch, tmp_path, prov)


def test_an_empty_provenance_is_refused_when_members_have_predictions(tmp_path, monkeypatch):
    _refused(monkeypatch, tmp_path, {})


@pytest.mark.parametrize("label", ["A", "B"])
@pytest.mark.parametrize("what", sorted(BAD_DECLARATIONS))
def test_a_member_declared_in_sample_or_final_board_informed_is_refused(
    tmp_path, monkeypatch, label, what
):
    prov = {**provenances(), member(label): declaration(label, **BAD_DECLARATIONS[what])}
    _refused(monkeypatch, tmp_path, prov)


def test_only_the_members_with_predictions_need_a_declaration(tmp_path, monkeypatch):
    """A alone has predictions and a declaration: the run scores it and records A's."""
    registry = frozen_registry(tmp_path)
    result = run(
        monkeypatch,
        tmp_path,
        registry,
        tmp_path / "data" / "reports",
        races=[make_race("R3")],
        preds={member("A"): predictions()[member("A")]},
        prov={member("A"): declaration("A")},
    )
    assert result.rows
    recorded = [r for r in registry.records() if r.params["member_id"] == member("A")]
    assert recorded
    assert all(r.params.get("member_provenance") == declaration("A") for r in recorded)
