"""Ticket #713: the rule families re-measured on the nowcast's predicted final board at T-10, T-5
and T-2 (pre-registered), on synthetic boards only.

Contract pinned here for `horsetrack.countries.japan.research.predicted_final` (the coder
implements what these tests call):

- `CHECKPOINTS == (10, 5, 2)`: minutes before the post (the owner's T-10 fill, T-5 narrow, T-2
  queue). `selection_label(k) == f"predicted_final@T-{k}"`.
- `RULES`: rule id -> rule, each with `.ticket` (the source study's ticket as #713 names it: #503
  N8 spreads, #435 cross-pool T-2, #82 L-rules, #131 E2/E15 and exacta cuts, #430 N1 segment
  cells; all five appear) and `.pool` (the pool it buys). Every #503 spread of
  `spread_scan.SPREADS` is the rule `n8:<spread id>`; #430's N1 cells are `n1:<POOL>:<cell id>`
  with #430's cell ids, among them `n1:WIN:band:5-10` (every Win runner priced in [5, 10)) and
  `n1:WIN:fav:le1.9` (every Win runner of a race whose favourite is priced under 2.0).
  `FAMILIES`: family id -> rule ids, a partition of `RULES` (the registry families);
  `family_spec(family_id)`, the plain-JSON spec each family is frozen with, naming its rules.
- `project_board(board, *, nowcast, checkpoint_min) -> RaceBoard`: the board at post - k projected
  to the final with the shadow page's ladder projector (`board.shadow_feed._projector`): each WIN,
  PLACE (low figure) and Wide (low figure) price O becomes round(max(1.0, exp(ln O + mu(k, b))), 2),
  b = `band_of(O, nowcast.bands)`, mu the nowcast cell mean at tau k, linear in tau between the
  neighbouring grid taus (k = 10 lies between 5 and 16 on the decision grid). Same keys as `board`.
- `RaceInput(race_id, race_date, meeting_id, race_class, surface, boards, final, paid, brackets)`:
  `boards` maps each checkpoint to the observed `RaceBoard` at post - k, `final` is the final
  board, `paid` the race's `dividends` rows (race_id, pool, combination, dividend per 100 yen,
  is_refund), `brackets` horse -> bracket.
- `select(rule_id, board, *, race) -> TicketSet | None`: the rule's tickets read on `board` (the
  projected board; `race` gives the card), None or an empty set when it buys nothing.
- `pick_stability(picks) -> {5: share, 2: share}`: `picks` maps each checkpoint to a set of picks;
  the share of the T-10 picks still picked at T-5 and at T-2; None when T-10 picked nothing.
- `accuracy_table(examples) -> DataFrame`: examples with columns checkpoint_min, pool, band,
  odds_t, odds_projected, odds_final; one row per (checkpoint_min, pool, band) with n, log_error
  (mean |ln projected - ln final|) and log_error_rw (the random walk: mean |ln odds_t - ln final|).
- `run_study(*, races, nowcast, registry, out_dir, seed, n_boot)`:
  - refuses with `FamilyNotFrozen`, before it reads a race, unless every family of `FAMILIES` is
    frozen in `registry` with `family_spec(family_id)`; a refusal appends no run and writes no file;
  - drops every race on a holdout day (`core.eval.holdout.is_holdout_day`) and never calls
    `LockedHoldout.open`;
  - for each race, checkpoint k and rule: projects `race.boards[k]` and selects on the projection;
    each ticket stakes 100 yen and is settled at the race's real final dividends (`paid`);
  - `.rows`: one dict per (rule, checkpoint), every rule including those with no bet, with
    rule_id, family_id, checkpoint_min, selection (`selection_label`), n_bets (races bought),
    n_tickets, n_meetings, staked_yen, returned_yen, return_per_100 (100 x returned / staked; None
    without a bet), ci_lo and ci_hi (a meeting bootstrap, per 100 yen);
  - `.stability`: rule id -> `pick_stability` of its tickets, a pick being (race, combination);
  - `.accuracy`: `accuracy_table` of every WIN, PLACE and QUINELLA_PLACE combination priced on both
    the T-k board and the final board (low figures; band of the T-k price on the nowcast's bands);
  - appends registry records (ticket 713, families of `FAMILIES`, sources naming jravan, the
    nowcast's fit id recorded) and writes the private report through `write_report` under
    `out_dir` (`.report_path`); an `out_dir` inside the repository raises `LicensedOutputInRepo`;
  - every rule x checkpoint is a row of the Japan leaderboard (`leaderboard_build.build_rows`)
    with rule the rule id, segment 'all' and selection `selection_label(k)`, a rule with no bet
    included as an unmeasured row.

The synthetic nowcast has hand-set drift cells; the three development races and one holdout race
are hand-built boards, and every expected selection, total, stability share and log error below is
computed by hand (the tables are checked against an independent reference here at import).
"""

# Ruff reads the module under test as third-party until it exists, then as first-party.
from __future__ import annotations  # noqa: I001

import itertools
import json
import math
import shutil
import uuid
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from horsetrack.core.eval import holdout
from horsetrack.core.eval.registry import Registry, _spec_hash
from horsetrack.core.eval.reports import LicensedOutputInRepo
from horsetrack.countries.japan.board.snapshot import DecisionGrid, RaceBoard
from horsetrack.countries.japan.models.v0.nowcast import DriftCell, NowcastV0
from horsetrack.countries.japan.research import spread_scan
from horsetrack.countries.japan.research import predicted_final as pf

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_VAR = "HORSETRACK_DATA_DIR"
CHECKPOINTS = (10, 5, 2)
SEED, N_BOOT = 20261129, 100
FIT_ID = "synthetic-nowcast-713"

EXACTA = "n8:exacta:2-4/1"  # ranks 2-4 first, the favourite second
BAND = "n1:WIN:band:5-10"  # every Win runner priced in [5, 10)
FAV = "n1:WIN:fav:le1.9"  # every Win runner when the favourite is under 2.0
NO_BET = "n8:trifecta:1/2-4/2-7"  # needs seven ranks; every synthetic field has six
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"


# --------------------------------------------------------------------------- the nowcast

GRID = DecisionGrid(minutes=(60, 30, 16, 5, 2, 1), primary=2)
BANDS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)  # band = cut points at or below the odds: 0..6
BASE = {1: 0.02, 2: 0.05, 5: 0.10, 16: 0.32, 30: 0.40, 60: 0.50}
# mu(pool, tau, band) = DIRECTION[pool][band] * BASE[tau]: WIN bands 1 (2-5) lengthen and band 2
# (5-10) shortens, so a 4.8 and a 5.2 swap places on the projection.
DIRECTION = {
    WIN: (0.5, 1.0, -1.0, 0.5, 1.0, 0.0, 0.0),
    PLACE: (-0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
    WIDE: (0.5,) * 7,
}


def make_nowcast() -> NowcastV0:
    cells = {
        (pool, tau, band): DriftCell(pool, tau, band, 40, DIRECTION[pool][band] * BASE[tau], 0.3)
        for pool in DIRECTION
        for tau in GRID.minutes
        for band in range(len(BANDS) + 1)
    }
    sigma0 = {(pool, tau): 0.3 for pool in DIRECTION for tau in GRID.minutes}
    return NowcastV0(BANDS, GRID, 50.0, ("2025-05-31",), sigma0, cells, FIT_ID)


def ref_mu(pool: str, k: int, band: int) -> float:
    """The ladder's drift: the grid cell's mean at k, linear in tau between grid neighbours."""
    if k in BASE:
        base = BASE[k]
    else:
        taus = sorted(BASE)
        hi = next(t for t in taus if t > k)
        lo = max(t for t in taus if t < k)
        base = BASE[lo] + (k - lo) / (hi - lo) * (BASE[hi] - BASE[lo])
    return DIRECTION[pool][band] * base


def ref_band(odds: float) -> int:
    return bisect_right(BANDS, odds)


def ref_project(pool: str, odds: float, k: int) -> float:
    return round(max(1.0, math.exp(math.log(odds) + ref_mu(pool, k, ref_band(odds)))), 2)


# --------------------------------------------------------------------------- the races

D1 = date(2025, 6, 7)  # before the holdout start: a development day
_AFTER = [date(2026, 11, 28) + timedelta(days=i) for i in range(60)]
D2 = next(d for d in _AFTER if not holdout.is_holdout_day("japan", d))  # development, after start
HELD = next(d for d in _AFTER if holdout.is_holdout_day("japan", d))

WIN_BOARDS = {
    "R1": {
        10: {1: 4.6, 2: 5.1, 3: 6.0, 4: 9.0, 5: 12.0, 6: 30.0},
        5: {1: 4.4, 2: 5.3, 3: 6.2, 4: 8.5, 5: 12.0, 6: 32.0},
        2: {1: 4.5, 2: 5.0, 3: 5.5, 4: 7.5, 5: 13.0, 6: 34.0},
        0: {1: 5.8, 2: 4.0, 3: 5.0, 4: 7.0, 5: 14.0, 6: 35.0},  # the final board
    },
    "R2": {
        10: {1: 1.8, 2: 4.0, 3: 7.0, 4: 15.0, 5: 25.0, 6: 60.0},
        5: {1: 1.85, 2: 3.8, 3: 10.5, 4: 15.0, 5: 25.0, 6: 60.0},
        2: {1: 2.1, 2: 3.5, 3: 7.0, 4: 14.0, 5: 25.0, 6: 55.0},
        0: {1: 2.0, 2: 3.6, 3: 6.8, 4: 14.0, 5: 26.0, 6: 50.0},
    },
    "R3": {
        10: {1: 1.85, 2: 3.2, 3: 9.5, 4: 11.0, 5: 20.0, 6: 45.0},
        5: {1: 1.8, 2: 3.4, 3: 10.2, 4: 10.5, 5: 20.0, 6: 45.0},
        2: {1: 1.7, 2: 3.5, 3: 10.5, 4: 9.8, 5: 21.0, 6: 50.0},
        0: {1: 1.6, 2: 3.9, 3: 10.0, 4: 9.0, 5: 22.0, 6: 48.0},
    },
}
WIN_BOARDS["H"] = WIN_BOARDS["R1"]  # the holdout race: R1's boards, planted dividends

# (pool, combination text, dividend per 100 yen): finishes R1 1-2-4, R2 3-1-2, R3 4-1-2.
PAID = {
    "R1": [
        ("WIN", "1", 580), ("PLACE", "1", 200), ("PLACE", "2", 160), ("PLACE", "4", 250),
        ("FORECAST", "1,2", 2400), ("QUINELLA", "1,2", 1100), ("QUINELLA_PLACE", "1,2", 450),
        ("QUINELLA_PLACE", "1,4", 700), ("QUINELLA_PLACE", "2,4", 600), ("TRIO", "1,2,4", 3000),
        ("TIERCE", "1,2,4", 15000), ("BRACKET_QUINELLA", "1,2", 1100),
    ],
    "R2": [
        ("WIN", "3", 680), ("PLACE", "3", 210), ("PLACE", "1", 110), ("PLACE", "2", 140),
        ("FORECAST", "3,1", 1500), ("QUINELLA", "1,3", 800), ("QUINELLA_PLACE", "1,3", 300),
        ("QUINELLA_PLACE", "2,3", 500), ("QUINELLA_PLACE", "1,2", 200), ("TRIO", "1,2,3", 900),
        ("TIERCE", "3,1,2", 6000), ("BRACKET_QUINELLA", "1,3", 800),
    ],
    "R3": [
        ("WIN", "4", 900), ("PLACE", "4", 300), ("PLACE", "1", 110), ("PLACE", "2", 150),
        ("FORECAST", "4,1", 3200), ("QUINELLA", "1,4", 1600), ("QUINELLA_PLACE", "1,4", 550),
        ("QUINELLA_PLACE", "2,4", 900), ("QUINELLA_PLACE", "1,2", 190), ("TRIO", "1,2,4", 2500),
        ("TIERCE", "4,1,2", 12000), ("BRACKET_QUINELLA", "1,4", 1600),
    ],
    "H": [(pool, combo, 99_990) for pool, combo, _ in [
        ("WIN", "1", 0), ("PLACE", "1", 0), ("PLACE", "2", 0), ("PLACE", "4", 0),
        ("FORECAST", "1,2", 0), ("QUINELLA", "1,2", 0), ("QUINELLA_PLACE", "1,2", 0),
        ("QUINELLA_PLACE", "1,4", 0), ("QUINELLA_PLACE", "2,4", 0), ("TRIO", "1,2,4", 0),
        ("TIERCE", "1,2,4", 0), ("BRACKET_QUINELLA", "1,2", 0),
    ]],
}  # fmt: skip
CARD = {
    # name: (race day, venue, race no, class, surface)
    "R1": (D1, "TOK", 1, "MAIDEN", "turf"),
    "R2": (D1, "TOK", 2, "C1", "dirt"),
    "R3": (D2, "NAK", 5, "MAIDEN", "turf"),
    "H": (HELD, "TOK", 1, "MAIDEN", "turf"),
}
DEVELOPMENT = ("R1", "R2", "R3")
# R2's T-10 Place on runner 1 sits low enough that the projection hits the 1.0 floor.
PLACE_OVERRIDE = {("R2", 10, 1): (1.05, 1.2)}


def race_id(name: str) -> str:
    day, venue, no, _, _ = CARD[name]
    return f"jra:{day.isoformat()}:{venue}:{no:02d}"


def meeting_id(name: str) -> str:
    day, venue, *_ = CARD[name]
    return f"{day.isoformat()}:{venue}"


def place_of(name: str, k: int, win: dict[int, float]) -> dict[int, tuple[float, float]]:
    out = {
        h: (round(1.0 + (w - 1.0) / 4.0, 2), round(1.0 + (w - 1.0) / 2.5, 2))
        for h, w in win.items()
    }
    for (n, kk, h), figure in PLACE_OVERRIDE.items():
        if (n, kk) == (name, k):
            out[h] = figure
    return out


def wide_of(win: dict[int, float]) -> dict[tuple[int, int], tuple[float, float]]:
    out = {}
    for a, b in itertools.combinations(sorted(h for h in win if h <= 4), 2):
        lo = round(1.0 + win[a] * win[b] / 5.0, 2)
        out[(a, b)] = (lo, round(lo * 1.6, 2))
    return out


def post_of(name: str) -> datetime:
    day, _, no, _, _ = CARD[name]
    return datetime(day.year, day.month, day.day, 1, 0, tzinfo=timezone.utc) + timedelta(
        minutes=30 * no
    )


def make_board(name: str, k: int) -> RaceBoard:
    post = post_of(name)
    win = WIN_BOARDS[name][k]
    at = post - timedelta(minutes=k)
    return RaceBoard(
        race_id=race_id(name),
        post_utc=post,
        at_utc=at,
        capture_utc=at,
        final=k == 0,
        win=dict(win),
        place=place_of(name, k, win),
        wide=wide_of(win),
        totals={WIN: 12_000_000.0, PLACE: 5_000_000.0, WIDE: 4_000_000.0},
        scratched=frozenset(),
    )


def make_race(name: str) -> pf.RaceInput:
    day, _, _, race_class, surface = CARD[name]
    paid = pd.DataFrame(
        {
            "race_id": [race_id(name)] * len(PAID[name]),
            "pool": [p for p, _, _ in PAID[name]],
            "combination": [c for _, c, _ in PAID[name]],
            "dividend": [float(d) for _, _, d in PAID[name]],
            "is_refund": [False] * len(PAID[name]),
        }
    )
    return pf.RaceInput(
        race_id=race_id(name),
        race_date=day,
        meeting_id=meeting_id(name),
        race_class=race_class,
        surface=surface,
        boards={k: make_board(name, k) for k in CHECKPOINTS},
        final=make_board(name, 0),
        paid=paid,
        brackets={h: h for h in WIN_BOARDS[name][0]},
    )


# --------------------------------------------------------------------------- hand values

# The projected Win boards, by hand (ref_project reproduces them at import).
PROJECTED_WIN = {
    ("R1", 10): {1: 5.62, 2: 4.18, 3: 4.91, 4: 7.37, 5: 13.26, 6: 36.64},
    ("R1", 5): {1: 4.86, 2: 4.80, 3: 5.61, 4: 7.69, 5: 12.62, 6: 35.37},
    ("R1", 2): {1: 4.73, 2: 4.76, 3: 5.23, 4: 7.13, 5: 13.33, 6: 35.74},
    ("R2", 10): {1: 1.99, 2: 4.89, 3: 5.73, 4: 16.58, 5: 30.54, 6: 60.0},
    ("R2", 5): {1: 1.94, 2: 4.20, 3: 11.04, 4: 15.77, 5: 27.63, 6: 60.0},
    ("R2", 2): {1: 2.21, 2: 3.68, 3: 6.66, 4: 14.35, 5: 26.28, 6: 55.0},
    ("R3", 10): {1: 2.04, 2: 3.91, 3: 7.78, 4: 12.16, 5: 24.43, 6: 54.96},
    ("R3", 5): {1: 1.89, 2: 3.76, 3: 10.72, 4: 11.04, 5: 22.10, 6: 49.73},
    ("R3", 2): {1: 1.74, 2: 3.68, 3: 10.77, 4: 9.32, 5: 22.08, 6: 50.0},
}
# The tickets each rule buys on those projections; a missing entry buys nothing.
PICKS = {
    (EXACTA, "R1", 10): {(3, 2), (1, 2), (4, 2)},  # the projection makes 2 the favourite
    (EXACTA, "R1", 5): {(1, 2), (3, 2), (4, 2)},
    (EXACTA, "R1", 2): {(2, 1), (3, 1), (4, 1)},  # back to 1 by 0.03
    **{(EXACTA, r, k): {(2, 1), (3, 1), (4, 1)} for r in ("R2", "R3") for k in CHECKPOINTS},
    (BAND, "R1", 10): {(1,), (4,)},
    (BAND, "R1", 5): {(3,), (4,)},
    (BAND, "R1", 2): {(3,), (4,)},
    (BAND, "R2", 10): {(3,)},
    (BAND, "R2", 2): {(3,)},
    (BAND, "R3", 10): {(3,)},
    (BAND, "R3", 2): {(4,)},
    (FAV, "R2", 10): {(h,) for h in range(1, 7)},
    (FAV, "R2", 5): {(h,) for h in range(1, 7)},
    (FAV, "R3", 5): {(h,) for h in range(1, 7)},
    (FAV, "R3", 2): {(h,) for h in range(1, 7)},
}
# (rule, k) -> (n_bets, n_tickets, n_meetings, staked_yen, returned_yen)
TOTALS = {
    (EXACTA, 10): (3, 9, 2, 900, 7100),
    (EXACTA, 5): (3, 9, 2, 900, 7100),
    (EXACTA, 2): (3, 9, 2, 900, 4700),
    (BAND, 10): (3, 4, 2, 400, 1260),
    (BAND, 5): (1, 2, 1, 200, 0),
    (BAND, 2): (3, 4, 2, 400, 1580),
    (FAV, 10): (1, 6, 1, 600, 680),
    (FAV, 5): (2, 12, 2, 1200, 1580),
    (FAV, 2): (1, 6, 1, 600, 900),
    **{(NO_BET, k): (0, 0, 0, 0, 0) for k in CHECKPOINTS},
}
STABILITY = {
    EXACTA: {5: 1.0, 2: 6 / 9},
    BAND: {5: 1 / 4, 2: 2 / 4},
    FAV: {5: 1.0, 2: 0.0},
    NO_BET: {5: None, 2: None},
}
POOL_OF = {EXACTA: "FORECAST", BAND: "WIN", FAV: "WIN", NO_BET: "TIERCE"}


def ref_select(rule: str, win: dict[int, float]) -> set[tuple[int, ...]]:
    """The three rules, read independently of the module (ties by the lower horse number)."""
    ranked = sorted(win, key=lambda h: (win[h], h))
    if rule == EXACTA:
        return {(ranked[i], ranked[0]) for i in (1, 2, 3)} if len(ranked) >= 4 else set()
    if rule == BAND:
        return {(h,) for h, o in win.items() if 5.0 <= o < 10.0}
    if rule == FAV:
        return {(h,) for h in win} if win[ranked[0]] < 2.0 else set()
    if rule == "n8:quinella:box1-4":
        return set(itertools.combinations(sorted(ranked[:4]), 2)) if len(ranked) >= 4 else set()
    if rule == "n8:wide:1/2-3":
        return {tuple(sorted((ranked[0], h))) for h in ranked[1:3]} if len(ranked) >= 3 else set()
    if rule == NO_BET:
        return set()
    raise KeyError(rule)


def ref_return(rule: str, name: str, tickets: set[tuple[int, ...]]) -> float:
    pool = POOL_OF[rule]
    paying = {(p, c): d for p, c, d in PAID[name]}
    ordered = pool in ("FORECAST", "TIERCE")
    total = 0.0
    for t in tickets:
        text = ",".join(str(h) for h in (t if ordered else sorted(t)))
        total += paying.get((pool, text), 0.0)
    return total


def _check_tables() -> None:
    for (name, k), expected in PROJECTED_WIN.items():
        got = {h: ref_project(WIN, o, k) for h, o in WIN_BOARDS[name][k].items()}
        assert got == pytest.approx(expected, abs=1e-9), (name, k)
    for rule in (EXACTA, BAND, FAV, NO_BET):
        for k in CHECKPOINTS:
            bets = tickets = staked = returned = 0
            meetings = set()
            for name in DEVELOPMENT:
                picks = ref_select(rule, PROJECTED_WIN[(name, k)])
                assert picks == PICKS.get((rule, name, k), set()), (rule, name, k)
                if picks:
                    bets, tickets = bets + 1, tickets + len(picks)
                    staked += 100 * len(picks)
                    returned += ref_return(rule, name, picks)
                    meetings.add(meeting_id(name))
            assert (bets, tickets, len(meetings), staked, returned) == TOTALS[(rule, k)], (rule, k)
        t10 = {(n, c) for n in DEVELOPMENT for c in PICKS.get((rule, n, 10), set())}
        for k in (5, 2):
            later = {(n, c) for n in DEVELOPMENT for c in PICKS.get((rule, n, k), set())}
            share = len(t10 & later) / len(t10) if t10 else None
            assert share == STABILITY[rule][k] or math.isclose(share, STABILITY[rule][k]), rule
    # the projection matters: the observed T-10 board and the final board pick otherwise
    assert ref_select(EXACTA, WIN_BOARDS["R1"][10]) != PICKS[(EXACTA, "R1", 10)]
    assert ref_select(BAND, WIN_BOARDS["R1"][10]) != PICKS[(BAND, "R1", 10)]
    assert ref_select(BAND, WIN_BOARDS["R1"][0]) != PICKS[(BAND, "R1", 10)]
    assert ref_select(EXACTA, WIN_BOARDS["R1"][0]) != PICKS[(EXACTA, "R1", 2)]
    assert ref_select(FAV, WIN_BOARDS["R3"][10]) != PICKS.get((FAV, "R3", 10), set())


_check_tables()


def combos(tickets) -> set[tuple[int, ...]]:
    """A TicketSet's combinations as tuples of ints; None or an empty set is no ticket."""
    if tickets is None:
        return set()
    out = set()
    for c in tickets.combinations:
        if isinstance(c, (int, np.integer)):
            out.add((int(c),))
        elif isinstance(c, str):
            out.add(tuple(int(x) for x in c.split(",")))
        else:
            out.add(tuple(int(x) for x in c))
    return out


# --------------------------------------------------------------------------- the rule set


def test_checkpoints_and_selection_labels():
    assert tuple(pf.CHECKPOINTS) == CHECKPOINTS
    assert [pf.selection_label(k) for k in CHECKPOINTS] == [
        "predicted_final@T-10",
        "predicted_final@T-5",
        "predicted_final@T-2",
    ]


def test_the_rules_cover_every_named_family_and_the_families_partition_them():
    rules = set(pf.RULES)
    assert {f"n8:{s}" for s in spread_scan.SPREADS} <= rules, "every #503 spread is a rule"
    assert {BAND, FAV} <= rules
    tickets = {pf.RULES[r].ticket for r in rules}
    assert {503, 435, 82, 131, 430} <= tickets, "#503, #435, #82, #131 and #430 are all measured"
    for rule_id, pool in POOL_OF.items():
        assert pf.RULES[rule_id].pool == pool, rule_id
    assert {pf.RULES[f"n8:{s}"].ticket for s in spread_scan.SPREADS} == {503}
    assert pf.RULES[BAND].ticket == pf.RULES[FAV].ticket == 430
    members = [r for family in pf.FAMILIES.values() for r in family]
    assert sorted(members) == sorted(rules), "every rule is in exactly one family"


def test_each_family_freezes_its_own_plain_json_spec_naming_its_rules():
    texts = []
    for family_id, members in pf.FAMILIES.items():
        spec = pf.family_spec(family_id)
        assert isinstance(spec, dict) and spec
        assert json.loads(json.dumps(spec)) == spec, "a family spec is plain JSON"
        text = json.dumps(spec)
        missing = [r for r in members if json.dumps(r)[1:-1] not in text]
        assert not missing, f"{family_id}'s spec does not name {missing[:3]}"
        texts.append(json.dumps(spec, sort_keys=True))
    assert len(set(texts)) == len(texts)


# --------------------------------------------------------------------------- projection


@pytest.mark.parametrize(("name", "k"), sorted(PROJECTED_WIN))
def test_the_projected_win_board_is_the_ladder_projection(name, k):
    projected = pf.project_board(make_board(name, k), nowcast=make_nowcast(), checkpoint_min=k)
    assert isinstance(projected, RaceBoard)
    assert set(projected.win) == set(WIN_BOARDS[name][k])
    assert projected.win == pytest.approx(PROJECTED_WIN[(name, k)], abs=1e-9)


PLACE_WIDE_CASES = [
    # (k, place lows in, wide lows in, place lows out, wide lows out); band 0 Place at 1.05 floors
    (10, {1: 1.05, 2: 2.4}, {(1, 2): 8.0}, {1: 1.0, 2: 2.65}, {(1, 2): 8.84}),
    (5, {1: 1.05, 2: 2.4}, {(1, 2): 8.0}, {1: 1.0, 2: 2.52}, {(1, 2): 8.41}),
    (2, {1: 1.5, 2: 6.0}, {(1, 2): 3.0, (1, 3): 25.0},
     {1: 1.46, 2: 6.31}, {(1, 2): 3.08, (1, 3): 25.63}),
]  # fmt: skip


@pytest.mark.parametrize(("k", "place", "wide", "place_out", "wide_out"), PLACE_WIDE_CASES)
def test_place_and_wide_project_on_their_low_figure(k, place, wide, place_out, wide_out):
    for h, o in place_out.items():
        assert ref_project(PLACE, place[h], k) == pytest.approx(o, abs=1e-9)
    for key, o in wide_out.items():
        assert ref_project(WIDE, wide[key], k) == pytest.approx(o, abs=1e-9)
    post = datetime(2025, 6, 7, 3, 0, tzinfo=timezone.utc)
    board = RaceBoard(
        race_id="jra:2025-06-07:TOK:09",
        post_utc=post,
        at_utc=post - timedelta(minutes=k),
        capture_utc=post - timedelta(minutes=k),
        final=False,
        win={1: 3.0, 2: 6.0, 3: 12.0},
        place={h: (lo, round(lo * 1.5, 2)) for h, lo in place.items()},
        wide={key: (lo, round(lo * 1.7, 2)) for key, lo in wide.items()},
        totals={},
        scratched=frozenset(),
    )
    projected = pf.project_board(board, nowcast=make_nowcast(), checkpoint_min=k)
    assert set(projected.place) == set(place) and set(projected.wide) == set(wide)
    assert {h: v[0] for h, v in projected.place.items()} == pytest.approx(place_out, abs=1e-9)
    assert {key: v[0] for key, v in projected.wide.items()} == pytest.approx(wide_out, abs=1e-9)
    assert projected.win == pytest.approx({h: ref_project(WIN, o, k) for h, o in board.win.items()})


def test_the_projection_follows_the_ladder_over_a_spread_of_prices():
    nowcast = make_nowcast()
    prices = [1.01, 1.4, 1.99, 2.0, 2.6, 4.99, 5.0, 7.3, 9.99, 10.0, 18.0, 20.0, 33.0, 64.0, 150.0]
    seen = set()
    for k in CHECKPOINTS:
        win = {h + 1: o for h, o in enumerate(prices)}
        place = {h + 1: (o, o * 1.3) for h, o in enumerate(prices)}
        post = datetime(2025, 6, 8, 3, 0, tzinfo=timezone.utc)
        board = RaceBoard(
            "jra:2025-06-08:TOK:03", post, post - timedelta(minutes=k), post - timedelta(minutes=k),
            False, win, place, {}, {}, frozenset(),
        )  # fmt: skip
        projected = pf.project_board(board, nowcast=nowcast, checkpoint_min=k)
        for h, o in win.items():
            assert projected.win[h] == pytest.approx(ref_project(WIN, o, k), abs=1e-9), (k, o)
            assert projected.place[h][0] == pytest.approx(ref_project(PLACE, o, k), abs=1e-9)
            seen.add((k, projected.win[h]))
    assert len(seen) > 2 * len(prices), "the projection changes with the checkpoint"


# --------------------------------------------------------------------------- selection


SELECT_CASES = [
    (rule, name, k)
    for rule in (EXACTA, BAND, FAV, NO_BET)
    for name in DEVELOPMENT
    for k in CHECKPOINTS
]


@pytest.mark.parametrize(("rule", "name", "k"), SELECT_CASES)
def test_a_rule_selects_on_the_projected_board(rule, name, k):
    race = make_race(name)
    projected = pf.project_board(race.boards[k], nowcast=make_nowcast(), checkpoint_min=k)
    tickets = pf.select(rule, projected, race=race)
    assert combos(tickets) == PICKS.get((rule, name, k), set())
    if tickets is not None and tickets.combinations:
        assert tickets.pool == POOL_OF[rule]


def _random_boards():
    rng = np.random.default_rng(713)
    out = []
    for n in (4, 5, 6, 8, 10, 12, 14, 16, 18, 7):
        prices = rng.choice(np.arange(12, 900) / 10.0, size=n, replace=False)
        if n in (5, 12):  # a short favourite
            prices[int(rng.integers(n))] = 1.6
        out.append({h + 1: float(p) for h, p in enumerate(rng.permutation(prices))})
    return out


@pytest.mark.parametrize("rule", [EXACTA, BAND, FAV, "n8:quinella:box1-4", "n8:wide:1/2-3"])
def test_selection_matches_an_independent_reading_over_varied_boards(rule):
    race = make_race("R1")
    seen = set()
    for i, win in enumerate(_random_boards()):
        post = datetime(2025, 6, 7, 6, 0, tzinfo=timezone.utc)
        board = RaceBoard(
            race.race_id, post, post - timedelta(minutes=2), post - timedelta(minutes=2), False,
            win, {}, {}, {}, frozenset(),
        )  # fmt: skip
        got = combos(pf.select(rule, board, race=race))
        assert got == ref_select(rule, win), (rule, i)
        seen.add(frozenset(got))
    assert len(seen) > 2, "the picks come from the board"


# --------------------------------------------------------------------------- the pure maths


STABILITY_CASES = [
    ({10: {"a", "b", "c", "d"}, 5: {"a", "b", "e"}, 2: {"a"}}, {5: 0.5, 2: 0.25}),
    ({10: {"a"}, 5: set(), 2: {"a", "b"}}, {5: 0.0, 2: 1.0}),
    ({10: {"a", "b", "c"}, 5: {"c"}, 2: {"b", "c", "x", "y"}}, {5: 1 / 3, 2: 2 / 3}),
    ({10: set(), 5: {"a"}, 2: {"a"}}, {5: None, 2: None}),
    ({10: {("R1", (3, 2)), ("R1", (1, 2))}, 5: {("R1", (1, 2))}, 2: {("R1", (2, 1))}},
     {5: 0.5, 2: 0.0}),
]  # fmt: skip


@pytest.mark.parametrize(("picks", "expected"), STABILITY_CASES)
def test_pick_stability_is_the_share_of_t10_picks_still_picked(picks, expected):
    got = pf.pick_stability(picks)
    assert set(got) == {5, 2}
    for k, value in expected.items():
        if value is None:
            assert got[k] is None, k
        else:
            assert got[k] == pytest.approx(value), k


def test_accuracy_table_hand_values():
    examples = pd.DataFrame(
        [
            (10, WIN, 1, 4.0, 5.0, 5.0),
            (10, WIN, 1, 3.0, 2.5, 2.0),
            (10, WIN, 2, 8.0, 8.0, 10.0),
            (2, WIN, 1, 4.0, 4.4, 4.0),
            (10, PLACE, 0, 1.5, 1.2, 1.5),
            (5, WIDE, 3, 12.0, 15.0, 10.0),
            (5, WIDE, 3, 18.0, 18.0, 12.0),
        ],
        columns=["checkpoint_min", "pool", "band", "odds_t", "odds_projected", "odds_final"],
    )
    l = math.log  # noqa: E741
    expected = {
        (10, WIN, 1): (2, (0.0 + l(1.25)) / 2, (l(5 / 4) + l(1.5)) / 2),
        (10, WIN, 2): (1, l(1.25), l(1.25)),
        (2, WIN, 1): (1, l(1.1), 0.0),
        (10, PLACE, 0): (1, l(1.25), 0.0),
        (5, WIDE, 3): (2, (l(1.5) + l(1.5)) / 2, (l(1.2) + l(1.5)) / 2),
    }
    table = pf.accuracy_table(examples)
    assert_accuracy(table, expected)


def assert_accuracy(table: pd.DataFrame, expected: dict) -> None:
    got = {}
    for r in table.to_dict("records"):
        band = r["band"]
        if isinstance(band, (int, np.integer)) and not isinstance(band, bool):
            key = (int(r["checkpoint_min"]), str(r["pool"]), int(band))
            assert key not in got, f"{key} is two rows"
            got[key] = r
    assert set(got) == set(expected)
    for key, (n, err, rw) in expected.items():
        assert int(got[key]["n"]) == n, key
        assert got[key]["log_error"] == pytest.approx(err, abs=1e-9), key
        assert got[key]["log_error_rw"] == pytest.approx(rw, abs=1e-9), key


# --------------------------------------------------------------------------- the run


class Forbidden(AssertionError):
    pass


def temp_registry(root: Path) -> Registry:
    return Registry(root / "registry" / "japan.jsonl", holdout_log=root / "holdout_log.jsonl")


def frozen_registry(root: Path) -> Registry:
    registry = temp_registry(root)
    for family_id in pf.FAMILIES:
        registry.freeze_family(family_id, pf.family_spec(family_id))
    return registry


def all_races() -> list:
    return [make_race(name) for name in ("R1", "R2", "H", "R3")]


def run(mp: pytest.MonkeyPatch, root: Path, registry: Registry, out_dir: Path, races=None):
    mp.setenv(DATA_VAR, str(root / "data"))
    return pf.run_study(
        races=all_races() if races is None else races,
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
    root = tmp_path_factory.mktemp("predicted-final").resolve()
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


def rows_by_key(rows) -> dict[tuple[str, int], dict]:
    out = {}
    for row in rows:
        key = (row["rule_id"], int(row["checkpoint_min"]))
        assert key not in out, f"{key} is two rows"
        out[key] = row
    return out


def test_every_rule_and_checkpoint_is_a_row_with_its_selection(study):
    rows = rows_by_key(study.result.rows)
    assert set(rows) == {(r, k) for r in pf.RULES for k in CHECKPOINTS}
    family_of = {r: f for f, members in pf.FAMILIES.items() for r in members}
    for (rule, k), row in rows.items():
        assert row["selection"] == f"predicted_final@T-{k}", (rule, k)
        assert row["family_id"] == family_of[rule], (rule, k)


@pytest.mark.parametrize(("rule", "k"), sorted(TOTALS))
def test_a_rules_return_is_settled_at_the_real_final_dividends(study, rule, k):
    """The hand totals hold the development races only: the holdout race (R1's boards with every
    dividend at 99,990 yen) would move each of them."""
    row = rows_by_key(study.result.rows)[(rule, k)]
    n_bets, n_tickets, n_meetings, staked, returned = TOTALS[(rule, k)]
    assert row["n_bets"] == n_bets
    assert row["n_tickets"] == n_tickets
    assert row["n_meetings"] == n_meetings
    assert row["staked_yen"] == pytest.approx(staked)
    assert row["returned_yen"] == pytest.approx(returned)
    if n_bets:
        assert row["return_per_100"] == pytest.approx(100.0 * returned / staked, rel=1e-9)
    else:
        assert row["return_per_100"] is None
        assert row["ci_lo"] is None and row["ci_hi"] is None


def test_the_meeting_bootstrap_brackets_the_return(study):
    checked = 0
    for row in study.result.rows:
        if row["n_meetings"] >= 2:
            lo, hi, value = row["ci_lo"], row["ci_hi"], row["return_per_100"]
            assert lo is not None and hi is not None, row["rule_id"]
            assert math.isfinite(lo) and math.isfinite(hi)
            assert lo <= value + 1e-9 and value <= hi + 1e-9, (row["rule_id"], lo, value, hi)
            checked += 1
    assert checked >= 5


@pytest.mark.parametrize("rule", sorted(STABILITY))
def test_pick_stability_from_t10_to_t5_and_t2(study, rule):
    got = study.result.stability[rule]
    for k, value in STABILITY[rule].items():
        if value is None:
            assert got[k] is None, (rule, k)
        else:
            assert got[k] == pytest.approx(value), (rule, k)
    assert set(study.result.stability) == set(pf.RULES)


def low(prices: dict) -> dict:
    return {key: figure[0] for key, figure in prices.items()}


def expected_accuracy() -> dict:
    """Every WIN, PLACE and Wide combination priced on both the T-k and final boards of the
    development races, by the ladder projection of its T-k low figure."""
    cells = defaultdict(list)
    for name in DEVELOPMENT:
        final = make_board(name, 0)
        for k in CHECKPOINTS:
            board = make_board(name, k)
            lows = {
                WIN: (board.win, final.win),
                PLACE: (low(board.place), low(final.place)),
                WIDE: (low(board.wide), low(final.wide)),
            }
            for pool, (now, fin) in lows.items():
                for key in now.keys() & fin.keys():
                    o, y = now[key], fin[key]
                    p = ref_project(pool, o, k)
                    cells[(k, pool, ref_band(o))].append(
                        (abs(math.log(p) - math.log(y)), abs(math.log(o) - math.log(y)))
                    )
    return {
        key: (len(v), sum(e for e, _ in v) / len(v), sum(r for _, r in v) / len(v))
        for key, v in cells.items()
    }


def test_the_nowcast_accuracy_by_pool_and_band_against_the_random_walk(study):
    expected = expected_accuracy()
    assert {pool for _, pool, _ in expected} == {WIN, PLACE, WIDE}
    assert any(err != rw for _, err, rw in expected.values())
    assert_accuracy(study.result.accuracy, expected)


def test_holdout_days_are_never_scored_nor_opened(study):
    assert study.opened == []
    assert holdout.is_holdout_day("japan", HELD) and not holdout.is_holdout_day("japan", D2)
    # the holdout race would add its 99,990-yen dividends to the band rule at T-10
    row = rows_by_key(study.result.rows)[(BAND, 10)]
    assert row["returned_yen"] == pytest.approx(TOTALS[(BAND, 10)][4])
    n_win = sum(n for (k, pool, _), (n, _, _) in expected_accuracy().items() if pool == WIN)
    banded = [
        r
        for r in study.result.accuracy.to_dict("records")
        if isinstance(r["band"], (int, np.integer)) and r["pool"] == WIN
    ]
    assert sum(int(r["n"]) for r in banded) == n_win


def test_the_run_is_recorded_in_the_registry(study):
    records = study.registry.records()
    assert records, "the run appended no record"
    assert {r.family_id for r in records} == set(pf.FAMILIES), "every family is recorded"
    for r in records:
        assert r.ticket == 713
        assert r.holdout_open is None
        assert "jravan" in {s.lower().replace("-", "") for s in r.sources}
        provenance = json.dumps(
            {"c": r.components, "p": r.params, "b": r.data_builds}, default=str, sort_keys=True
        )
        assert FIT_ID in provenance, "the nowcast's fit id is on the record"
    for family_id in pf.FAMILIES:
        assert study.registry.frozen_spec_hash(family_id) == _spec_hash(pf.family_spec(family_id))
    assert study.result.run_ids


def test_every_rule_and_checkpoint_is_a_leaderboard_row(study):
    from horsetrack.countries.japan.research import leaderboard_build

    candidate_dir = study.out_dir / "candidate-cells"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = study.root / "depth"
    depth_dir.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_VAR, str(study.root / "data"))
        rows = leaderboard_build.build_rows(
            registry=study.registry, candidate_dir=candidate_dir, depth_dir=depth_dir
        )
    ours = [r for r in rows if r.family_id in pf.FAMILIES]
    for rule in pf.RULES:
        for k in CHECKPOINTS:
            label = f"predicted_final@T-{k}"
            found = [
                r for r in ours if r.rule == rule and r.selection == label and r.segment == "all"
            ]
            assert len(found) == 1, (rule, k, len(found))
    for (rule, k), (n_bets, _, n_meetings, staked, returned) in TOTALS.items():
        (row,) = [
            r
            for r in ours
            if r.rule == rule and r.selection == f"predicted_final@T-{k}" and r.segment == "all"
        ]
        assert row.withheld, "built on JRA-VAN: withheld from the public leaderboard (D21)"
        if n_bets:
            assert row.return_per_100 == pytest.approx(100.0 * returned / staked, rel=1e-9)
            assert row.n_meetings == n_meetings
        else:
            assert row.return_per_100 is None, "a rule with no bet is an unmeasured row"


def test_the_report_is_private_and_under_out_dir(study):
    path = Path(study.result.report_path).resolve()
    assert path.is_file() and path.stat().st_size > 0
    assert path.is_relative_to(study.out_dir.resolve())
    assert not path.is_relative_to(REPO_ROOT)


# --------------------------------------------------------------------------- refusals


def test_the_run_refuses_before_reading_a_race_when_a_family_is_not_frozen(tmp_path, monkeypatch):
    registry = temp_registry(tmp_path)
    families = list(pf.FAMILIES)
    for family_id in families[1:]:  # every family but the first
        registry.freeze_family(family_id, pf.family_spec(family_id))
    read = []

    def races():
        read.append(True)
        yield from all_races()

    out_dir = tmp_path / "data" / "reports"
    with pytest.raises(pf.FamilyNotFrozen):
        run(monkeypatch, tmp_path, registry, out_dir, races=races())
    assert read == [], "a race was read before the freeze was checked"
    assert registry.records() == []
    assert not out_dir.exists() or not any(p.is_file() for p in out_dir.rglob("*"))


def test_the_run_refuses_a_family_frozen_with_another_spec(tmp_path, monkeypatch):
    registry = temp_registry(tmp_path)
    families = list(pf.FAMILIES)
    registry.freeze_family(families[0], {"not": "the spec", "ticket": 713})
    for family_id in families[1:]:
        registry.freeze_family(family_id, pf.family_spec(family_id))
    read = []

    def races():
        read.append(True)
        yield from all_races()

    with pytest.raises(pf.FamilyNotFrozen):
        run(monkeypatch, tmp_path, registry, tmp_path / "data" / "reports", races=races())
    assert read == []
    assert registry.records() == []


def test_an_out_dir_inside_the_repository_is_refused(tmp_path, monkeypatch):
    inside = REPO_ROOT / f".predicted-final-713-{uuid.uuid4().hex[:8]}"
    try:
        with pytest.raises(LicensedOutputInRepo):
            run(monkeypatch, tmp_path, frozen_registry(tmp_path), inside)
        assert not inside.exists() or not any(p.is_file() for p in inside.rglob("*"))
    finally:
        shutil.rmtree(inside, ignore_errors=True)
