"""Synthetic leaderboard rows, a hand-set nowcast and a two-venue day for the best bets (#714).

Nothing here imports the module under test: the expected best bets are worked out by hand.

The nowcast's drift is the same at every grid tau (so a projection does not depend on which tau,
1 min or more, the implementation reads it at) and moves WIN prices by band only: band 1 (2-5)
lengthens by exp(0.30), band 2 (5-10) shortens by exp(-0.25), every other band and every PLACE
and Wide price stays put. The projection therefore reorders the Win favourites, so a rule read on
the plain board buys other tickets than the same rule read on the predicted final board.

Boards (WIN odds of horses 1-8; SyntheticDay derives Place and Wide from them):
- A: raw ranks 4,1,2,3,...; projected {1: 5.4, 2: 4.28, 3: 4.67, 4: 4.86, 7: 7.01, ...}, ranks
  2,3,4,1,7,5,6,8.
- B: horse 1 at 2.4, projected 3.24: ranks 1,2,3,4,7,5,6,8.
- C: horse 3 at 2.2 as well, projected 2.97: ranks 3,1,2,4,... (only ever captured after post - 2,
  so a frozen best-bets list never reads it).

`TICKETS[board][rule]` lists, by hand, each rule's tickets on that projected board; `SHRUNK[k]`
is each rule's shrunk return per ¥100 at checkpoint T-k (distinct within a checkpoint).
`expected(board, k)` is the reference: every ticket once, its best row first and the others
under it by shrunk descending, the tickets by shrunk descending.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from horsetrack.core.eval.leaderboard import LeaderboardRow
from horsetrack.countries.japan.board.snapshot import DecisionGrid
from horsetrack.countries.japan.models.v0.nowcast import DriftCell, NowcastV0
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels
from horsetrack.countries.japan.research import predicted_final as pf

# isort: split
from shadow_fakes import FakeOrdering, FakeWin, Race, SyntheticDay

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
DAY = date(2025, 11, 29)
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
CHECKPOINTS = (10, 5, 2)
FIT_ID = "synthetic-nowcast-714"

# --------------------------------------------------------------------------- the nowcast

GRID = DecisionGrid(minutes=(60, 30, 16, 10, 5, 2, 1), primary=2)
BANDS = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
MU = {
    WIN: (0.0, 0.30, -0.25, 0.0, 0.0, 0.0, 0.0),
    PLACE: (0.0,) * 7,
    WIDE: (0.0,) * 7,
}


def make_nowcast() -> NowcastV0:
    cells = {
        (pool, tau, band): DriftCell(pool, tau, band, 40, MU[pool][band], 0.3)
        for pool in MU
        for tau in GRID.minutes
        for band in range(len(BANDS) + 1)
    }
    sigma0 = {(pool, tau): 0.3 for pool in MU for tau in GRID.minutes}
    return NowcastV0(BANDS, GRID, 50.0, ("2025-05-31",), sigma0, cells, FIT_ID)


def make_models() -> ShadowModels:
    ids = {"win": "win-a1", "ordering": "ord-b2", "nowcast": FIT_ID}
    return ShadowModels(win=FakeWin(), ordering=FakeOrdering(), nowcast=make_nowcast(), ids=ids)


def ref_project(odds: float) -> float:
    """The WIN projection by hand: round(max(1, O exp(mu(band(O)))), 2)."""
    return round(max(1.0, math.exp(math.log(odds) + MU[WIN][bisect_right(BANDS, odds)])), 2)


# --------------------------------------------------------------------------- the boards

BASES = {
    "A": (4.0, 5.5, 6.0, 3.6, 12.0, 25.0, 9.0, 60.0),
    "B": (2.4, 5.5, 6.0, 3.6, 12.0, 25.0, 9.0, 60.0),
    "C": (2.4, 5.5, 2.2, 3.6, 12.0, 25.0, 9.0, 60.0),
}
PROJECTED = {  # by hand from MU; checked against ref_project below
    "A": {1: 5.4, 2: 4.28, 3: 4.67, 4: 4.86, 5: 12.0, 6: 25.0, 7: 7.01, 8: 60.0},
    "B": {1: 3.24, 2: 4.28, 3: 4.67, 4: 4.86, 5: 12.0, 6: 25.0, 7: 7.01, 8: 60.0},
    "C": {1: 3.24, 2: 4.28, 3: 2.97, 4: 4.86, 5: 12.0, 6: 25.0, 7: 7.01, 8: 60.0},
}

TOP1 = "x435:offset:normalised:WIN:top1"
BAND35 = "n1:WIN:band:3-5"
WIDE12 = "n8:wide:1/2-3"
TRIO_MKT = "l4:TRIO:market-order-trio"
TRIO_BOX = "n8:trio:box1-4"
TIERCE_MKT = "l4:TIERCE:market-order-trifecta"
EXACTA12 = "n8:exacta:1/2-3"
QUIN12 = "n8:quinella:1/2-3"
E15 = "e15:PLACE:ranks2-3:fav-le1.5"  # a favourite at or under 1.5: never here, so no ticket

TICKETS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "A": {  # projected ranks 2, 3, 4, 1, 7, 5, 6, 8
        TOP1: [(WIN, "2")],
        BAND35: [(WIN, "2"), (WIN, "3"), (WIN, "4")],
        WIDE12: [(WIDE, "2,3"), (WIDE, "2,4")],
        TRIO_MKT: [("TRIO", "2,3,4")],
        TRIO_BOX: [("TRIO", "1,2,3"), ("TRIO", "1,2,4"), ("TRIO", "1,3,4"), ("TRIO", "2,3,4")],
        TIERCE_MKT: [("TIERCE", "2,3,4")],
        EXACTA12: [("FORECAST", "2,3"), ("FORECAST", "2,4")],
        QUIN12: [("QUINELLA", "2,3"), ("QUINELLA", "2,4")],
        E15: [],
    },
    "B": {  # projected ranks 1, 2, 3, 4, 7, 5, 6, 8
        TOP1: [(WIN, "1")],
        BAND35: [(WIN, "1"), (WIN, "2"), (WIN, "3"), (WIN, "4")],
        WIDE12: [(WIDE, "1,2"), (WIDE, "1,3")],
        TRIO_MKT: [("TRIO", "1,2,3")],
        TRIO_BOX: [("TRIO", "1,2,3"), ("TRIO", "1,2,4"), ("TRIO", "1,3,4"), ("TRIO", "2,3,4")],
        TIERCE_MKT: [("TIERCE", "1,2,3")],
        EXACTA12: [("FORECAST", "1,2"), ("FORECAST", "1,3")],
        QUIN12: [("QUINELLA", "1,2"), ("QUINELLA", "1,3")],
        E15: [],
    },
}

SHRUNK: dict[int, dict[str, float]] = {
    10: {
        TOP1: 131.0,
        BAND35: 95.0,
        WIDE12: 100.0,  # exactly 100: the strong colour
        TRIO_MKT: 97.0,
        TRIO_BOX: 98.5,
        TIERCE_MKT: 92.0,
        EXACTA12: 85.0,
        QUIN12: 60.0,  # raw 180: ranked by the shrunk value, not the raw one
        E15: 150.0,
    },
    5: {
        TOP1: 88.0,
        BAND35: 99.5,
        WIDE12: 79.0,
        TRIO_MKT: 104.0,
        TRIO_BOX: 70.0,
        TIERCE_MKT: 120.0,
        EXACTA12: 99.99,  # just under 100: the muted colour
        QUIN12: 99.0,
        E15: 150.0,
    },
    2: {
        TOP1: 99.0,
        BAND35: 76.0,
        WIDE12: 133.0,
        TRIO_MKT: 91.0,
        TRIO_BOX: 108.0,
        TIERCE_MKT: 66.0,
        EXACTA12: 140.0,
        QUIN12: 100.5,
        E15: 150.0,
    },
}
RAW = {QUIN12: 180.0}  # every other rule's raw return is its shrunk value + 7


def _check_tables() -> None:
    for name, base in BASES.items():
        assert PROJECTED[name] == {h: ref_project(o) for h, o in enumerate(base, 1)}, name
    raw_a = sorted(range(1, 9), key=lambda h: (BASES["A"][h - 1], h))
    projected_a = sorted(PROJECTED["A"], key=lambda h: (PROJECTED["A"][h], h))
    assert raw_a[:4] == [4, 1, 2, 3] and projected_a[:4] == [2, 3, 4, 1]  # the projection matters
    for k, table in SHRUNK.items():
        assert len(set(table.values())) == len(table), k
        assert set(table) == set(TICKETS["A"]) == set(TICKETS["B"])
    for rule in SHRUNK[10]:
        assert rule in pf.RULES and pf.RULES[rule].unmeasured is None, rule


# --------------------------------------------------------------------------- leaderboard rows


def option_id(rule: str, k: int, family: str | None = None) -> str:
    family = family or pf.RULES[rule].family_id
    return f"{family}/{rule}@T-{k}"


def row(
    rule: str,
    shrunk: float | None,
    *,
    k: int | None = 10,
    family: str | None = None,
    pool: str | None = None,
    selection: str | None = None,
    raw: float | None = None,
    n: int | None = 150,
    meetings: int | None = 30,
    licensed: bool = False,
    oid: str | None = None,
) -> LeaderboardRow:
    known = pf.RULES.get(rule)
    family = family or (known.family_id if known else pf.FAMILY_N8)
    pool = pool or (known.pool if known else WIN)
    selection = selection or f"predicted_final@T-{k}"
    if raw is None and shrunk is not None:
        raw = RAW.get(rule, shrunk + 7.0)
    measured = raw is not None
    return LeaderboardRow(
        option_id=oid or option_id(rule, k, family),
        family_id=family,
        ticket=713,
        kind="rule",
        pool=pool,
        segment="all",
        rule=rule,
        selection=selection,
        sources=("jravan", "jra"),
        variants_searched=len(pf.FAMILIES.get(family, ())) or 12,
        status="ok",
        return_per_100=raw,
        ci_lo=None if not measured else raw - 31.5,
        ci_hi=None if not measured else raw + 29.25,
        ci_level=None if not measured else 0.95,
        n=n if measured else None,
        n_unit="races" if measured else None,
        n_meetings=meetings if measured else None,
        payout_rate=0.8,
        shrunk_per_100=shrunk,
        licensed=licensed,
        note=None if measured else "no bet on the projected boards",
    )


UNKNOWN_RULE = "n8:retired:1/2"  # a #713-family row whose rule no longer exists
UNMEASURED_RULE = next(r for r, rule in pf.RULES.items() if rule.unmeasured)  # select raises
DECISION_FAMILY = "japan-82-family-l"  # a decision-time family with no matcher
DECISION_OID = f"{DECISION_FAMILY}/L1:TRIO:threshold_3.0"


def scored_rows(k: int) -> list[LeaderboardRow]:
    """The rules of TICKETS at T-k; TOP1 licensed (licensed rows are eligible), WIDE12 exactly at
    the thresholds (100 races, 20 meetings)."""
    out = []
    for rule, shrunk in SHRUNK[k].items():
        extra = {}
        if rule == TOP1:
            extra["licensed"] = True
        if rule == WIDE12:
            extra.update(n=100, meetings=20)
        out.append(row(rule, shrunk, k=k, **extra))
    return out


def unmatched_rows(k: int) -> list[LeaderboardRow]:
    """Eligible rows at T-k with no matcher: an unknown #713 rule and an unmeasured rule."""
    return [row(UNKNOWN_RULE, 105.0, k=k), row(UNMEASURED_RULE, 101.0, k=k)]


DECISION_ROW = row(
    "L1:TRIO:threshold_3.0",
    110.0,
    family=DECISION_FAMILY,
    pool="TRIO",
    selection="decision_time",
    n=400,
    meetings=40,
    oid=DECISION_OID,
)


def leak_rows(k: int) -> list[LeaderboardRow]:
    """Rows at T-k that are never eligible; each would buy a ticket of its own on A and B."""
    return [
        row("n1:WIN:band:5-10", 250.0, k=k, n=99),  # WIN 1 and 7 on A, WIN 7 on B
        row("n1:WIN:band:10-20", 240.0, k=k, meetings=19),  # WIN 5
        row("n1:WIN:band:20-50", 230.0, k=k, meetings=None),  # WIN 6
        row("n1:WIN:band:ge50", 260.0, k=k, n=None),  # WIN 8
        row("n8:wide:box1-4", None, k=k, raw=300.0),  # measured but no shrunk value
        row("n8:exacta:2-3/1", None, k=k),  # unmeasured
    ]


OTHER_SELECTIONS = [
    row("E2 ceiling", 300.0, family="japan-431-exacta-ceiling", pool="FORECAST",
        selection="look_ahead", n=500, meetings=50, oid="japan-431-exacta-ceiling/ceiling"),
    row("card rule", 290.0, family="japan-90-card", pool=WIN, selection="card_only", n=500,
        meetings=50, oid="japan-90-card/rule"),
    row("old rule", 280.0, family="japan-60-old", pool=WIN, selection="unknown", n=500,
        meetings=50, oid="japan-60-old/rule"),
]  # fmt: skip


def all_rows() -> list[LeaderboardRow]:
    out: list[LeaderboardRow] = []
    for k in CHECKPOINTS:
        out += scored_rows(k) + unmatched_rows(k) + leak_rows(k)
    return out + [DECISION_ROW] + OTHER_SELECTIONS


def eligible_ids(k: int) -> set[str]:
    return {r.option_id for r in scored_rows(k) + unmatched_rows(k)} | {DECISION_OID}


def unmatched_ids(k: int) -> set[str]:
    return {r.option_id for r in unmatched_rows(k)} | {DECISION_OID}


# --------------------------------------------------------------------------- the reference


def expected(board: str, k: int) -> list[dict]:
    buyers: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rule, tickets in TICKETS[board].items():
        for ticket in tickets:
            buyers[ticket].append(rule)
    out = []
    for (pool, combination), rules in buyers.items():
        rules.sort(key=lambda r: -SHRUNK[k][r])
        out.append(
            {
                "pool": pool,
                "combination": combination,
                "rule": rules[0],
                "option_id": option_id(rules[0], k),
                "duplicates": [option_id(r, k) for r in rules[1:]],
                "shrunk": SHRUNK[k][rules[0]],
            }
        )
    out.sort(key=lambda e: -e["shrunk"])
    return out


def check_bets(got: list[dict], board: str, k: int, where: str = "") -> None:
    """`got` (the best_bets list) against `expected(board, k)`: the same tickets once each, ranks
    1..n in order of shrunk return (a tie compared as a set), each ticket's best row with its
    figures, its duplicates under it and its highlight."""
    want = expected(board, k)
    rows = {r.option_id: r for r in scored_rows(k)}
    keys = [(e["pool"], e["combination"]) for e in got]
    assert len(keys) == len(set(keys)), f"{where}: a ticket appears twice: {keys}"
    wanted = [(w["pool"], w["combination"]) for w in want]
    assert len(got) == len(want), f"{where}: {keys} != {wanted}"
    assert [e["rank"] for e in got] == list(range(1, len(got) + 1)), where
    i = 0
    while i < len(want):
        j = i
        while j < len(want) and want[j]["shrunk"] == want[i]["shrunk"]:
            j += 1
        tie = {(w["pool"], w["combination"]) for w in want[i:j]}
        assert set(keys[i:j]) == tie, f"{where}: ranks {i + 1}-{j}: {keys[i:j]} != {tie}"
        i = j
    by_key = {(e["pool"], e["combination"]): e for e in got}
    for w in want:
        e = by_key[(w["pool"], w["combination"])]
        at = f"{where} {w['pool']} {w['combination']}"
        r = rows[w["option_id"]]
        assert e["option_id"] == w["option_id"], at
        assert e["rule"] == w["rule"], at
        assert e["family_id"] == r.family_id, at
        assert e["description"] == pf.RULES[w["rule"]].description, at
        assert math.isclose(e["shrunk_per_100"], w["shrunk"]), at
        assert math.isclose(e["return_per_100"], r.return_per_100), at
        assert math.isclose(e["ci_lo"], r.ci_lo) and math.isclose(e["ci_hi"], r.ci_hi), at
        assert (e["n"], e["n_meetings"]) == (r.n, r.n_meetings), at
        assert e["variants_searched"] == r.variants_searched, at
        assert e["highlight"] == ("strong" if w["shrunk"] >= 100 else "muted"), at
        dups = [d["option_id"] for d in e["duplicates"]]
        assert dups == w["duplicates"], f"{at}: duplicates {dups} != {w['duplicates']}"


# --------------------------------------------------------------------------- the day


def jst(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(DAY.year, DAY.month, DAY.day, hh, mm, ss, tzinfo=JST)


def rid(venue: str, no: int) -> str:
    return f"jra:{DAY.isoformat()}:{venue}:{no:02d}"


POSTS = {("TOK", 1): jst(10, 5), ("TOK", 2): jst(10, 40), ("KYO", 3): jst(10, 20),
         ("KYO", 4): jst(10, 55)}  # fmt: skip
# (venue, race no, board, capture time JST)
CAPTURES = [
    ("TOK", 1, "A", jst(9, 25)),
    ("TOK", 1, "B", jst(10, 1, 30)),  # post - 3.5
    ("TOK", 1, "C", jst(10, 3, 30)),  # post - 1.5: after the freeze
    ("TOK", 2, "A", jst(10, 0)),
    ("KYO", 3, "A", jst(9, 40)),
    ("KYO", 3, "B", jst(10, 12)),  # post - 8
    ("KYO", 4, "A", jst(10, 15)),
]


def races() -> dict[tuple[str, int], Race]:
    return {
        key: Race(DAY, key[0], key[1], post.astimezone(UTC), BASES["A"])
        for key, post in POSTS.items()
    }


def write_day() -> SyntheticDay:
    """The odds logs and post-time sidecars of the two-venue day under the data dir."""
    known = races()
    day = SyntheticDay(DAY, list(known.values()))
    for venue, no, board, at in sorted(CAPTURES, key=lambda c: c[3]):
        race = known[(venue, no)]
        day.capture(Race(DAY, venue, no, race.post, BASES[board]), at.astimezone(UTC), 1.0)
    return day


# (venue, now JST, race no or None, checkpoint, frozen, board)
TIMELINE = [
    ("TOK", jst(9, 54), 1, 10, False, "A"),
    ("TOK", jst(9, 59), 1, 10, False, "A"),  # post - 6
    ("TOK", jst(10, 0), 1, 5, False, "A"),  # post - 5 exactly: T-5
    ("TOK", jst(10, 2), 1, 5, False, "B"),  # provisional: the newer board
    ("TOK", jst(10, 3), 1, 2, True, "B"),  # post - 2 exactly: frozen
    ("TOK", jst(10, 4), 1, 2, True, "B"),  # C captured at post - 1.5 does not count
    ("TOK", jst(10, 4, 59), 1, 2, True, "B"),
    ("TOK", jst(10, 5), 2, 10, False, "A"),  # TOK 1's post: the tab moves on
    ("KYO", jst(9, 54), 3, 10, False, "A"),
    ("KYO", jst(10, 13), 3, 10, False, "B"),
    ("KYO", jst(10, 16), 3, 5, False, "B"),
    ("KYO", jst(10, 19, 59), 3, 2, True, "B"),
    ("KYO", jst(10, 20), 4, 10, False, "A"),  # KYO 3's post
]
DELAYED_KYO3 = jst(10, 30)  # KYO 3 delayed by ten minutes
DELAYED_TIMELINE = [
    ("KYO", jst(10, 20), 3, 10, False, "B"),  # the old post: still KYO 3
    ("KYO", jst(10, 27), 3, 5, False, "B"),
    ("KYO", jst(10, 28), 3, 2, True, "B"),  # the delayed post - 2: frozen
    ("KYO", jst(10, 30), 4, 10, False, "A"),  # the delayed post
    ("TOK", jst(10, 20), 2, 10, False, "A"),  # the other venue is not moved
]


_check_tables()
