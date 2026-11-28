"""The rule families re-measured on the nowcast's predicted final board at T-10, T-5, T-2 (#713).

The owner's workflow fills cards at T-10, narrows at T-5 and bets at T-2. At each checkpoint k
(`CHECKPOINTS`) the board at post - k is projected to the final with the shadow nowcast, the same
projector as the shadow page's ladder (`board.shadow_feed._projector`, `project_board`): each
WIN, PLACE (low figure) and Wide (low figure) price O becomes round(max(1, exp(ln O + mu(k, b))),
2), b its band on the nowcast's cut points and mu the cell mean, linear in tau between grid
neighbours. Each rule (`RULES`) reads its conditions on that projection (`select`); every ticket
stakes 100 yen and is settled at the race's real final dividends. A rule whose conditions read a
board the v0 nowcast does not project (exacta, Quinella, Trio, Trifecta, bracket boards) is a row
at every checkpoint, unmeasured (`NOT_PROJECTED`): the T-k exotic board is not a predicted final.

`run_study` takes the races (`RaceInput`) and the nowcast as arguments; it refuses
(`FamilyNotFrozen`) before it reads a race unless every family of `FAMILIES` is frozen in the
registry with `family_spec`, drops every holdout day (`is_holdout_day`; the holdout is never
opened), and gives per (rule, checkpoint) the return per 100 yen with the D7 meeting bootstrap,
the pick stability from T-10 (`pick_stability`) and the nowcast's accuracy against the random walk
by pool and band (`accuracy_table`). It writes the private report through `write_report` (an
out_dir inside the repository is refused before anything is written) and appends one registry
record per (family, checkpoint) whose metrics carry every rule; `leaderboard_backfill` reads them
as one leaderboard row per (rule, checkpoint), selection `predicted_final@T-k`. D21: the figures
are JRA-VAN-derived and stay under the data directory. Reading the store and the logs into
`RaceInput`s is separate (not in this module).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import typer

from horsetrack.core import config, env
from horsetrack.core.eval.holdout import _git_sha, is_holdout_day
from horsetrack.core.eval.leaderboard import PREDICTED_FINAL_SELECTIONS
from horsetrack.core.eval.registry import (
    FoldMetric,
    Registry,
    RunRecord,
    _canonical,
    _sha256,
    _spec_hash,
    default_registry,
)
from horsetrack.core.eval.reports import LicensedOutputInRepo, is_inside, write_report
from horsetrack.countries.japan.betting.formations import TicketSet
from horsetrack.countries.japan.board.shadow_feed import _low, _projector
from horsetrack.countries.japan.board.snapshot import RaceBoard
from horsetrack.countries.japan.models.v0.nowcast import NowcastV0, band_of
from horsetrack.countries.japan.research import exacta_rules, pool_scan, spread_scan
from horsetrack.countries.japan.studies import cross_pool_run

COUNTRY, TICKET, HYPOTHESIS_ID, STUDY = "japan", 713, "P713", "predicted-final"
FAMILY_PREFIX = "japan-713-predicted-final"
SOURCES = ("jravan", "jra")  # JRA-VAN odds history and our logged development days
FOLD_SCHEME = "in-sample:development-days"
CHECKPOINTS = (10, 5, 2)  # minutes before the post: T-10 fill, T-5 narrow, T-2 bet
STAKE = spread_scan.STAKE  # yen per ticket
DEFAULT_SEED, DEFAULT_N_BOOT = 20261129, 1000
WIN, PLACE, WIDE = "WIN", "PLACE", "QUINELLA_PLACE"
PROJECTED_POOLS = (WIN, PLACE, WIDE)
NOT_PROJECTED = "pool not projected by the v0 nowcast"
DEVELOPMENT = "development"
REPORT_BUILD = "report"


class FamilyNotFrozen(RuntimeError):
    """A family of `FAMILIES` is not frozen with `family_spec`: nothing was read or written."""


def selection_label(checkpoint_min: int) -> str:
    """The leaderboard selection of a rule read on the projection from the board at T-k."""
    label = f"predicted_final@T-{checkpoint_min}"
    if checkpoint_min not in CHECKPOINTS or label not in PREDICTED_FINAL_SELECTIONS:
        raise ValueError(f"no checkpoint T-{checkpoint_min} (have {CHECKPOINTS})")
    return label


# --------------------------------------------------------------------------- races and boards


@dataclass(frozen=True)
class RaceInput:
    """One development race: its card, the observed board at each checkpoint, the final board
    and its `dividends` rows (race_id, pool, combination, dividend per 100 yen, is_refund)."""

    race_id: str
    race_date: date
    meeting_id: str
    race_class: str
    surface: str
    boards: Mapping[int, RaceBoard]
    final: RaceBoard | None
    paid: pd.DataFrame
    brackets: Mapping[int, int] = field(default_factory=dict)


def project_board(board: RaceBoard, *, nowcast: NowcastV0, checkpoint_min: float) -> RaceBoard:
    """The board at post - k projected to the final by the ladder's projector (module docstring):
    WIN, PLACE and Wide low figures, the same keys; the high figures are not projected (None). A
    price the nowcast cannot project (a cell it lacks) is left off the projection."""
    project = _projector(SimpleNamespace(nowcast=nowcast))
    tau = float(checkpoint_min)

    def one(pool: str, figure: Any) -> float | None:
        odds = _low(figure)
        return None if odds is None else project(pool, odds, tau, 0)

    win = {h: p for h, o in board.win.items() if (p := one(WIN, o)) is not None}
    place = {h: (p, None) for h, f in board.place.items() if (p := one(PLACE, f)) is not None}
    wide = {k: (p, None) for k, f in board.wide.items() if (p := one(WIDE, f)) is not None}
    return replace(board, win=win, place=place, wide=wide, final=False)


def _ranked(board: RaceBoard) -> list[int]:
    """The horses by ascending Win price, ties by the lower horse number."""
    return sorted(board.win, key=lambda h: (board.win[h], h))


def _lows(board: RaceBoard, pool: str) -> dict[tuple[int, ...], float]:
    """(combination) -> low figure of a projected pool."""
    if pool == WIN:
        return {(h,): float(o) for h, o in board.win.items()}
    if pool == PLACE:
        return {(h,): float(f[0]) for h, f in board.place.items() if f and f[0] is not None}
    return {tuple(k): float(f[0]) for k, f in board.wide.items() if f and f[0] is not None}


def _tickets(pool: str, combos: Iterable[tuple[int, ...]]) -> TicketSet | None:
    found = tuple(sorted(set(combos)))
    return TicketSet(pool, found) if found else None


# --------------------------------------------------------------------------- the rules

Picker = Callable[[RaceBoard, RaceInput], "TicketSet | None"]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    family_id: str
    ticket: int  # the source study's ticket as #713 names it
    pool: str
    description: str
    unmeasured: str | None = None  # why no projected board can read it
    pick: Picker | None = field(default=None, compare=False, repr=False)


def _spread(spread_id: str) -> Picker:
    def pick(board: RaceBoard, race: RaceInput) -> TicketSet | None:
        try:
            return spread_scan.expand(spread_id, _ranked(board), brackets=race.brackets)
        except ValueError:  # fewer ranks than the spread's top rank, or a bracket unknown
            return None

    return pick


# #430 (N1, scan v6): each pool's odds-band edges; Place and Wide band on the low figure.
N1_BANDS = {
    WIN: (2, 3, 5, 10, 20, 50),
    PLACE: (1.2, 1.5, 2, 3, 5, 10),
    WIDE: (2, 5, 10, 20, 50, 100),
    "QUINELLA": (5, 10, 20, 50, 100, 300),
    "FORECAST": (10, 20, 50, 100, 300, 1000),
    "TRIO": (10, 30, 100, 300, 1000, 3000),
}
N1_RACE_CELLS = (
    "all",
    *(f"fav:{f}" for f in ("le1.9", "2.0-2.9", "3.0-4.9", "ge5.0")),
    *(f"field:{f}" for f in ("le4", "5-10", "11-14", "15-18")),
)


def _n1_cells(pool: str) -> tuple[str, ...]:
    bands = (f"band:{b}" for b in pool_scan._band_labels(N1_BANDS[pool]))
    return (N1_RACE_CELLS[0], *bands, *N1_RACE_CELLS[1:])


def _race_cells(board: RaceBoard, race: RaceInput) -> frozenset[str]:
    """#430's favourite and field cells of the race on the projected Win board."""
    ranked = _ranked(board)
    if not ranked:
        return frozenset()
    try:
        cells = pool_scan.race_cells(
            race_class=race.race_class,
            field_size=len(ranked),
            fav_odds=board.win[ranked[0]],
            pool_total=0.0,
            pool_edges=(0.0, 0.0),
            surface=race.surface,
            venue="",
        )
    except ValueError:  # an unknown class or a field outside 1-18: no segment cell
        return frozenset({"all"})
    return frozenset(c for c in cells if c == "all" or c.startswith(("fav:", "field:")))


def _n1(pool: str, cell: str) -> Picker:
    kind, _, level = cell.partition(":")

    def pick(board: RaceBoard, race: RaceInput) -> TicketSet | None:
        prices = _lows(board, pool)
        if kind == "band":
            edges = N1_BANDS[pool]
            return _tickets(pool, (c for c, o in prices.items() if _band(o, edges) == level))
        if cell not in _race_cells(board, race):
            return None
        return _tickets(pool, prices)

    return pick


def _band(odds: float, edges: tuple) -> str:
    return pool_scan.odds_band(odds, edges)


E15_FAV_MAX = 1.5  # #83's step 2: behind a favourite at or under 1.5


def _e15(board: RaceBoard, race: RaceInput) -> TicketSet | None:
    ranked = _ranked(board)
    if len(ranked) < 3 or board.win[ranked[0]] > E15_FAV_MAX:
        return None
    return _tickets(PLACE, ((ranked[1],), (ranked[2],)))


def _l4(kind: str) -> Picker:
    def pick(board: RaceBoard, race: RaceInput) -> TicketSet | None:
        ranked = _ranked(board)
        if len(ranked) < 3:
            return None
        top = tuple(ranked[:3])
        if kind == "trifecta":
            return _tickets("TIERCE", [top])
        if kind == "trio":
            return _tickets("TRIO", [tuple(sorted(top))])
        return _tickets("TIERCE", itertools.permutations(top))

    return pick


def _favourite(board: RaceBoard, race: RaceInput) -> TicketSet | None:
    ranked = _ranked(board)
    return _tickets(WIN, [(ranked[0],)]) if ranked else None


FAMILY_N8 = f"{FAMILY_PREFIX}-n8"
FAMILY_X435 = f"{FAMILY_PREFIX}-x435"
FAMILY_L82 = f"{FAMILY_PREFIX}-l82"
FAMILY_E131 = f"{FAMILY_PREFIX}-e131"
FAMILY_N1 = f"{FAMILY_PREFIX}-n1"


def _rules() -> dict[str, Rule]:
    rules: list[Rule] = []
    for spread_id, spread in spread_scan.SPREADS.items():
        rules.append(
            Rule(
                f"n8:{spread_id}",
                FAMILY_N8,
                503,
                spread.pool,
                f"#503's spread {spread_id} on the projected Win popularity ranks",
                pick=_spread(spread_id),
            )
        )
    for variant in cross_pool_run.VARIANTS:
        pools = ("QUINELLA",) if variant == cross_pool_run.PAIR_VARIANT else cross_pool_run.POOLS
        for pool, chosen in itertools.product(pools, cross_pool_run.SELECTIONS):
            rules.append(
                Rule(
                    f"x435:{variant}:{pool}:{chosen}",
                    FAMILY_X435,
                    435,
                    pool,
                    f"#435's {variant} scorer, {chosen} in {pool}: reads the T-k Quinella board",
                    NOT_PROJECTED,
                )
            )
    rules.append(
        Rule(
            "x435:offset:normalised:WIN:top1",
            FAMILY_X435,
            435,
            WIN,
            "#435's offset:normalised Win top1 (the market alone): the projected Win favourite",
            pick=_favourite,
        )
    )
    owner = config.load_country_yaml(COUNTRY, "avoidance")
    for pool, key in (("TRIO", "l1_threshold_trio"), ("QUINELLA", "l1_threshold_quinella")):
        for x in owner[key]:
            rules.append(
                Rule(
                    f"l1:{pool}:threshold_{x}",
                    FAMILY_L82,
                    82,
                    pool,
                    f"L1: every set priced under {x} in {pool}, else its ordered synthetic: reads "
                    f"the T-k {pool} board",
                    NOT_PROJECTED,
                )
            )
    for kind, pool, text in (
        ("trifecta", "TIERCE", "the market-order TIERCE (projected Win ranks 1-2-3)"),
        ("trio", "TRIO", "the TRIO on the market-order three (L4's trio comparator)"),
        ("box", "TIERCE", "the six TIERCE orders of the market-order three (L4's box)"),
    ):
        rule_id = f"l4:{pool}:market-order-{kind}"
        rules.append(Rule(rule_id, FAMILY_L82, 82, pool, f"L4: {text}", pick=_l4(kind)))
    for member in exacta_rules.MEMBERS:
        name = exacta_rules.variant(*member)
        rules.append(
            Rule(
                f"e2:FORECAST:{name}",
                FAMILY_E131,
                131,
                "FORECAST",
                f"#131's E2 member {name}: reads the T-k exacta board's ratio to Harville",
                NOT_PROJECTED,
            )
        )
    rules.append(
        Rule(
            "e15:PLACE:ranks2-3:fav-le1.5",
            FAMILY_E131,
            131,
            PLACE,
            "E15 step 2: Place on projected Win ranks 2 and 3 behind a favourite at or under 1.5",
            pick=_e15,
        )
    )
    for pool in pool_scan.POOLS:
        for cell in _n1_cells(pool):
            projected = pool in PROJECTED_POOLS
            rules.append(
                Rule(
                    f"n1:{pool}:{cell}",
                    FAMILY_N1,
                    430,
                    pool,
                    f"#430's flat box on every {pool} combination in {cell} of the projected board",
                    None if projected else NOT_PROJECTED,
                    _n1(pool, cell) if projected else None,
                )
            )
    return {rule.rule_id: rule for rule in rules}


RULES: dict[str, Rule] = _rules()
FAMILIES: dict[str, tuple[str, ...]] = {
    family: tuple(r for r, rule in RULES.items() if rule.family_id == family)
    for family in (FAMILY_N8, FAMILY_X435, FAMILY_L82, FAMILY_E131, FAMILY_N1)
}


def select(rule_id: str, board: RaceBoard, *, race: RaceInput) -> TicketSet | None:
    """The tickets `rule_id` buys read on `board` (the projected board); None buys nothing.
    ValueError for a rule no projected board can read (`Rule.unmeasured`)."""
    rule = RULES[rule_id]
    if rule.pick is None:
        raise ValueError(f"{rule_id} is unmeasured: {rule.unmeasured}")
    return rule.pick(board, race)


# --------------------------------------------------------------------------- the spec


def _level() -> float:
    boot = config.load_country_yaml(COUNTRY, "scoring")["bootstrap"]
    if boot["cluster_unit"] != "meeting" or boot["sided"] != "one-sided":
        raise ValueError(f"the study implements meeting clusters, one-sided; scoring.yaml: {boot}")
    return float(boot["level"])


def _common_spec() -> dict[str, Any]:
    return {
        "study": STUDY,
        "ticket": TICKET,
        "hypothesis_id": HYPOTHESIS_ID,
        "sources": list(SOURCES),
        "hypothesis": "some pre-registered rule, read on the nowcast's predicted final board at "
        "T-10, T-5 or T-2, returns more than its stake at the real final dividends; the null, the "
        "starting point: none beats the takeout",
        "checkpoints_min": list(CHECKPOINTS),
        "selection_labels": [selection_label(k) for k in CHECKPOINTS],
        "projection": "per checkpoint k the observed board at post - k (RaceBoard) projected to "
        "the final by board/shadow_feed.py _projector with the nowcast "
        "models/japan/v0/current.json names at run time (NowcastV0): O -> "
        "round(max(1.0, exp(ln O + mu(k, b))), 2), b = band_of(O, nowcast.bands), mu the cell "
        "mean at tau k, linear in tau between the grid's neighbouring taus; WIN, PLACE low and "
        "Wide low figures; Win not renormalised; tau is k, not each row's fetch age; a price "
        "whose cell the fit lacks leaves the projection",
        "nowcast_provenance": "the fit id is on each record's data_builds after its '@' (a refit "
        "is another run of the same variants)",
        "selection": "each rule's conditions read on the projected board only; ranks by ascending "
        "projected Win price, ties to the lower horse number; the card (class, surface, "
        "brackets) from the race",
        "unmeasured_rules": f"a rule whose conditions read a board the v0 nowcast does not "
        f"project (exacta, Quinella, Trio, Trifecta, bracket) is a row at every checkpoint, "
        f"unmeasured: '{NOT_PROJECTED}'; the T-k exotic board is never carried forward as a "
        "predicted final",
        "stake_yen_per_ticket": STAKE,
        "settlement": "each ticket returns its pool's paying dividends rows of its combination "
        "(dead heats and Wide pay several), its stake on an is_refund row, else 0 "
        "(spread_scan._settle); a race without a dividends row of the rule's pool is no bet",
        "metrics": {
            "return_per_100": "100 x returned / staked over the rule's bets; none without a bet",
            "n_bets": "races bought; n_tickets their tickets; n_meetings their meetings",
            "ci": "D7 meeting bootstrap over the rule's own meetings: rng.multinomial(m, uniform) "
            "per replicate, numpy default_rng(seed) per row, the ratio's (1 - level) and level "
            "quantiles x 100, one-sided each; none under two meetings",
            "level": _level(),
            "run_of_record": {"seed": DEFAULT_SEED, "n_boot": DEFAULT_N_BOOT},
            "pick_stability": "a pick is (race, combination); the share of the T-10 picks still "
            "picked at T-5 and at T-2, each apart; none when T-10 picked nothing",
            "accuracy": "every WIN, PLACE and Wide combination priced on both the T-k and the "
            "final board (low figures): per (k, pool, band of the T-k price on the nowcast's "
            "bands) n, mean |ln projected - ln final| and the random walk's mean "
            "|ln O_t - ln final|",
        },
        "data": "development race days with a time series: JRA-VAN odds history and our logged "
        "development days; is_holdout_day days are dropped and LockedHoldout is never opened",
        "ticket_numbers": "#82 names the L-rules, whose code and registry family are #81's "
        "(studies/avoidance.py); #131 names E2 and E15, E15's code being #83's "
        "(studies/coplacer.py); #435's code is studies/cross_pool_run.py; #503 "
        "research/spread_scan.py; #430 research/pool_scan.py's cells",
        "left_out": [
            "L5: report only, not a bet",
            "L4 same-bracket: card only, the projection never enters",
            "#435's offset:calibrated and its other offset rows: a D4 calibration fit, not a "
            "board reading",
            "#430's class, surface, venue and pool-size cells and the class crosses: card or "
            "final-pool cuts the projection does not change",
            "#83's step 1 and step 3 (descriptive and pricing, not selections)",
        ],
        "leaderboard": "one row per (rule, checkpoint): rule the rule id, segment 'all', "
        "selection predicted_final@T-k, withheld (D21); a rule without a bet unmeasured",
        "verdicts": "none: every rule x checkpoint is a ranked row (owner reframe, 2026-11-27)",
    }


def family_spec(family_id: str) -> dict[str, Any]:
    """The plain-JSON spec `family_id` is frozen with: its rules and the study's rules."""
    if family_id not in FAMILIES:
        raise KeyError(f"unknown family {family_id!r} (have {sorted(FAMILIES)})")
    rules = {
        r: {
            "ticket": RULES[r].ticket,
            "pool": RULES[r].pool,
            "rule": RULES[r].description,
            "unmeasured": RULES[r].unmeasured,
        }
        for r in FAMILIES[family_id]
    }
    return {"family_id": family_id, "rules": rules, **_common_spec()}


def spec_hash() -> str:
    return _spec_hash({family: family_spec(family) for family in FAMILIES})


def _check_frozen(registry: Registry) -> None:
    for family_id in FAMILIES:
        if registry.frozen_spec_hash(family_id) != _spec_hash(family_spec(family_id)):
            raise FamilyNotFrozen(
                f"{family_id} is not frozen with its spec: run `horsetrack japan predicted-final "
                "freeze` first; nothing was read"
            )


# --------------------------------------------------------------------------- the maths


def pick_stability(picks: Mapping[int, set]) -> dict[int, float | None]:
    """The share of the T-10 picks still picked at T-5 and at T-2; None when T-10 picked none."""
    base = set(picks.get(10, ()))
    return {k: len(base & set(picks.get(k, ()))) / len(base) if base else None for k in (5, 2)}


ACCURACY_COLUMNS = ["checkpoint_min", "pool", "band", "n", "log_error", "log_error_rw"]


def accuracy_table(examples: pd.DataFrame) -> pd.DataFrame:
    """One row per (checkpoint_min, pool, band) of `examples` (checkpoint_min, pool, band, odds_t,
    odds_projected, odds_final): n, the mean |ln projected - ln final| and the random walk's mean
    |ln odds_t - ln final|."""
    if examples.empty:
        return pd.DataFrame(columns=ACCURACY_COLUMNS)
    final = np.log(examples["odds_final"].to_numpy(dtype=float))
    frame = pd.DataFrame(
        {
            "checkpoint_min": examples["checkpoint_min"].to_numpy(dtype=np.int64),
            "pool": examples["pool"].astype(str).to_numpy(dtype=object),
            "band": examples["band"].to_numpy(dtype=np.int64),
            "err": np.abs(np.log(examples["odds_projected"].to_numpy(dtype=float)) - final),
            "rw": np.abs(np.log(examples["odds_t"].to_numpy(dtype=float)) - final),
        }
    )
    out = frame.groupby(["checkpoint_min", "pool", "band"], sort=True).agg(
        n=("err", "size"), log_error=("err", "mean"), log_error_rw=("rw", "mean")
    )
    out = out.reset_index().sort_values(
        ["checkpoint_min", "pool", "band"], ascending=[False, True, True], kind="stable"
    )
    return out.reset_index(drop=True)[ACCURACY_COLUMNS]


def _examples(
    board: RaceBoard, projected: RaceBoard, final: RaceBoard | None, k: int, bands
) -> list[tuple]:
    if final is None:
        return []
    out = []
    for pool in PROJECTED_POOLS:
        now, proj, fin = _lows(board, pool), _lows(projected, pool), _lows(final, pool)
        for key in sorted(now.keys() & proj.keys() & fin.keys()):
            o, y = now[key], fin[key]
            if o > 0 and y > 0:
                out.append((k, pool, band_of(o, bands), o, proj[key], y))
    return out


def _bootstrap(
    by_meeting: Mapping[str, list[float]], *, seed: int, n_boot: int, level: float
) -> tuple[float | None, float | None]:
    """The ratio's one-sided bounds per 100 yen over the meetings; none under two meetings."""
    if len(by_meeting) < 2:
        return None, None
    names = sorted(by_meeting)
    back = np.array([by_meeting[m][1] for m in names])
    cost = np.array([by_meeting[m][0] for m in names])
    m = len(names)
    w = np.random.default_rng(seed).multinomial(m, np.full(m, 1 / m), size=n_boot).astype(float)
    ratio = 100.0 * (w @ back) / (w @ cost)
    lo, hi = np.quantile(ratio, [1 - level, level])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- the run


@dataclass
class StudyResult:
    rows: list[dict[str, Any]]
    stability: dict[str, dict[int, float | None]]
    accuracy: pd.DataFrame
    report_path: Path
    run_ids: dict[str, str]
    counts: dict[str, int]


def _row(rule: Rule, k: int, bets: list[tuple], *, seed: int, n_boot: int, level: float) -> dict:
    by_meeting: dict[str, list[float]] = {}
    for meeting, cost, ret, _ in bets:
        acc = by_meeting.setdefault(meeting, [0.0, 0.0])
        acc[0], acc[1] = acc[0] + cost, acc[1] + ret
    staked = float(sum(b[1] for b in bets))
    returned = float(sum(b[2] for b in bets))
    lo, hi = _bootstrap(by_meeting, seed=seed, n_boot=n_boot, level=level)
    note = rule.unmeasured or (None if bets else "no bet on the projected boards")
    return {
        "rule_id": rule.rule_id,
        "family_id": rule.family_id,
        "ticket": rule.ticket,
        "pool": rule.pool,
        "checkpoint_min": k,
        "selection": selection_label(k),
        "n_bets": len(bets),
        "n_tickets": int(sum(b[3] for b in bets)),
        "n_meetings": len(by_meeting),
        "staked_yen": staked,
        "returned_yen": returned,
        "return_per_100": 100.0 * returned / staked if staked > 0 else None,
        "ci_lo": lo,
        "ci_hi": hi,
        "note": note,
    }


def run_study(
    *,
    races: Iterable[RaceInput],
    nowcast: NowcastV0,
    registry: Registry,
    out_dir: Path | str,
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_N_BOOT,
    sources: tuple[str, ...] = SOURCES,
) -> StudyResult:
    """Score every rule at every checkpoint on the development races (module docstring)."""
    _check_frozen(registry)
    out_dir = Path(out_dir)
    if is_inside(out_dir, env.repo_root()):
        raise LicensedOutputInRepo(f"{out_dir} is inside the repository; write under the data dir")
    level = _level()
    measured = [rule for rule in RULES.values() if rule.pick is not None]
    bets: dict[tuple[str, int], list[tuple]] = defaultdict(list)
    picks: dict[tuple[str, int], set] = defaultdict(set)
    examples: list[tuple] = []
    counts: Counter = Counter()
    scored: list[str] = []
    for race in races:
        if is_holdout_day(COUNTRY, race.race_date):
            counts["race: holdout day, dropped"] += 1
            continue
        own = race.paid.loc[race.paid["race_id"].to_numpy() == race.race_id]
        paid = spread_scan._paid(own)
        scored.append(race.race_id)
        for k in CHECKPOINTS:
            board = race.boards.get(k)
            if board is None:
                counts[f"race: no board at T-{k}"] += 1
                continue
            projected = project_board(board, nowcast=nowcast, checkpoint_min=k)
            examples += _examples(board, projected, race.final, k, nowcast.bands)
            for rule in measured:
                tickets = rule.pick(projected, race)
                if tickets is None or not tickets.combinations or tickets.pool not in paid:
                    continue
                settled = spread_scan._settle(tickets, paid)
                bets[(rule.rule_id, k)].append(
                    (race.meeting_id, settled.cost, settled.ret, tickets.n_tickets)
                )
                picks[(rule.rule_id, k)].update(
                    (race.race_id, tuple(int(h) for h in c)) for c in tickets.combinations
                )
    counts["race: scored"] = len(scored)
    rows = [
        _row(rule, k, bets.get((rule.rule_id, k), []), seed=seed, n_boot=n_boot, level=level)
        for rule in RULES.values()
        for k in CHECKPOINTS
    ]
    stability = {
        r: pick_stability({k: picks.get((r, k), set()) for k in CHECKPOINTS}) for r in RULES
    }
    accuracy = accuracy_table(
        pd.DataFrame(
            examples,
            columns=["checkpoint_min", "pool", "band", "odds_t", "odds_projected", "odds_final"],
        )
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = spec_hash()
    fingerprint = _sha256(_canonical(sorted(scored)))[:16]
    report_path = out_dir / f"predicted-final-japan-{digest[:12]}-{stamp}.md"
    text = _report(rows, stability, accuracy, counts, nowcast, digest, seed, n_boot)
    write_report(report_path, text, sources=sources)
    builds = {
        "nowcast": f"nowcast-v0@{nowcast.fit_id or 'unidentified'}",
        "races": f"race-inputs@{fingerprint}",
        REPORT_BUILD: f"predicted-final-report@{report_path.name}",
    }
    run_ids = {}
    git_sha = _git_sha()
    notes = {"run_utc": stamp, "seed": str(seed), "n_boot": str(n_boot)}  # run, not variant
    for family_id in FAMILIES:
        for k in CHECKPOINTS:
            mine = [
                row for row in rows if row["family_id"] == family_id and row["checkpoint_min"] == k
            ]
            record = _record(family_id, k, mine, stability, builds, git_sha, sources, level)
            run_ids[f"{family_id}@T-{k}"] = registry.append(replace(record, notes=notes))
    return StudyResult(rows, stability, accuracy, report_path, run_ids, dict(counts))


def _record(
    family_id: str,
    k: int,
    rows: list[dict],
    stability: Mapping[str, Mapping[int, float | None]],
    builds: dict[str, str],
    git_sha: str,
    sources: tuple[str, ...],
    level: float,
) -> RunRecord:
    metrics = []
    for row in rows:
        rule_id = row["rule_id"]
        if row["return_per_100"] is not None:
            n = row["n_bets"]
            metrics += [
                FoldMetric(
                    DEVELOPMENT,
                    f"return_per_100[{rule_id}]",
                    row["return_per_100"],
                    row["ci_lo"],
                    row["ci_hi"],
                    n,
                ),
                FoldMetric(DEVELOPMENT, f"n_meetings[{rule_id}]", row["n_meetings"], None, None, n),
                FoldMetric(DEVELOPMENT, f"n_tickets[{rule_id}]", row["n_tickets"], None, None, n),
                FoldMetric(DEVELOPMENT, f"staked_yen[{rule_id}]", row["staked_yen"], None, None, n),
            ]
        share = stability[rule_id].get(k) if k != 10 else None
        if share is not None:
            metrics.append(
                FoldMetric(DEVELOPMENT, f"stability_from_t10[{rule_id}]", share, None, None, 0)
            )
    rules = {
        row["rule_id"]: {
            "pool": row["pool"],
            "ticket": row["ticket"],
            "unmeasured": RULES[row["rule_id"]].unmeasured,
        }
        for row in rows
    }
    return RunRecord(
        hypothesis_id=HYPOTHESIS_ID,
        family_id=family_id,
        variant=selection_label(k),
        ticket=TICKET,
        git_sha=git_sha,
        data_builds=dict(builds),
        sources=tuple(sources),
        fold_scheme=FOLD_SCHEME,
        components={"projector": "board.shadow_feed._projector", "nowcast": "models.v0.NowcastV0"},
        params={
            "checkpoint_min": k,
            "selection": selection_label(k),
            "stake_yen": STAKE,
            "ci_level": level,
            "rules": rules,
        },
        metrics=tuple(metrics),
        status="scored" if any(row["return_per_100"] is not None for row in rows) else "failed",
    )


# --------------------------------------------------------------------------- the report


def _num(value: float | None, digits: int = 1) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _report(rows, stability, accuracy, counts, nowcast, digest, seed, n_boot) -> str:
    measured = [r for r in rows if r["return_per_100"] is not None]
    lines = [
        "# Predicted-final re-measure of the rule families (Japan, #713)",
        "",
        "Private: built on JRA-VAN (D21). Never commit this file.",
        "",
        f"- spec hash `{digest}`; nowcast fit `{nowcast.fit_id or 'unidentified'}`; seed {seed}, "
        f"{n_boot} bootstrap replicates",
        *(f"- {name}: {value}" for name, value in sorted(counts.items())),
        "",
        "## Hypothesis",
        "",
        "Some pre-registered rule, read on the nowcast's predicted final board at T-10, T-5 or "
        "T-2, returns more than its stake at the real final dividends. The null (none does) is the "
        "starting point.",
        "",
        "## Variants tried",
        "",
        f"{len(RULES)} rules x {len(CHECKPOINTS)} checkpoints = {len(rows)} rows, {len(measured)} "
        "with a bet; every row, failures and unmeasured rules included:",
        "",
        "| Rule | Family | T-k | Bets | Tickets | Meetings | Staked | Returned | Per 100 | CI | "
        "Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        ci = "—" if r["ci_lo"] is None else f"[{_num(r['ci_lo'])}, {_num(r['ci_hi'])}]"
        lines.append(
            f"| `{r['rule_id']}` | {r['family_id']} | {r['checkpoint_min']} | {r['n_bets']} | "
            f"{r['n_tickets']} | {r['n_meetings']} | {r['staked_yen']:.0f} | "
            f"{r['returned_yen']:.0f} | {_num(r['return_per_100'])} | {ci} | {r['note'] or ''} |"
        )
    lines += [
        "",
        "## Pick stability",
        "",
        "Share of the T-10 picks (race, combination) still picked at T-5 and at T-2.",
        "",
        "| Rule | T-10 to T-5 | T-10 to T-2 |",
        "| --- | --- | --- |",
        *(
            f"| `{rule}` | {_num(s[5], 3)} | {_num(s[2], 3)} |"
            for rule, s in stability.items()
            if s[5] is not None
        ),
        "",
        "## Nowcast accuracy",
        "",
        "Mean absolute log error of the projected against the final price, and of the random walk "
        "(the T-k price itself), by checkpoint, pool and band of the T-k price.",
        "",
        "| T-k | Pool | Band | n | Projected | Random walk |",
        "| --- | --- | --- | --- | --- | --- |",
        *(
            f"| {a['checkpoint_min']} | {a['pool']} | {a['band']} | {a['n']} | "
            f"{_num(a['log_error'], 4)} | {_num(a['log_error_rw'], 4)} |"
            for a in accuracy.to_dict("records")
        ),
        "",
        "## Alternatives rejected and why",
        "",
        "Carrying the T-k exotic boards forward as a predicted final: it is not one, so rules "
        f"that read them are unmeasured ('{NOT_PROJECTED}'). See the frozen spec's left_out.",
        "",
        "## What this rules out",
        "",
        "Read the rows above against the null; nothing is adopted or rejected (D19).",
        "",
        "## Leaderboard rows",
        "",
        "One row per rule x checkpoint, selection predicted_final@T-k, withheld (D21).",
        "",
        "## Follow-ups filed",
        "",
        "None from this run.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- the command

predicted_final_app = typer.Typer(
    no_args_is_help=True, help="#713: the rule families on the nowcast's predicted final board."
)


@predicted_final_app.command("freeze")
def freeze_command() -> None:
    """Freeze every family's spec in the japan registry (irreversible; again is a no-op)."""
    registry = default_registry(COUNTRY)
    for family_id in FAMILIES:
        typer.echo(
            f"frozen {family_id} {registry.freeze_family(family_id, family_spec(family_id))}"
        )
    typer.echo(f"spec_hash={spec_hash()}")
