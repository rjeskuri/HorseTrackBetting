"""Exotic formations and an EV-per-yen formation search, shadow mode only (ticket #139).

Pools are the jra.yaml codes TIERCE and FORECAST (ordered) and TRIO, QUINELLA and QUINELLA_PLACE
(unordered). A combination is a tuple of horse numbers: the finishing order for an ordered pool,
ascending for an unordered one. Expansion follows the JRA mark cards: a box (ボックス), a wheel
(ながし, with マルチ on the ordered pools) and a per-position formation (フォーメーション), each
counted once per distinct combination.

Probabilities and expected final dividends (per 100 yen, after takeout and breakage) come from
the caller. Budget, unit and the minimum value per combination are owner inputs with no default.
The search reports what it would have chosen (D19, shadow mode): never a stake suggestion, never a
bet.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from horsetrack.core import config
from horsetrack.countries.japan.shadow.log import NO_BOARD, ShadowLog, shadow_log_path

Combo = tuple[int, ...]

ORDERED = {
    "TIERCE": True,
    "FORECAST": True,
    "TRIO": False,
    "QUINELLA": False,
    "QUINELLA_PLACE": False,
}
UNIT_STEP = 100
LOG_KIND = "formation"

_EPS = 1e-9
_DIGITS = 9  # gains are compared after rounding, so summation order cannot break a tie


# ---------------------------------------------------------------- pools and combinations


def _legs(pool: str) -> int:
    """The number of horses in one combination of ``pool``; ValueError for any other pool."""
    if pool not in ORDERED:
        raise ValueError(f"{pool!r} is not a combination pool ({', '.join(ORDERED)})")
    return config.load_source("japan", "jra").pools[pool].legs


def _canon(pool: str, combo: Iterable[int]) -> Combo:
    t = tuple(combo)
    return t if ORDERED[pool] else tuple(sorted(t))


def _horses(horses: Iterable[int], what: str) -> tuple[int, ...]:
    out = tuple(horses)
    if any(type(h) is not int for h in out):
        raise ValueError(f"{what}: horse numbers must be integers, got {out!r}")
    if len(set(out)) != len(out):
        raise ValueError(f"{what}: a horse is named twice in {out!r}")
    return out


def _check_unit(unit: Any) -> int:
    if type(unit) is not int or unit <= 0 or unit % UNIT_STEP:
        raise ValueError(f"unit must be a positive multiple of {UNIT_STEP} yen, got {unit!r}")
    return unit


@dataclass(frozen=True)
class TicketSet:
    """The distinct combinations a mark card covers, one ticket each."""

    pool: str
    combinations: tuple[Combo, ...]

    @property
    def n_tickets(self) -> int:
        return len(self.combinations)

    def cost(self, *, unit: int) -> int:
        """Yen for every ticket at ``unit`` yen a point."""
        return self.n_tickets * _check_unit(unit)


def _ticket_set(pool: str, combos: Iterable[Combo]) -> TicketSet:
    return TicketSet(pool, tuple(sorted({_canon(pool, c) for c in combos})))


def box(pool: str, horses: Iterable[int]) -> TicketSet:
    """Box (ボックス): every combination of ``horses``."""
    legs = _legs(pool)
    hs = _horses(horses, "box")
    if len(hs) < legs:
        raise ValueError(f"a {pool} box needs at least {legs} horses, got {hs!r}")
    return _ticket_set(pool, itertools.permutations(hs, legs))


def wheel(
    pool: str,
    axis: Sequence[int],
    opponents: Iterable[int],
    *,
    positions: Sequence[int] | None = None,
    multi: bool = False,
) -> TicketSet:
    """Wheel (ながし): every combination holding all ``axis`` horses, the rest from ``opponents``.

    On an ordered pool ``axis[i]`` finishes at 1-based ``positions[i]`` (default 1, 2, ...);
    ``multi`` (マルチ) lets the axis horses take any positions. Unordered pools take neither.
    """
    legs = _legs(pool)
    ordered = ORDERED[pool]
    ax = _horses(axis, "wheel axis")
    opp = _horses(opponents, "wheel opponents")
    if not 1 <= len(ax) < legs:
        raise ValueError(f"a {pool} wheel takes 1 to {legs - 1} axis horses, got {ax!r}")
    if set(ax) & set(opp):
        raise ValueError(f"an axis horse is also an opponent: {ax!r}, {opp!r}")
    if not ordered and (multi or positions is not None):
        raise ValueError(f"{pool} is unordered: no multi and no axis positions")
    if not ordered or multi:
        if positions is not None:
            raise ValueError("multi lets the axis take any position: positions are not given")
        picks = itertools.permutations(ax + opp, legs)
        return _ticket_set(pool, (t for t in picks if set(ax) <= set(t)))
    pos = tuple(range(1, len(ax) + 1)) if positions is None else tuple(positions)
    if len(pos) != len(ax) or len(set(pos)) != len(pos) or not set(pos) <= set(range(1, legs + 1)):
        raise ValueError(f"positions {pos!r} must place each axis horse at a distinct 1..{legs}")
    spec: list[Sequence[int]] = [opp] * legs
    for horse, p in zip(ax, pos, strict=True):
        spec[p - 1] = (horse,)
    return formation(pool, spec)


def formation(pool: str, legs: Sequence[Iterable[int]]) -> TicketSet:
    """Formation (フォーメーション): one horse from each position's set, all distinct."""
    n = _legs(pool)
    spec = [_horses(leg, "formation leg") for leg in legs]
    if len(spec) != n or not all(spec):
        raise ValueError(f"a {pool} formation takes {n} non-empty legs, got {spec!r}")
    return _ticket_set(pool, (t for t in itertools.product(*spec) if len(set(t)) == n))


# ---------------------------------------------------------------- dividends and evaluation


def paid_dividend(pool: str, share: float, *, rate: float) -> int:
    """Yen paid per 100 on a winning ticket holding ``share`` of an equal-split pool.

    100 x rate / share, floored to the pool's dividend step and never under its min dividend
    (jra.yaml). ``rate`` is the race's payout rate (``takeout.race_takeout``). QUINELLA_PLACE
    returns the stake plus a share of the losers' money: see
    ``core.pricing.stake_back_share_dividend``.
    """
    _legs(pool)
    spec = config.load_source("japan", "jra").pools[pool]
    if spec.payout_rule != "equal_split":
        raise ValueError(f"{pool} is not an equal-split pool: use stake_back_share_dividend")
    if not share > 0 or not rate > 0:
        raise ValueError(f"share and rate must be positive, got {share!r} and {rate!r}")
    step = spec.dividend_step
    paid = math.floor(UNIT_STEP * rate / share / step + _EPS) * step
    floor = spec.min_dividend
    return int(paid if floor is None else max(floor, paid))


@dataclass(frozen=True)
class Evaluation:
    """Expected yen of a ticket set at ``unit`` yen a point; ``ev_per_yen`` None when empty."""

    pool: str
    combinations: tuple[Combo, ...]
    n_tickets: int
    cost: int
    expected_return: float
    expected_profit: float
    ev_per_yen: float | None


def _keyed(pool: str, mapping: Mapping[Any, float], what: str) -> dict[Combo, float]:
    out: dict[Combo, float] = {}
    for combo, x in mapping.items():
        key = _canon(pool, combo)
        if key in out:
            raise ValueError(f"{what}: {key!r} is given twice")
        out[key] = x
    return out


def _empty(pool: str) -> Evaluation:
    return Evaluation(pool, (), 0, 0, 0.0, 0.0, None)


def evaluate(
    tickets: TicketSet,
    probs: Mapping[Combo, float],
    dividends: Mapping[Combo, float],
    *,
    unit: int,
) -> Evaluation:
    """sum p x dividend / 100 x unit over the tickets, against their cost. Dividends per 100 yen."""
    _check_unit(unit)
    p = _keyed(tickets.pool, probs, "probs")
    d = _keyed(tickets.pool, dividends, "dividends")
    missing = [c for c in tickets.combinations if c not in p or c not in d]
    if missing:
        raise ValueError(f"no probability or dividend for {missing[:5]!r}")
    if not tickets.combinations:
        return _empty(tickets.pool)
    cost = tickets.cost(unit=unit)
    ret = math.fsum(p[c] * d[c] for c in tickets.combinations) / UNIT_STEP * unit
    return Evaluation(
        pool=tickets.pool,
        combinations=tickets.combinations,
        n_tickets=tickets.n_tickets,
        cost=cost,
        expected_return=ret,
        expected_profit=ret - cost,
        ev_per_yen=ret / cost,
    )


# ---------------------------------------------------------------- the search


@dataclass
class FormationChoice:
    """What the search would have chosen, for the shadow log. Shadow only: no stake suggestion.

    ``legs`` expand through ``formation`` to ``chosen`` (``()`` when nothing qualifies);
    ``best_single`` and ``best_box`` are the best single combination and box under the same rules.
    """

    pool: str
    legs: tuple[tuple[int, ...], ...]
    chosen: Evaluation
    best_single: Evaluation | None
    best_box: Evaluation | None
    shadow: bool = dataclasses.field(default=True, init=False)


class _Best:
    """The best candidate so far: largest gain, then fewer tickets, then smaller combinations."""

    def __init__(self) -> None:
        self.key: tuple[float, int, list[Combo]] | None = None
        self.found: Any = None

    def offer(self, gain: float, n: int, combos: Callable[[], Iterable[Combo]], found: Any) -> None:
        if not gain > 0:
            return
        g = round(gain, _DIGITS)
        listed: list[Combo] | None = None
        if self.key is not None:
            bg, bn, bc = self.key
            if g < bg or (g == bg and n > bn):
                return
            if g == bg and n == bn:
                listed = sorted(combos())
                if listed >= bc:
                    return
        self.key = (g, n, listed if listed is not None else sorted(combos()))
        self.found = found() if callable(found) else found


def _subsets(items: Sequence[Any], visit: Callable[[list[Any]], bool]) -> bool:
    """Depth-first over the non-empty subsets of ``items`` in index order. ``visit`` returns
    False to skip a subset's supersets that extend it. True when some singleton was visited
    true."""
    any_single = False

    def grow(start: int, chosen: list[Any]) -> None:
        nonlocal any_single
        for j in range(start, len(items)):
            chosen.append(items[j])
            keep = visit(chosen)
            if keep and len(chosen) == 1:
                any_single = True
            if keep:
                grow(j + 1, chosen)
            chosen.pop()

    grow(0, [])
    return any_single


def _search_formations(
    pool: str, q: set[Combo], gain: Mapping[Combo, float], most: int
) -> tuple[tuple[int, ...], ...] | None:
    """Legs of the best formation whose every combination is in ``q`` within ``most`` tickets.

    Exact: legs range over the horses ``q`` places at each position; the last leg is chosen per
    prefix of the others, and a prefix is dropped with its extensions once no last-leg horse can
    join it (a combination outside ``q`` or too many tickets stays so as legs grow).
    """
    n_legs = _legs(pool)
    ordered = ORDERED[pool]
    if ordered:
        sets = [sorted({c[i] for c in q}) for i in range(n_legs)]
    else:
        sets = [sorted({h for c in q for h in c})] * n_legs
    best = _Best()

    def last_leg(prefix: list[tuple[int, ...]]) -> bool:
        heads = [t for t in itertools.product(*prefix) if len(set(t)) == n_legs - 1]
        usable = False
        live: list[tuple[int, frozenset[Combo]]] = []
        for c in sets[-1]:
            group = frozenset(_canon(pool, (*t, c)) for t in heads if c not in t)
            if len(group) <= most and group <= q:
                usable = True
                if group:
                    live.append((c, group))
        if ordered:
            _choose_ordered(prefix, live)
        else:
            _choose_unordered(prefix, live)
        return usable

    def _choose_ordered(prefix: list[tuple[int, ...]], live: list[Any]) -> None:
        # groups are disjoint (they differ in the last position): gains add up
        items = [(c, g, math.fsum(gain[x] for x in g)) for c, g in live]
        items = [it for it in items if it[2] > 0]

        def visit(chosen: list[Any]) -> bool:
            n = sum(len(g) for _, g, _ in chosen)
            if n > most:
                return False
            union = [x for _, g, _ in chosen for x in g]
            best.offer(math.fsum(v for *_, v in chosen), n, lambda: union, lambda: union)
            return True

        _subsets(items, visit)

    def _choose_unordered(prefix: list[tuple[int, ...]], live: list[Any]) -> None:
        def visit(chosen: list[Any]) -> bool:
            union = frozenset().union(*(g for _, g in chosen))
            if len(union) > most:
                return False
            legs = (*prefix, tuple(c for c, _ in chosen))
            best.offer(math.fsum(gain[x] for x in union), len(union), lambda: union, legs)
            return True

        _subsets(live, visit)

    def leg(i: int, prefix: list[tuple[int, ...]]) -> bool:
        def visit(chosen: list[int]) -> bool:
            here = [*prefix, tuple(chosen)]
            return last_leg(here) if i == n_legs - 2 else leg(i + 1, here)

        return _subsets(sets[i], visit)

    leg(0, [])
    if best.key is None:
        return None
    if ordered:
        combos = best.found
        return tuple(tuple(sorted({c[i] for c in combos})) for i in range(n_legs))
    return best.found


def _search_boxes(
    pool: str, q: set[Combo], gain: Mapping[Combo, float], most: int
) -> tuple[int, ...] | None:
    """Horses of the best box inside ``q`` within ``most`` tickets (both rules stay broken as the
    box grows, so a failing box ends its branch)."""
    n_legs = _legs(pool)
    horses = sorted({h for c in q for h in c})
    best = _Best()
    held: list[set[Combo]] = [set()]

    def visit(chosen: list[int]) -> bool:
        while len(held) > len(chosen):
            held.pop()
        h = chosen[-1]
        new = {_canon(pool, t) for t in itertools.permutations(chosen, n_legs) if h in t}
        combos = held[-1] | new
        if not new <= q or len(combos) > most:
            return False
        held.append(combos)
        if combos:
            box_horses = tuple(chosen)
            best.offer(math.fsum(gain[c] for c in combos), len(combos), lambda: combos, box_horses)
        return True

    _subsets(horses, visit)
    return best.found if best.key is not None else None


def optimise_formation(
    pool: str,
    field: Iterable[int],
    probs: Mapping[Combo, float],
    dividends: Mapping[Combo, float],
    *,
    budget: int,
    unit: int,
    min_ev: float,
) -> FormationChoice:
    """The per-position formation over ``field`` with the largest expected profit, for the shadow
    log only.

    Feasible: cost <= ``budget`` at ``unit`` yen a point and every combination's value
    p x dividend / 100 >= ``min_ev``. The profit must be positive, else the choice is empty. Ties
    go to fewer tickets, then the lexicographically smallest sorted combination list. A
    combination missing from ``probs`` or ``dividends`` never qualifies.
    """
    _legs(pool)
    _check_unit(unit)
    horses = set(_horses(field, "field"))
    if not budget >= 0 or not min_ev == min_ev:
        raise ValueError(f"budget must be >= 0 and min_ev a number, got {budget!r}, {min_ev!r}")
    p = _keyed(pool, probs, "probs")
    d = _keyed(pool, dividends, "dividends")
    value = {c: p[c] * d[c] / UNIT_STEP for c in p if c in d and horses.issuperset(c)}
    q = {c for c, v in value.items() if v >= min_ev}
    gain = {c: value[c] - 1.0 for c in q}
    most = int(budget // unit)
    empty = FormationChoice(pool, (), _empty(pool), None, None)
    if most < 1 or not q:
        return empty

    legs = _search_formations(pool, q, gain, most)
    if legs is None:
        return empty
    chosen = evaluate(formation(pool, legs), p, d, unit=unit)
    singles = [c for c in q if gain[c] > 0]
    best_single = None
    if singles:
        top = min(singles, key=lambda c: (-round(gain[c], _DIGITS), c))
        best_single = evaluate(TicketSet(pool, (top,)), p, d, unit=unit)
    box_horses = _search_boxes(pool, q, gain, most)
    best_box = None if box_horses is None else evaluate(box(pool, box_horses), p, d, unit=unit)
    return FormationChoice(pool, legs, chosen, best_single, best_box)


# ---------------------------------------------------------------- the shadow log


def _legs_text(legs: Sequence[Sequence[int]]) -> str:
    """``formation:1/2,3/2,3,4``: the record's key text, apart from any single-combination pick."""
    body = "/".join(",".join(str(h) for h in leg) for leg in legs) if legs else "none"
    return f"{LOG_KIND}:{body}"


def log_choice(
    choice: FormationChoice,
    *,
    day: date,
    race_id: str,
    tau_min: int,
    decision_utc: str,
    board_capture_utc: str | None,
    written_utc: str,
    mode: str,
    models: Mapping[str, str] | None,
    git_sha: str,
) -> int:
    """Append ``choice`` as one record to the day's shadow log (docs/specs/shadow-log.md).

    An empty choice is logged too. Returns the number of records written (1).
    """
    chosen = choice.chosen
    record = {
        "race_id": race_id,
        "pool": choice.pool,
        "combination": _legs_text(choice.legs),
        "tau_min": tau_min,
        "decision_utc": decision_utc,
        "board_capture_utc": board_capture_utc,
        "written_utc": written_utc,
        "mode": mode,
        "status": NO_BOARD if board_capture_utc is None else "ok",
        "models": None if models is None else dict(models),
        "git_sha": git_sha,
        "kind": LOG_KIND,
        "shadow": True,
        "legs": [list(leg) for leg in choice.legs],
        "n_tickets": chosen.n_tickets,
        "combinations": [",".join(str(h) for h in c) for c in chosen.combinations],
        "cost": chosen.cost,
        "expected_return": chosen.expected_return,
        "expected_profit": chosen.expected_profit,
        "ev_per_yen": chosen.ev_per_yen,
    }
    return ShadowLog(shadow_log_path(day), day=day).append([record])
