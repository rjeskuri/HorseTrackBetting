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

import numpy as np

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
_SLACK = 1e-6  # the search cuts a branch only when its bound is this far below the best gain
# the search's Lagrange multipliers: the gains ranked at these multiples of the ticket limit
_RANKS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0)


# ---------------------------------------------------------------- pools and combinations


def _legs(pool: str) -> int:
    """The number of horses in one combination of ``pool``; ValueError for any other pool."""
    if pool not in ORDERED:
        raise ValueError(f"{pool!r} is not a combination pool ({', '.join(ORDERED)})")
    return config.load_source("japan", "jra").pools[pool].legs


def _canon(pool: str, combo: Iterable[int]) -> Combo:
    return _canon_order(ORDERED[pool], combo)


def _canon_order(ordered: bool, combo: Iterable[int]) -> Combo:
    t = tuple(combo)
    return t if ordered else tuple(sorted(t))


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


class _Search:
    """Exact branch and bound over the formations whose every combination is in ``q``.

    Horses are labelled one at a time, strongest first (by the positive gain of their
    combinations): a label is the set of legs a horse is in, as a bit mask (0 = out). The legs
    are read off the labels, so every formation has a labelling; a combination is in it when its
    horses can fill the legs, one each, and it is decided when its last horse is labelled. A
    labelling is dropped as soon as a decided combination is outside ``q`` or the tickets pass
    ``most``. On unordered pools the legs are interchangeable, so only labellings whose legs'
    membership columns do not increase from one leg to the next are searched.

    A branch is cut only when an upper bound on every formation under it falls below the best
    gain found (by ``_SLACK``), so the answer, ties included, is the one full enumeration gives.
    The bound relaxes the ticket limit (Lagrange, for each multiplier ``lam`` in ``lams``: the
    profit is at most ``lam * most`` plus the most ``gain - lam`` can add up to with no limit).
    That sum is the decided part plus, for each unlabelled horse z on its own, the best over its
    labels of: z's combinations whose other horses are labelled (exact), plus the positive part
    of z's other combinations that the labels already given still allow.
    """

    def __init__(self, pool: str, q: set[Combo], gain: Mapping[Combo, float], most: int) -> None:
        k = self.k = _legs(pool)
        self.ordered = ORDERED[pool]
        self.gain = gain
        self.most = most
        self.best = _Best()
        potential: dict[int, float] = {}
        for c in q:
            for h in c:
                potential[h] = potential.get(h, 0.0) + max(0.0, gain[c])
        self.order = sorted(potential, key=lambda h: (-potential[h], h))
        n = self.n = len(self.order)
        at = {h: i for i, h in enumerate(self.order)}
        # An ordered pool has one item per arrangement (member r, in search order, finishes at
        # position s[r]); an unordered pool one item. Cells are indexed by members in search order.
        arrangements = list(itertools.permutations(range(k)))
        items = len(arrangements) if self.ordered else 1
        nl = 1 << k
        cell = np.zeros((items,) + (n,) * k + (3,))  # gain, 1 if in q, 1 if not
        cell[..., 2] = 1.0
        for c in q:
            members = sorted(c, key=at.__getitem__)
            item = arrangements.index(tuple(c.index(h) for h in members)) if self.ordered else 0
            cell[(item, *(at[h] for h in members))] = (gain[c], 1.0, 0.0)
        self.cell = cell
        real = np.zeros((items,) + (nl,) * k)  # 1 when members with these labels fill the legs
        for labels in itertools.product(range(nl), repeat=k):
            fits = [all(labels[r] >> s[r] & 1 for r in range(k)) for s in arrangements]
            if self.ordered:
                for item, ok in enumerate(fits):
                    real[(item, *labels)] = ok
            else:
                real[(0, *labels)] = any(fits)
        self.real = real
        positive = sorted((v for v in gain.values() if v > 0), reverse=True)
        ranks = {max(1, round(most * f)) for f in _RANKS}
        self.lams = np.array(sorted({0.0, *(positive[r - 1] for r in ranks if r <= len(positive))}))
        # Positive parts of z's undecided combinations (second-last member j or later):
        # rest_open[j, z, label of z, lam] those with no labelled member, by what z's label allows;
        # after[i, x, j, z, lam] those whose first member x is labelled (k = 3), by can_xz.
        valid = cell[..., 1] > 0
        pos = np.where(valid[..., None], np.maximum(cell[..., 0, None] - self.lams, 0.0), 0.0)
        ax = k - 1  # the second-last member's axis
        tail = np.zeros(pos.shape[:ax] + (n + 1,) + pos.shape[ax + 1 :])
        tail[(slice(None),) * ax + (slice(0, n),)] = np.flip(np.cumsum(np.flip(pos, ax), ax), ax)
        can_z = real.reshape(items, -1, nl).max(axis=1)
        self.after = tail
        self.after_by: dict[tuple[int, int], Any] = {}
        if k == 3:
            self.can_xz = real.max(axis=2)
            unlabelled = np.tril(np.ones((n, n + 1)))  # [x, j] = 1 when x >= j
            tail = np.einsum("ixjzm,xj->ijzm", tail, unlabelled)
        self.rest_open = np.einsum("ijzm,il->jzlm", tail, can_z)

    def _after(self, x: int, label: int) -> Any:
        """``after`` for first member x labelled ``label``, by z's label: [j, z, label, lam]."""
        key = (x, label)
        if key not in self.after_by:
            self.after_by[key] = np.einsum("ijzm,il->jzlm", self.after[:, x], self.can_xz[:, label])
        return self.after_by[key]

    def hopeless(self, bound: float) -> bool:
        key = self.best.key
        return key is not None and bound < key[0] - _SLACK

    def run(self) -> tuple[tuple[int, ...], ...] | None:
        for c in sorted(self.gain):
            if self.gain[c] > 0:
                self.best.offer(self.gain[c], 1, lambda c=c: [c], tuple((h,) for h in c))
        if self.best.key is None:
            return None
        sums = np.zeros((self.n, 1 << self.k, 3))
        self._node([], sums, 0.0, 0, (True,) * (self.k - 1))
        legs = self.best.found
        if self.ordered:  # each leg trimmed to the horses its combinations use there
            combos = _expand(True, legs)
            legs = tuple(tuple(sorted({c[i] for c in combos})) for i in range(self.k))
        return legs

    def _offer(self, labels: list[int]) -> None:
        legs = tuple(
            tuple(sorted(h for h, lab in zip(self.order, labels, strict=False) if lab >> b & 1))
            for b in range(self.k)
        )
        if not all(legs):
            return
        if not self.ordered:
            legs = tuple(sorted(legs, key=lambda leg: (len(leg), leg)))
        combos = _expand(self.ordered, legs)
        self.best.offer(math.fsum(self.gain[c] for c in combos), len(combos), lambda: combos, legs)

    def _node(
        self, labels: list[int], sums: Any, gain: float, n: int, ties: tuple[bool, ...]
    ) -> None:
        """Label the next horse every way that keeps its decided combinations in ``q`` and within
        ``most``, best bound first. ``sums[z, label of z]`` holds (gain, count in q, count not in
        q) of z's combinations whose other members are labelled; ``gain`` and ``n`` the decided
        combinations'. All children are scored at once."""
        j, k, z0 = len(labels), self.k, len(labels) + 1
        if j == self.n:
            return
        cand = []
        for label in range(1 << k):
            if not self.ordered and any(
                tied and label >> (b + 1) & 1 and not label >> b & 1 for b, tied in enumerate(ties)
            ):
                continue  # leg b + 1 would come before leg b, its twin so far
            if label and (sums[j, label, 2] or n + sums[j, label, 1] > self.most):
                continue
            cand.append(label)
        ls = np.array(cand)
        gains = gain + sums[j, ls, 0]
        counts = n + sums[j, ls, 1]
        upd = None  # (child, z, label of z, 3): what labelling horse j adds to later sums
        if k == 2:  # tensordot: a BLAS product, far faster here than einsum
            upd = np.tensordot(self.cell[:, j, z0:], self.real[:, ls], axes=([0], [0]))
        elif j:
            mask = self.real[:, labels][:, :, ls]  # (items, a, child, label of z)
            upd = np.tensordot(self.cell[:, :j, j, z0:], mask, axes=([0, 1], [0, 1]))
        if upd is not None:
            upd = upd.transpose(2, 0, 3, 1)  # (z, 3, child, label of z) -> (child, z, label, 3)
        lams = self.lams
        base = lams * self.most + gains[:, None] - lams * counts[:, None]  # (child, lam)
        if z0 < self.n:
            s = sums[z0:, 1:] if upd is None else sums[z0:, 1:] + upd[:, :, 1:]
            rest = self.rest_open[z0, z0:, 1:]
            if k == 3:
                for x, lab in enumerate(labels):
                    if lab:
                        rest = rest + self._after(x, lab)[z0, z0:, 1:]
                if any(cand):
                    own = [self._after(j, lab)[z0, z0:, 1:] if lab else 0.0 for lab in cand]
                    rest = rest + np.stack(np.broadcast_arrays(*own))
            ok = (s[..., 2] == 0) & (counts[:, None, None] + s[..., 1] <= self.most)
            val = s[..., 0, None] - s[..., 1, None] * lams + rest
            val = np.where(ok[..., None], val, -np.inf).max(axis=-2)  # (child, z, lam)
            base = base + np.maximum(val, 0.0).sum(axis=-2)
        bounds = base.min(axis=1)
        for c in sorted(range(len(cand)), key=lambda c: (-bounds[c], cand[c])):
            if self.hopeless(bounds[c]):
                continue
            label = cand[c]
            here = [*labels, label]
            if label and not self.hopeless(gains[c]):
                self._offer(here)
            child = sums
            if label and upd is not None:
                child = sums.copy()
                child[z0:] += upd[c]
            twins = tuple(
                t and (label >> b & 1) == (label >> (b + 1) & 1) for b, t in enumerate(ties)
            )
            self._node(here, child, float(gains[c]), int(counts[c]), twins)


def _expand(ordered: bool, legs: Sequence[Sequence[int]]) -> list[Combo]:
    """The sorted distinct combinations of a per-position formation."""
    n = len(legs)
    return sorted({_canon_order(ordered, t) for t in itertools.product(*legs) if len(set(t)) == n})


def _search_formations(
    pool: str, q: set[Combo], gain: Mapping[Combo, float], most: int
) -> tuple[tuple[int, ...], ...] | None:
    """Legs of the best formation whose every combination is in ``q`` within ``most`` tickets."""
    return _Search(pool, q, gain, most).run()


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
        rest = itertools.combinations(chosen[:-1], n_legs - 1)
        if ORDERED[pool]:
            new = {t for c in rest for t in itertools.permutations((*c, h))}
        else:
            new = {tuple(sorted((*c, h))) for c in rest}
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
