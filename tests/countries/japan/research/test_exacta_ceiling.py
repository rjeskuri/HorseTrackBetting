"""Ticket #431 (N2): the exacta look-ahead ceiling on final boards, on synthetic data only.

The owner's reframe of 2026-11-27 governs: no kill rule and no verdict; every frozen member is a
ranked row; the family is frozen before any scored read; the selection is look-ahead (read from
the final boards).

Contract pinned here for ``horsetrack.countries.japan.research.exacta_ceiling`` (the coder
implements what these tests call):

- ``select_combinations(board, p, *, favourite, rate, lambda2, category, cut) -> list[(i, j)]``:
  ``board`` maps (first, second) to the final decimal exacta odds. A combination of ``category``
  (``exacta_consistency.combo_category`` against ``favourite``) is selected when
  r = q / p_c < cut, q = ``implied_share(odds x 100, rate=rate)`` and p_c the discounted
  Harville probability (``harville_exacta`` at ``lambda2``) on ``p``, renormalised. fav_second
  and no_fav are the rule categories; fav_first is never selected (``category="fav_first"``
  returns nothing or raises ValueError).
- ``race_return(selected, *, winner, dividend, pool_yen, money_yen, stake, rate) -> (staked,
  returned)``: staked = stake x n for the n selected combinations; returned is 0 unless
  ``winner`` is selected, else stake / 100 x dividend x (P + n s) / P x W / (W + s): the payout
  identity with our n stakes s in the pool P and s on the winner's money W = money_yen[winner],
  not rounded again.
- ``quinella_order_split(quinella, exacta, p, *, favourite, lambda2, quinella_rate, exacta_rate)
  -> {category: ratio}``: ``quinella`` maps a pair (i, j), i < j, to its final decimal Quinella
  odds and ``exacta`` maps (first, second) to the final decimal exacta odds. Each Quinella pair
  share (``implied_share`` at ``quinella_rate``) is split into (i, j) and (j, i) by Harville's
  order probabilities at ``lambda2`` on ``p``; for each of the three categories the ratio is the
  sum of the exacta shares (``implied_share`` at ``exacta_rate``) over the sum of the split
  Quinella shares.
- ``FAMILY_ID`` and ``family_specs() -> {family_id: plain-JSON spec}``: every family the run
  writes, ``FAMILY_ID`` among them (the 16 members: #131's E2 fav_second and no_fav x cut 0.6,
  0.7, 0.8, 0.9 x all races / headline cell, at the full-development lambda2). The tests freeze
  them with ``Registry.freeze_family``.
- ``RunRefused``: ``run_ceiling`` raises it, before it reads the store or records or writes
  anything, unless every family of ``family_specs()`` is frozen in the registry with that spec.
- The store seam ``load_races(days) -> (races, meta)``, which the tests replace with a synthetic
  store. ``run_ceiling`` passes it ``datetime.date`` days only, and only development days
  (``is_holdout_day`` False) in [max(date_from, 2014-06-07), date_to]; it may be called once or
  once per chunk of days, and returns those days' races. A race is a dict whose keys are all the
  run may read:
  race_id; race_date (date); meeting_id; race_no; race_class (an ``exacta_study.CLASS_GROUPS``
  key); race_class_raw; win_odds {horse: final Win odds} (the favourite and the headline cell,
  as #131: lowest odds, ties to the lowest number; ``segment_cells``); p {horse: the Win
  probability Harville is fed}; rate (the race's exacta payout rate); quinella_rate;
  boost_may_be_withheld; board {(i, j): final decimal exacta odds}; quinella_board {(i, j), i < j:
  final decimal Quinella odds}; pool_yen (the exacta pool's final money); money_yen {(i, j): the
  final money on the combination}; exacta [((i, j), dividend per 100 yen, is_refund)], the paying
  rows (``exacta_study.exclusion_reason``'s input). meta: lambda2_full_development (the lambda2
  every member scores at), win_fit, ordering_fit and data_builds {name: "<identity>@<digest>"}.
- ``run_ceiling(*, date_from, date_to, out_dir, seed, registry=None) -> CeilingResult``:
  ``result.rows`` is exactly the 16 members in rank order: rho descending, and a member without
  a bet (rho NaN or None) after every member with one. Each row (a mapping or an object) carries
  ``category``, ``cut``, ``cell`` ("all" or "headline"), ``rho`` (the look-ahead
  rho = sum returned / sum (staked x rate) over the member's bets: 100 yen per selected
  combination, ``race_return``'s own impact, each race's own rate), ``upper`` (rho's one-sided
  95% meeting-bootstrap upper bound), ``n_selected`` (combinations bought) and ``n_races`` and
  ``n_meetings`` (races and meetings with at least one bet). In the registry: one 'scored'
  record per member in ``FAMILY_ID`` (family size 16), every scored record in a family frozen
  before it, every record with withheld sources (jravan) and ticket 431. The private report is
  written under ``out_dir``; nothing is written into the repository.
- ``render_public_report(result) -> str``: carries no figure of the result (no rho, bound,
  count, rank or span): two results with different figures render the same text, which says
  'withheld (D21)' and labels the selection look-ahead.

Synthetic boards. A board is copied into M races and each combination c wins exactly m_c of them
(shuffled with the seed): its winning frequency is q_c = m_c / M, Harville's probability on a grid
of 1/M, so a sample's rho is its expectation and the tolerances are about the construction, not
luck. Every expected value is computed here from first principles (``_harville``, ``_share``,
``_reference``), never by the module under test.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from horsetrack.core import env
from horsetrack.core.eval import registry as registry_module
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.registry import Registry
from horsetrack.core.eval.reports import is_withheld
from horsetrack.countries.japan.research import exacta_ceiling as ceiling
from horsetrack.countries.japan.research import exacta_study as study
from horsetrack.countries.japan.research.exacta_consistency import harville_exacta, implied_share

RULE_CATEGORIES = ("fav_second", "no_fav")
CATEGORIES = ("fav_first", "fav_second", "no_fav")
CUTS = (0.6, 0.7, 0.8, 0.9)
CELLS = ("all", "headline")
MEMBERS = [(c, k, s) for c in RULE_CATEGORIES for k in CUTS for s in CELLS]
STAKE = 100.0
FIRST_DAY = date(2014, 6, 7)
VENUES = ("TOK", "HSN", "KYO")
RACES_PER_MEETING = 12
RAW_CLASS = {
    "C1": "3歳以上1勝クラス",
    "C2": "3歳以上2勝クラス",
    "MAIDEN": "3歳未勝利",
    "OP": "3歳以上オープン",
}
REPO = env.repo_root()
GUARDED = [REPO / "docs" / "ledger.md", REPO / "docs" / "experiments"]


# ---------------------------------------------------------------------------------------------
# References from first principles
# ---------------------------------------------------------------------------------------------


def _harville(p: Mapping[int, float], lam: float) -> dict[tuple[int, int], float]:
    """Discounted Harville: P(i first, j second) = p_i p_j^lam / (sum_k p_k^lam - p_i^lam)."""
    total = sum(p.values())
    q = {h: v / total for h, v in p.items()}
    w = {h: v**lam for h, v in q.items()}
    wsum = sum(w.values())
    return {(i, j): q[i] * w[j] / (wsum - w[i]) for i in q for j in q if i != j}


def _share(odds: float, rate: float) -> float:
    """The pool share of decimal odds: rate / ((100 odds + 5) / 100), half a 10-yen step back."""
    return rate / (odds + 0.05)


def _odds_for(share: float, rate: float) -> float:
    """The decimal odds whose implied share is exactly ``share``."""
    return rate / share - 0.05


def _category(combo: tuple[int, int], fav: int) -> str:
    return "fav_first" if combo[0] == fav else "fav_second" if combo[1] == fav else "no_fav"


def _favourite(win_odds: Mapping[int, float]) -> int:
    return min(win_odds, key=lambda h: (win_odds[h], h))


def _headline(race: dict) -> bool:
    """#131's headline cell: favourite's Win odds >= 3.0 and less popular (race 1-4 or maiden)."""
    fav_odds = race["win_odds"][_favourite(race["win_odds"])]
    less = race["race_no"] <= 4 or race["race_class"] in ("DEBUT", "MAIDEN")
    return fav_odds >= 3.0 and less


def _ref_select(board, p, *, favourite, rate, lam, category, cut) -> list[tuple[int, int]]:
    model = _harville(p, lam)
    return sorted(
        c
        for c, odds in board.items()
        if _category(c, favourite) == category and _share(odds, rate) / model[c] < cut
    )


def _ref_return(selected, winner, dividend, pool, money, stake) -> tuple[float, float]:
    n = len(selected)
    if winner not in selected:
        return n * stake, 0.0
    w = money[winner]
    return n * stake, stake / 100 * dividend * (pool + n * stake) / pool * w / (w + stake)


@dataclass
class MemberRef:
    meetings: list[str] = field(default_factory=list)
    staked: list[float] = field(default_factory=list)
    returned: list[float] = field(default_factory=list)
    rates: list[float] = field(default_factory=list)
    n_selected: int = 0

    @property
    def rho(self) -> float:
        if not self.staked:
            return math.nan
        cost = sum(s * r for s, r in zip(self.staked, self.rates, strict=True))
        return sum(self.returned) / cost


def _reference(races: list[dict], lam: float) -> dict[tuple, MemberRef]:
    """Each member's bets on ``races``: 100 yen on every selected combination of every race in
    the member's cell, paid at the dividend with the payout identity's own impact."""
    out = {m: MemberRef() for m in MEMBERS}
    chosen_by_board: dict[int, dict] = {}
    for race in races:
        fav = _favourite(race["win_odds"])
        key = id(race["board"])
        if key not in chosen_by_board:
            chosen_by_board[key] = {
                (c, k): _ref_select(race["board"], race["p"], favourite=fav, rate=race["rate"],
                                    lam=lam, category=c, cut=k)
                for c in RULE_CATEGORIES for k in CUTS
            }  # fmt: skip
        chosen = chosen_by_board[key]
        head = _headline(race)
        ((winner, dividend, _),) = race["exacta"]
        for member in MEMBERS:
            category, cut, cell = member
            if cell == "headline" and not head:
                continue
            selected = chosen[(category, cut)]
            if not selected:
                continue
            staked, returned = _ref_return(
                selected, winner, dividend, race["pool_yen"], race["money_yen"], STAKE
            )
            ref = out[member]
            ref.meetings.append(race["meeting_id"])
            ref.staked.append(staked)
            ref.returned.append(returned)
            ref.rates.append(race["rate"])
            ref.n_selected += len(selected)
    return out


@functools.cache
def _holdout_definition() -> tuple[date, str]:
    path = REPO / "src/horsetrack/countries/japan/config/holdout.yaml"
    block = yaml.safe_load(path.read_text(encoding="utf-8"))["holdout"]
    return date.fromisoformat(str(block["start"])), str(block["seed"])


def _ref_holdout(day: date) -> bool:
    """The committed A2 definition: from the start, a day is holdout when its seeded sha256 is
    even."""
    start, seed = _holdout_definition()
    digest = hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).hexdigest()
    return day >= start and int(digest, 16) % 2 == 0


# ---------------------------------------------------------------------------------------------
# Synthetic boards and stores (random.Random.random only: its stream is stable across Pythons)
# ---------------------------------------------------------------------------------------------


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()


def _gauss(rng: random.Random) -> float:
    u1, u2 = 1.0 - rng.random(), rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _shuffle(rng: random.Random, items: list) -> None:
    for i in range(len(items) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        items[i], items[j] = items[j], items[i]


def _normalised(w: Mapping[int, float]) -> dict[int, float]:
    total = sum(w.values())
    return {h: v / total for h, v in w.items()}


# The four race kinds: (race_no, race_class, favourite's Win odds, in the headline cell).
KINDS = (
    (2, "C2", 3.6, True),  # less popular by race number, favourite at 3.6
    (9, "MAIDEN", 4.2, True),  # less popular by class, favourite at 4.2
    (3, "C1", 2.4, False),  # less popular, but the favourite is under 3.0
    (10, "OP", 3.8, False),  # favourite at 3.8, but not less popular
)


@dataclass
class Board:
    """One final board, copied into ``sum(wins.values())`` races."""

    p: dict[int, float]
    win_odds: dict[int, float]
    rate: float
    quinella_rate: float
    race_no: int
    race_class: str
    odds: dict[tuple[int, int], float]
    quinella: dict[tuple[int, int], float]
    pool_yen: float
    money_yen: dict[tuple[int, int], float]
    wins: dict[tuple[int, int], int]  # m_c: how many of the copies combination c wins


def _make_board(rng, *, p, kind, rate, odds, truth, m0) -> Board:
    """A board with Win probabilities ``p`` (the run's p, also the Win board's normalised
    shares), exacta ``odds`` and winning frequencies ``truth`` (normalised, then put on a grid
    of about 1/m0 with every combination of ``truth`` winning at least once)."""
    race_no, race_class, fav_odds, _ = kind
    fav = max(p, key=lambda h: p[h])
    win_odds = {h: fav_odds * p[fav] / p[h] for h in p}
    total = sum(truth.values())
    wins = {c: max(1, round(m0 * v / total)) for c, v in truth.items()}
    board = Board(
        p=p, win_odds=win_odds, rate=rate, quinella_rate=0.775, race_no=race_no,
        race_class=race_class, odds={}, quinella={}, pool_yen=_uniform(rng, 30e6, 80e6),
        money_yen={}, wins=wins,
    )  # fmt: skip
    _set_odds(board, odds)
    return board


def _set_odds(board: Board, odds: dict[tuple[int, int], float]) -> None:
    """The exacta board, the money it implies on each combination and a Quinella board whose
    pair shares are the two orders' exacta shares."""
    board.odds = odds
    board.money_yen = {c: board.rate * board.pool_yen / o for c, o in odds.items()}
    pairs = {(min(c), max(c)) for c in odds}
    board.quinella = {
        (i, j): _odds_for(
            _share(odds[(i, j)], board.rate) + _share(odds[(j, i)], board.rate),
            board.quinella_rate,
        )
        for i, j in sorted(pairs)
    }


def _flat_p(rng, n: int) -> dict[int, float]:
    return _normalised({h: _uniform(rng, 0.6, 1.6) for h in range(1, n + 1)})


def _pool_board(rng, k: int, *, lam: float) -> Board:
    """The pool prices like Harville (odds = rate / q_c, winners at q_c); the run's p is a noisy
    view of the true Win probabilities, so r spreads and every cut selects something."""
    n = 5 + int(rng.random() * 4)
    p_true = _flat_p(rng, n)
    p_model = _normalised({h: v * math.exp(0.6 * _gauss(rng)) for h, v in p_true.items()})
    rate = 0.8 if k % 4 == 3 else 0.75
    truth = _harville(p_true, 1.0)
    board = _make_board(rng, p=p_model, kind=KINDS[k % 4], rate=rate, odds={}, truth=truth, m0=60)
    copies = sum(board.wins.values())
    _set_odds(board, {c: rate / (m / copies) for c, m in board.wins.items()})
    return board


def _underbet_board(rng, k: int) -> Board:
    """The Win board is right (the run's p is the truth); the exacta pool prices no_fav at 0.6
    of its Harville share and the rest at it; winners are Harville's."""
    p = _flat_p(rng, 5 + int(rng.random() * 3))
    fav = max(p, key=lambda h: p[h])
    board = _make_board(rng, p=p, kind=KINDS[k % 4], rate=0.75, odds={},
                        truth=_harville(p, 1.0), m0=60)  # fmt: skip
    copies = sum(board.wins.values())
    underbet = {"fav_first": 1.0, "fav_second": 1.0, "no_fav": 0.6}
    _set_odds(board, {c: 0.75 / (underbet[_category(c, fav)] * m / copies)
                      for c, m in board.wins.items()})  # fmt: skip
    return board


FACTORS = (0.45, 0.55, 0.65, 0.75)  # r of a rule combination: 0.6 buys two, 0.7 three, 0.8+ four


def _ranked_board(rng, k: int, *, lam: float, good: str, m0: int = 120) -> Board:
    """Every rule combination's r sits near one of FACTORS (the exacta board at that fraction of
    the run's own Harville); the winners overweight the ``good`` category (each of its bets is
    above break-even) and give the other rule category 0.3 of its weight (each bet below)."""
    p = _flat_p(rng, 5 + int(rng.random() * 2))
    fav = max(p, key=lambda h: p[h])
    model = _harville(p, lam)
    odds, truth, seen = {}, {}, dict.fromkeys(CATEGORIES, 0)
    for c in sorted(model):
        category = _category(c, fav)
        if category == "fav_first":
            factor, weight = 1.0, 1.0
        else:
            factor = FACTORS[seen[category] % len(FACTORS)]
            weight = 1.0 if category == good else 0.3
        seen[category] += 1
        odds[c] = 0.75 / (factor * model[c])
        truth[c] = weight * model[c]
    return _make_board(rng, p=p, kind=KINDS[k % 4], rate=0.75, odds=odds, truth=truth, m0=m0)


def _race_days(first: date, count: int) -> list[date]:
    """``count`` weekend development days from ``first``."""
    days, day = [], first
    while len(days) < count:
        if day.weekday() >= 5 and day >= FIRST_DAY and not _ref_holdout(day):
            days.append(day)
        day += timedelta(days=1)
    return days


def _races(rng, boards: list[Board], days: list[date], *, tag: str = "R") -> list[dict]:
    """Every copy of every board as a race, shuffled over the meetings of ``days`` (three venues,
    up to twelve races a meeting); a copy's winner is one of its board's m_c draws."""
    copies = []
    for board in boards:
        winners = [c for c in sorted(board.wins) for _ in range(board.wins[c])]
        _shuffle(rng, winners)
        copies += [(board, w) for w in winners]
    _shuffle(rng, copies)
    meetings = [(d, v) for d in days for v in VENUES]
    if len(copies) > len(meetings) * RACES_PER_MEETING:
        raise AssertionError("not enough meetings for the synthetic races")
    races = []
    for i, (board, winner) in enumerate(copies):
        day, venue = meetings[i % len(meetings)]
        races.append(_race(board, winner, day, venue, i // len(meetings) + 1, tag))
    return races


def _race(board: Board, winner, day: date, venue: str, slot: int, tag: str) -> dict:
    """One race in the store seam's schema, paying ``winner`` at 100 x its final odds."""
    return {
        "race_id": f"{tag}{day:%Y%m%d}{venue}{slot:02d}",
        "race_date": day,
        "meeting_id": f"{day.isoformat()}:{venue}",
        "race_no": board.race_no,
        "race_class": board.race_class,
        "race_class_raw": RAW_CLASS[board.race_class],
        "win_odds": board.win_odds,
        "p": board.p,
        "rate": board.rate,
        "quinella_rate": board.quinella_rate,
        "boost_may_be_withheld": False,
        "board": board.odds,
        "quinella_board": board.quinella,
        "pool_yen": board.pool_yen,
        "money_yen": board.money_yen,
        "exacta": [(winner, 100.0 * board.odds[winner], False)],
    }


def _meta(lam: float, tag: str) -> dict:
    digest = hashlib.sha256(tag.encode()).hexdigest()[:16]
    return {
        "lambda2_full_development": lam,
        "win_fit": "win-synthetic",
        "ordering_fit": "ordering-synthetic",
        "data_builds": {
            "jravan_final_odds": f"processed/jravan/final_odds.parquet@{digest}",
            "jravan_pool_votes": f"processed/jravan/pool_votes.parquet@{digest}",
            "jravan_dividends": f"processed/jravan/dividends.parquet@{digest}",
            "jravan_races": f"processed/jravan/races.parquet@{digest}",
        },
    }


class FakeStore:
    """The store seam: the races of the days asked for, and every day asked for."""

    def __init__(self, races: list[dict], meta: dict) -> None:
        self.races = races
        self.meta = meta
        self.asked: list[date] = []
        self.calls = 0

    def __call__(self, days):
        self.calls += 1
        asked = list(days)
        for day in asked:
            assert isinstance(day, date) and not isinstance(day, datetime), day
        self.asked += asked
        wanted = set(asked)
        return [r for r in self.races if r["race_date"] in wanted], copy.deepcopy(self.meta)


# ---------------------------------------------------------------------------------------------
# Environment, registry and run helpers
# ---------------------------------------------------------------------------------------------


def _snapshot() -> dict[Path, bytes]:
    found = {}
    for target in GUARDED:
        files = [target] if target.is_file() else sorted(target.rglob("*"))
        files = [f for f in files if f.is_file()]
        found.update({f: f.read_bytes() for f in files})
    return found


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """The data directory and the committed summary folder live under tmp_path; the test fails
    (and restores the files) if the run wrote the ledger or anything under docs/experiments."""
    before = _snapshot()
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("HORSETRACK_DATA_DIR", str(data))
    summary = tmp_path / "summary" / "japan"
    summary.mkdir(parents=True)
    monkeypatch.setattr(ceiling, "summary_dir", lambda country: summary, raising=False)
    monkeypatch.setattr(registry_module, "summary_dir", lambda country: summary)
    yield {"data": data, "summary": summary}
    after = _snapshot()
    if after != before:
        for path in set(after) - set(before):
            path.unlink()
        for path, raw in before.items():
            path.write_bytes(raw)
        pytest.fail("the test wrote the repository's ledger or docs/experiments (restored)")


def _registry(root: Path, *, frozen: bool = True) -> Registry:
    registry = Registry(root / "registry" / "japan.jsonl", holdout_log=root / "holdout_log.jsonl")
    if frozen:
        for family_id, spec in ceiling.family_specs().items():
            registry.freeze_family(family_id, spec)
    return registry


def _run(monkeypatch, root: Path, races: list[dict], meta: dict, *, date_from: date,
         date_to: date, registry: Registry | None = None):  # fmt: skip
    store = FakeStore(races, meta)
    monkeypatch.setattr(ceiling, "load_races", store)
    registry = registry if registry is not None else _registry(root)
    result = ceiling.run_ceiling(
        date_from=date_from, date_to=date_to, out_dir=root / "reports", seed=431, registry=registry
    )
    return result, store, registry


def _get(row, name: str):
    return row[name] if isinstance(row, Mapping) else getattr(row, name)


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def _rows(result) -> dict[tuple, object]:
    rows = list(result.rows)
    keys = [(_get(r, "category"), round(float(_get(r, "cut")), 2), _get(r, "cell")) for r in rows]
    assert len(rows) == 16 and sorted(keys) == sorted(MEMBERS), keys
    return dict(zip(keys, rows, strict=True))


def _assert_rank_order(result) -> None:
    """rho descending; a member without a bet (rho NaN or None) after every member with one."""
    values = [_get(r, "rho") for r in result.rows]
    finite = [_finite(v) for v in values]
    assert finite == sorted(finite, reverse=True), "a member without a bet ranks above one with"
    ranked = [float(v) for v, f in zip(values, finite, strict=True) if f]
    assert ranked == sorted(ranked, reverse=True), f"rows not in rho order: {ranked}"


def _assert_matches_reference(result, ref: dict[tuple, MemberRef]) -> None:
    for member, row in _rows(result).items():
        want = ref[member]
        assert _get(row, "n_selected") == want.n_selected, member
        assert _get(row, "n_races") == len(want.staked), member
        assert _get(row, "n_meetings") == len(set(want.meetings)), member
        if want.staked:
            assert float(_get(row, "rho")) == pytest.approx(want.rho, rel=1e-9), member
        else:
            assert not _finite(_get(row, "rho")), member


def _span(store_days: list[date]) -> tuple[date, date]:
    return min(store_days) - timedelta(days=3), max(store_days) + timedelta(days=3)


def _ranked_store(seed: int, *, boards: int, good: str, lam: float, first: date, m0: int = 120):
    rng = random.Random(seed)
    made = [_ranked_board(rng, k, lam=lam, good=good, m0=m0) for k in range(boards)]
    copies = sum(sum(b.wins.values()) for b in made)
    days = _race_days(first, math.ceil(copies / (len(VENUES) * RACES_PER_MEETING)))
    return _races(rng, made, days), _meta(lam, f"ranked-{seed}"), days


# ---------------------------------------------------------------------------------------------
# 1. The selection reads the final board
# ---------------------------------------------------------------------------------------------

TICKET_P = {1: 0.5, 2: 0.3, 3: 0.2}
TICKET_BOARD = {(1, 2): 2.5, (1, 3): 4.0, (2, 1): 5.0, (3, 1): 8.0, (2, 3): 12.0, (3, 2): 6.0}
TICKET_R = {(1, 2): 0.980, (1, 3): 0.926, (2, 1): 0.693, (3, 1): 0.745, (2, 3): 0.726,
            (3, 2): 1.653}  # fmt: skip


def _select(board, p, *, favourite, rate, lam, category, cut) -> list[tuple[int, int]]:
    try:
        chosen = ceiling.select_combinations(
            board, p, favourite=favourite, rate=rate, lambda2=lam, category=category, cut=cut
        )
    except ValueError:
        if category != "fav_first":
            raise
        return []
    return sorted(tuple(c) for c in chosen)


def _random_selection_case(rng: random.Random) -> dict:
    n = 5 + int(rng.random() * 5)
    p = {h: _uniform(rng, 0.2, 2.0) for h in range(1, n + 1)}  # not normalised on purpose
    p_pool = _normalised({h: v * math.exp(0.5 * _gauss(rng)) for h, v in p.items()})
    rate = 0.8 if rng.random() < 0.3 else 0.75
    lam = (0.8, 1.0, 1.2)[int(rng.random() * 3)]
    pool = _harville(p_pool, 1.0)
    board = {c: max(1.3, rate / s * _uniform(rng, 0.85, 1.2)) for c, s in pool.items()}
    by_p = max(p, key=lambda h: p[h])
    favourite = by_p if rng.random() < 0.7 else 1 + int(rng.random() * n)
    return {"board": board, "p": p, "rate": rate, "lam": lam, "favourite": favourite}


def test_selection_reads_the_final_board():
    # The ticket's race: r = q / p_c, checked against #131's own implied_share and harville_exacta.
    model = harville_exacta(np.array([0.5, 0.3, 0.2]), 1.0)
    reference = _harville(TICKET_P, 1.0)
    for (i, j), odds in TICKET_BOARD.items():
        r = implied_share(odds * 100, rate=0.75) / model[i - 1, j - 1]
        assert r == pytest.approx(_share(odds, 0.75) / reference[(i, j)], rel=1e-12)
        assert round(r, 3) == TICKET_R[(i, j)]

    def pick(category, cut):
        return _select(TICKET_BOARD, TICKET_P, favourite=1, rate=0.75, lam=1.0,
                       category=category, cut=cut)  # fmt: skip

    assert pick("fav_second", 0.8) == [(2, 1), (3, 1)]
    assert pick("no_fav", 0.8) == [(2, 3)]
    assert pick("fav_second", 0.7) == [(2, 1)]
    assert pick("no_fav", 0.7) == []
    # fav_first's r (0.980, 0.926) sits under a cut of 1.0, and it is still never selected.
    for cut in (*CUTS, 1.0, 2.0):
        assert pick("fav_first", cut) == []
        for category in RULE_CATEGORIES:
            assert all(first != 1 for first, _ in pick(category, cut))

    # A spread of races: field sizes, unnormalised p, lambda2, payout rate and favourite vary.
    rng = random.Random(431)
    hits = dict.fromkeys(((c, k) for c in RULE_CATEGORIES for k in CUTS), 0)
    lam_matters = rate_matters = cut_matters = False
    for _ in range(40):
        case = _random_selection_case(rng)
        args = {k: case[k] for k in ("board", "p", "favourite")}
        for category in RULE_CATEGORIES:
            picked = {}
            for cut in (*CUTS, _uniform(rng, 0.5, 1.2)):
                want = _ref_select(**args, rate=case["rate"], lam=case["lam"], category=category,
                                   cut=cut)  # fmt: skip
                got = _select(**args, rate=case["rate"], lam=case["lam"], category=category,
                              cut=cut)  # fmt: skip
                assert got == want, (case, category, cut)
                picked[cut] = want
                if cut in CUTS:
                    hits[(category, cut)] += len(want)
                plain = _ref_select(**args, rate=case["rate"], lam=1.0, category=category, cut=cut)
                lam_matters |= plain != want
                other = 0.75 if case["rate"] == 0.8 else 0.8
                moved = _ref_select(**args, rate=other, lam=case["lam"], category=category, cut=cut)
                rate_matters |= moved != want
            cut_matters |= picked[0.6] != picked[0.9]
            assert _select(**args, rate=case["rate"], lam=case["lam"], category="fav_first",
                           cut=2.0) == []  # fmt: skip
    assert all(hits.values()), hits  # every rule category and cut selects in the spread
    assert lam_matters and rate_matters and cut_matters


# ---------------------------------------------------------------------------------------------
# 2. Own impact: the payout identity
# ---------------------------------------------------------------------------------------------


def _money(rng, horses: int, pool: float, winner, w: float) -> dict[tuple[int, int], float]:
    """Money on every combination of ``horses`` summing to ``pool``, ``w`` on ``winner``."""
    combos = [(i, j) for i in range(1, horses + 1) for j in range(1, horses + 1) if i != j]
    rest = {c: _uniform(rng, 0.5, 1.5) for c in combos if c != winner}
    scale = (pool - w) / sum(rest.values())
    return {winner: w, **{c: v * scale for c, v in rest.items()}}


def test_own_impact_lowers_the_paid_return():
    # The ticket's case: P = 1,000,000 yen, W = 10,000 yen on the winner, 100 yen staked on it.
    rng = random.Random(4310)
    pool, w, s = 1_000_000.0, 10_000.0, 100.0
    dividend = 100 * 0.75 * pool / w  # the no-stake dividend per 100 yen: 7,500 yen
    money = _money(rng, 5, pool, (1, 2), w)
    staked, returned = ceiling.race_return(
        [(1, 2)],
        winner=(1, 2),
        dividend=dividend,
        pool_yen=pool,
        money_yen=money,
        stake=s,
        rate=0.75,
    )
    assert staked == pytest.approx(100.0)
    assert returned < dividend * s / 100
    assert returned == pytest.approx(dividend * (pool + s) / pool * w / (w + s), rel=1e-9)

    # A table: pool and winner money, stake, rate, how many combinations, winner bought or not.
    cases = [
        ([(1, 2), (2, 1), (3, 1)], (2, 1), 2_500_000.0, 40_000.0, 100.0, 0.8),
        ([(3, 1)], (3, 1), 500_000.0, 2_000.0, 1_000.0, 0.75),
        ([(1, 2), (1, 3)], (2, 3), 3_000_000.0, 25_000.0, 100.0, 0.75),
        ([], (2, 3), 3_000_000.0, 25_000.0, 100.0, 0.75),
    ]
    for _ in range(20):
        horses = 5 + int(rng.random() * 4)
        combos = [(i, j) for i in range(1, horses + 1) for j in range(1, horses + 1) if i != j]
        draws = 1 + int(rng.random() * 6)
        chosen = sorted({combos[int(rng.random() * len(combos))] for _ in range(draws)})
        winner = chosen[0] if rng.random() < 0.7 else combos[int(rng.random() * len(combos))]
        pool = _uniform(rng, 2e5, 5e7)
        cases.append((chosen, winner, pool, pool * _uniform(rng, 0.002, 0.2),
                      (100.0, 200.0, 1000.0)[int(rng.random() * 3)],
                      0.8 if rng.random() < 0.3 else 0.75))  # fmt: skip
    for chosen, winner, pool, w, s, rate in cases:
        horses = max(max(c) for c in [*chosen, winner])
        money = _money(rng, max(horses, 5), pool, winner, w)
        dividend = 100 * rate * pool / w
        staked, returned = ceiling.race_return(
            chosen,
            winner=winner,
            dividend=dividend,
            pool_yen=pool,
            money_yen=money,
            stake=s,
            rate=rate,
        )
        want_staked, want_returned = _ref_return(chosen, winner, dividend, pool, money, s)
        assert staked == pytest.approx(want_staked, rel=1e-12), (chosen, winner)
        assert returned == pytest.approx(want_returned, rel=1e-9, abs=1e-9), (chosen, winner)
        if winner in chosen:
            assert returned < dividend * s / 100


# ---------------------------------------------------------------------------------------------
# 3. rho is 1 when the pool prices like Harville
# ---------------------------------------------------------------------------------------------


def test_rho_is_near_one_when_the_pool_prices_like_harville(isolated, tmp_path, monkeypatch):
    """Each board's exacta shares are Harville's on the true Win probabilities (on its 1/M grid)
    and its copies' winners are drawn from them; the run's p is a noisy Win model, so r spreads
    and every member bets. Payout rates differ by race (0.75 and 0.8), lambda2 is 0.85."""
    lam = 0.85
    rng = random.Random(43101)
    boards = [_pool_board(rng, k, lam=lam) for k in range(28)]
    copies = sum(sum(b.wins.values()) for b in boards)
    days = _race_days(date(2024, 1, 6), math.ceil(copies / (len(VENUES) * RACES_PER_MEETING)))
    races = _races(rng, boards, days)
    ref = _reference(races, lam)
    assert all(ref[m].n_selected > 0 and len(set(ref[m].meetings)) >= 10 for m in MEMBERS)

    result, _, _ = _run(monkeypatch, tmp_path, races, _meta(lam, "pool"), date_from=days[0],
                        date_to=days[-1])  # fmt: skip
    _assert_matches_reference(result, ref)
    for member, row in _rows(result).items():
        rho, upper = float(_get(row, "rho")), float(_get(row, "upper"))
        assert abs(rho - 1.0) < 0.05, member
        # One-sided 95% upper bound of a meeting bootstrap: above rho, about as far above it as
        # the meeting bootstrap of the reference bets.
        want = ref[member]
        costs = [s * r for s, r in zip(want.staked, want.rates, strict=True)]
        boot = study.rho_interval(want.returned, costs, want.meetings, alternative="less",
                                  n_boot=4000, seed=7)  # fmt: skip
        width = boot.bound - boot.rho
        assert width > 0
        assert 0.5 * width <= upper - rho <= 2.0 * width, (member, upper - rho, width)


# ---------------------------------------------------------------------------------------------
# 4. A planted underbet category clears break-even
# ---------------------------------------------------------------------------------------------


def test_a_planted_underbet_category_clears_break_even(isolated, tmp_path, monkeypatch):
    lam = 0.9
    rng = random.Random(43102)
    boards = [_underbet_board(rng, k) for k in range(12)]
    copies = sum(sum(b.wins.values()) for b in boards)
    days = _race_days(date(2023, 3, 4), math.ceil(copies / (len(VENUES) * RACES_PER_MEETING)))
    races = _races(rng, boards, days)
    ref = _reference(races, lam)

    result, _, _ = _run(monkeypatch, tmp_path, races, _meta(lam, "underbet"), date_from=days[0],
                        date_to=days[-1])  # fmt: skip
    rows = _rows(result)
    _assert_matches_reference(result, ref)
    _assert_rank_order(result)
    for cell in CELLS:
        row = rows[("no_fav", 0.9, cell)]
        assert _get(row, "n_selected") > 0
        assert float(_get(row, "rho")) > 1 / 0.75, cell
    # fav_second is priced at its (grid) Harville share: a cut that buys some of it returns
    # about 1, below break-even; k = 0.6 buys none, and those members keep their rows, last.
    empty = [m for m in MEMBERS if not ref[m].staked]
    assert empty and all(m[0] == "fav_second" for m in empty)
    for member in empty:
        assert _get(rows[member], "n_selected") == 0
        assert not _finite(_get(rows[member], "rho"))
    for cut in CUTS:
        for cell in CELLS:
            rho = _get(rows[("fav_second", cut, cell)], "rho")
            assert not _finite(rho) or abs(float(rho) - 1.0) < 0.01


# ---------------------------------------------------------------------------------------------
# 5. The Quinella x order split
# ---------------------------------------------------------------------------------------------


def _split(race: dict, exacta_share: dict[tuple[int, int], float]) -> dict[str, float]:
    exacta = {c: _odds_for(s, race["exacta_rate"]) for c, s in exacta_share.items()}
    return ceiling.quinella_order_split(
        race["quinella"],
        exacta,
        race["p"],
        favourite=race["favourite"],
        lambda2=race["lambda2"],
        quinella_rate=race["quinella_rate"],
        exacta_rate=race["exacta_rate"],
    )


def test_quinella_order_split_matches_a_consistent_exacta_board():
    rng = random.Random(43103)
    planted = {"fav_first": 1.0, "fav_second": 1.25, "no_fav": 0.6}
    for _ in range(25):
        n = 5 + int(rng.random() * 5)
        p = {h: _uniform(rng, 0.2, 2.0) for h in range(1, n + 1)}
        lam = (0.8, 1.0, 1.25)[int(rng.random() * 3)]
        fav = max(p, key=lambda h: p[h]) if rng.random() < 0.7 else 1 + int(rng.random() * n)
        q_rate, e_rate = (0.775, 0.75) if rng.random() < 0.6 else (0.8, 0.775)
        pairs = {(i, j): _uniform(rng, 0.3, 1.7) for i in p for j in p if i < j}
        pair_share = _normalised(pairs)  # Quinella shares, not Harville's pair sums
        model = _harville(p, lam)
        exacta_share = {}
        for (i, j), s in pair_share.items():
            split = model[(i, j)] / (model[(i, j)] + model[(j, i)])
            exacta_share[(i, j)] = s * split
            exacta_share[(j, i)] = s * (1 - split)
        quinella = {pair: _odds_for(s, q_rate) for pair, s in pair_share.items()}
        race = {"quinella": quinella, "p": p, "favourite": fav, "lambda2": lam,
                "quinella_rate": q_rate, "exacta_rate": e_rate}  # fmt: skip

        consistent = _split(race, exacta_share)
        assert set(consistent) == set(CATEGORIES)
        for category in CATEGORIES:
            assert consistent[category] == pytest.approx(1.0, rel=1e-9), (category, n, lam)
        # An exacta board off the split by a category factor gives that factor back.
        off = _split(race, {c: s * planted[_category(c, fav)] for c, s in exacta_share.items()})
        for category in CATEGORIES:
            assert off[category] == pytest.approx(planted[category], rel=1e-9), (category, n)


# ---------------------------------------------------------------------------------------------
# 6. Every member is a ranked row (replaces the kill rule, owner reframe 2026-11-27)
# ---------------------------------------------------------------------------------------------


def test_every_member_is_a_ranked_row(isolated, tmp_path, monkeypatch):
    lam = 1.1
    races, meta, days = _ranked_store(43104, boards=6, good="no_fav", lam=lam,
                                      first=date(2022, 4, 2))  # fmt: skip
    ref = _reference(races, lam)
    above = {m for m in MEMBERS if ref[m].rho > 1 / 0.75}
    below = {m for m in MEMBERS if ref[m].rho < 1 / 0.75}
    assert above == {m for m in MEMBERS if m[0] == "no_fav"} and below == set(MEMBERS) - above

    result, _, _ = _run(monkeypatch, tmp_path, races, meta, date_from=days[0], date_to=days[-1])
    _assert_matches_reference(result, ref)
    _assert_rank_order(result)
    keys = [(_get(r, "category"), round(float(_get(r, "cut")), 2), _get(r, "cell"))
            for r in result.rows]  # fmt: skip
    last_above = max(i for i, k in enumerate(keys) if k in above)
    first_below = min(i for i, k in enumerate(keys) if k in below)
    assert last_above < first_below, keys
    for row in result.rows:
        assert _finite(_get(row, "upper")) and float(_get(row, "upper")) >= float(_get(row, "rho"))


# ---------------------------------------------------------------------------------------------
# 7. The freeze comes before every scored record
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["nothing_frozen", "frozen_with_another_spec"])
def test_run_refuses_an_unfrozen_family(isolated, tmp_path, monkeypatch, mode):
    races, meta, days = _ranked_store(43105, boards=2, good="no_fav", lam=1.0,
                                      first=date(2022, 9, 3), m0=40)  # fmt: skip
    registry = _registry(tmp_path, frozen=False)
    specs = ceiling.family_specs()
    assert ceiling.FAMILY_ID in specs
    if mode == "frozen_with_another_spec":
        for family_id, spec in specs.items():
            if family_id == ceiling.FAMILY_ID:
                spec = {**spec, "changed_after_the_freeze": True}
            registry.freeze_family(family_id, spec)
    store = FakeStore(races, meta)
    monkeypatch.setattr(ceiling, "load_races", store)
    out = tmp_path / "reports"
    with pytest.raises(ceiling.RunRefused):
        ceiling.run_ceiling(date_from=days[0], date_to=days[-1], out_dir=out, seed=431,
                            registry=registry)  # fmt: skip
    assert store.calls == 0, "the store was read before the freeze was checked"
    assert registry.records() == []
    assert not out.exists() or not any(out.rglob("*"))


def test_the_freeze_comes_before_every_scored_record(isolated, tmp_path, monkeypatch):
    races, meta, days = _ranked_store(43106, boards=3, good="fav_second", lam=0.95,
                                      first=date(2021, 5, 1), m0=60)  # fmt: skip
    result, _, registry = _run(monkeypatch, tmp_path, races, meta, date_from=days[0],
                               date_to=days[-1])  # fmt: skip
    _rows(result)
    specs = ceiling.family_specs()
    assert all("/" not in family_id for family_id in specs)
    frozen, scored = set(), 0
    for line in registry.path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["kind"] == "freeze":
            frozen.add(entry["family_id"])
        elif entry["record"]["status"] == "scored":
            scored += 1
            assert entry["record"]["family_id"] in frozen, entry["record"]["family_id"]
    records = registry.records()
    assert scored >= 16
    assert {r.family_id for r in records} <= set(specs)
    assert registry.family_size(ceiling.FAMILY_ID) == 16
    assert len([r for r in registry.records(ceiling.FAMILY_ID) if r.status == "scored"]) == 16
    assert all(is_withheld(r.sources) and r.ticket == 431 for r in records)
    assert any(p.is_file() for p in (tmp_path / "reports").rglob("*")), "no private report"


# ---------------------------------------------------------------------------------------------
# 8. No holdout day is read, nothing before 2014-06-07
# ---------------------------------------------------------------------------------------------


def _leaked_races(rng, days: list[date], lam: float) -> list[dict]:
    """Races every member would buy (every rule combination at r 0.45, headline kind), on days
    the run must not read: one race each a fav_second and a no_fav winner per meeting."""
    races = []
    for day in days:
        for venue in VENUES:
            for slot, wanted in enumerate(RULE_CATEGORIES, start=1):
                p = _flat_p(rng, 6)
                fav = max(p, key=lambda h: p[h])
                model = _harville(p, lam)
                odds = {c: 0.75 / ((1.0 if _category(c, fav) == "fav_first" else 0.45) * model[c])
                        for c in model}  # fmt: skip
                winner = next(c for c in sorted(model) if _category(c, fav) == wanted)
                board = _make_board(rng, p=p, kind=KINDS[0], rate=0.75, odds=odds,
                                    truth={winner: 1.0}, m0=1)  # fmt: skip
                races.append(_race(board, winner, day, venue, slot, "X"))
    return races


def test_run_reads_no_holdout_day(isolated, tmp_path, monkeypatch):
    lam = 1.0
    window = [date(2026, 11, 21) + timedelta(days=k) for k in range(41)]
    weekend = [d for d in window if d.weekday() >= 5]
    assert all(_ref_holdout(d) == is_holdout_day("japan", d) for d in weekend)
    held = [d for d in weekend if _ref_holdout(d)]
    dev_late = [d for d in weekend if not _ref_holdout(d)]
    early = [date(2014, 5, 31), date(2014, 6, 1)]  # before D10's first day
    dev_early = [date(2014, 6, 7), date(2014, 6, 8), date(2014, 6, 14)]
    assert len(held) >= 3 and len(dev_late) >= 3 and not any(map(_ref_holdout, dev_early))

    rng = random.Random(43107)
    made = [_ranked_board(rng, k, lam=lam, good="no_fav", m0=40) for k in range(4)]
    dev_days = dev_early + dev_late
    clean = _races(rng, made, dev_days)
    leaked = _leaked_races(rng, held + early, lam)
    date_from, date_to = date(2014, 5, 24), window[-1]

    base, base_store, _ = _run(monkeypatch, tmp_path / "clean", clean, _meta(lam, "clean"),
                               date_from=date_from, date_to=date_to)  # fmt: skip
    result, store, _ = _run(monkeypatch, tmp_path / "leaky", clean + leaked, _meta(lam, "clean"),
                            date_from=date_from, date_to=date_to)  # fmt: skip
    for asked in (base_store.asked, store.asked):
        assert asked, "the store was never read"
        assert not any(_ref_holdout(d) for d in asked), "a holdout day was read"
        assert all(FIRST_DAY <= d <= date_to for d in asked), "a day outside the span was read"
        assert {r["race_date"] for r in clean} <= set(asked)
    _assert_matches_reference(result, _reference(clean, lam))
    want = _rows(base)
    for member, row in _rows(result).items():
        for name in ("n_selected", "n_races", "n_meetings"):
            assert _get(row, name) == _get(want[member], name), (member, name)
        assert _get(row, "n_selected") > 0
        for name in ("rho", "upper"):
            assert float(_get(row, name)) == pytest.approx(float(_get(want[member], name))), member


# ---------------------------------------------------------------------------------------------
# 9. The public report carries no figure
# ---------------------------------------------------------------------------------------------


def test_public_report_carries_no_figures(isolated, tmp_path, monkeypatch):
    """Two stores with different spans, sizes and winners (no_fav ranks first in one, fav_second
    in the other) give results that differ in every figure; their public reports are the same."""
    a_races, a_meta, a_days = _ranked_store(43108, boards=3, good="no_fav", lam=1.0,
                                            first=date(2020, 2, 1), m0=50)  # fmt: skip
    b_races, b_meta, b_days = _ranked_store(43109, boards=5, good="fav_second", lam=1.0,
                                            first=date(2021, 10, 2), m0=80)  # fmt: skip
    date_from, date_to = date(2020, 1, 1), date(2022, 12, 31)
    first, _, _ = _run(monkeypatch, tmp_path / "a", a_races, a_meta, date_from=date_from,
                       date_to=date_to)  # fmt: skip
    second, _, _ = _run(monkeypatch, tmp_path / "b", b_races, b_meta, date_from=date_from,
                        date_to=date_to)  # fmt: skip
    rows_a, rows_b = _rows(first), _rows(second)
    for member in MEMBERS:
        for name in ("rho", "n_selected", "n_races"):
            assert _get(rows_a[member], name) != _get(rows_b[member], name), (member, name)
    assert [_get(r, "category") for r in first.rows] != [_get(r, "category") for r in second.rows]

    text_a = ceiling.render_public_report(first)
    text_b = ceiling.render_public_report(second)
    assert isinstance(text_a, str) and "withheld (D21)" in text_a
    assert "look-ahead" in text_a.lower()
    assert text_a == text_b, "the public report changes with the result's figures"
    for day in (*a_days, *b_days):  # no span: no race day of either store
        assert day.isoformat() not in text_a
