"""Ticket #618: run #100's C6 ``ShadowDecider`` on a development day and log its decisions through
#101's ``log_decisions``, with a C4 ``MetaFit`` behind the decider's ``meta[r].predict(q, view)``.

Contract pinned here:

- ``horsetrack.countries.japan.models.meta_adapter.MetaAdapter(fit, members)``: ``fit`` a
  ``models.meta.MetaFit``; ``members`` a frame of one row per runner with ``race_id``,
  ``horse_no`` (the saddle number the Win board's combinations name), the fit's member columns
  and its segment columns (other columns and other races allowed). ``predict(q, view)`` returns
  P(win) of the runners on ``view.pools["WIN"]``, one float per Win combination in the board's
  order, from ``fit.predict`` on those runners' rows with ``q`` (one Win draw's share row, in the
  board's order): the rows of ``view.race_id`` only, matched by horse number (never by row
  order), a runner of the race not on the Win board (scratched) left out, so the values sum to
  1 over the board's runners. ValueError when a Win-board runner has no row, or has two.
  The exotic side is ``models.exotic.Ordering`` itself: ``Ordering.trifecta(p_win)`` already is
  ``exotic[r].trifecta(p_win)``, indexed as the adapter's output.
- ``horsetrack.countries.japan.models.shadow.DecisionPoint(view, decision_utc,
  board_capture_utc, field_at_sale=None)``: one grid point of a day, its ``RaceView``, the
  aware decision and board-capture times, and the field at sale Place and Wide pay from.
- ``horsetrack.countries.japan.models.shadow.replay_day(day, points, *, decider, models,
  git_sha) -> int``: decides every point with ``decider.decide(point.view,
  field_at_sale=point.field_at_sale)`` and appends the decisions of each point through
  ``log_decisions`` (``mode`` 'replay', that point's own times) to the day's v0 ``ShadowLog``
  (``shadow_log_path(day)``), returning the records appended. On a holdout day (the committed
  rule, ``is_holdout_day``) nothing is decided and nothing is logged; a point whose race is not on
  ``day`` is a ValueError, checked before any point is decided or logged.

Every board, member value and fit here is synthetic; HORSETRACK_DATA_DIR points at tmp_path.
"""

from __future__ import annotations  # noqa: I001 (isort counts models.meta_adapter third-party until it exists)

import hashlib
import itertools
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from horsetrack.core import paths
from horsetrack.countries.japan.models.decision import PLACE, WIDE, WIN, ShadowDecider
from horsetrack.countries.japan.models.exotic import Ordering
from horsetrack.countries.japan.models.meta import MetaFit, MetaSpec
from horsetrack.countries.japan.models.meta_adapter import MetaAdapter
from horsetrack.countries.japan.models.nowcast import Nowcast, PoolBoard, RaceView
from horsetrack.countries.japan.models.shadow import DecisionPoint, replay_day
from horsetrack.countries.japan.shadow.log import ShadowLog, shadow_log_path

UTC = timezone.utc
REPO = Path(__file__).resolve().parents[4]
HOLDOUT_YAML = REPO / "src" / "horsetrack" / "countries" / "japan" / "config" / "holdout.yaml"
STAKE_LIKE = ("stake", "bet", "kelly", "amount", "wager")
MODELS = {"meta": "meta-618a", "nowcast": "now-618", "exotic": "ord-618"}
GIT_SHA = "618c0de"
N_DRAWS = 24


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    return data


# ---------------------------------------------------------------- helpers


def rule_days() -> tuple[date, date]:
    """(the first development day on or after the holdout start, the first holdout day), from
    the committed rule in holdout.yaml itself: a post-start day is held when
    int(sha256(f"{seed}:{day}"), 16) is even."""
    section = yaml.safe_load(HOLDOUT_YAML.read_text(encoding="utf-8"))["holdout"]
    start, seed = section["start"], section["seed"]
    start = start if isinstance(start, date) else date.fromisoformat(start)
    dev = held = None
    for offset in range(60):
        day = start + timedelta(days=offset)
        even = int(hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).hexdigest(), 16) % 2 == 0
        if even and held is None:
            held = day
        if not even and dev is None:
            dev = day
    assert dev is not None and held is not None
    return dev, held


def spec(members, *, segments=(), form="stage2", quadratic=False) -> MetaSpec:
    return MetaSpec(list(members), quadratic, list(segments), form, 0.0)


def meta_fit(s: MetaSpec, coef: dict[str, float]) -> MetaFit:
    names = [name for name, _, _ in s.terms()]
    assert set(coef) == set(names), (sorted(coef), names)
    series = pd.Series([coef[n] for n in names], index=names, dtype=float)
    return MetaFit(spec=s, coef=series, se=series * 0.0, loglik=0.0, n_races=1)


def expected_p(s: MetaSpec, coef: dict[str, float], rows: dict[int, dict], horses, q) -> np.ndarray:
    """P(win) from first principles: q normalised over the board's runners, u_i = sum_k (a_k0 +
    sum_j a_kj z_ij) m_ki (+ b ln q_i in stage2, + ln q_i in offset) (+ c (ln q_i)^2), softmax."""
    q = np.asarray(q, dtype=float)
    lq = np.log(q / q.sum())
    u = np.zeros(len(horses))
    for i, h in enumerate(horses):
        row = rows[h]
        for m in s.members:
            alpha = coef[m] + sum(coef[f"{m}:{z}"] * row[z] for z in s.segment_cols)
            u[i] += alpha * row[m]
    u += coef["ln_q_final"] * lq if s.form == "stage2" else lq
    if s.quadratic:
        u += coef["ln_q_final^2"] * lq * lq
    e = np.exp(u - u.max())
    return e / e.sum()


def board(combinations, shares, total) -> PoolBoard:
    shares = np.asarray(shares, dtype=float)
    shares = shares / shares.sum()
    return PoolBoard(
        combinations=tuple(combinations),
        shares=shares,
        total=float(total),
        shares_lag5=None,
        shares_lag15=None,
        board_min=2.0,
        odds=0.8 / shares,
        odds_hi=None,
    )


def win_view(race_id: str, horses, shares, *, tau: float = 2.0) -> RaceView:
    _, _, venue, race_no = race_id.split(":")
    pools = {WIN: board([str(h) for h in horses], shares, 400_000.0)}
    return RaceView(race_id, venue, int(race_no), None, float(tau), pools)


def full_view(race_id: str, horses, *, tau: float, tilt: float) -> RaceView:
    """Win, Place and Wide boards of ``horses`` (in number order), the market's strength falling
    with the horse's position in ``horses`` and bent by ``tilt``."""
    _, _, venue, race_no = race_id.split(":")
    rank = np.arange(len(horses), 0, -1, dtype=float)
    win = rank**tilt
    place = (rank + 3.0) / (rank + 3.0).sum()
    strength = dict(zip(horses, place, strict=True))
    pairs = [f"{a},{b}" for a, b in itertools.combinations(sorted(horses), 2)]
    wide = [strength[int(a)] * strength[int(b)] for a, b in (p.split(",") for p in pairs)]
    pools = {
        WIN: board([str(h) for h in horses], win, 300_000.0),
        PLACE: board([str(h) for h in horses], place, 150_000.0),
        WIDE: board(pairs, wide, 120_000.0),
    }
    return RaceView(race_id, venue, int(race_no), None, float(tau), pools)


def members_frame(by_race: dict[str, dict[int, dict]], seed: int) -> pd.DataFrame:
    """One row per (race, horse) with its member and segment values, rows shuffled so that row
    order says nothing about the board's order."""
    rows = [
        {"race_id": race_id, "horse_no": h, "horse_id": f"h-{race_id}-{h}", **v}
        for race_id, horses in by_race.items()
        for h, v in horses.items()
    ]
    order = np.random.default_rng(seed).permutation(len(rows))
    return pd.DataFrame([rows[i] for i in order]).reset_index(drop=True)


def stake_like_key(value) -> str | None:
    if isinstance(value, dict):
        for name, inner in value.items():
            if any(part in str(name).lower() for part in STAKE_LIKE):
                return str(name)
            found = stake_like_key(inner)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for inner in value:
            found = stake_like_key(inner)
            if found is not None:
                return found
    return None


def instant(text: str) -> datetime:
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert stamp.utcoffset() is not None
    return stamp


# ---------------------------------------------------------------- the adapter's probabilities

RACE_A = "jra:2026-11-21:TOK:11"
RACE_B = "jra:2026-11-21:KYO:05"
BOARD_A = (7, 2, 8, 1, 5, 3, 6)  # the Win board's order, not the frame's; horse 4 scratched


def adapter_rows(rng: np.random.Generator) -> dict[str, dict[int, dict]]:
    """Members m_speed, m_form (log-probabilities) and segment z_turf for race A (horses 1-8, 4
    scratched but still in the frame) and race B (horses 1-8, wildly different values)."""
    a = {
        h: {
            "m_speed": float(np.log(rng.uniform(0.03, 0.4))),
            "m_form": float(np.log(rng.uniform(0.03, 0.4))),
            "z_turf": 1.0,
        }
        for h in range(1, 9)
    }
    b = {
        h: {"m_speed": float(5.0 * h), "m_form": float(-3.0 * h), "z_turf": 0.0}
        for h in range(1, 9)
    }
    return {RACE_A: a, RACE_B: b}


ADAPTER_CASES = [
    pytest.param(
        spec(["m_speed", "m_form"], segments=["z_turf"], quadratic=True),
        {
            "m_speed": 0.45,
            "m_form": 0.2,
            "m_speed:z_turf": -0.15,
            "m_form:z_turf": 0.1,
            "ln_q_final": 0.8,
            "ln_q_final^2": -0.03,
        },
        id="stage2-segment-quadratic",
    ),
    pytest.param(
        spec(["m_speed"], form="offset"),
        {"m_speed": 0.7},
        id="offset",
    ),
    pytest.param(
        spec(["m_form", "m_speed"], form="offset", quadratic=True),
        {"m_form": -0.4, "m_speed": 0.25, "ln_q_final^2": 0.05},
        id="offset-quadratic",
    ),
    pytest.param(
        spec(["m_speed"]),
        {"m_speed": 0.0, "ln_q_final": 1.0},
        id="market-alone",  # P is q normalised over the board's runners
    ),
]


@pytest.mark.parametrize(("s", "coef"), ADAPTER_CASES)
def test_the_adapter_gives_p_win_of_the_win_boards_runners(s, coef):
    """DoD 1: P(win) of the Win board's runners from a Win draw, in board order, summing to 1."""
    rng = np.random.default_rng(618)
    by_race = adapter_rows(rng)
    frame = members_frame(by_race, seed=1)
    fit = meta_fit(s, coef)
    adapter = MetaAdapter(fit, frame)
    view = win_view(RACE_A, BOARD_A, rng.dirichlet(np.full(len(BOARD_A), 3.0)))

    draws = [view.pools[WIN].shares, *rng.dirichlet(np.full(len(BOARD_A), 1.5), size=4)]
    seen = []
    for q in draws:
        got = np.asarray(adapter.predict(q, view), dtype=float)
        assert got.shape == (len(BOARD_A),), "one P(win) per Win-board runner"
        assert np.all(got > 0) and got.sum() == pytest.approx(1.0, abs=1e-12)
        want = expected_p(s, coef, by_race[RACE_A], BOARD_A, q)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)
        seen.append(got)
    spread = max(np.abs(a - b).max() for a, b in itertools.combinations(seen, 2))
    assert spread > 1e-3, "P(win) must move with the Win draw q"

    # The same board in number order gives the same horses the same chances.
    sorted_view = win_view(RACE_A, sorted(BOARD_A), np.ones(len(BOARD_A)))
    q = draws[1]
    by_horse = dict(zip(BOARD_A, q, strict=True))
    q_sorted = np.array([by_horse[h] for h in sorted(BOARD_A)])
    again = dict(zip(sorted(BOARD_A), adapter.predict(q_sorted, sorted_view), strict=True))
    first = dict(zip(BOARD_A, adapter.predict(q, view), strict=True))
    for h in BOARD_A:
        assert again[h] == pytest.approx(first[h], abs=1e-12), f"horse {h} moved with the order"


def test_the_adapter_reads_its_own_race_and_its_members():
    """The rows are the view's race only, matched by horse number: race B's values or another
    fit's coefficients change the answer, the frame's row order does not."""
    rng = np.random.default_rng(7)
    by_race = adapter_rows(rng)
    s = spec(["m_speed", "m_form"], segments=["z_turf"])
    coef = {"m_speed": 0.6, "m_form": 0.3, "m_speed:z_turf": 0.2, "m_form:z_turf": -0.1}
    coef["ln_q_final"] = 0.9
    fit = meta_fit(s, coef)
    q = rng.dirichlet(np.full(len(BOARD_A), 2.0))
    view_a = win_view(RACE_A, BOARD_A, q)
    p_a = np.asarray(MetaAdapter(fit, members_frame(by_race, seed=2)).predict(q, view_a))
    shuffled = np.asarray(MetaAdapter(fit, members_frame(by_race, seed=99)).predict(q, view_a))
    np.testing.assert_allclose(p_a, shuffled, atol=1e-12)
    np.testing.assert_allclose(p_a, expected_p(s, coef, by_race[RACE_A], BOARD_A, q), atol=1e-10)

    # Race B on the same horse numbers: its own rows, so its own (very different) chances.
    view_b = win_view(RACE_B, BOARD_A, q)
    p_b = np.asarray(MetaAdapter(fit, members_frame(by_race, seed=2)).predict(q, view_b))
    np.testing.assert_allclose(p_b, expected_p(s, coef, by_race[RACE_B], BOARD_A, q), atol=1e-10)
    assert np.abs(p_a - p_b).max() > 0.05

    # A different refit of C4 gives different chances from the same draw.
    other = dict(coef, m_speed=-0.5)
    p_other = np.asarray(
        MetaAdapter(meta_fit(s, other), members_frame(by_race, 2)).predict(q, view_a)
    )
    np.testing.assert_allclose(
        p_other, expected_p(s, other, by_race[RACE_A], BOARD_A, q), atol=1e-10
    )
    assert np.abs(p_other - p_a).max() > 1e-3


def test_the_exotic_trifecta_is_indexed_as_the_adapters_output():
    """``Ordering.trifecta`` of the adapter's P(win) is the (n, n, n) array the decider reads,
    indexed as the Win board: its first-place margin is the adapter's P(win), in board order."""
    rng = np.random.default_rng(11)
    by_race = adapter_rows(rng)
    s = spec(["m_speed"])
    fit = meta_fit(s, {"m_speed": 0.5, "ln_q_final": 0.85})
    adapter = MetaAdapter(fit, members_frame(by_race, seed=3))
    view = win_view(RACE_A, BOARD_A, np.ones(len(BOARD_A)))
    q = rng.dirichlet(np.full(len(BOARD_A), 2.0))
    p = np.asarray(adapter.predict(q, view), dtype=float)
    n = len(BOARD_A)
    for ordering in (
        Ordering("harville"),
        Ordering("discounted_harville", {"lambda2": 0.8, "lambda3": 0.7}),
    ):
        t = np.asarray(ordering.trifecta(p), dtype=float)
        assert t.shape == (n, n, n)
        assert t.sum() == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(t.sum(axis=(1, 2)), p, atol=1e-9)
        for i in range(n):
            assert t[i, i, :].sum() == 0 and t[i, :, i].sum() == 0 and t[:, i, i].sum() == 0


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(
            lambda f: f[~((f.race_id == RACE_A) & (f.horse_no == 5))], id="runner-missing"
        ),
        pytest.param(
            lambda f: pd.concat([f, f[(f.race_id == RACE_A) & (f.horse_no == 2)]]),
            id="runner-twice",
        ),
        pytest.param(lambda f: f[f.race_id != RACE_A], id="race-missing"),
    ],
)
def test_the_adapter_refuses_a_win_runner_it_cannot_price_once(broken):
    rng = np.random.default_rng(5)
    frame = broken(members_frame(adapter_rows(rng), seed=4)).reset_index(drop=True)
    fit = meta_fit(spec(["m_speed"]), {"m_speed": 0.5, "ln_q_final": 1.0})
    view = win_view(RACE_A, BOARD_A, np.ones(len(BOARD_A)))
    with pytest.raises(ValueError):
        MetaAdapter(fit, frame).predict(np.full(len(BOARD_A), 1.0 / len(BOARD_A)), view)


# ---------------------------------------------------------------- a replayed day


def day_points(day: date) -> tuple[list[DecisionPoint], pd.DataFrame]:
    """Two races of ``day`` at two grid points each: race 11 with eight runners, race 3 with
    horse 4 scratched after sales opened to eight (Place pays three from the field at sale)."""
    rng = np.random.default_rng(int(day.strftime("%Y%m%d")))
    races = {
        f"jra:{day.isoformat()}:TOK:11": ((1, 2, 3, 4, 5, 6, 7, 8), None),
        f"jra:{day.isoformat()}:KYO:03": ((1, 2, 3, 5, 6, 7, 8), 8),
    }
    by_race = {
        race_id: {
            h: {
                "m_speed": float(np.log(rng.uniform(0.04, 0.35))),
                "m_form": float(np.log(rng.uniform(0.04, 0.35))),
                "z_turf": float(k % 2),
            }
            for h in (1, 2, 3, 4, 5, 6, 7, 8)  # the scratched horse keeps its member rows
        }
        for k, race_id in enumerate(races)
    }
    points = []
    for k, (race_id, (horses, field)) in enumerate(races.items()):
        post = datetime(day.year, day.month, day.day, 5 + k, 40, tzinfo=UTC)
        for tau, tilt in ((5, 1.1), (2, 1.3)):
            decided = post - timedelta(minutes=tau)
            points.append(
                DecisionPoint(
                    view=full_view(race_id, horses, tau=tau, tilt=tilt + 0.1 * k),
                    decision_utc=decided,
                    board_capture_utc=decided - timedelta(seconds=11 + 7 * k + tau),
                    field_at_sale=field,
                )
            )
    return points, members_frame(by_race, seed=8)


def c6_decider(members: pd.DataFrame) -> ShadowDecider:
    s = spec(["m_speed", "m_form"], segments=["z_turf"])
    fits = [
        meta_fit(
            s,
            {
                "m_speed": 0.5,
                "m_form": 0.2,
                "m_speed:z_turf": 0.1,
                "m_form:z_turf": 0.0,
                "ln_q_final": 0.85,
            },
        ),  # fmt: skip
        meta_fit(
            s,
            {
                "m_speed": 0.3,
                "m_form": 0.35,
                "m_speed:z_turf": -0.1,
                "m_form:z_turf": 0.05,
                "ln_q_final": 0.95,
            },
        ),  # fmt: skip
    ]
    return ShadowDecider(
        mode="shadow",
        meta=[MetaAdapter(fit, members) for fit in fits],
        nowcast=Nowcast.from_params(phi=0.6, sigma=0.5),
        exotic=[
            Ordering("harville"),
            Ordering("discounted_harville", {"lambda2": 0.8, "lambda3": 0.7}),
        ],
        n_draws=N_DRAWS,
        seed=3,
    )


class Spy:
    """A decider that records every view it is asked to decide and delegates to the real one."""

    def __init__(self, inner: ShadowDecider) -> None:
        self.inner = inner
        self.decided: list[str] = []

    def decide(self, view, *, field_at_sale=None):
        self.decided.append(view.race_id)
        return self.inner.decide(view, field_at_sale=field_at_sale)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_a_replayed_day_logs_one_decision_record_per_combination():
    """DoD 2: every combination of every pool at every grid point is one record in the day's v0
    shadow log, carrying the C6 decider's EV per yen over its refits, and no stake-like key."""
    dev, _ = rule_days()
    points, members = day_points(dev)
    decider = c6_decider(members)
    appended = replay_day(dev, points, decider=decider, models=MODELS, git_sha=GIT_SHA)

    expected = {}
    for point in points:
        for d in c6_decider(members).decide(point.view, field_at_sale=point.field_at_sale):
            key = (d.race_id, d.pool, d.combination, int(d.tau_min))
            expected[key] = (point, np.asarray(d.ev, dtype=float))
    n_combinations = sum(len(b.combinations) for p in points for b in p.view.pools.values())
    assert len(expected) == n_combinations, "the decider prices every combination once"
    assert appended == n_combinations

    log = ShadowLog(shadow_log_path(dev), day=dev)
    records = log.read()
    keys = [(r["race_id"], r["pool"], r["combination"], r["tau_min"]) for r in records]
    assert len(keys) == len(set(keys)) == n_combinations, "one record per combination, once each"
    assert set(keys) == set(expected)
    assert log.verify().day_status == "development"
    for record, key in zip(records, keys, strict=True):
        point, ev = expected[key]
        assert stake_like_key(record) is None, f"stake-like key {stake_like_key(record)!r}"
        assert type(record["tau_min"]) is int
        assert record["mode"] == "replay" and record["status"] == "ok"
        assert record["models"] == MODELS and record["git_sha"] == GIT_SHA
        assert instant(record["decision_utc"]) == point.decision_utc, "the point's own time"
        assert instant(record["board_capture_utc"]) == point.board_capture_utc
        assert record["n_refits"] == 2
        assert record["ev_mean"] == pytest.approx(float(ev.mean()), abs=1e-12)
        lo, hi = np.quantile(ev, (0.05, 0.95))
        assert record["ev_q05"] == pytest.approx(float(lo), abs=1e-12)
        assert record["ev_q95"] == pytest.approx(float(hi), abs=1e-12)
        board_ = point.view.pools[key[1]]
        j = board_.combinations.index(key[2])
        assert record["board_odds"] == pytest.approx(float(board_.odds[j]))

    # The field at sale reaches the decider: race 3's Place pays three, not the two of seven.
    race3 = points[2]
    assert race3.field_at_sale == 8
    two_places = c6_decider(members).decide(race3.view)
    by_key = {(r["race_id"], r["pool"], r["combination"], r["tau_min"]): r for r in records}
    moved = [
        abs(by_key[(d.race_id, d.pool, d.combination, int(d.tau_min))]["ev_mean"] - d.ev.mean())
        for d in two_places
        if d.pool == PLACE
    ]
    assert max(moved) > 1e-3, "Place was priced as if sold to the seven left"

    # The EVs spread: a constant or v0's value would not match a decider's refits this way.
    assert len({round(r["ev_mean"], 9) for r in records}) > n_combinations // 2
    assert any(r["ev_q95"] - r["ev_q05"] > 1e-6 for r in records), "refits disagree somewhere"
    assert {r["pool"] for r in records} == {WIN, PLACE, WIDE}


def test_no_holdout_day_is_decided_or_logged():
    """DoD 3: a holdout day by the committed rule is neither decided nor logged through this
    path, and a holdout race cannot ride in under a development day."""
    dev, held = rule_days()
    held_points, held_members = day_points(held)
    spy = Spy(c6_decider(held_members))
    try:
        appended = replay_day(held, held_points, decider=spy, models=MODELS, git_sha=GIT_SHA)
    except (ValueError, PermissionError):
        appended = 0
    assert appended == 0
    assert spy.decided == [], "a holdout day was decided"
    held_log = shadow_log_path(held)
    assert not held_log.exists() or ShadowLog(held_log, day=held).verify().n_records == 0

    # A development day's batch carrying a race of the holdout day is refused before anything is
    # decided or written, on either day's log.
    dev_points, dev_members = day_points(dev)
    members = pd.concat([dev_members, held_members], ignore_index=True)
    mixed = Spy(c6_decider(members))
    with pytest.raises(ValueError):
        replay_day(dev, [*dev_points[:2], held_points[0], *dev_points[2:]], decider=mixed,
                   models=MODELS, git_sha=GIT_SHA)  # fmt: skip
    assert held_points[0].view.race_id not in mixed.decided
    for day in (dev, held):
        path = shadow_log_path(day)
        assert not path.exists() or ShadowLog(path, day=day).verify().n_records == 0, day

    # The same development points alone are decided and logged: the refusal was the holdout race.
    assert replay_day(dev, dev_points, decider=mixed, models=MODELS, git_sha=GIT_SHA) > 0
    assert math.isfinite(ShadowLog(shadow_log_path(dev), day=dev).read()[0]["ev_mean"])
