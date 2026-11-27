"""Ticket #101: log shadow decisions, settle them at the final dividends, score their calibration,
drift and log growth, and report the owner's hand-played bets by day beside each option's rank.

The owner's reframe of 2026-11-27 applies: no Kelly or other staking variants (E46 is out), flat
stakes under the race-day cap, the shadow log is the v0 ``ShadowLog`` (#119) and the manual bet log
is #453's ``BetLog``, settled by #454's ``settle_bets``. Nothing here places a bet.

Contract pinned here, module ``horsetrack.countries.japan.models.shadow``:

- ``log_decisions(log, decisions, *, decision_utc, board_capture_utc, mode, models, git_sha) ->
  int``: appends one v0 shadow-log record per #100 ``ShadowDecision`` through
  ``ShadowLog.append`` (so the batch is written whole or not at all, a repeated
  ``(race_id, pool, combination, tau_min)`` is refused, a sealed log refuses more) and returns the
  count. A record carries the decision's key (``tau_min`` as the log's int), its ``board_odds``
  (the board's figure at t for its combination), the claimed EV per yen (the mean of
  ``decision.ev`` over the refits) and no stake-like key at any depth (the flat stake priced into
  the dividend is not logged under a stake name).
- ``read_shadow(days) -> DataFrame``: the records of each development day's shadow log, read
  through ``ShadowLog.read``; a holdout day gives no record (its log stays sealed, ruling) and an
  absent day none.
- ``settle_shadow(log, dividends) -> DataFrame``: one row per record, in order, its columns kept,
  plus ``outcome``, ``return_per_100``, ``claimed_ev`` and ``realised``. ``dividends`` are #454's
  rows (``race_id, pool, combination, dividend, is_refund``). Each record is a paper ticket of
  100 yen:
  - 'won': its combination is a paying row of its race and pool; ``return_per_100`` is that row's
    dividend (a dead heat, Place and Wide pay each paying combination its own);
  - 'lost': its race and pool have rows, none its combination; ``return_per_100`` 0;
  - 'refunded': its combination has an ``is_refund`` row (a runner scratched before the start);
    the stake back, no return, so ``return_per_100`` and ``realised`` are null;
  - 'unsettled': no dividend row of its pool in its race yet;
  - 'no_board': a ``no_board`` record (pool and combination ``*``);
  - 'holdout': labelled holdout, or of a day the committed holdout rule holds, whatever dividends
    are passed; never settled here.
  ``claimed_ev`` is the net EV per yen the record claims: the logged decision's mean EV, or a v0
  record's ``value_mean - 1`` (``value_mean`` is the gross return per yen); null without one.
  ``realised`` is the net return per yen at the final dividend, ``return_per_100 / 100 - 1``, for
  won and lost records only.
- ``calibration_slope(claimed_ev, realised, clusters, *, n_boot, level, seed)``: an
  ``Interval``-shaped result (``mean``, ``lo``, ``hi``, ``n_clusters``, the semantics of
  ``core.eval.bootstrap.Interval``): ``mean`` the OLS slope, with an intercept, of realised on
  claimed over every pair, ``lo``/``hi`` the one-sided (1 - level) and level quantiles of the slope
  over replicates that resample whole clusters. Pairs with a null claimed or realised value are
  left out (so refunded records never enter it).
- ``flat_stakes(picks, *, limits) -> DataFrame``: the picks, same rows and order, with
  ``stake_yen``: the flat stake while the race day's total (every venue of the day together, the
  day from ``race_id``) stays within ``limits.day_cap_yen``, else 0. Picks take the cap in order
  of ``decision_utc`` (as instants, whatever the offset), then ``rank`` (1 first) (ruling).
  ValueError without a flat stake.
- ``log_growth_by_day(settled, *, bankroll_yen) -> DataFrame``: one row per race day with
  ``day`` and ``log_growth`` = ln(ending / starting money), the day starting with
  ``bankroll_yen`` (ruling): ending = bankroll - stakes of won, lost and refunded rows + stake /
  100 x ``return_per_100`` of won rows + the stakes refunded. Pending rows (unsettled, no_board,
  holdout) are left out; a day with nothing settled has no growth figure (no row, or null).
- ``selection_drift(log, *, lag_minutes, n_boot, level, seed)`` (E20): regresses the fall in log
  odds from t to the final, ln(``board_odds``) - ln(``final_odds``) of the record at t, on the
  overlay score (its claimed EV) of the record of the same race, pool and combination
  ``lag_minutes`` before t (``tau_min + lag_minutes``), never the record at t itself. Rows without
  that record, a score, a board or a final figure, and holdout rows, are dropped, never filled.
  The result has ``slope``, ``lo``, ``hi`` (as ``calibration_slope``) and ``n``, the rows used.
  ValueError for a lag that is not positive.
- ``read_bets(days) -> DataFrame``: each day's ``BetLog.read()`` rows (holdout days too: stakes are
  not outcomes); ``played_by_day(settled, ranked) -> DataFrame``: per race day and ``option_id``
  of ``settle_bets`` rows (``off-board`` for a null one), #454's ``n_bets``, ``staked_yen``,
  ``paid_yen``, ``return_per_100``, ``n_refunded``, ``n_pending``, and ``rank``, the option's rank
  on the given leaderboard (``leaderboard.rank`` rows), null off the board.
- The module imports no HTTP client, browser driver, network, subprocess or ingest code.

Every record, bet and dividend here is synthetic. HORSETRACK_DATA_DIR points at tmp_path in every
test.
"""

from __future__ import annotations  # noqa: I001 (isort counts models.shadow third-party until it exists)

import ast
import hashlib
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.eval.leaderboard import LeaderboardRow, rank
from horsetrack.countries.japan.bets.log import BetLog, Limits, bet_log_path, load_limits
from horsetrack.countries.japan.bets.settle import settle_bets
from horsetrack.countries.japan.models.decision import ShadowDecision
from horsetrack.countries.japan.models.nowcast import PoolBoard
from horsetrack.countries.japan.models.shadow import (
    calibration_slope,
    flat_stakes,
    log_decisions,
    log_growth_by_day,
    played_by_day,
    read_bets,
    read_shadow,
    selection_drift,
    settle_shadow,
)
from horsetrack.countries.japan.shadow.log import ShadowLog, shadow_log_path

UTC = timezone.utc
JST = timezone(timedelta(hours=9))
REPO = Path(__file__).resolve().parents[3]
JAPAN_CONFIG = REPO / "src" / "horsetrack" / "countries" / "japan" / "config"
SHADOW_MODULE = REPO / "src" / "horsetrack" / "countries" / "japan" / "models" / "shadow.py"
D1 = date(2026, 11, 21)  # before the holdout start: development by any reading
D2 = date(2026, 11, 22)
D3 = date(2026, 11, 15)
DIVIDEND_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
SETTLE_ADDED = {"outcome", "return_per_100", "claimed_ev", "realised"}
OUTCOMES = {"won", "lost", "refunded", "unsettled", "no_board", "holdout"}
STAKE_LIKE = ("stake", "bet", "kelly", "amount", "wager")
FORBIDDEN_IMPORTS = (
    "requests",
    "httpx",
    "urllib",
    "urllib3",
    "aiohttp",
    "socket",
    "http",
    "ftplib",
    "smtplib",
    "playwright",
    "selenium",
    "webbrowser",
    "subprocess",
    "horsetrack.core.ingest",
)


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture(autouse=True)
def D(tmp_path, monkeypatch) -> Path:
    data = (tmp_path / "data").resolve()
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(data))
    return data


def first_held_day() -> date:
    """The first holdout day, from the committed rule in holdout.yaml itself."""
    section = yaml.safe_load((JAPAN_CONFIG / "holdout.yaml").read_text(encoding="utf-8"))
    start, seed = section["holdout"]["start"], section["holdout"]["seed"]
    start = start if isinstance(start, date) else date.fromisoformat(start)
    for offset in range(60):
        day = start + timedelta(days=offset)
        if int(hashlib.sha256(f"{seed}:{day.isoformat()}".encode()).hexdigest(), 16) % 2 == 0:
            assert is_holdout_day("japan", day)
            return day
    raise AssertionError("no holdout day in the 60 days from the start")


def race(n: int, venue: str = "TOK", day: date = D1) -> str:
    return f"jra:{day.isoformat()}:{venue}:{n:02d}"


def paying(race_id: str, pool: str, combination: str, dividend: float, refund: bool = False):
    return {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "dividend": float(dividend),
        "is_refund": refund,
    }


def refund_row(race_id: str, pool: str, combination: str) -> dict:
    return paying(race_id, pool, combination, float("nan"), refund=True)


def dividends(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DIVIDEND_COLUMNS)


def v0_record(
    race_id: str,
    pool: str,
    combination: str,
    *,
    tau: int = 2,
    value_mean: float | None = 0.9,
    board_odds: float | None = 3.4,
    day_status: str = "development",
    status: str = "ok",
    **extra,
) -> dict:
    """A v0 shadow-day record as ``ShadowLog.read()`` returns it (docs/specs/shadow-log.md)."""
    day = race_id.split(":")[1]
    ok = status == "ok"
    record = {
        "race_id": race_id,
        "pool": pool if ok else "*",
        "combination": combination if ok else "*",
        "tau_min": tau,
        "decision_utc": f"{day}T05:38:00+00:00",
        "board_capture_utc": f"{day}T05:37:44+00:00" if ok else None,
        "written_utc": f"{day}T05:38:16+00:00",
        "mode": "live",
        "status": status,
        "models": {"win": "a1", "ordering": "b2", "nowcast": "c3"} if ok else None,
        "git_sha": "0123abc",
        "board_odds": board_odds if ok else None,
        "board_odds_hi": None,
        "value_mean": value_mean if ok else None,
        "value_q05": None,
        "value_q95": None,
        "value_at_low": None,
        "value_at_high": None,
        "flags": [],
        "n_draws": 400 if ok else None,
        "seed": 1 if ok else None,
        "day_status": day_status,
    }
    record.update(extra)
    return record


def key_of(row: dict) -> tuple:
    return (row["race_id"], row["pool"], row["combination"], int(row["tau_min"]))


def settled_by_key(records: list[dict], rows: list[dict] | pd.DataFrame) -> dict[tuple, dict]:
    """``settle_shadow`` on the records, checked for shape, as {record key: settled row}."""
    frame = pd.DataFrame(records)
    out = settle_shadow(frame, rows if isinstance(rows, pd.DataFrame) else dividends(rows))
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(frame), "one settled row per record: none dropped, none added"
    assert set(frame.columns) | SETTLE_ADDED <= set(out.columns), "the record's columns are kept"
    got = out.to_dict("records")
    assert [key_of(r) for r in got] == [key_of(r) for r in records], "rows stay in record order"
    assert {r["outcome"] for r in got} <= OUTCOMES
    return {key_of(r): r for r in got}


def null(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


def stake_like_key(value) -> str | None:
    """The first key, at any depth, naming a stake, bet, kelly, amount or wager."""
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


def pool_board(combinations: list[str], odds: list[float]) -> PoolBoard:
    k = len(combinations)
    return PoolBoard(
        combinations=tuple(combinations),
        shares=np.full(k, 1.0 / k),
        total=1_000_000.0,
        shares_lag5=None,
        shares_lag15=None,
        board_min=30.0,
        odds=np.array(odds, dtype=float),
        odds_hi=None,
    )


def decisions_of(race_id: str, tau: float, pools: dict[str, tuple]) -> list[ShadowDecision]:
    """#100's ShadowDecision rows for one race at one tau; pools maps a pool to (combinations,
    odds at t, per-combination EV draws over the refits)."""
    out = []
    for pool, (combinations, odds, evs) in pools.items():
        board = pool_board(combinations, odds)
        for combination, ev in zip(combinations, evs, strict=True):
            out.append(
                ShadowDecision(
                    race_id=race_id,
                    pool=pool,
                    combination=combination,
                    tau_min=float(tau),
                    board=board,
                    ev=np.array(ev, dtype=float),
                    stake_yen=1_000,
                )
            )
    return out


def instant(text: str) -> datetime:
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert stamp.utcoffset() is not None, f"{text} has no UTC offset"
    return stamp


def board_row(option_id: str, per_100: float | None) -> LeaderboardRow:
    """A synthetic leaderboard row; ``per_100`` None is an unmeasured (pending) option."""
    measured = per_100 is not None
    return LeaderboardRow(
        option_id=option_id,
        family_id="japan-test-family",
        ticket=101,
        kind="rule",
        pool="WIN",
        segment="all",
        rule=f"rule {option_id}",
        selection="decision_time",
        sources=("jra",),
        variants_searched=1,
        status="ok" if measured else "pending",
        return_per_100=per_100,
        ci_lo=per_100 - 10.0 if measured else None,
        ci_hi=per_100 + 10.0 if measured else None,
        ci_level=0.95 if measured else None,
        n=500 if measured else None,
        n_unit="bets" if measured else None,
    )


def bet(race_id: str, pool: str, combination: str, stake: int, option_id: str | None, note: str):
    day = race_id.split(":")[1]
    return {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "stake_yen": stake,
        "placed_utc": f"{day}T05:40:00+00:00",
        "option_id": option_id,
        "note": note,
    }


def day_text(value) -> str:
    """A ``day`` cell as YYYY-MM-DD, whether it is a date, a Timestamp or ISO text."""
    return str(value)[:10]


# ---------------------------------------------------------------- calibration


def synthetic_claims(n: int, seed: int, truth, *, meetings: int = 50) -> pd.DataFrame:
    """n paper bets in ``meetings`` clusters. Claimed EV (net, per yen) is uniform on [-0.6, 0.6];
    each bet wins with probability (1 + truth(claimed)) / odds at final odds on [1.6, 3.2], so its
    realised net return (odds - 1 if it wins, -1 if not) has expectation truth(claimed)."""
    rng = np.random.default_rng(seed)
    odds = rng.uniform(1.6, 3.2, n)
    claimed = rng.uniform(-0.6, 0.6, n)
    p = (1.0 + truth(claimed)) / odds
    assert ((p >= 0) & (p <= 1)).all()
    won = rng.random(n) < p
    index = [f"b{i:05d}" for i in range(n)]
    cluster = [f"jra:2026-11-{1 + m % 28:02d}:M{m:02d}" for m in np.arange(n) % meetings]
    return pd.DataFrame(
        {"claimed": claimed, "realised": np.where(won, odds - 1.0, -1.0), "cluster": cluster},
        index=index,
    )


def slope_of(frame: pd.DataFrame, **kw):
    return calibration_slope(frame["claimed"], frame["realised"], frame["cluster"], **kw)


def test_calibration_slope_is_one_on_synthetic_calibrated_claims():
    boot = {"n_boot": 300, "level": 0.99, "seed": 7}
    calibrated = synthetic_claims(5_000, 2026, lambda c: c)
    got = slope_of(calibrated, **boot)
    ols = np.polyfit(calibrated["claimed"], calibrated["realised"], 1)[0]  # first principles
    assert got.mean == pytest.approx(ols, abs=1e-9), "mean is the OLS slope with an intercept"
    assert got.lo <= got.mean <= got.hi
    assert got.lo < 1.0 < got.hi, f"calibrated claims: [{got.lo:.3f}, {got.hi:.3f}] misses 1"
    assert got.hi - got.lo < 0.5, "an interval this wide proves nothing at 5,000 bets"
    assert got.n_clusters == 50
    again = slope_of(calibrated, **boot)
    assert (again.mean, again.lo, again.hi, again.n_clusters) == (
        got.mean,
        got.lo,
        got.hi,
        got.n_clusters,
    ), "the same seed gives the same interval"

    # Miscalibrated claims: the slope moves with the truth, so a constant 1 cannot pass.
    halved = synthetic_claims(5_000, 2027, lambda c: 0.5 * c)
    half = slope_of(halved, **boot)
    assert half.mean == pytest.approx(np.polyfit(halved["claimed"], halved["realised"], 1)[0])
    assert half.hi < 1.0, f"realised = 0.5 x claimed: [{half.lo:.3f}, {half.hi:.3f}] covers 1"

    # A slope of 0.6 with an intercept of -0.1: the ratio of mean gross returns is 0.9, so only a
    # regression with an intercept lands near 0.6.
    shifted = synthetic_claims(5_000, 2028, lambda c: -0.1 + 0.6 * c)
    tilt = slope_of(shifted, **boot)
    assert tilt.mean == pytest.approx(np.polyfit(shifted["claimed"], shifted["realised"], 1)[0])
    assert tilt.hi < 0.85, f"slope 0.6: [{tilt.lo:.3f}, {tilt.hi:.3f}]"
    assert tilt.lo < tilt.mean < tilt.hi


def test_calibration_slope_resamples_whole_clusters():
    # Ten meetings, each with its own true slope (1 + w, w from -0.9 to 0.9): the slope's spread
    # is between meetings, which only a bootstrap of whole meetings sees.
    rng = np.random.default_rng(11)
    n, meetings = 5_000, 10
    meeting = np.repeat(np.arange(meetings), n // meetings)
    odds = rng.uniform(1.8, 3.2, n)
    claimed = rng.uniform(-0.4, 0.4, n)
    w = np.linspace(-0.9, 0.9, meetings)[rng.permutation(meetings)][meeting]
    p = (1.0 + claimed * (1.0 + w)) / odds
    realised = np.where(rng.random(n) < p, odds - 1.0, -1.0)
    index = [f"b{i:05d}" for i in range(n)]
    x, y = pd.Series(claimed, index=index), pd.Series(realised, index=index)
    by_meeting = pd.Series([f"jra:2026-11-21:M{m}" for m in meeting], index=index)
    by_bet = pd.Series(index, index=index)

    boot = {"n_boot": 200, "level": 0.99, "seed": 3}
    clustered = calibration_slope(x, y, by_meeting, **boot)
    per_bet = calibration_slope(x, y, by_bet, **boot)
    assert clustered.n_clusters == meetings and per_bet.n_clusters == n
    assert clustered.mean == pytest.approx(per_bet.mean), "the point slope ignores the clusters"
    width, narrow = clustered.hi - clustered.lo, per_bet.hi - per_bet.lo
    assert width > 1.5 * narrow, f"meeting bootstrap {width:.3f} vs per-bet {narrow:.3f}"


# ---------------------------------------------------------------- the shadow log


def test_the_shadow_log_is_append_only(D):
    path = shadow_log_path(D1)
    log = ShadowLog(path, day=D1)
    r11, r12 = race(11), race(12, "KYO")
    meta = {
        "decision_utc": datetime(2026, 11, 21, 5, 38, tzinfo=UTC),
        "board_capture_utc": datetime(2026, 11, 21, 5, 37, 44, tzinfo=UTC),
        "mode": "replay",
        "models": {"meta": "m-1", "nowcast": "n-1", "exotic": "x-1"},
        "git_sha": "0123abc",
    }
    first = decisions_of(
        r11,
        5,
        {
            "WIN": (["3", "4", "7"], [5.6, 2.1, 11.0], [[0.12, -0.05, 0.31], [-0.2], [0.4, 0.5]]),
            "PLACE": (["3", "4"], [1.8, 1.2], [[0.02, 0.04], [-0.11, -0.09]]),
        },
    )
    second = decisions_of(
        r11, 2, {"WIN": (["3", "4", "7"], [5.2, 2.2, 12.5], [[0.08], [-0.25, -0.15], [0.61]])}
    ) + decisions_of(r12, 5, {"QUINELLA_PLACE": (["1,9", "3,12"], [7.4, 40.0], [[0.0], [0.9]])})

    assert log_decisions(log, first, **meta) == len(first)
    raw1, inode = path.read_bytes(), os.stat(path).st_ino
    assert log_decisions(log, second, **meta) == len(second)
    raw2 = path.read_bytes()
    assert raw2.startswith(raw1) and len(raw2) > len(raw1), "a second batch only appends"
    assert os.stat(path).st_ino == inode, "the file is appended in place, never replaced"

    # A repeated key is refused, alone or beside a new decision, and nothing is written.
    fresh = decisions_of(r12, 2, {"WIN": (["5"], [3.3], [[0.1]])})
    for batch in (first[:1], fresh + second[-1:]):
        with pytest.raises(ValueError):
            log_decisions(log, batch, **meta)
        assert path.read_bytes() == raw2, "a refused batch wrote to the log"
    assert (r12, "WIN", "5", 2) not in log.logged_keys()

    # What was logged: each decision once, its key and board at t, the log's own day status,
    # and no stake-like key anywhere (the log refuses them).
    records = ShadowLog(path, day=D1).read()
    logged = first + second
    assert len(records) == len(logged)
    for record, decision in zip(records, logged, strict=True):
        assert stake_like_key(record) is None, f"stake-like key {stake_like_key(record)!r}"
        assert key_of(record) == (
            decision.race_id,
            decision.pool,
            decision.combination,
            int(decision.tau_min),
        )
        assert type(record["tau_min"]) is int
        j = decision.board.combinations.index(decision.combination)
        assert record["board_odds"] == pytest.approx(float(decision.board.odds[j]))
        assert record["status"] == "ok" and record["mode"] == "replay"
        assert record["models"] == meta["models"] and record["git_sha"] == "0123abc"
        assert instant(record["decision_utc"]) == meta["decision_utc"]
        assert instant(record["board_capture_utc"]) == meta["board_capture_utc"]
        assert record["day_status"] == "development"

    # Reading, settling, staking and the day's growth read the log and never write it.
    paid = [
        paying(r11, "WIN", "3", 560),
        paying(r11, "PLACE", "3", 190),
        paying(r11, "PLACE", "7", 330),
        paying(r12, "QUINELLA_PLACE", "3,12", 4_020),
    ]
    frame = read_shadow([D1])
    assert len(frame) == len(logged)
    settled = settle_shadow(frame, dividends(paid))
    by_key = {key_of(r): r for r in settled.to_dict("records")}
    for decision in logged:
        row = by_key[(decision.race_id, decision.pool, decision.combination, int(decision.tau_min))]
        claimed = float(np.mean(decision.ev))
        assert row["claimed_ev"] == pytest.approx(claimed), "claimed EV is the decision's mean EV"
    assert by_key[(r11, "WIN", "3", 5)]["outcome"] == "won"
    assert by_key[(r11, "WIN", "3", 5)]["realised"] == pytest.approx(4.6)
    assert by_key[(r11, "PLACE", "4", 5)]["outcome"] == "lost"
    assert by_key[(r12, "QUINELLA_PLACE", "3,12", 5)]["return_per_100"] == pytest.approx(4_020)
    picks = settled.assign(rank=range(1, len(settled) + 1))
    staked = flat_stakes(picks, limits=load_limits())
    log_growth_by_day(staked, bankroll_yen=100_000)
    assert path.read_bytes() == raw2 and os.stat(path).st_ino == inode, "a reader wrote the log"
    assert ShadowLog(path, day=D1).verify().n_records == len(logged)

    # A holdout day is logged prospectively and labelled by the log, but never read back here.
    held = first_held_day()
    held_path = shadow_log_path(held)
    held_log = ShadowLog(held_path, day=held)
    held_decisions = decisions_of(
        race(11, day=held), 2, {"WIN": (["1", "2"], [2.0, 4.0], [[0.1], [0.2]])}
    )
    at = datetime.combine(held, datetime.min.time(), UTC) + timedelta(hours=5, minutes=38)
    held_meta = {**meta, "decision_utc": at, "board_capture_utc": at - timedelta(seconds=16)}
    assert log_decisions(held_log, held_decisions, **held_meta) == 2
    held_raw = held_path.read_bytes()
    assert held_log.verify().day_status == "holdout"
    both = read_shadow([D1, held])
    assert len(both) == len(logged) and set(both["race_id"]) == {r11, r12}
    assert held_path.read_bytes() == held_raw

    # A sealed log takes no more decisions.
    log.seal(sealed_utc=datetime(2026, 11, 21, 9, 0, tzinfo=UTC))
    sealed = path.read_bytes()
    with pytest.raises(ValueError):
        log_decisions(log, fresh, **meta)
    assert path.read_bytes() == sealed


# ---------------------------------------------------------------- settling shadow records


def test_every_shadow_record_settles_against_the_final_dividends():
    r1, r2, r3 = race(11), race(5, "KYO"), race(1, "NAK", D2)
    held = first_held_day()
    rh = race(11, day=held)
    rows = [
        paying(r1, "WIN", "4", 350),
        paying(r1, "PLACE", "4", 150),
        paying(r1, "PLACE", "7", 230),
        paying(r1, "PLACE", "12", 410),
        paying(r1, "QUINELLA_PLACE", "4,7", 520),
        paying(r1, "QUINELLA_PLACE", "4,12", 1_030),
        paying(r1, "QUINELLA_PLACE", "7,12", 1_880),
        paying(r2, "WIN", "1", 1_740),  # a dead heat: each winner pays its own dividend
        paying(r2, "WIN", "9", 980),
        paying(rh, "WIN", "4", 350),  # a holdout day's dividend is never used here
    ]
    # (race, pool, combination, tau, value_mean, outcome, return_per_100), worked out by hand
    cases = [
        (r1, "WIN", "4", 2, 1.12, "won", 350),
        (r1, "WIN", "4", 5, 1.05, "won", 350),  # every tau settles on its own
        (r1, "WIN", "2", 2, 0.83, "lost", 0),
        (r1, "WIN", "12", 2, None, "lost", 0),  # no values: settled, no claimed EV
        (r1, "PLACE", "7", 2, 0.97, "won", 230),
        (r1, "PLACE", "12", 2, 1.31, "won", 410),
        (r1, "PLACE", "3", 2, 0.88, "lost", 0),
        (r1, "QUINELLA_PLACE", "4,12", 2, 1.4, "won", 1_030),
        (r1, "QUINELLA_PLACE", "3,12", 2, 0.7, "lost", 0),  # 12 placed, 3 did not
        (r2, "WIN", "1", 2, 0.95, "won", 1_740),
        (r2, "WIN", "9", 2, 1.21, "won", 980),
        (r2, "WIN", "4", 2, 1.02, "lost", 0),  # 4 won TOK 11, not KYO 5
        (r2, "QUINELLA_PLACE", "1,9", 2, 1.1, "unsettled", None),  # no Wide rows in KYO 5 yet
        (r3, "WIN", "4", 2, 0.9, "unsettled", None),  # no rows in NAK 1 at all
    ]
    records = [
        v0_record(r, pool, comb, tau=tau, value_mean=vm) for r, pool, comb, tau, vm, _, _ in cases
    ]
    records += [
        v0_record(r1, "*", "*", tau=10, status="no_board"),
        v0_record(rh, "WIN", "4", day_status="holdout", value_mean=1.5),
        v0_record(
            rh, "WIN", "5", tau=5, value_mean=1.5
        ),  # labelled development, but the rule holds
    ]
    got = settled_by_key(records, rows)
    for r, pool, comb, tau, vm, outcome, per_100 in cases:
        row = got[(r, pool, comb, tau)]
        where = f"{pool} {comb} T-{tau} on {r}"
        assert row["outcome"] == outcome, where
        if vm is None:
            assert null(row["claimed_ev"]), where
        else:
            assert row["claimed_ev"] == pytest.approx(vm - 1.0), where
        if per_100 is None:
            assert null(row["return_per_100"]) and null(row["realised"]), where
        else:
            assert row["return_per_100"] == pytest.approx(per_100), where
            assert row["realised"] == pytest.approx(per_100 / 100 - 1.0), where
    assert got[(r1, "*", "*", 10)]["outcome"] == "no_board"
    for key in ((rh, "WIN", "4", 2), (rh, "WIN", "5", 5)):
        assert got[key]["outcome"] == "holdout", f"{key} was settled on a holdout day"
        assert null(got[key]["return_per_100"]) and null(got[key]["realised"])


def test_settlement_refunds_scratched_runners():
    r = race(7)
    field = [h for h in range(1, 11) if h != 5]  # horse 5 scratched before the start
    rows = [
        paying(r, "WIN", "3", 450),
        paying(r, "PLACE", "3", 140),
        paying(r, "PLACE", "8", 220),
        paying(r, "PLACE", "1", 310),
        paying(r, "QUINELLA_PLACE", "3,8", 610),
        paying(r, "QUINELLA_PLACE", "1,3", 480),
        paying(r, "QUINELLA_PLACE", "1,8", 990),
        # the refund rows #454's day_dividends adds for a scratched runner, in the store's order
        refund_row(r, "WIN", "5"),
        refund_row(r, "PLACE", "5"),
        *[refund_row(r, "QUINELLA_PLACE", ",".join(map(str, sorted((5, h))))) for h in field],
    ]
    cases = [  # pool, combination, tau, outcome, return_per_100
        ("WIN", "5", 2, "refunded", None),
        ("WIN", "5", 5, "refunded", None),
        ("WIN", "3", 2, "won", 450),
        ("WIN", "7", 2, "lost", 0),
        ("PLACE", "5", 2, "refunded", None),
        ("PLACE", "8", 2, "won", 220),
        ("QUINELLA_PLACE", "3,5", 2, "refunded", None),
        ("QUINELLA_PLACE", "5,10", 2, "refunded", None),
        ("QUINELLA_PLACE", "1,3", 2, "won", 480),
        ("QUINELLA_PLACE", "2,7", 2, "lost", 0),
    ]
    records = [
        v0_record(r, pool, comb, tau=tau, value_mean=1.0 + 0.05 * i)
        for i, (pool, comb, tau, _, _) in enumerate(cases)
    ]
    got = settled_by_key(records, rows)
    for pool, comb, tau, outcome, per_100 in cases:
        row = got[(r, pool, comb, tau)]
        where = f"{pool} {comb} T-{tau}"
        assert row["outcome"] == outcome, where
        if outcome == "refunded":
            assert null(row["return_per_100"]), f"{where}: a refund has no return"
            assert null(row["realised"]), f"{where}: a refund is not a realised return"
        else:
            assert row["return_per_100"] == pytest.approx(per_100), where
            assert row["realised"] == pytest.approx(per_100 / 100 - 1.0), where

    # The calibration never sees a refunded record: with the refunds in or out, the same slope.
    rng = np.random.default_rng(5)
    many, paid = [], []
    for n in range(40):
        rid = race(1 + n % 12, ("TOK", "KYO", "HSN", "NAK")[n // 12 % 4], D1 if n < 24 else D2)
        scratched, winner = rng.choice(np.arange(1, 13), size=2, replace=False)
        paid += [paying(rid, "WIN", str(winner), float(rng.integers(20, 300) * 10))]
        paid += [refund_row(rid, "WIN", str(scratched))]
        many += [
            v0_record(rid, "WIN", str(h), value_mean=float(rng.uniform(0.5, 1.5)))
            for h in range(1, 13)
        ]
    settled = settle_shadow(pd.DataFrame(many), dividends(paid))
    assert (settled["outcome"] == "refunded").sum() == 40
    assert settled.loc[settled["outcome"] == "refunded", "realised"].isna().all()
    boot = {"n_boot": 200, "level": 0.95, "seed": 1}
    with_refunds = calibration_slope(
        settled["claimed_ev"], settled["realised"], settled["race_id"], **boot
    )
    kept = settled[settled["outcome"].isin(["won", "lost"])]
    without = calibration_slope(kept["claimed_ev"], kept["realised"], kept["race_id"], **boot)
    assert len(kept) == 40 * 11
    for name in ("mean", "lo", "hi", "n_clusters"):
        assert getattr(with_refunds, name) == pytest.approx(getattr(without, name)), name


# ---------------------------------------------------------------- flat stakes and growth


def picks_of(day: date, venues: dict[str, tuple[str, int]], races: int, rng) -> list[dict]:
    """Two picks per race: venues maps a venue to (the text form of its decision times, its
    minutes after 01:00 UTC for race 1); ranks are filled in by the caller."""
    out = []
    for venue, (form, offset) in venues.items():
        for n in range(1, races + 1):
            post = datetime.combine(day, datetime.min.time(), UTC) + timedelta(
                hours=1, minutes=offset + 30 * (n - 1)
            )
            for tau, combination in ((5, str(rng.integers(1, 9))), (2, str(rng.integers(9, 17)))):
                at = post - timedelta(minutes=tau)
                text = {
                    "Z": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "utc": at.isoformat(),
                    "jst": at.astimezone(JST).isoformat(),
                }[form]
                out.append(
                    {
                        "race_id": race(n, venue, day),
                        "pool": "WIN",
                        "combination": combination,
                        "decision_utc": text,
                        "instant": at,
                    }
                )
    return out


def test_stakes_respect_the_cap():
    rng = np.random.default_rng(17)
    plan = [
        # three venues, two on the same post times: 72 picks for 30 places
        (D3, {"TOK": ("Z", 0), "KYO": ("jst", 0), "HSN": ("utc", 10)}, 12),
        (D1, {"NAK": ("utc", 5)}, 6),  # 12 picks: all fit
        (D2, {"TOK": ("jst", 0), "KYO": ("Z", 15)}, 8),  # 32 picks
    ]
    rows = []
    for day, venues, races in plan:
        day_picks = picks_of(day, venues, races, rng)
        for pick, r in zip(day_picks, rng.permutation(len(day_picks)) + 1, strict=True):
            pick["rank"] = int(r)  # unique within the day: no tie in (time, rank)
        rows += day_picks
    rows = [rows[i] for i in rng.permutation(len(rows))]  # input order is not time order
    picks = pd.DataFrame(rows).drop(columns="instant")
    picks.index = [f"p{i:03d}" for i in range(len(picks))]

    for limits in (
        load_limits(),
        Limits(day_cap_yen=2_500, flat_stake_yen=1_000),
        Limits(day_cap_yen=10_000, flat_stake_yen=300),
    ):
        flat, cap = limits.flat_stake_yen, limits.day_cap_yen
        out = flat_stakes(picks, limits=limits)
        assert list(out.index) == list(picks.index), "same picks, same order"
        for name in ("race_id", "combination", "decision_utc"):
            assert out[name].tolist() == picks[name].tolist(), name
        stakes = out["stake_yen"].tolist()
        assert all(isinstance(s, (int, np.integer)) for s in stakes)
        assert all(s in (0, flat) and s % 100 == 0 for s in stakes)

        # Expected, from first principles: per race day, by decision instant then rank, the
        # first cap // flat picks take the flat stake and the rest nothing.
        expected = {}
        for day, _, _ in plan:
            mine = [
                (row["instant"], row["rank"], i)
                for i, row in enumerate(rows)
                if row["race_id"].split(":")[1] == day.isoformat()
            ]
            for place, (_, _, i) in enumerate(sorted(mine)):
                expected[picks.index[i]] = flat if place < cap // flat else 0
        assert dict(zip(out.index, stakes, strict=True)) == expected, f"cap {cap}, flat {flat}"
        by_day = out.assign(day=out["race_id"].str.split(":").str[1]).groupby("day")["stake_yen"]
        assert (by_day.sum() <= cap).all(), f"a race day staked over {cap}"
        counts = {
            day.isoformat(): sum(1 for r in rows if day.isoformat() in r["race_id"])
            for day, _, _ in plan
        }
        assert (by_day.apply(lambda s: int((s > 0).sum()))).to_dict() == {
            day: min(n, cap // flat) for day, n in counts.items()
        }

    with pytest.raises(ValueError):
        flat_stakes(picks, limits=Limits(day_cap_yen=30_000, flat_stake_yen=None))


def test_log_growth_by_day_is_the_log_of_ending_over_starting_money():
    held = first_held_day()

    def row(day, n, outcome, stake, per_100):
        return {
            "race_id": race(n, day=day),
            "pool": "WIN",
            "combination": str(n),
            "outcome": outcome,
            "stake_yen": stake,
            "return_per_100": per_100,
        }

    nan = float("nan")
    settled = pd.DataFrame(
        [
            row(D3, 1, "won", 1_000, 350.0),  # paid 3,500
            row(D3, 2, "lost", 1_000, 0.0),
            row(D3, 3, "lost", 1_000, 0.0),
            row(D1, 1, "refunded", 1_000, nan),  # the stake back
            row(D1, 2, "lost", 1_000, 0.0),
            row(D1, 3, "won", 0, 900.0),  # past the cap: no stake, nothing paid
            row(D2, 1, "unsettled", 1_000, nan),  # pending: left out
            row(D2, 2, "won", 2_000, 1_740.0),  # paid 34,800
            row(held, 1, "holdout", 1_000, nan),
            row(date(2026, 11, 8), 1, "unsettled", 1_000, nan),
        ]
    )
    for bankroll in (100_000, 40_000):
        out = log_growth_by_day(settled, bankroll_yen=bankroll)
        growth = {day_text(d): g for d, g in zip(out["day"], out["log_growth"], strict=True)}
        expected = {  # ln(ending / starting), worked out by hand
            D3.isoformat(): math.log((bankroll - 3_000 + 3_500) / bankroll),
            D1.isoformat(): math.log((bankroll - 2_000 + 1_000) / bankroll),
            D2.isoformat(): math.log((bankroll - 2_000 + 34_800) / bankroll),
        }
        for day, value in expected.items():
            assert growth[day] == pytest.approx(value), f"{day} at bankroll {bankroll}"
        for day in (held.isoformat(), "2026-11-08"):
            assert day not in growth or null(growth[day]), f"{day} has nothing settled"


# ---------------------------------------------------------------- E20: selection drift


TAUS = (5, 10, 15, 20, 25, 30)
DRIFT_VENUES = ("TOK", "KYO", "HSN")


def drift_row(
    race_id, pool, combination, tau, *, board, final, score, post, status="ok", held=False
):
    record = v0_record(
        race_id,
        pool,
        combination,
        tau=tau,
        value_mean=score,
        board_odds=board,
        status=status,
        day_status="holdout" if held else "development",
        decision_utc=(post - timedelta(minutes=tau)).isoformat(),
        board_capture_utc=(post - timedelta(minutes=tau, seconds=16)).isoformat()
        if status == "ok"
        else None,
    )
    record["final_odds"] = final
    return record


def drift_races(n: int):
    """n race ids over two development days and three venues, each with its post time."""
    out = []
    for i in range(n):
        day = (D1, D2)[i % 2]
        venue = DRIFT_VENUES[i // 2 % 3]
        no = 1 + i // 6
        post = datetime.combine(day, datetime.min.time(), UTC) + timedelta(hours=1, minutes=30 * no)
        out.append((race(no, venue, day), post))
    return out


def mechanical_log(seed: int) -> pd.DataFrame:
    """Boards that wobble around the final at every tau (iid noise n_tau in log odds) and a score
    read off that same wobbling board (score = 1 + 0.8 n_tau + noise): the fall from t to the
    final is n_t, so a score from the board at t predicts it mechanically and a score from an
    earlier board does not. PLACE scores at tau carry the WIN noise of tau - 5, so a join that
    ignores the pool would find a relation too."""
    rng = np.random.default_rng(seed)
    out = []
    for race_id, post in drift_races(60):
        for h in range(1, 11):
            win_final, place_final = rng.uniform(2.0, 30.0), rng.uniform(1.1, 6.0)
            wobble = {t: rng.normal(0.0, 0.25) for t in TAUS}
            other = {t: rng.normal(0.0, 0.25) for t in TAUS}
            for t in TAUS:
                out.append(
                    drift_row(
                        race_id,
                        "WIN",
                        str(h),
                        t,
                        board=win_final * np.exp(wobble[t]),
                        final=win_final,
                        score=1 + 0.8 * wobble[t] + rng.normal(0, 0.05),
                        post=post,
                    )
                )
                trap = wobble.get(t - 5)
                place_score = 1 + 0.8 * trap if trap is not None else 1 + rng.normal(0.0, 0.2)
                out.append(
                    drift_row(
                        race_id,
                        "PLACE",
                        str(h),
                        t,
                        board=place_final * np.exp(other[t]),
                        final=place_final,
                        score=place_score,
                        post=post,
                    )
                )
    return pd.DataFrame(out)


def planted_log(seed: int, beta: float, held: date) -> pd.DataFrame:
    """Scores iid across taus; the fall from t to the final is beta x (the score at t + 10 - 1)
    plus small noise wherever a record at t + 10 exists, noise elsewhere. Plus rows that must be
    dropped: combinations logged from T-15 only, a T-30 record without a score, a record without
    a board figure, a combination without a final figure, no_board records and a holdout race."""
    rng = np.random.default_rng(seed)
    out = []

    def combination(
        race_id, post, h, taus, *, final=None, held_day=False, unscored=(), unpriced=()
    ):
        final_odds = rng.uniform(2.0, 30.0) if final is None else final
        score = {t: 1 + rng.normal(0.0, 0.25) for t in taus}
        for t in taus:
            fall = (
                beta * (score[t + 10] - 1) + rng.normal(0.0, 0.05)
                if t + 10 in score
                else rng.normal(0.0, 0.25)
            )
            if held_day:
                fall = -3.0 * (score[t] - 1)  # a strong relation that must never be read
            board = None if t in unpriced else float(final_odds * np.exp(fall))
            out.append(
                drift_row(
                    race_id,
                    "WIN",
                    str(h),
                    t,
                    board=board,
                    final=final_odds,
                    score=None if t in unscored else score[t],
                    post=post,
                    held=held_day,
                )
            )

    for race_id, post in drift_races(60):
        for h in range(1, 9):
            combination(race_id, post, h, TAUS)
        combination(race_id, post, 9, (5, 10, 15))  # the logger started at T-15
        combination(race_id, post, 10, TAUS, unscored=(30,))
        combination(race_id, post, 11, TAUS, unpriced=(5,))
        combination(race_id, post, 12, TAUS, final=float("nan"))
        for t in TAUS:
            out.append(
                drift_row(
                    race_id,
                    "*",
                    "*",
                    t,
                    board=None,
                    final=float("nan"),
                    score=None,
                    post=post,
                    status="no_board",
                )
            )
    held_post = datetime.combine(held, datetime.min.time(), UTC) + timedelta(hours=3)
    for h in range(1, 9):
        combination(race(11, "TOK", held), held_post, h, TAUS, held_day=True)
    return pd.DataFrame(out)


def usable(frame: pd.DataFrame, lag: int) -> pd.DataFrame:
    """The rows E20 can use at ``lag``, joined to their lagged score, by hand: a development row
    with a board and a final figure, and a scored record of its race, pool and combination at
    tau + lag."""
    dev = frame[frame["day_status"] == "development"]
    scores = {key_of(r): r["value_mean"] for r in dev.to_dict("records")}
    rows = []
    for r in dev.to_dict("records"):
        lagged = scores.get((r["race_id"], r["pool"], r["combination"], r["tau_min"] + lag))
        if null(r["board_odds"]) or null(r["final_odds"]) or null(lagged):
            continue
        rows.append(
            {
                "fall": math.log(r["board_odds"]) - math.log(r["final_odds"]),
                "score": lagged - 1.0,
                "same_board": r["value_mean"],
            }
        )
    return pd.DataFrame(rows)


def test_selection_drift_scores_from_a_lagged_board():
    boot = {"n_boot": 200, "level": 0.975, "seed": 4}

    # No relation but the mechanical one: a same-board score would find a strong slope.
    mech = mechanical_log(5)
    win = mech[mech["pool"] == "WIN"]
    same_board = np.polyfit(win["value_mean"], np.log(win["board_odds"] / win["final_odds"]), 1)[0]
    assert same_board > 0.8, "the fixture's same-board score is mechanically tied to the fall"
    for lag in (5, 10):
        got = selection_drift(mech, lag_minutes=float(lag), **boot)
        assert got.n == len(usable(mech, lag)), f"lag {lag}: rows used"
        assert abs(got.slope) < 0.1, f"lag {lag}: slope {got.slope:.3f} from the board at t"
        assert got.lo <= got.slope <= got.hi and got.hi - got.lo < 0.3

    # A planted relation with the score at t + 10 is recovered at lag 10 and absent at lag 5.
    held = first_held_day()
    beta = 0.6
    planted = planted_log(9, beta, held)
    at_10 = selection_drift(planted, lag_minutes=10, **boot)
    used = usable(planted, 10)
    assert at_10.n == len(used), "rows without a lagged score, board or final, or held, dropped"
    assert at_10.slope == pytest.approx(np.polyfit(used["score"], used["fall"], 1)[0], abs=1e-6)
    assert at_10.slope == pytest.approx(beta, abs=0.03)
    assert 0.4 < at_10.lo <= at_10.slope <= at_10.hi
    at_5 = selection_drift(planted, lag_minutes=5, **boot)
    assert at_5.n == len(usable(planted, 5)) and at_5.n > at_10.n
    assert abs(at_5.slope) < 0.06, "lag 5 reads another record's score"

    for bad in (0, -5):
        with pytest.raises(ValueError):
            selection_drift(planted, lag_minutes=bad, **boot)


# ---------------------------------------------------------------- the manual bet log


def test_the_bet_log_is_append_only(D):
    limits = load_limits()
    log1 = BetLog(bet_log_path(D1), day=D1, limits=limits)
    log2 = BetLog(bet_log_path(D2), day=D2, limits=limits)
    log1.record(bet(race(11), "WIN", "4", 1_000, "opt-a", "a"))
    log1.record(bet(race(11), "WIN", "2", 1_000, "opt-a", "b"))
    log1.record(bet(race(5, "KYO"), "QUINELLA", "4,7", 1_000, None, "c"))
    log2.record(bet(race(1, day=D2), "PLACE", "6", 1_000, "opt-b", "d"))
    log2.record(bet(race(2, day=D2), "WIN", "1", 500, "opt-a", "e"))
    folder = bet_log_path(D1).parent
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    inodes = {p.name: os.stat(p).st_ino for p in folder.iterdir()}

    frame = read_bets([D1, D2, D3])  # D3 has no log
    assert frame["note"].tolist() == ["a", "b", "c", "d", "e"]
    for row, stored in zip(frame.to_dict("records"), log1.read() + log2.read(), strict=True):
        assert {k: row[k] for k in stored} == stored, "each bet as the log stores it"
    paid = [paying(race(11), "WIN", "4", 350), paying(race(1, day=D2), "PLACE", "6", 180)]
    settled = settle_bets(frame, dividends(paid))
    played_by_day(settled, rank([board_row("opt-a", 112.0), board_row("opt-b", 95.0)]))
    after = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert after == before, "reading, settling and the report changed or added a bet-log file"
    assert {p.name: os.stat(p).st_ino for p in folder.iterdir()} == inodes
    assert not bet_log_path(D3).exists()

    log1.record(bet(race(12), "WIN", "9", 1_000, "opt-b", "f"))
    grown = bet_log_path(D1).read_bytes()
    old = before[bet_log_path(D1).name]
    assert grown.startswith(old) and len(grown) > len(old), "a later bet only appends"
    again = read_bets([D1, D2])
    assert again["note"].tolist() == ["a", "b", "c", "f", "d", "e"]
    assert again["seq"].tolist() == [1, 2, 3, 4, 1, 2]


def test_manual_bets_settle_at_final_dividends(D):
    held = first_held_day()
    limits = load_limits()
    t11, k5, n1 = race(11), race(5, "KYO"), race(1, "NAK")
    p1, p2 = race(1, day=D2), race(2, day=D2)
    h11 = race(11, day=held)
    placed = {
        D1: [
            bet(t11, "WIN", "4", 1_000, "opt-a", "won at 350"),
            bet(t11, "WIN", "2", 1_000, "opt-a", "lost"),
            bet(k5, "QUINELLA", "7,4", 1_000, "opt-b", "won at 2590"),
            bet(k5, "WIN", "5", 1_000, None, "scratched, refunded"),
            bet(n1, "WIN", "3", 1_000, "opt-c", "no dividends yet"),
        ],
        D2: [
            bet(p1, "PLACE", "6", 1_000, "opt-a", "place at 180"),
            bet(p1, "PLACE", "9", 2_000, "opt-b", "place lost"),
            bet(p2, "WIN", "1", 500, "opt-a", "won at 1740"),
        ],
        held: [bet(h11, "WIN", "4", 1_000, "opt-a", "holdout")],
    }
    for day, bets in placed.items():
        log = BetLog(bet_log_path(day), day=day, limits=limits)
        for b in bets:
            log.record(b)
    rows = [
        paying(t11, "WIN", "4", 350),
        paying(k5, "QUINELLA", "4,7", 2_590),
        paying(k5, "WIN", "8", 1_200),
        refund_row(k5, "WIN", "5"),
        paying(p1, "PLACE", "6", 180),
        paying(p1, "PLACE", "2", 120),
        paying(p1, "PLACE", "11", 300),
        paying(p2, "WIN", "1", 1_740),
        paying(h11, "WIN", "4", 350),  # never used: the day is held out
    ]
    settled = settle_bets(read_bets([D1, D2, held]), dividends(rows))
    expected_bets = {  # note: (outcome, paid_yen), paid = stake / 100 x the dividend
        "won at 350": ("won", 3_500),
        "lost": ("lost", 0),
        "won at 2590": ("won", 25_900),
        "scratched, refunded": ("refunded", 1_000),
        "no dividends yet": ("unsettled", None),
        "place at 180": ("won", 1_800),
        "place lost": ("lost", 0),
        "won at 1740": ("won", 8_700),
        "holdout": ("holdout", None),
    }
    got_bets = {r["note"]: r for r in settled.to_dict("records")}
    assert set(got_bets) == set(expected_bets), "every logged bet is read and settled"
    for note, (outcome, paid) in expected_bets.items():
        assert got_bets[note]["outcome"] == outcome, note
        if paid is None:
            assert null(got_bets[note]["paid_yen"]), note
        else:
            assert got_bets[note]["paid_yen"] == pytest.approx(paid), note

    # opt-d leads the board, opt-a is second, opt-b third, opt-u is unmeasured, opt-c not on it.
    ranked = rank(
        [
            board_row("opt-a", 112.0),
            board_row("opt-b", 95.0),
            board_row("opt-d", 130.0),
            board_row("opt-u", None),
        ]
    )
    report = played_by_day(settled, ranked)
    got = {(day_text(r["day"]), r["option_id"]): r for r in report.to_dict("records")}
    d1, d2, dh = D1.isoformat(), D2.isoformat(), held.isoformat()
    expected = {  # (n_bets, staked, paid, per 100, n_refunded, n_pending, rank)
        (d1, "opt-a"): (2, 2_000, 3_500, 175.0, 0, 0, 2),
        (d1, "opt-b"): (1, 1_000, 25_900, 2_590.0, 0, 0, 3),
        (d1, "off-board"): (0, 0, 0, None, 1, 0, None),
        (d1, "opt-c"): (0, 0, 0, None, 0, 1, None),
        (d2, "opt-a"): (2, 1_500, 10_500, 700.0, 0, 0, 2),
        (d2, "opt-b"): (1, 2_000, 0, 0.0, 0, 0, 3),
        (dh, "opt-a"): (0, 0, 0, None, 0, 1, 2),
    }
    assert set(got) == set(expected), f"rows by day and option: {sorted(got)}"
    for key, (n, staked, paid, per_100, n_refunded, n_pending, place) in expected.items():
        row = got[key]
        assert (row["n_bets"], row["n_refunded"], row["n_pending"]) == (n, n_refunded, n_pending), (
            key
        )
        assert row["staked_yen"] == staked and row["paid_yen"] == pytest.approx(paid), key
        if per_100 is None:
            assert null(row["return_per_100"]), key
        else:
            assert row["return_per_100"] == pytest.approx(per_100), key
        if place is None:
            assert null(row["rank"]), f"{key}: off the board has no rank"
        else:
            assert row["rank"] == place, key


# ---------------------------------------------------------------- no code places a bet


def imported_names(source: str, module: str) -> set[str]:
    """Every module ``source`` imports, relative imports resolved against ``module``'s package."""
    package = module.split(".")[:-1]
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            parent = ".".join([*base, *([node.module] if node.module else [])])
            names.add(parent)
            names.update(f"{parent}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called in ("import_module", "__import__") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
    return names


def forbidden(names: set[str]) -> set[str]:
    return {
        name
        for name in names
        for bad in FORBIDDEN_IMPORTS
        if name == bad or name.startswith(f"{bad}.")
    }


def test_the_shadow_module_cannot_place_a_bet():
    probe = "\n".join(
        [
            "import requests",
            "import urllib.request",
            "from http import client",
            "import subprocess",
            "from selenium import webdriver",
            "from ....core.ingest import http",
            "import importlib; importlib.import_module('socket')",
        ]
    )
    caught = forbidden(imported_names(probe, "horsetrack.countries.japan.models.shadow"))
    assert {"requests", "urllib.request", "http", "subprocess", "selenium", "socket"} <= caught
    assert "horsetrack.core.ingest" in caught, "a relative import of core.ingest is caught"
    allowed = "import numpy\nfrom horsetrack.countries.japan.bets.log import BetLog\nimport httpxx"
    assert not forbidden(imported_names(allowed, "horsetrack.countries.japan.models.shadow"))

    assert SHADOW_MODULE.is_file(), f"{SHADOW_MODULE} does not exist"
    names = imported_names(
        SHADOW_MODULE.read_text(encoding="utf-8"), "horsetrack.countries.japan.models.shadow"
    )
    assert not forbidden(names), f"models/shadow.py imports {sorted(forbidden(names))}"
