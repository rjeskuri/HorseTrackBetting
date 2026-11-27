"""Shadow decisions logged, settled at the final dividends and scored (C6 in shadow, #101).

Nothing here places, sizes for or sends a bet: every figure is paper, and the module imports no
HTTP client, socket, browser driver, subprocess or ingest code. It writes no file but the shadow
log, and that only through ``ShadowLog.append``; the bet log (#453) is only read.

- ``log_decisions``: one v0 shadow-log record (docs/specs/shadow-log.md) per #100
  ``ShadowDecision``, appended through ``ShadowLog.append``, so a batch is written whole or not at
  all, a repeated key or a sealed log is refused, and a holdout day's records are labelled by the
  log. A record keeps the decision's key, the board's figure at t for its combination
  (``board_odds``) and the claimed EV per yen over the refits (``ev_mean``, ``ev_q05``, ``ev_q95``,
  ``n_refits``). The flat stake priced inside the dividend is not logged: it is betting.yaml's,
  pinned by ``git_sha``, and no key of the log names a stake.
- ``read_shadow``: the records of development days through ``ShadowLog.read``; a holdout day gives
  none, its log stays sealed.
- ``settle_shadow``: each record a paper ticket of 100 yen, settled by #454's ``settle_bets`` (the
  one place a combination is matched to the dividends that paid), with ``claimed_ev`` and
  ``realised`` beside it. Holdout records are never settled here.
- ``calibration_slope``: realised net return regressed on claimed EV (OLS with an intercept), its
  interval from a bootstrap of whole clusters (D7: the meeting), reported, never a gate.
- ``flat_stakes``: the flat stake per pick under the race-day cap (betting.yaml, ``Limits``); no
  Kelly, no threshold. ``log_growth_by_day``: ln(ending / starting money) per race day.
- ``selection_drift`` (E20): the fall in log odds from t to the final regressed on the claimed EV
  of the record ``lag_minutes`` earlier, never the board at t itself.
- ``read_bets`` and ``played_by_day``: the owner's hand-played bets by day and option, beside each
  option's rank on the leaderboard.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from horsetrack.core.eval.bootstrap import Interval
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.bets.log import (
    ADDED,
    OPTIONAL,
    BetLog,
    Limits,
    bet_log_path,
    load_limits,
)
from horsetrack.countries.japan.bets.log import REQUIRED as BET_FIELDS
from horsetrack.countries.japan.bets.settle import (
    LOST,
    OFF_BOARD,
    OPTION_COLUMNS,
    REFUNDED,
    SETTLED,
    WON,
    _day_of,
    played_by_option,
    settle_bets,
)
from horsetrack.countries.japan.shadow.log import (
    COUNTRY,
    HOLDOUT,
    NO_BOARD,
    REQUIRED,
    HoldoutSealed,
    ShadowLog,
    _instant,
    shadow_log_path,
)

if TYPE_CHECKING:
    from horsetrack.core.eval.leaderboard import RankedRow
    from horsetrack.countries.japan.models.decision import ShadowDecision

TICKET_YEN = 100  # a paper ticket: dividends are quoted per 100 yen
ANY = "*"  # pool and combination of a no_board record
OK = "ok"
MOVED = (WON, LOST, REFUNDED)  # the outcomes whose stake left the day's money
QUANTILES = (0.05, 0.95)  # the claimed EV's spread over refits, as v0's value_q05 / value_q95
_MAX_DRAWS_PER_CHUNK = 2_000_000  # replicate-by-cluster draws built at once, as core.eval.bootstrap


class NotEstimable(ValueError):
    """Too few pairs, no spread in the score, or too few clusters for a slope or its interval.

    ``slope`` is the point slope when only the interval is not estimable (one cluster, or a
    replicate with no spread), else None; ``n`` is the number of pairs either way."""

    def __init__(self, message: str, *, n: int, slope: float | None = None) -> None:
        super().__init__(message)
        self.n = n
        self.slope = slope


@dataclass(frozen=True)
class RegressionResult:
    """E20's slope of the fall in log odds on the lagged score, with ``lo`` and ``hi`` as
    ``calibration_slope``'s (one-sided quantiles at ``level`` over replicates of whole meetings),
    ``n`` the rows used and ``n_clusters`` the meetings resampled."""

    slope: float
    lo: float
    hi: float
    n: int
    n_clusters: int


# ---------------------------------------------------------------- logging decisions


def _utc_text(name: str, value: Any) -> str:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime, got {value!r}")
    return value.astimezone(timezone.utc).isoformat()


def _record(decision: ShadowDecision, meta: Mapping[str, Any]) -> dict[str, Any]:
    """The shadow-log record of one decision; ValueError for a tau that is not whole minutes, a
    combination not on its board, or an EV that is not one finite value per refit."""
    tau = decision.tau_min
    where = f"{decision.race_id} {decision.pool} {decision.combination}"
    if isinstance(tau, bool) or not isinstance(tau, Real) or not float(tau).is_integer():
        raise ValueError(f"{where}: tau_min {tau!r} is not whole minutes, as the log keys it")
    ev = np.asarray(decision.ev, dtype=float).ravel()
    if ev.size == 0 or not np.isfinite(ev).all():
        raise ValueError(f"{where}: the EV must be one finite value per refit")
    board = decision.board
    combinations = list(board.combinations)
    if decision.combination not in combinations:
        raise ValueError(f"{where}: the combination is not on its board")
    odds = None
    if board.odds is not None:
        figure = float(
            np.asarray(board.odds, dtype=float)[combinations.index(decision.combination)]
        )
        odds = figure if math.isfinite(figure) else None
    lo, hi = np.quantile(ev, QUANTILES)
    return {
        "race_id": decision.race_id,
        "pool": decision.pool,
        "combination": decision.combination,
        "tau_min": int(tau),
        **meta,
        "board_odds": odds,
        "ev_mean": float(ev.mean()),
        "ev_q05": float(lo),
        "ev_q95": float(hi),
        "n_refits": int(ev.size),
    }


def log_decisions(
    log: ShadowLog,
    decisions: Iterable[ShadowDecision],
    *,
    decision_utc: datetime,
    board_capture_utc: datetime,
    mode: str,
    models: Mapping[str, str] | None,
    git_sha: str,
) -> int:
    """Append one record per decision to ``log`` through ``ShadowLog.append`` and return the count.

    The records share the decision and board times (timezone-aware, written in UTC), ``mode``
    ('live' or 'replay'), ``models`` (the fit ids) and ``git_sha``; ``written_utc`` is now and
    ``status`` 'ok'. ``ev_mean`` is the claimed EV per yen (net: 0 is break-even), the mean of
    ``decision.ev`` over the refits, with its 5th and 95th percentiles and ``n_refits``. Every
    record is built before any is appended, and the log checks the batch whole, so a bad decision,
    a repeated (race_id, pool, combination, tau_min) or a sealed log (ValueError, from the log's
    ``RecordRefused`` and ``LogSealed``) writes nothing."""
    meta = {
        "decision_utc": _utc_text("decision_utc", decision_utc),
        "board_capture_utc": _utc_text("board_capture_utc", board_capture_utc),
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "status": OK,
        "models": None if models is None else dict(models),
        "git_sha": git_sha,
    }
    return log.append([_record(decision, meta) for decision in decisions])


def read_shadow(days: Iterable[date]) -> pd.DataFrame:
    """The records of each day's shadow log, in day then log order, read through
    ``ShadowLog.read``. A holdout day (by the committed rule, or a log with a record labelled
    holdout) gives no record: its log stays sealed and is not read here. A day without a log gives
    none. Each day is read once, however often it is given."""
    records: list[dict[str, Any]] = []
    for day in dict.fromkeys(days):
        try:
            records += ShadowLog(shadow_log_path(day), day=day).read()
        except HoldoutSealed:
            continue
    if not records:
        return pd.DataFrame({name: pd.Series(dtype=object) for name in (*REQUIRED, "day_status")})
    return pd.DataFrame(records)


# ---------------------------------------------------------------- settling


def _is_null(value: Any) -> bool:
    return value is None or (not isinstance(value, (list, dict, tuple)) and bool(pd.isna(value)))


def _held(record: Mapping[str, Any]) -> bool:
    """Labelled holdout, or of a day the committed holdout rule holds."""
    return record.get("day_status") == HOLDOUT or is_holdout_day(
        COUNTRY, _day_of(record["race_id"])
    )


def _no_board(record: Mapping[str, Any]) -> bool:
    return record.get("status") == NO_BOARD or ANY in (
        record.get("pool"),
        record.get("combination"),
    )


def _numbers(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name]).astype("float64")


def claimed_ev(frame: pd.DataFrame) -> pd.Series:
    """The net EV per yen each record claims: its ``ev_mean`` (a logged #100 decision), else its
    v0 ``value_mean - 1`` (``value_mean`` is the gross return per yen), else null."""
    ev = _numbers(frame, "ev_mean")
    return ev.where(ev.notna(), _numbers(frame, "value_mean") - 1.0)


def settle_shadow(log: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """``log`` (``read_shadow`` rows), one row per record in order with its columns kept, plus
    ``outcome``, ``return_per_100``, ``claimed_ev`` and ``realised``.

    ``dividends`` are #454's rows (``race_id, pool, combination, dividend, is_refund``, e.g. from
    ``bets.settle.day_dividends``). Each record is a paper ticket of 100 yen settled by
    ``settle_bets``: 'won' at its combination's dividend (each paying combination of a dead heat,
    Place or Wide its own), 'lost' (0) when its race and pool have rows but not its combination,
    'refunded' on a refund row (a runner scratched before the start: the stake back, no return)
    and 'unsettled' when its race has no row of its pool yet. A ``no_board`` record (pool and
    combination ``*``) is 'no_board'. A record labelled holdout, or of a day the committed holdout
    rule holds, is 'holdout' whatever ``dividends`` holds, and its claimed EV is left null too, so
    nothing scored here reads it.

    ``return_per_100`` is set for won and lost records only, as is ``realised``, the net return
    per yen at the final dividend (``return_per_100 / 100 - 1``). ``claimed_ev`` is
    ``claimed_ev(log)``."""
    out = log.copy()
    records = out.to_dict("records")
    held = np.array([_held(record) for record in records], dtype=bool)
    boardless = np.array([_no_board(record) for record in records], dtype=bool) & ~held
    outcome = np.where(held, HOLDOUT, np.where(boardless, NO_BOARD, None)).astype(object)
    per_100 = np.full(len(out), np.nan)
    paper = np.flatnonzero(~held & ~boardless)
    if len(paper):
        rows = out.iloc[paper]
        tickets = pd.DataFrame(
            {
                name: rows[name].astype(object).to_numpy()
                for name in ("race_id", "pool", "combination")
            }
        ).assign(stake_yen=TICKET_YEN)
        settled = settle_bets(tickets, dividends)
        outcome[paper] = settled["outcome"].to_numpy(dtype=object)
        per_100[paper] = settled["return_per_100"].to_numpy(dtype=float)
    measured = pd.Series(outcome, index=out.index).isin(SETTLED).to_numpy()
    out["outcome"] = pd.Series(outcome, index=out.index, dtype=object)
    out["return_per_100"] = pd.Series(np.where(measured, per_100, np.nan), index=out.index)
    out["claimed_ev"] = claimed_ev(out).where(~held)
    out["realised"] = (out["return_per_100"] / TICKET_YEN - 1.0).where(measured)
    return out


# ---------------------------------------------------------------- calibration


def _check_boot(n_boot: Any, level: Any) -> None:
    if isinstance(level, bool) or not isinstance(level, Real) or not 0.0 < level < 1.0:
        raise ValueError(f"level must be a fraction in (0, 1), got {level!r}")
    if isinstance(n_boot, bool) or not isinstance(n_boot, Integral) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")


def _cluster_ols(
    x: np.ndarray, y: np.ndarray, clusters: Any, *, n_boot: int, level: float, seed: int
) -> tuple[float, float, float, int]:
    """(slope, lo, hi, n_clusters): the OLS slope of y on x with an intercept, and the (1 - level)
    and level quantiles of the slope over ``n_boot`` replicates that each draw as many clusters
    as there are, with replacement, and fit the rows of the drawn clusters. NotEstimable for fewer
    than two rows, no spread in x, one cluster, or a replicate with no spread in x."""
    pairs = len(x)
    if pairs < 2:
        raise NotEstimable(f"{pairs} pair(s): a slope needs at least two", n=pairs)
    xc, yc = x - x.mean(), y - y.mean()  # a slope with an intercept does not move with a shift
    sxx = float((xc * xc).sum())
    if not sxx > 0:
        raise NotEstimable("the score has no spread: the slope is not defined", n=pairs)
    slope = float((xc * yc).sum() / sxx)
    codes, _ = pd.factorize(pd.Series(clusters), sort=True)
    k = int(codes.max()) + 1
    if k < 2:
        raise NotEstimable(
            "one cluster: a bootstrap of whole clusters has no spread", n=pairs, slope=slope
        )
    sums = [np.bincount(codes, weights=w, minlength=k) for w in (None, xc, yc, xc * xc, xc * yc)]
    rng = np.random.default_rng(seed)
    chunk = max(1, _MAX_DRAWS_PER_CHUNK // k)
    slopes = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        drawn = rng.integers(0, k, size=(stop - start, k))
        size, sx, sy, sx2, sxy = (s[drawn].sum(axis=1) for s in sums)
        spread = sx2 - sx * sx / size
        if not (spread > 0).all():
            raise NotEstimable(
                "a replicate's score has no spread: too few clusters", n=pairs, slope=slope
            )
        slopes[start:stop] = (sxy - sx * sy / size) / spread
    lo, hi = np.quantile(slopes, [1.0 - level, level])
    return slope, float(lo), float(hi), k


def _aligned(
    x: pd.Series, y: pd.Series, clusters: pd.Series
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """The pairs of ``x`` and ``y`` matched by label, null pairs dropped, with their clusters;
    ValueError for a repeated label, or a kept pair without a cluster."""
    for name, series in (("claimed_ev", x), ("realised", y), ("clusters", clusters)):
        if not series.index.is_unique:
            raise ValueError(f"{name} has a repeated index label")
    pairs = pd.concat(
        {"x": pd.to_numeric(x).astype("float64"), "y": pd.to_numeric(y).astype("float64")}, axis=1
    ).dropna()
    missing = pairs.index.difference(clusters.index)
    if len(missing):
        raise ValueError(f"{len(missing)} pair(s) have no cluster, e.g. {missing[0]!r}")
    cluster_of = clusters.reindex(pairs.index)
    if cluster_of.isna().any():
        raise ValueError("a pair has a missing cluster")
    return pairs["x"].to_numpy(), pairs["y"].to_numpy(), cluster_of


def calibration_slope(
    claimed_ev: pd.Series,
    realised: pd.Series,
    clusters: pd.Series,
    *,
    n_boot: int,
    level: float,
    seed: int,
) -> Interval:
    """The calibration slope of realised on claimed EV, as a ``core.eval.bootstrap.Interval``.

    ``mean`` is the OLS slope, with an intercept, of ``realised`` on ``claimed_ev`` over every pair
    (matched by index label; a pair with a null on either side is left out, so refunded and
    pending records never enter it): 1 when the claims are calibrated in scale. ``lo`` and ``hi``
    are the (1 - level) and level quantiles of the slope over ``n_boot`` replicates that resample
    whole clusters (``clusters`` maps a label to its unit: D7 signs the meeting, venue x day), each
    one-sided at ``level`` as ``Interval``'s; ``n_clusters`` counts the clusters of the pairs used.
    ``n_boot``, ``level`` and ``seed`` have no defaults (D7: the caller passes the country's eval
    config). NotEstimable (a ValueError) for fewer than two pairs, claims with no spread or one
    cluster; ValueError for a repeated label or a pair without a cluster."""
    _check_boot(n_boot, level)
    x, y, cluster_of = _aligned(claimed_ev, realised, clusters)
    slope, lo, hi, k = _cluster_ols(x, y, cluster_of, n_boot=n_boot, level=level, seed=seed)
    return Interval(mean=slope, lo=lo, hi=hi, n_clusters=k)


# ---------------------------------------------------------------- flat stakes and growth


def _when(value: Any) -> datetime:
    if isinstance(value, datetime) and value.utcoffset() is not None:
        return value
    stamp = _instant(value)
    if stamp is None:
        raise ValueError(f"a pick's decision_utc is not a time with a UTC offset: {value!r}")
    return stamp


def _rank_key(value: Any) -> tuple[int, float]:
    """Rank 1 first; a pick without a rank after every ranked one."""
    return (1, math.inf) if _is_null(value) else (0, float(value))


def flat_stakes(picks: pd.DataFrame, *, limits: Limits) -> pd.DataFrame:
    """``picks``, the same rows in the same order, with ``stake_yen``: ``limits.flat_stake_yen``
    while the race day's staked total stays within ``limits.day_cap_yen``, else 0.

    The race day is the date of ``race_id`` (every venue of the day together). Within a day, picks
    take the cap in order of ``decision_utc`` as instants (ISO text with any UTC offset, or an
    aware datetime), then ``rank`` (1 first; a pick without one after the ranked), then input
    order. No Kelly, no EV threshold: a paper figure, never an order. ValueError without a flat
    stake, or for a pick without a decision time."""
    if not isinstance(limits, Limits):
        raise TypeError(f"limits must be Limits, got {limits!r}")
    flat = limits.flat_stake_yen
    if flat is None:
        raise ValueError("the limits have no flat stake: flat_stakes stakes the flat stake only")
    if "rank" not in picks.columns:
        raise ValueError("picks need a rank: they take the cap by decision time, then rank")
    days = [_day_of(race_id) for race_id in picks["race_id"]]
    whens = [_when(value) for value in picks["decision_utc"]]
    ranks = [_rank_key(value) for value in picks["rank"]]
    order = sorted(range(len(picks)), key=lambda i: (days[i], whens[i], ranks[i], i))
    stakes = [0] * len(picks)
    staked: dict[date, int] = {}
    for i in order:
        total = staked.get(days[i], 0)
        if total + flat <= limits.day_cap_yen:
            stakes[i] = flat
            staked[days[i]] = total + flat
    out = picks.copy()
    out["stake_yen"] = pd.Series(stakes, index=out.index, dtype="int64")
    return out


def log_growth_by_day(settled: pd.DataFrame, *, bankroll_yen: float) -> pd.DataFrame:
    """One row per race day of ``settled`` (``settle_shadow`` or ``settle_bets`` rows with
    ``stake_yen``), in day order: ``day``, ``n_settled`` (won, lost and refunded rows with a
    stake), ``staked_yen``, ``returned_yen``, ``ending_yen`` and ``log_growth`` = ln(ending /
    starting).

    The day starts with ``bankroll_yen``; ending = bankroll - the stakes of won, lost and refunded
    rows + stake / 100 x ``return_per_100`` of won rows + the stakes refunded. Pending rows
    (unsettled, no_board, holdout) and rows with no stake (past the cap) are left out, so a day
    with nothing staked and settled has a null ``log_growth``. A day that loses all its money has
    -inf. ValueError for a bankroll that is not positive, a frame without ``stake_yen``, or a day
    that staked more than its starting money and lost it (no logarithm)."""
    if isinstance(bankroll_yen, bool) or not isinstance(bankroll_yen, Real):
        raise ValueError(f"bankroll_yen must be a number, got {bankroll_yen!r}")
    if not (math.isfinite(bankroll_yen) and bankroll_yen > 0):
        raise ValueError(f"bankroll_yen must be positive, got {bankroll_yen!r}")
    if "stake_yen" not in settled.columns:
        raise ValueError("settled rows need stake_yen: the growth is of the money staked")
    days: dict[date, dict[str, float]] = {}
    columns = (settled["race_id"], settled["outcome"], settled["stake_yen"])
    for race_id, outcome, stake, per_100 in zip(
        *columns, _numbers(settled, "return_per_100"), strict=True
    ):
        day = days.setdefault(_day_of(race_id), {"n": 0, "staked": 0.0, "returned": 0.0})
        if outcome not in MOVED or not float(stake) > 0:
            continue
        day["n"] += 1
        day["staked"] += float(stake)
        if outcome == WON:
            day["returned"] += float(stake) * per_100 / TICKET_YEN
        elif outcome == REFUNDED:
            day["returned"] += float(stake)
    rows = []
    for day, sums in sorted(days.items()):
        ending = bankroll_yen - sums["staked"] + sums["returned"]
        if ending < 0:
            raise ValueError(f"{day.isoformat()} ends below 0 yen: it staked over its bankroll")
        if not sums["n"]:
            growth = math.nan
        else:
            growth = -math.inf if ending == 0 else math.log(ending / bankroll_yen)
        rows.append(
            {
                "day": day,
                "n_settled": int(sums["n"]),
                "staked_yen": sums["staked"],
                "returned_yen": sums["returned"],
                "ending_yen": ending,
                "log_growth": growth,
            }
        )
    names = ["day", "n_settled", "staked_yen", "returned_yen", "ending_yen", "log_growth"]
    return pd.DataFrame(rows, columns=names).astype({"log_growth": "float64"})


# ---------------------------------------------------------------- E20: selection drift


def _meeting(race_id: str) -> str:
    """The meeting (venue x day, D7) of a race: its race_id without the race number."""
    return race_id.rsplit(":", 1)[0]


def selection_drift(
    log: pd.DataFrame, *, lag_minutes: float, n_boot: int, level: float, seed: int
) -> RegressionResult:
    """E20: does the overlay score predict the fall in log odds to the final?

    For each record at t (``tau_min``), the fall ln(``board_odds``) - ln(``final_odds``) is
    regressed (OLS with an intercept) on the score of the record of the same race, pool and
    combination at ``tau_min + lag_minutes``, its claimed EV (``claimed_ev``): a board
    ``lag_minutes`` older, never the record at t itself, so noise in the board at t, which the
    fall contains, cannot drive the slope mechanically. ``final_odds`` is the final board's figure,
    joined to the frame by the caller. Rows without that lagged record or its score, without a
    positive board or final figure, no_board rows and holdout rows (labelled, or by the rule) are
    dropped, never filled; holdout rows give no score either. ``lo`` and ``hi`` come from a
    bootstrap of whole meetings (venue x day), as ``calibration_slope``'s. ValueError for a lag
    that is not a positive number, or a frame without ``final_odds``; NotEstimable when too few
    rows, scores or meetings remain."""
    if (
        isinstance(lag_minutes, bool)
        or not isinstance(lag_minutes, Real)
        or not math.isfinite(lag_minutes)
        or lag_minutes <= 0
    ):
        raise ValueError(f"lag_minutes must be a positive number of minutes, got {lag_minutes!r}")
    _check_boot(n_boot, level)
    if "final_odds" not in log.columns:
        raise ValueError("selection_drift needs final_odds, the final board's figure per record")
    records = log.to_dict("records")
    keep = [not _held(r) and not _no_board(r) for r in records]
    frame = log.loc[np.array(keep, dtype=bool)]
    scores = claimed_ev(frame)
    board, final = _numbers(frame, "board_odds"), _numbers(frame, "final_odds")
    keys = list(
        zip(frame["race_id"], frame["pool"], frame["combination"], frame["tau_min"], strict=True)
    )
    lookup = {
        (race, pool, combination, float(tau)): float(score)
        for (race, pool, combination, tau), score in zip(keys, scores, strict=True)
        if not math.isnan(score)
    }
    x, y, meetings = [], [], []
    for (race, pool, combination, tau), at_t, at_final in zip(keys, board, final, strict=True):
        lagged = lookup.get((race, pool, combination, float(tau) + lag_minutes))
        if lagged is None or not (at_t > 0 and at_final > 0):  # NaN fails too
            continue
        x.append(lagged)
        y.append(math.log(at_t) - math.log(at_final))
        meetings.append(_meeting(race))
    slope, lo, hi, k = _cluster_ols(
        np.array(x, dtype=float),
        np.array(y, dtype=float),
        meetings,
        n_boot=n_boot,
        level=level,
        seed=seed,
    )
    return RegressionResult(slope=slope, lo=lo, hi=hi, n=len(x), n_clusters=k)


# ---------------------------------------------------------------- the hand-played bets


def read_bets(days: Iterable[date]) -> pd.DataFrame:
    """Each day's ``BetLog.read()`` rows, in day then log order, each as the log stores it (object
    columns, so a null option_id stays None). Holdout days are read too: stakes are not
    outcomes, and ``settle_bets`` keeps their outcomes 'holdout'. A day without a log gives none
    and creates none. Each day is read once, however often it is given."""
    limits = load_limits()
    rows: list[dict[str, Any]] = []
    for day in dict.fromkeys(days):
        path = bet_log_path(day)
        if path.is_file():
            rows += BetLog(path, day=day, limits=limits).read()
    return pd.DataFrame(rows, columns=[*BET_FIELDS, *OPTIONAL, *ADDED], dtype=object)


def played_by_day(settled: pd.DataFrame, ranked: Iterable[RankedRow]) -> pd.DataFrame:
    """Per race day (the ``race_id`` date) and ``option_id`` of ``settled`` (``settle_bets``
    rows; ``off-board`` for a null option), #454's ``played_by_option`` totals (``n_bets``,
    ``staked_yen``, ``paid_yen``, ``return_per_100`` over won and lost bets, ``n_refunded``,
    ``n_pending``) and ``rank``, the option's place on ``ranked`` (``leaderboard.rank`` rows):
    null for an unmeasured option, one not on the board, and the owner's own off-board picks."""
    places = {row.row.option_id: row.rank for row in ranked}
    days = np.array([_day_of(race_id) for race_id in settled["race_id"]], dtype=object)
    frames = []
    for day in sorted(set(days)):
        by_option = played_by_option(settled.loc[days == day])
        by_option.insert(0, "day", day)
        frames.append(by_option)
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame({name: pd.Series(dtype=object) for name in ("day", *OPTION_COLUMNS)})
    out["rank"] = pd.array(
        [None if option == OFF_BOARD else places.get(option) for option in out["option_id"]],
        dtype="Int64",
    )
    return out
