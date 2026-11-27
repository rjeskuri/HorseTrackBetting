"""Settle the owner's hand-placed bets against the final dividends (#454).

``settle_bets`` joins a day's ``BetLog.read()`` rows to the dividends that paid: a bet whose
combination is a paying row of its race and pool won stake / 100 x that row's dividend, one with a
refund row gets its stake back, one whose race and pool have rows but not its combination lost.
A bet whose race has no row of its pool yet waits ('unsettled'), and a holdout day's bet is never
settled here ('holdout'): the outcomes of holdout days stay behind ``LockedHoldout.open``, which
this module never opens. ``played_by_option`` sums the settled bets per leaderboard option, and
``day_dividends`` reads a development day's dividends from the jra store, with a refund row for
each combination a runner scratched before the start returns (the store holds none). The format
is in docs/specs/bet-log.md (Settlement). Nothing here writes a file.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from datetime import date
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.core.paths import processed_dir
from horsetrack.countries.japan.bets.log import (
    _COMBINATION,
    _RACE_ID,
    COUNTRY,
    HOLDOUT,
    POOLS,
    SOURCE,
    WIN5,
    _check_day,
    _in_store_order,
    _jra,
    _kind,
    _legs_in_order,
)
from horsetrack.countries.japan.models.v0.store import read_store_table

WON, LOST, REFUNDED, UNSETTLED = "won", "lost", "refunded", "unsettled"
OUTCOMES = (WON, LOST, REFUNDED, UNSETTLED, HOLDOUT)
SETTLED = (WON, LOST)  # the bets a return is measured over
PENDING = (UNSETTLED, HOLDOUT)  # the bets still waiting for an outcome
OFF_BOARD = "off-board"  # the option of the owner's own pick (a null option_id)
DIVIDEND_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
OPTION_COLUMNS = [
    "option_id",
    "n_bets",
    "staked_yen",
    "paid_yen",
    "return_per_100",
    "n_refunded",
    "n_pending",
]
RACE_COLUMNS = ["race_id", "race_date", "venue", "race_no"]
RUNNER_COLUMNS = ["race_id", "horse_no", "bracket", "is_scratched"]


def _day_of(race_id: Any) -> date:
    match = _RACE_ID.fullmatch(race_id) if isinstance(race_id, str) else None
    if match is None:
        raise ValueError(f"a bet's race_id is not of the bet log's form: {race_id!r}")
    return date.fromisoformat(match["day"])


def _race_key(race_id: str, pool: str) -> str:
    """The race a bet's dividends are keyed by: WIN5 by its day, as the log keys it (a format-1
    WIN5 record carries a venue race_id), every other pool by its race."""
    return f"jra:{_day_of(race_id).isoformat()}:{WIN5}" if pool == WIN5 else race_id


def _paid_key(pool: str, combination: Any) -> str:
    """A dividend row's combination as the log stores a bet's: in the store's order."""
    if not isinstance(combination, str) or not _COMBINATION.fullmatch(combination):
        raise ValueError(
            f"a {pool} dividend row has a combination of no horse numbers: {combination!r}"
        )
    return _in_store_order(pool, combination)


def _is_refund(value: Any) -> bool:
    return not pd.isna(value) and bool(value)


def _paying_rows(
    dividends: pd.DataFrame, wanted: set[tuple[str, str]]
) -> dict[tuple[str, str], dict[str, tuple[float | None, bool]]]:
    """{(race key, pool): {combination: (dividend, is_refund)}} for the races and pools of
    ``wanted``; a refund's dividend, and a missing one, is None. A combination with two different
    rows in one race and pool is refused."""
    index: dict[tuple[str, str], dict[str, tuple[float | None, bool]]] = {}
    for row in dividends[DIVIDEND_COLUMNS].to_dict("records"):
        race_pool = (row["race_id"], row["pool"])
        if race_pool not in wanted:
            continue
        combination = _paid_key(row["pool"], row["combination"])
        refund = _is_refund(row["is_refund"])
        dividend = None if refund or pd.isna(row["dividend"]) else float(row["dividend"])
        seen = index.setdefault(race_pool, {}).setdefault(combination, (dividend, refund))
        if seen != (dividend, refund):
            raise ValueError(
                f"{race_pool[0]} {race_pool[1]} {combination} has two dividend rows: "
                f"{seen} and {(dividend, refund)}"
            )
    return index


def _settled(stake: Any, paid: tuple[float | None, bool] | None, rows: Mapping | None) -> tuple:
    """(outcome, paid_yen) of a development bet: ``rows`` are its race and pool's dividend rows
    (None when there are none yet), ``paid`` the row of its combination among them."""
    if rows is None:
        return UNSETTLED, None
    if paid is None:
        return LOST, 0.0
    dividend, refund = paid
    if refund:
        return REFUNDED, float(stake)
    if dividend is None:
        raise ValueError("a paying dividend row has no dividend")
    return WON, stake * dividend / 100


def settle_bets(bets: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """``bets`` (``BetLog.read()`` rows) with ``outcome``, ``paid_yen`` and ``return_per_100``
    added, one row per bet in order. ``dividends`` has ``DIVIDEND_COLUMNS``, its race_id in the
    log's form (``jra:<day>:<VENUE>:<NN>``, ``jra:<day>:WIN5``), its dividend in yen per 100 yen.
    A bet labelled holdout, or of a day the committed holdout rule holds, is 'holdout' whatever
    ``dividends`` holds; ``paid_yen`` and ``return_per_100`` are null for it and for an
    'unsettled' bet, and ``return_per_100`` is null for a refunded one."""
    out = bets.copy()
    records = out.to_dict("records")
    held = [
        row.get("day_status") == HOLDOUT or is_holdout_day(COUNTRY, _day_of(row["race_id"]))
        for row in records
    ]
    keys = [(_race_key(row["race_id"], row["pool"]), row["pool"]) for row in records]
    wanted = {key for key, hold in zip(keys, held, strict=True) if not hold}
    index = _paying_rows(dividends, wanted)
    outcomes: list[str] = []
    paid_yen: list[float | None] = []
    for row, key, hold in zip(records, keys, held, strict=True):
        if hold:
            outcome, paid = HOLDOUT, None
        else:
            rows = index.get(key)
            bet = _in_store_order(row["pool"], row["combination"])  # a format-1 record's order
            outcome, paid = _settled(
                row["stake_yen"], None if rows is None else rows.get(bet), rows
            )
        outcomes.append(outcome)
        paid_yen.append(paid)
    out["outcome"] = pd.Series(outcomes, index=out.index, dtype=object)
    out["paid_yen"] = pd.Series(paid_yen, index=out.index, dtype="float64")
    stake = pd.to_numeric(out.get("stake_yen", pd.Series(index=out.index, dtype="float64")))
    measured = out["outcome"].isin(SETTLED)
    out["return_per_100"] = (out["paid_yen"] / stake * 100).where(measured)
    return out


def totals(settled: pd.DataFrame) -> dict[str, Any]:
    """n_bets, staked_yen, paid_yen and return_per_100 (paid / staked x 100, None when nothing
    was staked) over the won and lost bets of ``settled``, with n_refunded and n_pending."""
    outcome = settled.get("outcome", pd.Series(index=settled.index, dtype=object))
    measured = outcome.isin(SETTLED)
    staked = int(pd.to_numeric(settled.loc[measured, "stake_yen"]).sum()) if measured.any() else 0
    paid = float(pd.to_numeric(settled.loc[measured, "paid_yen"]).sum()) if measured.any() else 0.0
    return {
        "n_bets": int(measured.sum()),
        "staked_yen": staked,
        "paid_yen": paid,
        "return_per_100": paid / staked * 100 if staked else None,
        "n_refunded": int(outcome.eq(REFUNDED).sum()),
        "n_pending": int(outcome.isin(PENDING).sum()),
    }


def _options(settled: pd.DataFrame) -> list[str]:
    if "option_id" not in settled.columns:
        return [OFF_BOARD] * len(settled)
    return [OFF_BOARD if pd.isna(o) else str(o) for o in settled["option_id"].tolist()]


def played_by_option(settled: pd.DataFrame) -> pd.DataFrame:
    """One row per option_id (``OFF_BOARD`` for a null or absent one), sorted, with
    ``totals`` of its bets: return per 100 yen over won and lost bets, refunded and pending
    (unsettled or holdout) bets counted beside them."""
    options = _options(settled)
    rows = []
    for option in sorted(set(options)):
        mine = [o == option for o in options]
        rows.append({"option_id": option, **totals(settled.loc[mine])})
    frame = pd.DataFrame(rows, columns=OPTION_COLUMNS)
    frame["return_per_100"] = frame["return_per_100"].astype("float64")
    return frame


def _venue_codes() -> dict[str, str]:
    """The odds log's venue code for each code, number and name of the jra config's venues."""
    lookup: dict[str, str] = {}
    for code, venue in _jra()["venues"].items():
        lookup.update({str(venue["number"]): code, str(venue["name"]): code, code: code})
    return lookup


def _log_race_ids(races: Iterable[dict], day: date) -> dict[str, str]:
    """{store race_id: jra:<day>:<VENUE>:<NN>} of the store's ``races`` rows."""
    codes = _venue_codes()
    return {
        row["race_id"]: f"jra:{day.isoformat()}:{codes.get(row['venue'], row['venue'])}:"
        f"{int(row['race_no']):02d}"
        for row in races
    }


def _no_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{c: pd.Series(dtype=object) for c in ("race_id", "pool", "combination")},
            "dividend": pd.Series(dtype="float64"),
            "is_refund": pd.Series(dtype=bool),
        }
    )


def _refunded(pool: str, field: list[tuple[int, int | None, bool]]) -> Iterator[str]:
    """The combinations of ``pool``, in the store's order, that a race's scratches return;
    ``field`` is its entered runners as (horse_no, bracket, is_scratched). A combination is
    returned when it uses one of its numbers more often than that number has runners left running,
    though it had that many entered: a scratched horse; for BRACKET_QUINELLA a bracket with no
    runner left, or for a pair of one bracket (7,7, JRA's ゾロ目) fewer than two."""
    legs = int(_jra()["pools"][pool]["legs"])
    if _kind(pool).endswith("_bracket"):
        runners = [(bracket, gone) for _, bracket, gone in field if bracket is not None]
        draw = itertools.combinations_with_replacement
    else:
        runners = [(horse_no, gone) for horse_no, _, gone in field]
        draw = itertools.permutations if _legs_in_order(pool) else itertools.combinations
    entered = Counter(number for number, _ in runners)
    left = Counter(number for number, gone in runners if not gone)
    for numbers in draw(sorted(entered), legs):
        if any(left[n] < k <= entered[n] for n, k in Counter(numbers).items()):
            yield ",".join(map(str, numbers))


def _scratch_refunds(paid: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    """Refund rows (``is_refund`` true, no dividend) of the combinations JRA returns for runners
    scratched before the start (取消 WV, 除外 WX: ``runners.is_scratched``; a runner pulled up,
    競走中止, or disqualified, 失格, ran), which the store's ``dividends`` never holds. In each
    race with such a runner, for each pool of its ``paid`` rows but WIN5 (JRA puts the leg's
    favourite in for a scratched WIN5 horse): every combination ``_refunded`` gives that the store
    has no row of. A pool without rows (not sold, 不成立, or no results yet) gets none."""
    scratched = runners["is_scratched"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    fields: dict[str, list[tuple[int, int | None, bool]]] = {}
    for race_id, horse_no, bracket, gone in zip(
        runners["race_id"], runners["horse_no"], runners["bracket"], scratched, strict=True
    ):
        if not pd.isna(horse_no):
            entry = (int(horse_no), None if pd.isna(bracket) else int(bracket), bool(gone))
            fields.setdefault(race_id, []).append(entry)
    hit = {race_id for race_id, field in fields.items() if any(gone for *_, gone in field)}
    stored: dict[tuple[str, str], set[str]] = {}
    for race_id, pool, combination in paid[["race_id", "pool", "combination"]].itertuples(
        index=False
    ):
        if race_id in hit and pool in POOLS and pool != WIN5:
            valid = isinstance(combination, str) and _COMBINATION.fullmatch(combination)
            stored.setdefault((race_id, pool), set()).add(
                _in_store_order(pool, combination) if valid else combination
            )
    rows = [
        (race_id, pool, combination)
        for (race_id, pool), present in sorted(stored.items())
        for combination in _refunded(pool, fields[race_id])
        if combination not in present
    ]
    return pd.DataFrame(
        {
            "race_id": pd.Series([row[0] for row in rows], dtype=object),
            "pool": pd.Series([row[1] for row in rows], dtype=object),
            "combination": pd.Series([row[2] for row in rows], dtype=object),
            "dividend": pd.Series([float("nan")] * len(rows), dtype="float64"),
            "is_refund": pd.Series([True] * len(rows), dtype=bool),
        }
    )


def _day_runners(day: date, race_ids: list[str]) -> pd.DataFrame | None:
    """The jra store's ``runners`` rows of ``race_ids`` (``RUNNER_COLUMNS``), read through
    ``read_store_table``; None when the store has no runners table."""
    if not (processed_dir(SOURCE) / "runners.parquet").is_file():
        return None
    return read_store_table(
        "runners",
        columns=RUNNER_COLUMNS,
        start=day,
        end=day,
        touches_price=False,
        where=ds.field("race_id").isin(race_ids),
    ).to_pandas()


def day_dividends(day: date) -> pd.DataFrame:
    """The jra store's dividend rows of ``day``'s races, keyed by the log's race_id, and a refund
    row (``_scratch_refunds``) for each combination a runner scratched before the start returns.
    A holdout day gives an empty frame of ``DIVIDEND_COLUMNS`` without opening the store. ``races``
    and ``runners`` are read through ``read_store_table``, which drops holdout days; a day without
    races gives the empty frame too, and a store without a runners table no refund rows."""
    _check_day(day)
    if is_holdout_day(COUNTRY, day):
        return _no_dividends()
    races = read_store_table(
        "races", columns=RACE_COLUMNS, start=day, end=day, touches_price=False
    ).to_pylist()
    keyed = _log_race_ids(races, day)
    if not keyed:
        return _no_dividends()
    paid = pq.read_table(
        processed_dir(SOURCE) / "dividends.parquet",
        columns=DIVIDEND_COLUMNS,
        filters=ds.field("race_id").isin(list(keyed)),
    ).to_pandas()
    runners = _day_runners(day, list(keyed))
    refunds = None if runners is None else _scratch_refunds(paid, runners)
    if refunds is not None and len(refunds):
        paid = pd.concat([paid, refunds.astype(paid.dtypes.to_dict())], ignore_index=True)
    paid["race_id"] = paid["race_id"].map(keyed)
    return paid[DIVIDEND_COLUMNS].reset_index(drop=True)
