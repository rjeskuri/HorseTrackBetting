"""Ticket #454: settle the owner's hand-placed bets against the final dividends.

Contract pinned here, module ``horsetrack.countries.japan.bets.settle``:

- ``settle_bets(bets, dividends) -> DataFrame``. ``bets`` are ``BetLog.read()`` rows; ``dividends``
  have ``race_id`` (the log's ``jra:<day>:<VENUE>:<NN>``, or ``jra:<day>:WIN5``), ``pool``,
  ``combination``, ``dividend`` (yen per 100 yen staked) and ``is_refund``. One row comes back per
  bet, its columns kept, plus ``outcome``, ``paid_yen`` and ``return_per_100``:
  - 'won': its combination is a paying row of its race and pool; paid stake/100 x that row's
    dividend (a dead heat, Place and Wide pay each paying combination its own dividend);
  - 'refunded': its combination has an ``is_refund`` row; paid its stake;
  - 'lost': its race and pool have rows and none is its combination; paid 0;
  - 'unsettled': its race has no dividend row yet (ruling: or none of its pool); no paid figure;
  - 'holdout': labelled holdout (ruling: or on a day the committed holdout rule holds), whatever
    dividends are passed; no paid figure.
  ``return_per_100`` is paid / stake x 100 for a won or lost bet, null without a paid figure.
  PLACE, BRACKET_QUINELLA, QUINELLA, QUINELLA_PLACE and TRIO match in any order; WIN, FORECAST and
  TIERCE only as given; WIN5 as text, in race order. A format-1 WIN5 record (a565ac0, keyed by a
  venue race) settles against its day's ``jra:<day>:WIN5`` rows (docs/specs/bet-log.md).
- ``played_by_option(settled) -> DataFrame``: one row per option_id ('off-board' for null or
  absent) with n_bets, staked_yen, paid_yen and return_per_100 (paid / staked x 100, null when
  nothing was staked) over won and lost bets, and n_refunded and n_pending beside them.
- ``day_dividends(day) -> DataFrame``: the jra store's dividend rows of the day's development races
  (``races`` read through ``models.v0.store.read_store_table``, which drops holdout days), each
  race keyed ``jra:<date>:<VENUE>:<NN>`` from its race_date, venue and race_no.

Every bet and dividend here is synthetic. HORSETRACK_DATA_DIR points at tmp_path in every test.
"""

from __future__ import annotations  # noqa: I001 (isort counts bets.settle third-party until it exists)

import hashlib
import itertools
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from horsetrack.core import paths
from horsetrack.core.eval.holdout import is_holdout_day
from horsetrack.countries.japan.bets.settle import day_dividends, played_by_option, settle_bets

REPO = Path(__file__).resolve().parents[4]
JAPAN_CONFIG = REPO / "src" / "horsetrack" / "countries" / "japan" / "config"
DAY = date(2026, 11, 22)  # before the holdout start: development by any reading
DIVIDEND_COLUMNS = ["race_id", "pool", "combination", "dividend", "is_refund"]
ADDED = {"outcome", "paid_yen", "return_per_100"}
NULL = None  # the expected paid figure of a bet that is not settled

_seq = itertools.count(1)


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


def race(n: int, venue: str = "TOK", day: date = DAY) -> str:
    return f"jra:{day.isoformat()}:{venue}:{n:02d}"


def placed(
    race_id: str,
    pool: str,
    combination: str,
    stake: int,
    *,
    tag: str,
    option_id: str | None = None,
    day_status: str = "development",
) -> dict:
    """A bet as ``BetLog.read()`` returns it (synthetic); ``tag`` is its note, to find it again."""
    day = race_id.split(":")[1]
    return {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "stake_yen": stake,
        "placed_utc": f"{day}T05:40:00+00:00",
        "option_id": option_id,
        "note": tag,
        "seq": next(_seq),
        "written_utc": f"{day}T05:41:00+00:00",
        "day_status": day_status,
        "over_cap": False,
        "off_flat": stake != 1_000,
    }


def paying(race_id: str, pool: str, combination: str, dividend: float, refund: bool = False):
    return {
        "race_id": race_id,
        "pool": pool,
        "combination": combination,
        "dividend": float(dividend),
        "is_refund": refund,
    }


def dividends(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=DIVIDEND_COLUMNS)


def settle(bets: list[dict], rows: list[dict] | pd.DataFrame) -> dict[str, dict]:
    """``settle_bets`` on the bets and dividend rows, as {note: settled row}."""
    frame = pd.DataFrame(bets)
    out = settle_bets(frame, rows if isinstance(rows, pd.DataFrame) else dividends(rows))
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(frame), "one settled row per bet"
    assert set(frame.columns) | ADDED <= set(out.columns), "the bet's columns are kept"
    by_tag = {row["note"]: row for row in out.to_dict("records")}
    assert set(by_tag) == {b["note"] for b in bets}
    return by_tag


def assert_settled(row: dict, outcome: str, paid: float | None) -> None:
    where = f"{row['note']}: {row['pool']} {row['combination']} on {row['race_id']}"
    assert row["outcome"] == outcome, where
    if paid is None:
        assert pd.isna(row["paid_yen"]), f"{where}: no paid figure, got {row['paid_yen']!r}"
        assert pd.isna(row["return_per_100"]), f"{where}: no return, got {row['return_per_100']!r}"
        return
    assert row["paid_yen"] == pytest.approx(paid), where
    if outcome in ("won", "lost"):
        expected = paid / row["stake_yen"] * 100
        assert row["return_per_100"] == pytest.approx(expected), where


def run_table(cases: list[tuple], rows: list[dict]) -> None:
    """cases: (tag, race_id, pool, combination, stake, outcome, paid)."""
    bets = [placed(r, pool, comb, stake, tag=tag) for tag, r, pool, comb, stake, _, _ in cases]
    got = settle(bets, rows)
    for tag, _, _, _, _, outcome, paid in cases:
        assert_settled(got[tag], outcome, paid)


# ---------------------------------------------------------------- won and lost


def test_bets_settle_against_the_final_dividends():
    r1, r2, r3, r4 = race(11), race(12), race(1, "KYO"), race(5, "NAK")
    rows = [
        paying(r1, "WIN", "4", 350),
        paying(r2, "WIN", "12", 1_740),
        paying(r3, "WIN", "1", 110),
        paying(r4, "WIN", "7", 44_340),
        paying(r4, "QUINELLA", "7,10", 2_590),
    ]
    # paid = stake / 100 x the dividend of the row it matches, worked out by hand
    cases = [
        ("the ticket's won bet", r1, "WIN", "4", 200, "won", 700),
        ("the ticket's lost bet", r1, "WIN", "2", 100, "lost", 0),
        ("1000 at 1740", r2, "WIN", "12", 1_000, "won", 17_400),
        ("4 won race 11, not 12", r2, "WIN", "4", 1_000, "lost", 0),
        ("300 at the 110 floor", r3, "WIN", "1", 300, "won", 330),
        ("12 won TOK 12, not KYO 1", r3, "WIN", "12", 500, "lost", 0),
        ("100 at 44340", r4, "WIN", "7", 100, "won", 44_340),
        ("1500 on the quinella", r4, "QUINELLA", "7,10", 1_500, "won", 38_850),
        ("one horse of the pair", r4, "QUINELLA", "4,7", 200, "lost", 0),
    ]
    run_table(cases, rows)


def test_unordered_pools_match_in_any_order_and_ordered_pools_do_not():
    rows, cases = [], []
    table = [  # pool, paid combination, dividend, bet combination, stake, outcome
        ("QUINELLA", "4,7", 1_230, "7,4", 200, "won"),  # the ticket's pair
        ("FORECAST", "4,7", 2_610, "7,4", 200, "lost"),  # the ticket's pair
        ("FORECAST", "4,7", 2_610, "4,7", 300, "won"),
        ("QUINELLA", "4,7", 1_230, "4,9", 100, "lost"),  # one shared horse is not a match
        ("QUINELLA", "11,2", 880, "2,11", 100, "won"),  # the paid row's order does not matter
        ("QUINELLA_PLACE", "3,12", 990, "12,3", 500, "won"),  # by number: 3 before 12
        ("BRACKET_QUINELLA", "2,8", 1_300, "8,2", 100, "won"),
        ("BRACKET_QUINELLA", "3,3", 770, "3,3", 100, "won"),
        ("TRIO", "2,5,9", 4_470, "9,2,5", 100, "won"),
        ("TRIO", "2,5,9", 4_470, "2,5,8", 100, "lost"),
        ("TIERCE", "2,5,9", 44_340, "9,2,5", 100, "lost"),
        ("TIERCE", "2,5,9", 44_340, "5,2,9", 100, "lost"),
        ("TIERCE", "2,5,9", 44_340, "2,5,9", 200, "won"),
        ("PLACE", "6", 130, "6", 1_000, "won"),
    ]
    for n, (pool, paid_comb, dividend, comb, stake, outcome) in enumerate(table, 1):
        r = race(1 + (n - 1) % 12, "HSN" if n <= 12 else "CKY")
        rows.append(paying(r, pool, paid_comb, dividend))
        paid = stake * dividend / 100 if outcome == "won" else 0
        cases.append((f"{pool} {comb} vs {paid_comb}", r, pool, comb, stake, outcome, paid))

    # WIN5 compares as text, its legs in race order, against the day's WIN5 key
    w5 = f"jra:{DAY.isoformat()}:WIN5"
    rows.append(paying(w5, "WIN5", "3,11,1,7,2", 1_528_000))
    cases += [
        ("win5 as paid", w5, "WIN5", "3,11,1,7,2", 100, "won", 1_528_000),
        ("win5 legs reordered", w5, "WIN5", "1,2,3,7,11", 100, "lost", 0),
    ]
    run_table(cases, rows)


def test_place_and_a_dead_heat_pay_each_paying_combination():
    r, other = race(6, "KYO"), race(7, "KYO")
    rows = [  # a dead heat for first between 3 and 6, horse 1 third (synthetic)
        paying(r, "WIN", "3", 120),
        paying(r, "WIN", "6", 160),
        paying(r, "PLACE", "3", 110),
        paying(r, "PLACE", "6", 130),
        paying(r, "PLACE", "1", 140),
        paying(r, "QUINELLA", "3,6", 330),
        paying(r, "QUINELLA_PLACE", "3,6", 150),
        paying(r, "QUINELLA_PLACE", "1,3", 170),
        paying(r, "QUINELLA_PLACE", "1,6", 290),
        paying(r, "FORECAST", "3,6", 290),
        paying(r, "FORECAST", "6,3", 420),
        paying(r, "TRIO", "1,3,6", 440),
        paying(r, "TIERCE", "3,6,1", 750),
        paying(r, "TIERCE", "6,3,1", 1_310),
        paying(other, "WIN", "1", 990),
        paying(other, "PLACE", "9", 270),
    ]
    cases = [
        ("win on 3", r, "WIN", "3", 1_000, "won", 1_200),
        ("win on 6", r, "WIN", "6", 500, "won", 800),
        ("win on 3 again", r, "WIN", "3", 200, "won", 240),  # each bet pays on its own
        ("win on the third", r, "WIN", "1", 100, "lost", 0),
        ("place on 1", r, "PLACE", "1", 300, "won", 420),
        ("place on 6", r, "PLACE", "6", 200, "won", 260),
        ("place on 3", r, "PLACE", "3", 100, "won", 110),
        ("place on 9 here", r, "PLACE", "9", 100, "lost", 0),  # 9 placed in the other race
        ("wide 6,1", r, "QUINELLA_PLACE", "6,1", 100, "won", 290),
        ("wide 3,1", r, "QUINELLA_PLACE", "3,1", 400, "won", 680),
        ("wide 1,9", r, "QUINELLA_PLACE", "1,9", 100, "lost", 0),
        ("exacta 6,3", r, "FORECAST", "6,3", 100, "won", 420),
        ("exacta 3,6", r, "FORECAST", "3,6", 200, "won", 580),
        ("exacta 3,1", r, "FORECAST", "3,1", 100, "lost", 0),
        ("trifecta 6,3,1", r, "TIERCE", "6,3,1", 100, "won", 1_310),
        ("trifecta 1,3,6", r, "TIERCE", "1,3,6", 100, "lost", 0),
        ("trio 6,1,3", r, "TRIO", "6,1,3", 100, "won", 440),
    ]
    run_table(cases, rows)


def test_a_refunded_combination_returns_the_stake():
    r = race(9, "NAK")
    rows = [  # horse 5 scratched after sales: its tickets are refunded (synthetic)
        paying(r, "WIN", "4", 350),
        paying(r, "WIN", "5", 100, refund=True),
        paying(r, "PLACE", "4", 150),
        paying(r, "PLACE", "5", float("nan"), refund=True),  # a refund row may carry no dividend
        paying(r, "QUINELLA", "4,8", 1_870),
        paying(r, "QUINELLA", "5,9", float("nan"), refund=True),
    ]
    cases = [
        ("win on the scratched horse", r, "WIN", "5", 300, "refunded", 300),
        ("place on the scratched horse", r, "PLACE", "5", 700, "refunded", 700),
        ("quinella with it, either order", r, "QUINELLA", "9,5", 200, "refunded", 200),
        ("win on the winner", r, "WIN", "4", 100, "won", 350),
        ("place on the winner", r, "PLACE", "4", 1_000, "won", 1_500),
        ("win on a loser", r, "WIN", "2", 100, "lost", 0),
        ("quinella 5,8 is not refunded", r, "QUINELLA", "5,8", 100, "lost", 0),
    ]
    run_table(cases, rows)


# ---------------------------------------------------------------- not settled


def test_holdout_day_bets_are_never_settled():
    r, quiet = race(11), race(3)
    rows = [paying(r, "WIN", "4", 350), paying(r, "QUINELLA", "4,7", 1_230)]
    bets = [
        placed(r, "WIN", "4", 200, tag="held winner", day_status="holdout"),
        placed(r, "WIN", "2", 100, tag="held loser", day_status="holdout"),
        placed(r, "QUINELLA", "7,4", 1_000, tag="held quinella", day_status="holdout"),
        placed(quiet, "WIN", "4", 100, tag="held, no rows", day_status="holdout"),
        placed(r, "WIN", "4", 200, tag="development winner"),
        placed(r, "WIN", "2", 100, tag="development loser"),
    ]
    got = settle(bets, rows)
    for tag in ("held winner", "held loser", "held quinella", "held, no rows"):
        assert_settled(got[tag], "holdout", NULL)
    assert_settled(got["development winner"], "won", 700)
    assert_settled(got["development loser"], "lost", 0)


def test_a_bet_on_a_day_the_holdout_rule_holds_is_never_settled():
    # Ruling: fail closed as the log does: the committed rule holding for the race's day is
    # enough, whatever the record's label says.
    held = first_held_day()
    r = race(11, "NAK", held)
    rows = [paying(r, "WIN", "4", 350), paying(race(11), "WIN", "4", 350)]
    bets = [
        placed(r, "WIN", "4", 200, tag="labelled development on a held day"),
        placed(r, "WIN", "2", 100, tag="its loser"),
        placed(race(11), "WIN", "4", 200, tag="a development day"),
    ]
    got = settle(bets, rows)
    assert_settled(got["labelled development on a held day"], "holdout", NULL)
    assert_settled(got["its loser"], "holdout", NULL)
    assert_settled(got["a development day"], "won", 700)


def test_a_race_without_dividends_is_unsettled():
    r = race(11)
    rows = [paying(r, "WIN", "4", 350), paying(r, "QUINELLA", "4,7", 1_230)]
    cases = [
        ("the next race", race(12), "WIN", "4", 100, "unsettled", NULL),
        ("another venue", race(11, "KYO"), "QUINELLA", "4,7", 1_000, "unsettled", NULL),
        ("another day", race(11, day=DAY - timedelta(days=1)), "WIN", "4", 100, "unsettled", NULL),
        ("the settled race", r, "WIN", "4", 200, "won", 700),
        ("its loser", r, "WIN", "2", 100, "lost", 0),
    ]
    run_table(cases, rows)

    # no dividend rows at all yet: every bet waits
    bets = [placed(r, "WIN", "4", 200, tag="a"), placed(r, "QUINELLA", "4,7", 100, tag="b")]
    got = settle(bets, dividends([]))
    assert_settled(got["a"], "unsettled", NULL)
    assert_settled(got["b"], "unsettled", NULL)


def test_a_bet_whose_pool_has_no_rows_in_its_race_is_unsettled():
    # Ruling: a bet settles only once its race has rows of its own pool; a pool missing from the
    # store is not a loss.
    r = race(11)
    rows = [paying(r, "WIN", "4", 350), paying(r, "PLACE", "4", 150)]
    cases = [
        ("no quinella rows", r, "QUINELLA", "4,7", 1_000, "unsettled", NULL),
        ("no trifecta rows", r, "TIERCE", "4,7,1", 100, "unsettled", NULL),
        (
            "no win5 rows",
            f"jra:{DAY.isoformat()}:WIN5",
            "WIN5",
            "1,2,3,4,5",
            100,
            "unsettled",
            NULL,
        ),
        ("the win pool is there", r, "WIN", "7", 100, "lost", 0),
        ("so is place", r, "PLACE", "4", 300, "won", 450),
    ]
    run_table(cases, rows)


def test_a_previous_format_win5_bet_settles_by_its_day():
    # A565ac0's records key a WIN5 bet by a venue race; the reader keys it by its day
    # (docs/specs/bet-log.md, Format version).
    w5, venue_race = f"jra:{DAY.isoformat()}:WIN5", race(11)
    rows = [paying(w5, "WIN5", "3,11,1,7,2", 98_760), paying(venue_race, "WIN", "3", 210)]
    bets = [
        placed(venue_race, "WIN5", "3,11,1,7,2", 200, tag="format 1, won"),
        placed(venue_race, "WIN5", "3,11,1,7,9", 100, tag="format 1, lost"),
        placed(w5, "WIN5", "3,11,1,7,2", 100, tag="format 2, won"),
    ]
    got = settle(bets, rows)
    assert_settled(got["format 1, won"], "won", 197_520)
    assert_settled(got["format 1, lost"], "lost", 0)
    assert_settled(got["format 2, won"], "won", 98_760)


# ---------------------------------------------------------------- played by option


def settled_row(option_id, stake: int, outcome: str, paid: float | None, *, tag: str) -> dict:
    """A row as ``settle_bets`` returns it, built by hand."""
    row = placed(race(11), "WIN", "4", stake, tag=tag, option_id=option_id)
    returned = None if paid is None or outcome not in ("won", "lost") else paid / stake * 100
    row.update(outcome=outcome, paid_yen=paid, return_per_100=returned)
    return row


def by_option(frame: pd.DataFrame) -> dict[str, dict]:
    out = played_by_option(frame)
    assert isinstance(out, pd.DataFrame)
    if "option_id" not in out.columns:
        out = out.rename_axis("option_id").reset_index()
    rows = {row["option_id"]: row for row in out.to_dict("records")}
    assert len(rows) == len(out), "one row per option_id"
    return rows


def assert_option(row: dict, n_bets, staked, paid, per_100, n_refunded, n_pending) -> None:
    assert (row["n_bets"], row["n_refunded"], row["n_pending"]) == (n_bets, n_refunded, n_pending)
    assert row["staked_yen"] == pytest.approx(staked)
    assert row["paid_yen"] == pytest.approx(paid)
    if per_100 is None:
        assert pd.isna(row["return_per_100"]), row
    else:
        assert row["return_per_100"] == pytest.approx(per_100)


def test_played_by_option_sums_settled_bets():
    frame = pd.DataFrame(
        [
            # option A: two won and one lost, one refunded and one unsettled
            settled_row("A", 1_000, "won", 3_500, tag="a1"),
            settled_row("A", 200, "won", 3_480, tag="a2"),
            settled_row("A", 1_000, "lost", 0, tag="a3"),
            settled_row("A", 300, "refunded", 300, tag="a4"),
            settled_row("A", 500, "unsettled", None, tag="a5"),
            # option B: one won, one lost, one refunded
            settled_row("B", 300, "won", 330, tag="b1"),
            settled_row("B", 500, "lost", 0, tag="b2"),
            settled_row("B", 1_000, "refunded", 1_000, tag="b3"),
            # the owner's own picks: a null option_id
            settled_row(None, 100, "won", 44_340, tag="o1"),
            settled_row(None, 200, "lost", 0, tag="o2"),
            settled_row(None, 400, "unsettled", None, tag="o3"),
        ]
    )
    rows = by_option(frame)
    assert set(rows) == {"A", "B", "off-board"}
    # over won and lost only: A staked 1000 + 200 + 1000, paid 3500 + 3480 + 0
    assert_option(rows["A"], 3, 2_200, 6_980, 6_980 / 2_200 * 100, 1, 1)
    assert_option(rows["B"], 2, 800, 330, 41.25, 1, 0)
    assert_option(rows["off-board"], 2, 300, 44_340, 44_340 / 300 * 100, 0, 1)


def test_an_option_with_nothing_won_or_lost_has_no_return():
    frame = pd.DataFrame(
        [
            settled_row("pending", 1_000, "unsettled", None, tag="p1"),
            settled_row("pending", 500, "unsettled", None, tag="p2"),
            settled_row("refunded", 300, "refunded", 300, tag="r1"),
            settled_row("lost", 700, "lost", 0, tag="l1"),
        ]
    )
    rows = by_option(frame)
    assert_option(rows["pending"], 0, 0, 0, None, 0, 2)
    assert_option(rows["refunded"], 0, 0, 0, None, 1, 0)
    assert_option(rows["lost"], 1, 700, 0, 0, 0, 0)

    # records written without an option_id key at all are the owner's own picks
    bare = [
        settled_row(None, 1_000, "won", 2_100, tag="x"),
        settled_row(None, 1_000, "lost", 0, tag="y"),
    ]
    for row in bare:
        del row["option_id"]
    rows = by_option(pd.DataFrame(bare))
    assert set(rows) == {"off-board"}
    assert_option(rows["off-board"], 2, 2_000, 2_100, 105, 0, 0)


# ---------------------------------------------------------------- the store's dividends of a day


def write_store(folder: Path, races: list[tuple], rows: list[tuple]) -> None:
    """A synthetic jra store: ``races`` rows (race_id, race_date, venue, race_no) and
    ``dividends`` rows (race_id, pool, combination, dividend, is_refund), its own race_id form."""
    folder.mkdir(parents=True)
    rid, day, venue, no = zip(*races, strict=True)
    race_table = pa.table(
        {
            "race_id": pa.array(rid, pa.string()),
            "race_date": pa.array(day, pa.date32()),
            "venue": pa.array(venue, pa.string()),
            "race_no": pa.array(no, pa.int64()),
            "race_name": pa.array([f"synthetic {r}" for r in rid], pa.string()),
        }
    )
    pq.write_table(race_table, folder / "races.parquet")
    rid, pool, comb, dividend, refund = zip(*rows, strict=True)
    pq.write_table(
        pa.table(
            {
                "jurisdiction": pa.array(["jra"] * len(rid), pa.string()),
                "race_id": pa.array(rid, pa.string()),
                "pool": pa.array(pool, pa.string()),
                "combination": pa.array(comb, pa.string()),
                "dividend": pa.array(dividend, pa.float64()),
                "is_refund": pa.array(refund, pa.bool_()),
            }
        ),
        folder / "dividends.parquet",
    )


def as_set(frame: pd.DataFrame) -> set[tuple]:
    assert set(DIVIDEND_COLUMNS) <= set(frame.columns)
    return {
        (r["race_id"], r["pool"], r["combination"], float(r["dividend"]), bool(r["is_refund"]))
        for r in frame[DIVIDEND_COLUMNS].to_dict("records")
    }


def test_day_dividends_keys_the_days_store_races_as_the_log_does(D):
    held, before = first_held_day(), DAY - timedelta(days=1)
    write_store(
        D / "processed" / "jra",
        races=[  # the store's race_id is its own key: the log's race_id comes from the race
            ("S-0001", DAY, "TOK", 11),
            ("S-0002", DAY, "KYO", 1),
            ("S-0003", before, "TOK", 11),
            ("S-0004", held, "NAK", 5),
        ],
        rows=[
            ("S-0001", "WIN", "4", 350.0, False),
            ("S-0001", "QUINELLA", "4,7", 1_230.0, False),
            ("S-0002", "WIN", "9", 110.0, False),
            ("S-0002", "WIN", "3", 100.0, True),
            ("S-0003", "WIN", "1", 500.0, False),
            ("S-0004", "WIN", "2", 700.0, False),
        ],
    )
    got = day_dividends(DAY)
    assert isinstance(got, pd.DataFrame)
    expected = {
        (f"jra:{DAY.isoformat()}:TOK:11", "WIN", "4", 350.0, False),
        (f"jra:{DAY.isoformat()}:TOK:11", "QUINELLA", "4,7", 1_230.0, False),
        (f"jra:{DAY.isoformat()}:KYO:01", "WIN", "9", 110.0, False),
        (f"jra:{DAY.isoformat()}:KYO:01", "WIN", "3", 100.0, True),
    }
    assert as_set(got) == expected and len(got) == len(expected)

    got = day_dividends(before)
    assert as_set(got) == {(f"jra:{before.isoformat()}:TOK:11", "WIN", "1", 500.0, False)}

    for quiet in (held, DAY + timedelta(days=1)):  # a holdout day's rows are dropped; no races
        got = day_dividends(quiet)
        assert as_set(got) == set(), quiet

    # what day_dividends returns is what settle_bets takes
    bets = [
        placed(f"jra:{DAY.isoformat()}:KYO:01", "WIN", "9", 1_000, tag="kyo win"),
        placed(f"jra:{DAY.isoformat()}:KYO:01", "WIN", "3", 500, tag="kyo refund"),
        placed(f"jra:{DAY.isoformat()}:TOK:11", "QUINELLA", "7,4", 200, tag="tok quinella"),
    ]
    settled = settle(bets, day_dividends(DAY))
    assert_settled(settled["kyo win"], "won", 1_100)
    assert_settled(settled["kyo refund"], "refunded", 500)
    assert_settled(settled["tok quinella"], "won", 2_460)
