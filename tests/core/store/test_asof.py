"""As-of feature views and leakage tests (ticket #80, C0).

Every table here is a tiny synthetic frame written into an in-memory DuckDB connection, and every
rule is passed explicitly (or read from a tmp-path YAML): nothing depends on the shipped
`countries/japan/config/availability.yaml`, whose values are owner decisions.

Pinned semantics, derived here by brute force rather than from the code under test:

- a 'column' rule makes a row available at `row[column]`; an 'offset' rule at
  `row[anchor] + offset`. A row whose availability time is null is never available.
- `read_asof(con, table, at_utc, rules=...)` returns every row available at or before `at_utc`
  (the whole history, not only the latest). `at_utc` must be timezone-aware; any zone is read as
  the instant it names.
- `asof_join(keys, table, on=..., rules=..., con=..., t_col=...)` returns one row per key row, in
  the keys' order, with the keys' columns unchanged plus the columns of the latest table row whose
  `on` columns match and whose availability time is <= the key's t. A key with no such row is kept,
  with the table's columns null (an ASOF LEFT JOIN).
- 'label_only' tables raise LabelOnlyTable and 'current_value' tables CurrentValueTable from both
  `read_asof` and `asof_join`; `read_label` returns a label-only table whole. A table with no rule
  raises NoAvailabilityRule from all three.
- label columns: a 'column' or 'offset' rule may list `label_columns` (a tuple, default ()), the
  table's result or look-ahead columns (odds_timeseries' `dividend`, `is_refund`,
  `is_final_capture`). `read_asof` and `asof_join` never return them: they are dropped from the
  result, not nulled. A listed column the table does not have is a ValueError at query time. A
  'label_only' or 'current_value' rule with label_columns, or label_columns that list the rule's
  own `column` or `anchor`, is a ValueError. `read_label` still refuses every table that is not
  'label_only', label columns or not.
- an offset rule on a DATE anchor (the local race date) takes `tz` (an IANA zone): the row is
  available at midnight of the date in `tz` plus the offset (an absolute timedelta). A DATE anchor
  without `tz`, `tz` on a timezone-aware anchor, or an unknown zone is a ValueError; `tz` on any
  other kind is a ValueError at construction.
- a column rule may take `fallback`, a second timezone-aware column: the row is available at
  coalesce(column, fallback). A non-null column wins even when the fallback is earlier; both null
  is never available. A fallback of another type, one listed in `label_columns`, or `fallback` on
  any other kind is a ValueError.
- kind 'static': every row is available at any t (no time field; `label_columns` allowed).
  `read_asof`/`asof_join` return it without its label columns; `read_label` refuses it.
- the session time zone of the connection never changes what a view returns.

YAML format of `load_availability` (pinned below):

    tables:
      <table>:
        kind: column | offset | label_only | current_value | static
        column: <name>        # kind column only, required
        fallback: <name>      # kind column only, optional
        anchor: <name>        # kind offset only, required
        offset_min: <number>  # kind offset only, required; minutes, int or float
        tz: <IANA zone>       # kind offset only, optional (required at query time on a DATE anchor)
        label_columns: [<name>, ...]  # kinds column, offset and static only, optional

A key that does not belong to the entry's kind, an unknown key, an unknown kind or a missing
required key is a ValueError, and so is a `label_columns` that is not a list of strings.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pytest

# isort: split
# A block of its own: ruff counts horsetrack.core.store.asof as third-party until it exists.
from horsetrack.core.store.asof import (
    Availability,
    CurrentValueTable,
    LabelOnlyTable,
    NoAvailabilityRule,
    asof_join,
    load_availability,
    read_asof,
    read_label,
)

UTC = timezone.utc
DAY = datetime(2026, 11, 22, tzinfo=UTC)


def at(hh: int, mm: int = 0, ss: int = 0, us: int = 0) -> datetime:
    return DAY.replace(hour=hh, minute=mm, second=ss, microsecond=us)


def utc_series(values) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values), utc=True))


@pytest.fixture
def con():
    connection = duckdb.connect(":memory:")
    yield connection
    connection.close()


def put(con, name: str, frame: pd.DataFrame) -> None:
    """Create a real DuckDB table `name` from a pandas frame (as the store's views would hold)."""
    con.register("_frame", frame)
    con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _frame')
    con.unregister("_frame")


def ids(frame: pd.DataFrame, column: str = "row_id") -> list[int]:
    return sorted(int(v) for v in frame[column])


def instant(value) -> pd.Timestamp | None:
    return None if pd.isna(value) else pd.Timestamp(value).tz_convert("UTC")


# --- broad behaviour: read_asof over many rows and many t, column and offset rules -------------


def _history(rng: np.random.Generator, stamp: str, n: int = 40) -> pd.DataFrame:
    minutes = rng.choice(np.arange(0, 600), size=n, replace=False)
    stamps = [
        DAY + timedelta(hours=6, minutes=int(m), seconds=int(rng.integers(0, 60))) for m in minutes
    ]
    for i in rng.choice(n, size=4, replace=False):
        stamps[int(i)] = None  # never available
    return pd.DataFrame(
        {
            "row_id": np.arange(n, dtype="int64"),
            "race_id": [f"jra:2026-11-22:TOK:{1 + int(r):02d}" for r in rng.integers(0, 6, n)],
            "horse_no": rng.integers(1, 9, n).astype("int64"),
            stamp: utc_series(stamps),
            "value": rng.normal(size=n),
        }
    )


RULE_CASES = [
    pytest.param("jrdb_pre_race", "column", "released_utc", None, id="column-released_utc"),
    pytest.param("odds_timeseries", "column", "fetched_utc", None, id="column-fetched_utc"),
    pytest.param("runners", "offset", "post_utc", 0.0, id="offset-0min"),
    pytest.param("runners", "offset", "post_utc", 7.0, id="offset-7min"),
    pytest.param("sectionals", "offset", "off_utc", 90.0, id="offset-90min-other-anchor"),
    pytest.param("runners", "offset", "post_utc", 0.5, id="offset-30s"),
]


def _rule(table: str, kind: str, stamp: str, offset_min: float | None) -> Availability:
    if kind == "column":
        return Availability(table=table, kind="column", column=stamp)
    return Availability(
        table=table, kind="offset", anchor=stamp, offset=timedelta(minutes=offset_min)
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("table", "kind", "stamp", "offset_min"), RULE_CASES)
def test_read_asof_returns_exactly_the_rows_available_by_t(
    con, seed, table, kind, stamp, offset_min
):
    rng = np.random.default_rng(seed)
    frame = _history(rng, stamp)
    put(con, table, frame)
    rules = {table: _rule(table, kind, stamp, offset_min)}
    shift = timedelta(minutes=offset_min or 0.0)

    available_at = {
        int(r.row_id): (None if pd.isna(r[stamp]) else r[stamp].to_pydatetime() + shift)
        for _, r in frame.iterrows()
    }
    times = [a for a in available_at.values() if a is not None]
    probes = [at(5), at(23, 59)]
    probes += [a for a in times[:10]]  # exactly at a row's availability
    probes += [a - timedelta(microseconds=1) for a in times[:10]]  # just before it
    probes += [DAY + timedelta(hours=6, minutes=int(m)) for m in rng.integers(0, 700, 8)]

    for t in probes:
        expected = sorted(i for i, a in available_at.items() if a is not None and a <= t)
        got = read_asof(con, table, t, rules=rules)
        assert ids(got) == expected, f"t={t.isoformat()}"
        # nothing available after t, and no never-available row, ever reaches the view
        assert set(ids(got)) <= {i for i, a in available_at.items() if a is not None and a <= t}


def test_read_asof_reads_at_utc_as_an_instant_and_refuses_a_naive_time(con):
    put(
        con,
        "runners",
        pd.DataFrame({"row_id": [1, 2], "post_utc": utc_series([at(3, 0), at(3, 40)])}),
    )
    rules = {"runners": Availability("runners", "offset", anchor="post_utc", offset=timedelta(0))}
    jst = timezone(timedelta(hours=9))
    assert ids(
        read_asof(con, "runners", datetime(2026, 11, 22, 12, 30, tzinfo=jst), rules=rules)
    ) == [1]
    assert ids(
        read_asof(con, "runners", datetime(2026, 11, 22, 12, 45, tzinfo=jst), rules=rules)
    ) == [
        1,
        2,
    ]
    with pytest.raises(ValueError):
        read_asof(con, "runners", datetime(2026, 11, 22, 12, 45), rules=rules)


# --- broad behaviour: asof_join against a brute-force point-in-time lookup ----------------------


def _brute_force(table: pd.DataFrame, keys: pd.DataFrame, on, t_col, avail) -> list[int | None]:
    out: list[int | None] = []
    for _, key in keys.iterrows():
        t = key[t_col].to_pydatetime()
        best: tuple[datetime, int] | None = None
        for _, row in table.iterrows():
            if any(row[c] != key[c] for c in on):
                continue
            a = avail(row)
            if a is None or a > t:
                continue
            if best is None or a > best[0]:
                best = (a, int(row.row_id))
        out.append(None if best is None else best[1])
    return out


@pytest.mark.parametrize("seed", [3, 4, 5])
@pytest.mark.parametrize(("table", "kind", "stamp", "offset_min"), RULE_CASES)
@pytest.mark.parametrize("t_col", ["t_utc", "decision_utc"])
def test_asof_join_matches_a_brute_force_point_in_time_lookup(
    con, seed, table, kind, stamp, offset_min, t_col
):
    rng = np.random.default_rng(seed)
    frame = _history(rng, stamp, n=60)
    put(con, table, frame)
    rules = {table: _rule(table, kind, stamp, offset_min)}
    shift = timedelta(minutes=offset_min or 0.0)

    def avail(row):
        return None if pd.isna(row[stamp]) else row[stamp].to_pydatetime() + shift

    n_keys = 30
    races = [
        f"jra:2026-11-22:TOK:{1 + int(r):02d}" for r in rng.integers(0, 7, n_keys)
    ]  # 07 unseen
    horses = rng.integers(1, 10, n_keys).astype("int64")  # horse 9 unseen
    times = [DAY + timedelta(hours=5, minutes=int(m)) for m in rng.integers(0, 720, n_keys)]
    for j in range(12):  # some keys exactly at a row's availability, on that row's keys
        row = frame.iloc[j]
        if avail(row) is not None:
            races[j], horses[j], times[j] = row.race_id, int(row.horse_no), avail(row)
    keys = pd.DataFrame(
        {
            "race_id": races,
            "horse_no": horses,
            t_col: utc_series(times),
            "key_tag": [f"k{j}" for j in range(n_keys)],
        }
    )

    expected = _brute_force(frame, keys, ["race_id", "horse_no"], t_col, avail)
    kwargs = {} if t_col == "t_utc" else {"t_col": t_col}
    got = asof_join(keys, table, on=["race_id", "horse_no"], rules=rules, con=con, **kwargs)
    got = got.reset_index(drop=True)

    assert len(got) == len(keys)
    assert list(got["key_tag"]) == list(keys["key_tag"])  # one row per key, in the keys' order
    assert list(got["race_id"]) == list(keys["race_id"])
    assert [int(h) for h in got["horse_no"]] == [int(h) for h in keys["horse_no"]]
    assert [instant(v) for v in got[t_col]] == [instant(v) for v in keys[t_col]]
    assert any(e is None for e in expected) and any(e is not None for e in expected)
    by_id = frame.set_index("row_id")["value"]
    for j, want in enumerate(expected):
        if want is None:
            assert pd.isna(got["row_id"].iloc[j]), f"key {j} has no row available by its t"
            assert pd.isna(got["value"].iloc[j])
        else:
            assert int(got["row_id"].iloc[j]) == want, f"key {j}"
            assert got["value"].iloc[j] == pytest.approx(by_id[want])


# --- the named tests -----------------------------------------------------------------------------


def _three_releases(con) -> dict[str, Availability]:
    put(
        con,
        "jrdb_pre_race",
        pd.DataFrame(
            {
                "row_id": [9, 10, 11, 12],
                "horse_id": ["H1", "H1", "H1", "H2"],
                "released_utc": utc_series([at(9), at(10), at(11), at(9, 30)]),
                "rating": [61.0, 64.5, 70.25, 55.0],
            }
        ),
    )
    return {"jrdb_pre_race": Availability("jrdb_pre_race", "column", column="released_utc")}


def test_join_takes_latest_row_available_before_t(con):
    rules = _three_releases(con)
    keys = pd.DataFrame(
        {
            "horse_id": ["H1", "H1", "H1", "H2", "H1"],
            "t_utc": utc_series([at(10, 30), at(8, 59), at(11, 0, 1), at(10, 30), at(9, 59, 59)]),
        }
    )
    got = asof_join(keys, "jrdb_pre_race", on=["horse_id"], rules=rules, con=con)
    got = got.reset_index(drop=True)
    assert len(got) == 5
    assert int(got["row_id"][0]) == 10 and got["rating"][0] == 64.5  # the 10:00 row at 10:30
    assert pd.isna(got["row_id"][1]) and pd.isna(got["rating"][1])  # nothing yet at 08:59
    assert int(got["row_id"][2]) == 11
    assert int(got["row_id"][3]) == 12  # the on-key separates horses
    assert int(got["row_id"][4]) == 9


def test_a_row_available_exactly_at_t_is_included(con):
    rules = _three_releases(con)
    keys = pd.DataFrame(
        {
            "horse_id": ["H1", "H1"],
            "t_utc": utc_series([at(10), at(10) - timedelta(microseconds=1)]),
        }
    )
    got = asof_join(keys, "jrdb_pre_race", on=["horse_id"], rules=rules, con=con)
    got = got.reset_index(drop=True)
    assert [int(v) for v in got["row_id"]] == [10, 9]
    assert ids(read_asof(con, "jrdb_pre_race", at(10), rules=rules)) == [9, 10, 12]
    assert ids(
        read_asof(con, "jrdb_pre_race", at(10) - timedelta(microseconds=1), rules=rules)
    ) == [
        9,
        12,
    ]


def test_final_odds_never_enter_a_view_at_t(con):
    final = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "race_id": ["jra:2026-11-22:TOK:01"] * 3,
            "horse_no": [1, 2, 3],
            "final_odds": [2.4, 5.1, 13.7],
            "fetched_utc": utc_series([at(3, 1)] * 3),  # a usable-looking stamp: still refused
        }
    )
    put(con, "final_odds", final)
    rules = {"final_odds": Availability("final_odds", "label_only")}
    keys = pd.DataFrame(
        {
            "race_id": ["jra:2026-11-22:TOK:01"],
            "horse_no": [1],
            "t_utc": utc_series([at(23)]),
        }
    )
    with pytest.raises(LabelOnlyTable):
        asof_join(keys, "final_odds", on=["race_id", "horse_no"], rules=rules, con=con)
    with pytest.raises(LabelOnlyTable):
        read_asof(con, "final_odds", at(23), rules=rules)
    assert issubclass(LabelOnlyTable, ValueError)

    labels = read_label(con, "final_odds", rules=rules)
    assert ids(labels) == [1, 2, 3]
    odds = dict(zip((int(i) for i in labels["row_id"]), labels["final_odds"], strict=True))
    assert odds == {1: 2.4, 2: 5.1, 3: 13.7}


def test_same_day_results_follow_the_availability_rule(con):
    results = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "race_id": ["jra:2026-11-22:TOK:01"] * 2 + ["jra:2026-11-22:TOK:02"] * 2,
            "horse_id": ["H1", "H2", "H1", "H3"],
            "finish_pos": [1, 2, 4, 1],
            "post_utc": utc_series([at(3, 0)] * 2 + [at(3, 35)] * 2),
        }
    )
    put(con, "runners", results)
    rules = {
        "runners": Availability(
            "runners", "offset", anchor="post_utc", offset=timedelta(minutes=20)
        )
    }
    assert ids(read_asof(con, "runners", at(3, 19, 59), rules=rules)) == []
    assert ids(read_asof(con, "runners", at(3, 20), rules=rules)) == [1, 2]
    assert ids(read_asof(con, "runners", at(3, 54, 59), rules=rules)) == [1, 2]
    assert ids(read_asof(con, "runners", at(3, 55), rules=rules)) == [1, 2, 3, 4]

    # a horse's last result as a feature at t
    keys = pd.DataFrame(
        {
            "horse_id": ["H1", "H1", "H1", "H1"],
            "t_utc": utc_series([at(3, 19, 59), at(3, 20), at(3, 54, 59), at(3, 55)]),
        }
    )
    got = asof_join(keys, "runners", on=["horse_id"], rules=rules, con=con).reset_index(drop=True)
    assert pd.isna(got["finish_pos"][0])
    assert [int(v) for v in got["finish_pos"][1:]] == [1, 1, 4]


def test_current_value_tables_are_refused(con):
    put(
        con,
        "horses",
        pd.DataFrame(
            {
                "horse_id": ["H1", "H2"],
                "career_wins": [7, 0],
                "fetched_utc": utc_series([at(1), at(1)]),
            }
        ),
    )
    rules = {"horses": Availability("horses", "current_value")}
    keys = pd.DataFrame({"horse_id": ["H1"], "t_utc": utc_series([at(23)])})
    with pytest.raises(CurrentValueTable):
        read_asof(con, "horses", at(23), rules=rules)
    with pytest.raises(CurrentValueTable):
        asof_join(keys, "horses", on=["horse_id"], rules=rules, con=con)
    assert issubclass(CurrentValueTable, ValueError)
    assert not issubclass(CurrentValueTable, LabelOnlyTable)


def test_a_table_without_an_availability_rule_is_refused(con):
    put(
        con,
        "incidents",
        pd.DataFrame({"horse_id": ["H1"], "noted_utc": utc_series([at(2)])}),
    )
    other = {"runners": Availability("runners", "column", column="post_utc")}
    keys = pd.DataFrame({"horse_id": ["H1"], "t_utc": utc_series([at(23)])})
    for rules in (other, {}):
        with pytest.raises(NoAvailabilityRule):
            read_asof(con, "incidents", at(23), rules=rules)
        with pytest.raises(NoAvailabilityRule):
            asof_join(keys, "incidents", on=["horse_id"], rules=rules, con=con)
        with pytest.raises(NoAvailabilityRule):
            read_label(con, "incidents", rules=rules)
    assert issubclass(NoAvailabilityRule, KeyError)


def test_board_at_t_is_the_latest_capture_fetched_by_t(con):
    race, other = "jra:2026-11-22:TOK:05", "jra:2026-11-22:TOK:06"
    rows = []

    def capture(race_id, snap, fetched, prices):
        for (pool, combination), odds in prices.items():
            rows.append(
                {
                    "race_id": race_id,
                    "pool": pool,
                    "combination": combination,
                    "snapshot_utc": snap,
                    "fetched_utc": fetched,
                    "odds": odds,
                }
            )

    capture(race, at(9, 0), at(9, 0, 5), {("WIN", "1"): 3.0, ("WIN", "2"): 4.0, ("WIN", "3"): 9.0,
                                          ("PLACE", "1"): 1.3})  # fmt: skip
    # combination 3 missing from this capture: its latest row stays the 09:00 one
    capture(race, at(9, 10), at(9, 10, 4), {("WIN", "1"): 2.8, ("WIN", "2"): 4.4,
                                            ("PLACE", "1"): 1.2})  # fmt: skip
    # snapshot before t, fetched after it: not on the board at t
    capture(race, at(9, 20), at(9, 20, 30), {("WIN", "1"): 2.1, ("WIN", "2"): 5.0,
                                             ("WIN", "3"): 7.5, ("PLACE", "1"): 1.1})  # fmt: skip
    # another race, fetched by t, must not leak across the on-keys
    capture(other, at(9, 15), at(9, 15, 2), {("WIN", "1"): 99.0, ("WIN", "2"): 98.0})
    frame = pd.DataFrame(rows)
    frame["snapshot_utc"] = utc_series(frame["snapshot_utc"])
    frame["fetched_utc"] = utc_series(frame["fetched_utc"])
    put(con, "odds_timeseries", frame)
    rules = {"odds_timeseries": Availability("odds_timeseries", "column", column="fetched_utc")}

    board = [("WIN", "1"), ("WIN", "2"), ("WIN", "3"), ("PLACE", "1")]

    def board_at(t):
        keys = pd.DataFrame(
            {
                "race_id": [race] * len(board),
                "pool": [p for p, _ in board],
                "combination": [c for _, c in board],
                "t_utc": utc_series([t] * len(board)),
            }
        )
        got = asof_join(
            keys, "odds_timeseries", on=["race_id", "pool", "combination"], rules=rules, con=con
        ).reset_index(drop=True)
        return [float(v) for v in got["odds"]], [instant(v) for v in got["snapshot_utc"]]

    odds, snaps = board_at(at(9, 20, 10))
    assert odds == [2.8, 4.4, 9.0, 1.2]
    assert snaps == [instant(at(9, 10))] * 2 + [instant(at(9, 0)), instant(at(9, 10))]

    odds, snaps = board_at(at(9, 20, 30))  # fetched exactly at t
    assert odds == [2.1, 5.0, 7.5, 1.1]
    assert snaps == [instant(at(9, 20))] * 4

    odds, _ = board_at(at(9, 10, 3))  # the 09:10 capture is a second short
    assert odds == [3.0, 4.0, 9.0, 1.3]

    view = read_asof(con, "odds_timeseries", at(9, 20, 10), rules=rules)
    assert len(view) == 4 + 3 + 2  # the whole history fetched by t, not only the latest
    assert all(instant(v) <= instant(at(9, 20, 10)) for v in view["fetched_utc"])


# --- label columns: result and look-ahead columns never reach a view ----------------------------

ODDS_LABELS = ("dividend", "is_refund", "is_final_capture")


def test_board_at_t_carries_no_payout_or_final_capture_columns(con):
    """odds_timeseries (logger/compact.py SCHEMA) puts the result on every capture: `dividend`
    and `is_refund`, and `is_final_capture` worked out from the whole day's log. Listed as label
    columns, none of them is on the board at t, while the prices still are."""
    race = "jra:2026-11-22:TOK:05"
    rows = []
    payout = {"1": 280.0, "2": 0.0, "3": 0.0}  # the result: combination 1 won

    def capture(snap, fetched, prices, final):
        for combination, odds in prices.items():
            rows.append(
                {
                    "race_id": race,
                    "pool": "WIN",
                    "combination": combination,
                    "snapshot_utc": snap,
                    "fetched_utc": fetched,
                    "odds": odds,
                    "investment": 1000.0 * odds,
                    "is_final_capture": final,
                    "dividend": payout[combination],
                    "is_refund": combination == "3",
                }
            )

    capture(at(9, 0), at(9, 0, 5), {"1": 3.0, "2": 4.0, "3": 9.0}, False)
    capture(at(9, 10), at(9, 10, 4), {"1": 2.8, "2": 4.4, "3": 8.0}, True)
    frame = pd.DataFrame(rows)
    frame["snapshot_utc"] = utc_series(frame["snapshot_utc"])
    frame["fetched_utc"] = utc_series(frame["fetched_utc"])
    put(con, "odds_timeseries", frame)
    rules = {
        "odds_timeseries": Availability(
            "odds_timeseries", "column", column="fetched_utc", label_columns=ODDS_LABELS
        )
    }
    assert rules["odds_timeseries"].label_columns == ODDS_LABELS

    keys = pd.DataFrame(
        {
            "race_id": [race] * 3,
            "pool": ["WIN"] * 3,
            "combination": ["1", "2", "3"],
            "t_utc": utc_series([at(9, 5), at(9, 10, 4), at(8)]),
        }
    )
    got = asof_join(
        keys, "odds_timeseries", on=["race_id", "pool", "combination"], rules=rules, con=con
    ).reset_index(drop=True)
    for column in ODDS_LABELS:
        assert column not in got.columns, f"{column} is on the board at t"
    assert {"odds", "investment", "snapshot_utc", "fetched_utc"} <= set(got.columns)
    assert list(got["combination"]) == ["1", "2", "3"]
    assert float(got["odds"][0]) == 3.0  # the 09:00 capture at 09:05
    assert float(got["odds"][1]) == 4.4  # the 09:10 capture, fetched exactly at t
    assert pd.isna(got["odds"][2])  # nothing fetched by 08:00

    view = read_asof(con, "odds_timeseries", at(9, 5), rules=rules)
    assert len(view) == 3
    assert set(view.columns) == set(frame.columns) - set(ODDS_LABELS)
    assert sorted(float(v) for v in view["odds"]) == [3.0, 4.0, 9.0]


LABEL_SETS = [
    pytest.param(("payout",), id="one-label"),
    pytest.param(("payout", "settled", "is_last"), id="three-labels"),
]


@pytest.mark.parametrize("labels", LABEL_SETS)
@pytest.mark.parametrize(("table", "kind", "stamp", "offset_min"), RULE_CASES)
def test_label_columns_never_reach_a_view_and_the_rest_is_point_in_time(
    con, table, kind, stamp, offset_min, labels
):
    rng = np.random.default_rng(11)
    frame = _history(rng, stamp, n=50)
    n = len(frame)
    frame["payout"] = rng.uniform(100, 5000, n)  # what the result paid: known only after it
    frame["settled"] = rng.integers(0, 2, n).astype(bool)
    frame["is_last"] = rng.integers(0, 2, n).astype(bool)
    put(con, table, frame)
    base = _rule(table, kind, stamp, offset_min)
    rule = Availability(
        table, kind, column=base.column, anchor=base.anchor, offset=base.offset,
        label_columns=labels,
    )  # fmt: skip
    rules = {table: rule}
    shift = timedelta(minutes=offset_min or 0.0)

    def avail(row):
        return None if pd.isna(row[stamp]) else row[stamp].to_pydatetime() + shift

    kept = set(frame.columns) - set(labels)
    for t in [at(5), at(9, 30), at(12), at(23, 59)]:
        view = read_asof(con, table, t, rules=rules)
        assert set(view.columns) == kept, f"t={t.isoformat()}"
        expected = sorted(
            int(r.row_id) for _, r in frame.iterrows() if avail(r) is not None and avail(r) <= t
        )
        assert ids(view) == expected, f"t={t.isoformat()}"

    n_keys = 25
    keys = pd.DataFrame(
        {
            "race_id": [f"jra:2026-11-22:TOK:{1 + int(r):02d}" for r in rng.integers(0, 7, n_keys)],
            "horse_no": rng.integers(1, 10, n_keys).astype("int64"),
            "t_utc": utc_series(
                DAY + timedelta(hours=5, minutes=int(m)) for m in rng.integers(0, 720, n_keys)
            ),
        }
    )
    on = ["race_id", "horse_no"]
    expected = _brute_force(frame, keys, on, "t_utc", avail)
    got = asof_join(keys, table, on=on, rules=rules, con=con).reset_index(drop=True)
    assert set(got.columns) == set(keys.columns) | (kept - set(on))
    assert any(e is not None for e in expected)
    by_id = frame.set_index("row_id")["value"]
    for j, want in enumerate(expected):
        if want is None:
            assert pd.isna(got["row_id"].iloc[j]), f"key {j}"
        else:
            assert int(got["row_id"].iloc[j]) == want, f"key {j}"
            assert got["value"].iloc[j] == pytest.approx(by_id[want])


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(("dividnd",), id="typo"),
        pytest.param(("dividend", "payout_yen"), id="one-real-one-missing"),
    ],
)
def test_a_label_column_the_table_lacks_is_an_error(con, labels):
    put(
        con,
        "odds_timeseries",
        pd.DataFrame(
            {
                "race_id": ["jra:2026-11-22:TOK:01"],
                "fetched_utc": utc_series([at(9)]),
                "odds": [2.5],
                "dividend": [250.0],
            }
        ),
    )
    rules = {
        "odds_timeseries": Availability(
            "odds_timeseries", "column", column="fetched_utc", label_columns=labels
        )
    }
    keys = pd.DataFrame({"race_id": ["jra:2026-11-22:TOK:01"], "t_utc": utc_series([at(10)])})
    with pytest.raises(ValueError):
        read_asof(con, "odds_timeseries", at(10), rules=rules)
    with pytest.raises(ValueError):
        asof_join(keys, "odds_timeseries", on=["race_id"], rules=rules, con=con)


def test_read_label_still_refuses_a_table_with_label_columns(con):
    put(
        con,
        "odds_timeseries",
        pd.DataFrame({"fetched_utc": utc_series([at(9)]), "dividend": [250.0]}),
    )
    rules = {
        "odds_timeseries": Availability(
            "odds_timeseries", "column", column="fetched_utc", label_columns=("dividend",)
        )
    }
    with pytest.raises(ValueError):
        read_label(con, "odds_timeseries", rules=rules)


def test_availability_label_columns_default_and_refusals():
    assert Availability("runners", "column", column="post_utc").label_columns == ()
    off = Availability(
        "runners", "offset", anchor="post_utc", offset=timedelta(minutes=20),
        label_columns=("finish_pos", "finish_time_s"),
    )  # fmt: skip
    assert off.label_columns == ("finish_pos", "finish_time_s")
    bad = [
        {"kind": "label_only", "label_columns": ("dividend",)},
        {"kind": "current_value", "label_columns": ("career_wins",)},
        {"kind": "column", "column": "fetched_utc", "label_columns": ("fetched_utc",)},
        {"kind": "column", "column": "fetched_utc", "label_columns": ("odds", "fetched_utc")},
        {
            "kind": "offset",
            "anchor": "post_utc",
            "offset": timedelta(minutes=20),
            "label_columns": ("post_utc",),
        },
    ]
    for fields in bad:
        with pytest.raises(ValueError):
            Availability("t", **fields)


# --- existing refusals ---------------------------------------------------------------------------


def test_a_naive_availability_column_is_refused(con):
    put(
        con,
        "odds_timeseries",
        pd.DataFrame(
            {
                "race_id": ["jra:2026-11-22:TOK:01"],
                "fetched_utc": pd.to_datetime(["2026-11-22 09:00:00"]),  # naive TIMESTAMP
                "odds": [2.5],
            }
        ),
    )
    rules = {"odds_timeseries": Availability("odds_timeseries", "column", column="fetched_utc")}
    keys = pd.DataFrame({"race_id": ["jra:2026-11-22:TOK:01"], "t_utc": utc_series([at(10)])})
    with pytest.raises(ValueError):
        read_asof(con, "odds_timeseries", at(10), rules=rules)
    with pytest.raises(ValueError):
        asof_join(keys, "odds_timeseries", on=["race_id"], rules=rules, con=con)


def test_asof_join_refuses_naive_key_times_and_clashing_columns(con):
    rules = _three_releases(con)
    naive = pd.DataFrame({"horse_id": ["H1"], "t_utc": pd.to_datetime(["2026-11-22 10:30:00"])})
    with pytest.raises(ValueError):
        asof_join(naive, "jrdb_pre_race", on=["horse_id"], rules=rules, con=con)
    clash = pd.DataFrame({"horse_id": ["H1"], "rating": [1.0], "t_utc": utc_series([at(10, 30)])})
    with pytest.raises(ValueError):
        asof_join(clash, "jrdb_pre_race", on=["horse_id"], rules=rules, con=con)


# --- load_availability ---------------------------------------------------------------------------

GOOD_YAML = """\
tables:
  odds_timeseries:
    kind: column
    column: fetched_utc
  runners:
    kind: offset
    anchor: post_utc
    offset_min: 20
  sectionals:
    kind: offset
    anchor: off_utc
    offset_min: 1.5
  dividends:
    kind: label_only
  horses:
    kind: current_value
"""


def test_load_availability_parses_each_kind(tmp_path: Path):
    path = tmp_path / "availability.yaml"
    path.write_text(GOOD_YAML, encoding="utf-8")
    rules = load_availability("japan", path=path)
    assert set(rules) == {"odds_timeseries", "runners", "sectionals", "dividends", "horses"}
    for name, rule in rules.items():
        assert isinstance(rule, Availability)
        assert rule.table == name

    ts = rules["odds_timeseries"]
    assert (ts.kind, ts.column, ts.anchor, ts.offset) == ("column", "fetched_utc", None, None)
    run = rules["runners"]
    assert (run.kind, run.column, run.anchor, run.offset) == (
        "offset",
        None,
        "post_utc",
        timedelta(minutes=20),
    )
    sec = rules["sectionals"]
    assert (sec.kind, sec.anchor, sec.offset) == ("offset", "off_utc", timedelta(seconds=90))
    assert (rules["dividends"].kind, rules["dividends"].column) == ("label_only", None)
    assert (rules["horses"].kind, rules["horses"].offset) == ("current_value", None)


def test_rules_loaded_from_yaml_drive_the_views(tmp_path: Path, con):
    path = tmp_path / "availability.yaml"
    path.write_text(GOOD_YAML, encoding="utf-8")
    rules = load_availability("japan", path=path)
    put(con, "runners", pd.DataFrame({"row_id": [1], "post_utc": utc_series([at(3)])}))
    put(con, "dividends", pd.DataFrame({"row_id": [5, 6], "payout": [240, 1310]}))
    assert ids(read_asof(con, "runners", at(3, 19, 59), rules=rules)) == []
    assert ids(read_asof(con, "runners", at(3, 20), rules=rules)) == [1]
    with pytest.raises(LabelOnlyTable):
        read_asof(con, "dividends", at(23), rules=rules)
    assert ids(read_label(con, "dividends", rules=rules)) == [5, 6]


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("kind: bogus", id="unknown-kind"),
        pytest.param("column: fetched_utc", id="no-kind"),
        pytest.param("kind: column", id="column-without-column"),
        pytest.param("kind: offset\n    offset_min: 20", id="offset-without-anchor"),
        pytest.param("kind: offset\n    anchor: post_utc", id="offset-without-offset_min"),
        pytest.param(
            "kind: offset\n    anchor: post_utc\n    offset_min: twenty",
            id="offset_min-not-a-number",
        ),
        pytest.param("kind: label_only\n    column: fetched_utc", id="key-foreign-to-kind"),
        pytest.param("kind: column\n    column: fetched_utc\n    colum: x", id="unknown-key"),
        pytest.param(
            "kind: column\n    column: fetched_utc\n    anchor: post_utc", id="column-with-anchor"
        ),
    ],
)
def test_load_availability_refuses_a_malformed_entry(tmp_path: Path, entry: str):
    _refuse_entry(tmp_path, entry)


def test_load_availability_reads_label_columns(tmp_path: Path, con):
    path = tmp_path / "availability.yaml"
    path.write_text(
        "tables:\n"
        "  odds_timeseries:\n"
        "    kind: column\n"
        "    column: fetched_utc\n"
        "    label_columns: [dividend, is_refund, is_final_capture]\n"
        "  runners:\n"
        "    kind: offset\n"
        "    anchor: post_utc\n"
        "    offset_min: 0\n"
        "    label_columns:\n"
        "      - finish_pos\n"
        "  horse_pedigree:\n"
        "    kind: column\n"
        "    column: scraped_utc\n",
        encoding="utf-8",
    )
    rules = load_availability("japan", path=path)
    assert rules["odds_timeseries"].label_columns == ODDS_LABELS
    assert isinstance(rules["odds_timeseries"].label_columns, tuple)
    assert rules["runners"].label_columns == ("finish_pos",)
    assert rules["horse_pedigree"].label_columns == ()

    put(
        con,
        "runners",
        pd.DataFrame(
            {"row_id": [1], "post_utc": utc_series([at(3)]), "draw": [4], "finish_pos": [1]}
        ),
    )
    view = read_asof(con, "runners", at(3), rules=rules)
    assert ids(view) == [1] and set(view.columns) == {"row_id", "post_utc", "draw"}


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(
            "kind: column\n    column: post_utc\n    label_columns: finish_pos", id="a-string"
        ),
        pytest.param("kind: column\n    column: post_utc\n    label_columns:", id="null"),
        pytest.param(
            "kind: column\n    column: post_utc\n    label_columns: [1, 2]", id="not-strings"
        ),
        pytest.param(
            "kind: column\n    column: post_utc\n    label_columns: [finish_pos, [x]]",
            id="nested-list",
        ),
        pytest.param(
            "kind: column\n    column: post_utc\n    label_columns: {finish_pos: 1}",
            id="a-mapping",
        ),
        pytest.param("kind: label_only\n    label_columns: [finish_pos]", id="on-label_only"),
        pytest.param("kind: current_value\n    label_columns: [finish_pos]", id="on-current_value"),
        pytest.param(
            "kind: column\n    column: post_utc\n    label_columns: [post_utc]",
            id="lists-the-column",
        ),
        pytest.param(
            "kind: offset\n    anchor: post_utc\n    offset_min: 20\n"
            "    label_columns: [finish_pos, post_utc]",
            id="lists-the-anchor",
        ),
    ],
)
def test_load_availability_refuses_malformed_label_columns(tmp_path: Path, entry: str):
    _refuse_entry(tmp_path, entry)


def _refuse_entry(tmp_path: Path, entry: str) -> None:
    path = tmp_path / "availability.yaml"
    path.write_text(
        "tables:\n  dividends:\n    kind: label_only\n  runners:\n    " + entry + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_availability("japan", path=path)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("runners:\n  kind: label_only\n", id="no-tables-key"),
        pytest.param("tables:\n  - runners\n", id="tables-not-a-mapping"),
        pytest.param("tables:\n  runners: label_only\n", id="entry-not-a-mapping"),
    ],
)
def test_load_availability_refuses_a_malformed_file(tmp_path: Path, text: str):
    path = tmp_path / "availability.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_availability("japan", path=path)


def test_load_availability_of_a_missing_file_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_availability("japan", path=tmp_path / "nope.yaml")


# --- offset rules on a DATE anchor: midnight of the local date in `tz`, plus the offset ----------


def put_typed(con, name: str, frame: pd.DataFrame, casts: dict[str, str]) -> None:
    """Like `put`, with the named columns cast to a DuckDB type (DATE, VARCHAR, TIMESTAMP...)."""
    con.register("_frame", frame)
    replaced = ", ".join(f'CAST("{c}" AS {kind}) AS "{c}"' for c, kind in casts.items())
    con.execute(f'CREATE TABLE "{name}" AS SELECT * REPLACE ({replaced}) FROM _frame')
    con.unregister("_frame")


def local_midnight(day: date, tz: str) -> datetime:
    """00:00 of `day` on the wall clock of `tz`, as a UTC instant (derived with zoneinfo)."""
    return datetime.combine(day, time(0), tzinfo=ZoneInfo(tz)).astimezone(UTC)


def set_session_zone(con, zone: str | None) -> None:
    if zone is not None:
        con.execute(f"SET TimeZone = '{zone}'")


US = timedelta(microseconds=1)
RACE_DATES = [
    date(2026, 1, 15),  # London on GMT
    date(2026, 7, 4),  # London on BST: its midnight is 23:00 UTC of the day before
    date(2026, 11, 27),
    date(2026, 11, 28),
    date(2026, 11, 29),
    date(2026, 12, 31),
    date(2027, 1, 1),
]
SESSION_ZONES = [
    pytest.param(None, id="session-default"),
    pytest.param("America/Los_Angeles", id="session-LA"),
    pytest.param("Asia/Tokyo", id="session-JST"),
]
DATE_CASES = [
    pytest.param("Asia/Tokyo", 0.0, id="jst-0"),
    pytest.param("Asia/Tokyo", 1440.0, id="jst-next-day"),
    pytest.param("Asia/Tokyo", -90.0, id="jst-minus-90min"),
    pytest.param("UTC", 30.0, id="utc-30min"),
    pytest.param("Europe/London", 0.0, id="london-gmt-and-bst"),
    pytest.param("America/New_York", 600.5, id="new-york-10h-30s"),
]


def _dated() -> pd.DataFrame:
    """Two rows per race date on different horses (no ties within a horse), and two null dates."""
    dates = RACE_DATES * 2 + [None, None]
    return pd.DataFrame(
        {
            "row_id": np.arange(len(dates), dtype="int64"),
            "horse_id": [f"H{i % 3 + 1}" for i in range(len(dates))],
            "race_date": dates,
            "value": [1.25 * i for i in range(len(dates))],
        }
    )


@pytest.mark.parametrize("session", SESSION_ZONES)
@pytest.mark.parametrize(("tz", "offset_min"), DATE_CASES)
def test_an_offset_on_a_date_anchor_is_local_midnight_plus_the_offset(con, tz, offset_min, session):
    set_session_zone(con, session)
    frame = _dated()
    put_typed(con, "runners", frame, {"race_date": "DATE"})
    shift = timedelta(minutes=offset_min)
    rules = {"runners": Availability("runners", "offset", anchor="race_date", offset=shift, tz=tz)}

    def avail(row):
        return None if pd.isna(row["race_date"]) else local_midnight(row["race_date"], tz) + shift

    available_at = {int(r["row_id"]): avail(r) for _, r in frame.iterrows()}
    times = sorted({a for a in available_at.values() if a is not None})
    probes = [datetime(2025, 6, 1, tzinfo=UTC), datetime(2028, 1, 1, tzinfo=UTC)]
    probes += times + [a - US for a in times]
    for t in probes:
        expected = sorted(i for i, a in available_at.items() if a is not None and a <= t)
        assert ids(read_asof(con, "runners", t, rules=rules)) == expected, f"t={t.isoformat()}"

    horses = ["H1", "H2", "H3", "H9"]  # H9 has no rows
    keys = pd.DataFrame(
        {
            "horse_id": [h for h in horses for _ in probes],
            "t_utc": utc_series([t for _ in horses for t in probes]),
        }
    )
    expected = _brute_force(frame, keys, ["horse_id"], "t_utc", avail)
    got = asof_join(keys, "runners", on=["horse_id"], rules=rules, con=con).reset_index(drop=True)
    assert len(got) == len(keys)
    assert any(e is None for e in expected) and any(e is not None for e in expected)
    by_id = frame.set_index("row_id")["value"]
    for j, want in enumerate(expected):
        if want is None:
            assert pd.isna(got["row_id"].iloc[j]), f"key {j}"
        else:
            assert int(got["row_id"].iloc[j]) == want, f"key {j}"
            assert got["value"].iloc[j] == pytest.approx(by_id[want])


def test_a_race_date_in_jst_opens_at_midnight_jst(con):
    put_typed(
        con,
        "runners",
        pd.DataFrame(
            {
                "row_id": [1, 2],
                "horse_id": ["H1", "H1"],
                "race_date": [date(2026, 11, 28), date(2026, 11, 29)],
                "draw": [3, 7],
            }
        ),
        {"race_date": "DATE"},
    )

    def rule(minutes: int) -> dict[str, Availability]:
        return {
            "runners": Availability(
                "runners", "offset", anchor="race_date", offset=timedelta(minutes=minutes),
                tz="Asia/Tokyo",
            )
        }  # fmt: skip

    same_day, next_day = rule(0), rule(1440)
    opens = datetime(2026, 11, 27, 15, tzinfo=UTC)  # 2026-11-28 00:00 JST
    later = datetime(2026, 11, 28, 15, tzinfo=UTC)  # 2026-11-29 00:00 JST
    assert ids(read_asof(con, "runners", opens - US, rules=same_day)) == []
    assert ids(read_asof(con, "runners", opens, rules=same_day)) == [1]
    assert ids(read_asof(con, "runners", later - US, rules=same_day)) == [1]
    assert ids(read_asof(con, "runners", later, rules=same_day)) == [1, 2]
    assert ids(read_asof(con, "runners", later - US, rules=next_day)) == []
    assert ids(read_asof(con, "runners", later, rules=next_day)) == [1]

    keys = pd.DataFrame(
        {"horse_id": ["H1"] * 4, "t_utc": utc_series([opens - US, opens, later - US, later])}
    )
    got = asof_join(keys, "runners", on=["horse_id"], rules=same_day, con=con)
    assert [None if pd.isna(v) else int(v) for v in got["draw"]] == [None, 3, 3, 7]
    got = asof_join(keys, "runners", on=["horse_id"], rules=next_day, con=con)
    assert [None if pd.isna(v) else int(v) for v in got["draw"]] == [None, None, None, 3]


def test_tz_is_required_on_a_date_anchor_and_refused_elsewhere(con):
    put_typed(
        con,
        "runners",
        pd.DataFrame(
            {
                "row_id": [1],
                "horse_id": ["H1"],
                "race_date": [date(2026, 11, 28)],
                "post_utc": utc_series([datetime(2026, 11, 28, 5, 40, tzinfo=UTC)]),
                "post_text": ["2026-11-28"],
            }
        ),
        {"race_date": "DATE", "post_text": "VARCHAR"},
    )
    keys = pd.DataFrame({"horse_id": ["H1"], "t_utc": utc_series([at(23)])})
    t = datetime(2026, 11, 28, 12, tzinfo=UTC)

    def offset(anchor: str, **extra) -> Availability:
        return Availability("runners", "offset", anchor=anchor, offset=timedelta(0), **extra)

    assert offset("post_utc").tz is None
    good = offset("race_date", tz="Asia/Tokyo")
    assert good.tz == "Asia/Tokyo"
    assert ids(read_asof(con, "runners", t, rules={"runners": good})) == [1]

    for rule in (
        offset("race_date"),  # a DATE anchor without a zone
        offset("post_utc", tz="Asia/Tokyo"),  # a zone on an anchor that already is an instant
        offset("post_text", tz="Asia/Tokyo"),  # a string anchor stays refused
    ):
        rules = {"runners": rule}
        with pytest.raises(ValueError):
            read_asof(con, "runners", t, rules=rules)
        with pytest.raises(ValueError):
            asof_join(keys, "runners", on=["horse_id"], rules=rules, con=con)

    for fields in (
        {"kind": "column", "column": "post_utc", "tz": "Asia/Tokyo"},
        {"kind": "label_only", "tz": "Asia/Tokyo"},
        {"kind": "current_value", "tz": "Asia/Tokyo"},
        {"kind": "static", "tz": "Asia/Tokyo"},
    ):
        with pytest.raises(ValueError):
            Availability("runners", **fields)

    for zone in ("Asia/Tokio", "JST+9", "", 9):
        with pytest.raises(ValueError):  # at construction or at query time
            rules = {"runners": offset("race_date", tz=zone)}
            read_asof(con, "runners", t, rules=rules)


def test_load_availability_reads_tz_and_refuses_it_off_an_offset_rule(tmp_path: Path, con):
    path = tmp_path / "availability.yaml"
    path.write_text(
        "tables:\n"
        "  runners:\n"
        "    kind: offset\n"
        "    anchor: race_date\n"
        "    offset_min: 1440\n"
        "    tz: Asia/Tokyo\n"
        "    label_columns: [finish_pos]\n"
        "  sectionals:\n"
        "    kind: offset\n"
        "    anchor: off_utc\n"
        "    offset_min: 30\n",
        encoding="utf-8",
    )
    rules = load_availability("japan", path=path)
    run = rules["runners"]
    assert (run.kind, run.anchor, run.offset, run.tz) == (
        "offset", "race_date", timedelta(days=1), "Asia/Tokyo",
    )  # fmt: skip
    assert rules["sectionals"].tz is None

    put_typed(
        con,
        "runners",
        pd.DataFrame(
            {"row_id": [1], "race_date": [date(2026, 11, 28)], "draw": [4], "finish_pos": [1]}
        ),
        {"race_date": "DATE"},
    )
    opens = datetime(2026, 11, 28, 15, tzinfo=UTC)  # 2026-11-29 00:00 JST
    assert ids(read_asof(con, "runners", opens - US, rules=rules)) == []
    view = read_asof(con, "runners", opens, rules=rules)
    assert ids(view) == [1] and set(view.columns) == {"row_id", "race_date", "draw"}

    for entry in (
        "kind: column\n    column: fetched_utc\n    tz: Asia/Tokyo",
        "kind: label_only\n    tz: Asia/Tokyo",
        "kind: current_value\n    tz: Asia/Tokyo",
        "kind: static\n    tz: Asia/Tokyo",
        "kind: offset\n    anchor: race_date\n    offset_min: 0\n    tz: 9",
        "kind: offset\n    anchor: race_date\n    offset_min: 0\n    tz: [Asia/Tokyo]",
        "kind: offset\n    anchor: race_date\n    offset_min: 0\n    tz:",
    ):
        _refuse_entry(tmp_path, entry)

    path.write_text(
        "tables:\n  runners:\n    kind: offset\n    anchor: race_date\n    offset_min: 0\n"
        "    tz: Asia/Tokio\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):  # at load or at query time
        read_asof(con, "runners", opens, rules=load_availability("japan", path=path))


# --- column rules with a fallback column: available at coalesce(column, fallback) ---------------


def _captures(rng: np.random.Generator) -> pd.DataFrame:
    """Captures of every shape the fallback must handle, on four (race, combination) keys."""
    rows = []
    starts = rng.choice(np.arange(0, 900, 3), size=36, replace=False)
    for i, m in enumerate(starts):
        base = DAY + timedelta(hours=1, minutes=int(m), seconds=int(rng.integers(0, 60)))
        shape = i % 6
        fetched, snapshot = {
            0: (base + timedelta(minutes=20), base),  # fetched well after the snapshot
            1: (base + timedelta(seconds=4), base),
            2: (None, base),  # an HKTrack row: no fetched_utc, so the snapshot counts
            3: (base, base + timedelta(minutes=20)),  # a snapshot after the fetch: fetched wins
            4: (base, None),
            5: (None, None),  # never available
        }[shape]
        rows.append(
            {
                "row_id": i,
                "race_id": f"jra:2026-11-22:TOK:{1 + i % 2:02d}",
                "pool": "WIN",
                "combination": str(1 + (i // 2) % 2),
                "fetched_utc": fetched,
                "snapshot_utc": snapshot,
                "odds": round(float(rng.uniform(1.1, 80.0)), 1),
            }
        )
    frame = pd.DataFrame(rows)
    frame["fetched_utc"] = utc_series(frame["fetched_utc"])
    frame["snapshot_utc"] = utc_series(frame["snapshot_utc"])
    return frame


def _coalesced(row) -> datetime | None:
    for column in ("fetched_utc", "snapshot_utc"):
        if not pd.isna(row[column]):
            return row[column].to_pydatetime()
    return None


@pytest.mark.parametrize("session", SESSION_ZONES[:2])
@pytest.mark.parametrize("seed", [7, 8])
def test_a_fallback_column_counts_only_where_the_column_is_null(con, seed, session):
    set_session_zone(con, session)
    frame = _captures(np.random.default_rng(seed))
    put(con, "odds_timeseries", frame)
    rules = {
        "odds_timeseries": Availability(
            "odds_timeseries", "column", column="fetched_utc", fallback="snapshot_utc"
        )
    }
    available_at = {int(r["row_id"]): _coalesced(r) for _, r in frame.iterrows()}
    on = ["race_id", "pool", "combination"]
    per_key: dict[tuple, list[datetime]] = {}
    for _, r in frame.iterrows():
        if _coalesced(r) is not None:
            per_key.setdefault(tuple(r[c] for c in on), []).append(_coalesced(r))
    assert all(len(v) == len(set(v)) for v in per_key.values())  # no ties: the pick is defined

    times = sorted({a for a in available_at.values() if a is not None})
    snapshots = [s.to_pydatetime() for s in frame["snapshot_utc"].dropna()]
    probes = [at(0), at(23, 59), *times, *[a - US for a in times], *snapshots]
    for t in probes:
        expected = sorted(i for i, a in available_at.items() if a is not None and a <= t)
        got = read_asof(con, "odds_timeseries", t, rules=rules)
        assert ids(got) == expected, f"t={t.isoformat()}"

    keys = pd.DataFrame(
        [
            {"race_id": race, "pool": "WIN", "combination": comb, "t_utc": t}
            for race in ("jra:2026-11-22:TOK:01", "jra:2026-11-22:TOK:02")
            for comb in ("1", "2", "3")  # combination 3 has no rows
            for t in probes[::2]
        ]
    )
    keys["t_utc"] = utc_series(keys["t_utc"])
    expected = _brute_force(frame, keys, on, "t_utc", _coalesced)
    got = asof_join(keys, "odds_timeseries", on=on, rules=rules, con=con).reset_index(drop=True)
    assert any(e is None for e in expected) and any(e is not None for e in expected)
    for j, want in enumerate(expected):
        if want is None:
            assert pd.isna(got["row_id"].iloc[j]), f"key {j}"
        else:
            assert int(got["row_id"].iloc[j]) == want, f"key {j}"


def test_a_fallback_must_be_a_utc_column_of_a_column_rule(con):
    put_typed(
        con,
        "odds_timeseries",
        pd.DataFrame(
            {
                "row_id": [1, 2],
                "race_id": ["jra:2026-11-22:TOK:01"] * 2,
                "fetched_utc": utc_series([None, at(9, 30)]),
                "snapshot_utc": utc_series([at(9, 10), at(9)]),
                "snap_naive": pd.to_datetime(["2026-11-22 09:10:00", "2026-11-22 09:00:00"]),
                "snap_date": [date(2026, 11, 22)] * 2,
                "snap_text": ["2026-11-22T09:10:00Z", "2026-11-22T09:00:00Z"],
                "dividend": [250.0, 0.0],
            }
        ),
        {"snap_date": "DATE", "snap_text": "VARCHAR"},
    )
    keys = pd.DataFrame({"race_id": ["jra:2026-11-22:TOK:01"], "t_utc": utc_series([at(10)])})

    def column(**extra) -> dict[str, Availability]:
        rule = Availability("odds_timeseries", "column", column="fetched_utc", **extra)
        return {"odds_timeseries": rule}

    assert column()["odds_timeseries"].fallback is None
    good = column(fallback="snapshot_utc")
    assert good["odds_timeseries"].fallback == "snapshot_utc"
    assert ids(read_asof(con, "odds_timeseries", at(9, 10), rules=good)) == [1]
    assert ids(read_asof(con, "odds_timeseries", at(9, 29), rules=good)) == [1]

    for fallback in ("snap_naive", "snap_date", "snap_text"):
        rules = column(fallback=fallback)
        with pytest.raises(ValueError):
            read_asof(con, "odds_timeseries", at(10), rules=rules)
        with pytest.raises(ValueError):
            asof_join(keys, "odds_timeseries", on=["race_id"], rules=rules, con=con)

    with pytest.raises(ValueError):  # at construction or at query time
        rules = column(fallback="snapshot_utc", label_columns=("dividend", "snapshot_utc"))
        read_asof(con, "odds_timeseries", at(10), rules=rules)

    for fields in (
        {"kind": "offset", "anchor": "fetched_utc", "offset": timedelta(0),
         "fallback": "snapshot_utc"},
        {"kind": "label_only", "fallback": "snapshot_utc"},
        {"kind": "current_value", "fallback": "snapshot_utc"},
        {"kind": "static", "fallback": "snapshot_utc"},
    ):  # fmt: skip
        with pytest.raises(ValueError):
            Availability("odds_timeseries", **fields)


def test_load_availability_reads_a_fallback(tmp_path: Path, con):
    path = tmp_path / "availability.yaml"
    path.write_text(
        "tables:\n"
        "  odds_timeseries:\n"
        "    kind: column\n"
        "    column: fetched_utc\n"
        "    fallback: snapshot_utc\n"
        "    label_columns: [dividend]\n"
        "  jrdb_pre_race:\n"
        "    kind: column\n"
        "    column: released_utc\n",
        encoding="utf-8",
    )
    rules = load_availability("japan", path=path)
    odds = rules["odds_timeseries"]
    assert (odds.kind, odds.column, odds.fallback, odds.label_columns) == (
        "column", "fetched_utc", "snapshot_utc", ("dividend",),
    )  # fmt: skip
    assert rules["jrdb_pre_race"].fallback is None

    put(
        con,
        "odds_timeseries",
        pd.DataFrame(
            {
                "row_id": [1, 2],
                "fetched_utc": utc_series([None, at(9, 30)]),
                "snapshot_utc": utc_series([at(9, 10), at(9)]),
                "dividend": [250.0, 0.0],
            }
        ),
    )
    assert ids(read_asof(con, "odds_timeseries", at(9, 10) - US, rules=rules)) == []
    view = read_asof(con, "odds_timeseries", at(9, 10), rules=rules)
    assert ids(view) == [1] and "dividend" not in view.columns
    assert ids(read_asof(con, "odds_timeseries", at(9, 30), rules=rules)) == [1, 2]

    for entry in (
        "kind: offset\n    anchor: post_utc\n    offset_min: 0\n    fallback: snapshot_utc",
        "kind: label_only\n    fallback: snapshot_utc",
        "kind: current_value\n    fallback: snapshot_utc",
        "kind: static\n    fallback: snapshot_utc",
        "kind: column\n    column: fetched_utc\n    fallback: 3",
        "kind: column\n    column: fetched_utc\n    fallback: [snapshot_utc]",
        "kind: column\n    column: fetched_utc\n    fallback:",
    ):
        _refuse_entry(tmp_path, entry)

    path.write_text(
        "tables:\n  odds_timeseries:\n    kind: column\n    column: fetched_utc\n"
        "    fallback: snapshot_utc\n    label_columns: [snapshot_utc]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):  # at load or at query time
        read_asof(con, "odds_timeseries", at(10), rules=load_availability("japan", path=path))


# --- static tables: every row available at any t -----------------------------------------------

PEDIGREE = pd.DataFrame(
    {
        "jurisdiction": ["jra"] * 4,
        "horse_id": ["H1", "H2", "H3", "H4"],
        "sire_name": ["Sire A", "Sire B", "Sire A", "Sire C"],
        "dam_name": ["Dam 1", "Dam 2", "Dam 3", "Dam 4"],
        "source_race_id": [f"jra:2026-11-22:TOK:{i:02d}" for i in (1, 2, 3, 4)],
        # scraped after the decision times below: a static table ignores it
        "scraped_utc": utc_series([datetime(2026, 11, 30, 3, tzinfo=UTC)] * 4),
    }
)
STATIC_TIMES = [
    datetime(1990, 1, 1, tzinfo=UTC),
    at(0),
    at(23, 59),
    datetime(2100, 1, 1, tzinfo=UTC),
]


@pytest.mark.parametrize("session", SESSION_ZONES[:2])
@pytest.mark.parametrize(
    "labels",
    [pytest.param((), id="no-labels"), pytest.param(("source_race_id", "scraped_utc"), id="two")],
)
def test_a_static_table_is_whole_in_every_view_at_any_t(con, labels, session):
    set_session_zone(con, session)
    put(con, "horse_pedigree", PEDIGREE)
    rule = Availability("horse_pedigree", "static", label_columns=labels)
    assert (rule.column, rule.anchor, rule.offset, rule.tz, rule.fallback) == (None,) * 5
    rules = {"horse_pedigree": rule}
    kept = set(PEDIGREE.columns) - set(labels)

    for t in STATIC_TIMES:
        view = read_asof(con, "horse_pedigree", t, rules=rules)
        assert sorted(view["horse_id"]) == ["H1", "H2", "H3", "H4"], f"t={t.isoformat()}"
        assert set(view.columns) == kept

    keys = pd.DataFrame(
        {
            "horse_id": ["H2", "H4", "H9", "H1", "H3"],
            "t_utc": utc_series([STATIC_TIMES[0], STATIC_TIMES[3], at(12), at(3), at(0)]),
        }
    )
    got = asof_join(keys, "horse_pedigree", on=["horse_id"], rules=rules, con=con)
    got = got.reset_index(drop=True)
    assert set(got.columns) == {"horse_id", "t_utc"} | (kept - {"horse_id"})
    sires = [None if pd.isna(v) else v for v in got["sire_name"]]
    assert sires == ["Sire B", "Sire C", None, "Sire A", "Sire A"]
    assert [None if pd.isna(v) else v for v in got["dam_name"]] == [
        "Dam 2", "Dam 4", None, "Dam 1", "Dam 3",
    ]  # fmt: skip

    with pytest.raises(ValueError):
        read_label(con, "horse_pedigree", rules=rules)


def test_a_static_rule_takes_no_time_fields(con):
    put(con, "horse_pedigree", PEDIGREE)
    assert Availability("horse_pedigree", "static").label_columns == ()
    for fields in (
        {"column": "scraped_utc"},
        {"anchor": "scraped_utc", "offset": timedelta(0)},
        {"offset": timedelta(minutes=5)},
        {"tz": "Asia/Tokyo"},
        {"fallback": "scraped_utc"},
    ):
        with pytest.raises(ValueError):
            Availability("horse_pedigree", "static", **fields)
    rules = {"horse_pedigree": Availability("horse_pedigree", "static", label_columns=("sir",))}
    with pytest.raises(ValueError):  # a misspelt label column cannot let the real one through
        read_asof(con, "horse_pedigree", at(12), rules=rules)


def test_load_availability_reads_a_static_rule(tmp_path: Path, con):
    path = tmp_path / "availability.yaml"
    path.write_text(
        "tables:\n"
        "  horse_pedigree:\n"
        "    kind: static\n"
        "    label_columns: [source_race_id, scraped_utc]\n"
        "  dividends:\n"
        "    kind: label_only\n",
        encoding="utf-8",
    )
    rules = load_availability("japan", path=path)
    ped = rules["horse_pedigree"]
    assert (ped.kind, ped.column, ped.anchor, ped.offset) == ("static", None, None, None)
    assert ped.label_columns == ("source_race_id", "scraped_utc")
    put(con, "horse_pedigree", PEDIGREE)
    view = read_asof(con, "horse_pedigree", datetime(1990, 1, 1, tzinfo=UTC), rules=rules)
    assert len(view) == 4 and not {"source_race_id", "scraped_utc"} & set(view.columns)

    for entry in (
        "kind: static\n    column: scraped_utc",
        "kind: static\n    anchor: scraped_utc\n    offset_min: 0",
        "kind: static\n    offset_min: 0",
        "kind: static\n    tz: Asia/Tokyo",
        "kind: static\n    fallback: scraped_utc",
        "kind: static\n    label_columns: source_race_id",
    ):
        _refuse_entry(tmp_path, entry)


# --- joining on a label column -------------------------------------------------------------------

JOIN_ON_LABEL_RULES = [
    pytest.param({"kind": "column", "column": "fetched_utc"}, id="column"),
    pytest.param(
        {"kind": "offset", "anchor": "fetched_utc", "offset": timedelta(minutes=20)}, id="offset"
    ),
    pytest.param(
        {"kind": "offset", "anchor": "race_date", "offset": timedelta(0), "tz": "Asia/Tokyo"},
        id="offset-on-date",
    ),
    pytest.param({"kind": "static"}, id="static"),
]


@pytest.mark.parametrize("fields", JOIN_ON_LABEL_RULES)
def test_asof_join_on_a_label_column_is_refused(con, fields):
    put_typed(
        con,
        "runners",
        pd.DataFrame(
            {
                "row_id": [1, 2],
                "race_id": ["jra:2026-11-22:TOK:01"] * 2,
                "horse_no": [1, 2],
                "race_date": [date(2026, 11, 22)] * 2,
                "fetched_utc": utc_series([at(1), at(1)]),
                "finish_pos": [2, 1],
            }
        ),
        {"race_date": "DATE"},
    )
    rules = {"runners": Availability("runners", label_columns=("finish_pos",), **fields)}
    keys = pd.DataFrame(
        {
            "race_id": ["jra:2026-11-22:TOK:01"] * 2,
            "horse_no": [1, 2],
            "finish_pos": [1, 2],
            "t_utc": utc_series([at(23), at(23)]),
        }
    )
    with pytest.raises(ValueError):
        asof_join(keys, "runners", on=["race_id", "finish_pos"], rules=rules, con=con)
    got = asof_join(keys, "runners", on=["race_id", "horse_no"], rules=rules, con=con)
    assert [int(v) for v in got["row_id"]] == [1, 2]
    assert list(got["finish_pos"]) == [1, 2]  # the keys' own column, untouched
