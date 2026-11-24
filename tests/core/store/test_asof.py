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

YAML format of `load_availability` (pinned below):

    tables:
      <table>:
        kind: column | offset | label_only | current_value
        column: <name>        # kind column only, required
        anchor: <name>        # kind offset only, required
        offset_min: <number>  # kind offset only, required; minutes, int or float

A key that does not belong to the entry's kind, an unknown key, an unknown kind or a missing
required key is a ValueError.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

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
