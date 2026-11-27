"""Japan's shipped availability rules (ticket #80; the owner's answer on #80, 2026-11-24).

`load_availability("japan")` reads `src/horsetrack/countries/japan/config/availability.yaml`. The
owner accepted the proposed defaults, each conservative relative to JRA's publication:

1. odds_timeseries: column on fetched_utc, falling back to snapshot_utc (HKTrack rows have no
   fetched_utc); dividend, is_refund and is_final_capture are label columns.
2. meetings: offset on race_date, 0 (00:00 JST of the race date).
3. races and runners: offset on race_date, 0 (00:00 JST); the result columns are label columns.
   races.result_published_utc and races.result_unpublished_utc are label columns too (the owner's
   answer on #80, 2026-11-25); races.rail_setting stays in the view. Since #227 races carry a
   derived post_utc, and the horse weights (runners horse_wt_kg, horse_wt_change_kg,
   declared_horse_wt_lb) and the same-day going (races going, going_raw) are in the view, null
   until post_utc - 60 min of their race. Since #305 the finish (races finish_time_s; runners
   finish_pos, dead_heat, finish_time_s) is in the view too, null until post_utc + 30 min of its
   race, the same-day dividends rule (the owner's answer on #96, 2026-11-25). Every other result
   column stays a label column.
4. dividends, sectionals, incidents, running_comments: available from post_utc + 30 min of their
   race (#227). Their boundaries are pinned in tests/core/store/test_asof.py.
5. horses: current_value. horse_pedigree: static (sire, dam and broodmare sire never change),
   with source_race_id and scraped_utc kept out as label columns.
6. horse_runs: offset on race_date + 1 day (00:00 JST of the next day).
7. jrdb_pre_race: offset on race_date, 0 (00:00 JST), with every `tyb_` column (JRDB's day-of
   TYB file: race-day body weight, the board, same-day going, paddock) kept out through
   `label_prefixes: [tyb_]` (the orchestrator's answer on #80, 2026-11-25, extending answer 3);
   the kyi_, cyb_ and joa_ columns stay in the view. 8. jrdb_post_race: label_only.
9. Nothing banned.
10. track_condition (#90): column on available_utc (= published_utc, the page's 発表 time), no
    label columns. The owner's words: #136's request ('ingest them as of their publication time
    and use them as features', 2026-11-24), #228's outline step 4 from his option-A split
    ('available_utc = published_utc as in #136') and his answer on #87 (usable 'once JRA has
    officially published it (with its publication timestamp)', 2026-11-24).
11. win_sales (#278): column on published_utc, each publication of a race's Win sales from its
    own time: a live page's clock, a final page's post + 30 min (the owner's answer on #278).
12. The JRA-VAN-only tables (#376, the owner's request of 2026-11-26), not store tables (not in
    `schemas.TABLES`; their schemas are the jravan modules'): `final_odds` and `pool_votes`
    label_only (labels or conditioners only, never features at t); `jockeys` and `trainers`
    current_value with `in_view` ("the master row as exported is used, with no history": in
    every view at any t, the look-ahead documented in docs/specs/as-of.md); `breeding_horses`,
    `produce` and `lineages` static (known at any t). None has label columns. `horses` stays
    current_value without `in_view`, refused (#80).

The expected rules below are written from that answer, never read back from the config. The
leakage tests run the shipped rules over small synthetic tables built from the store's column
contracts (`core.store.schemas`, `core.store.tables.ARROW_TYPES`), the odds log's
(`countries/japan/logger/compact.py` SCHEMA) and the jravan modules' schemas.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from horsetrack.core.store import schemas
from horsetrack.core.store.asof import (
    CurrentValueTable,
    LabelOnlyTable,
    NoAvailabilityRule,
    asof_join,
    load_availability,
    read_asof,
    read_label,
)
from horsetrack.core.store.tables import ARROW_TYPES
from horsetrack.countries.japan.jravan import bloodlines, masters, odds, votes
from horsetrack.countries.japan.logger.compact import SCHEMA as ODDS_SCHEMA

UTC = timezone.utc
US = timedelta(microseconds=1)
JST = "Asia/Tokyo"
SPEC = Path(__file__).resolve().parents[3] / "docs" / "specs" / "as-of.md"

RUNNER_LABELS = frozenset(
    {
        "placing_raw", "placing_code", "lbw_raw", "lbw_lengths", "running_positions", "last_3f_s",
        "win_odds", "win_fav_rank",
    }
)  # fmt: skip
# The finish left the label columns in #305: in the view from 00:00 JST, null until post_utc + 30
# min of its race (the owner's answer on #96, 2026-11-25: the same rule as same-day dividends).
POST_FINISH = {
    "runners": ("finish_pos", "dead_heat", "finish_time_s"),
    "races": ("finish_time_s",),
}
# The race-day body weight (horse_wt_kg, horse_wt_change_kg and declared_horse_wt_lb, its copy in
# pounds) and the same-day going (going, going_raw) left the label columns in #227: they are in the
# view, null until post_utc - 60 min (the owner's answer on #80, 2026-11-24).
POST_GATED = {
    "runners": ("horse_wt_kg", "horse_wt_change_kg", "declared_horse_wt_lb"),
    "races": ("going", "going_raw"),
}
# result_published_utc and result_unpublished_utc (#225): never in a view at t (the owner's answer
# on #80, 2026-11-25); #227 and #305 left them labels. rail_setting stays in.
RACE_LABELS = frozenset({"split_times_s", "result_published_utc", "result_unpublished_utc"})
# the columns a view adds to a table's own: races' post_utc, derived from race_date and
# post_time_local in Asia/Tokyo (#227)
VIEW_EXTRA = {"races": frozenset({"post_utc"})}
# available from post_utc + 30 min of their race (#227); how the rule reaches the race's post_utc
# is the implementation's, so only the behaviour is pinned (tests/core/store/test_asof.py)
POST_RESULTS = ("dividends", "sectionals", "incidents", "running_comments")
ODDS_LABELS = frozenset({"dividend", "is_refund", "is_final_capture"})
PEDIGREE_LABELS = frozenset({"source_race_id", "scraped_utc"})
# the JRA-VAN-only tables (#376): parquet under processed/jravan, not store tables
JRAVAN_SCHEMAS = {
    "final_odds": odds.FINAL_ODDS_SCHEMA,
    "pool_votes": votes.POOL_VOTES_SCHEMA,
    "jockeys": masters.JOCKEYS_SCHEMA,
    "trainers": masters.TRAINERS_SCHEMA,
    "breeding_horses": bloodlines.BREEDING_HORSES_SCHEMA,
    "produce": bloodlines.PRODUCE_SCHEMA,
    "lineages": bloodlines.LINEAGES_SCHEMA,
}


def _offset(minutes: int, labels=frozenset(), prefixes: tuple[str, ...] = ()) -> dict:
    return {"kind": "offset", "anchor": "race_date", "offset_min": minutes, "tz": JST,
            "labels": labels, "prefixes": prefixes}  # fmt: skip


OWNER: dict[str, dict] = {
    "odds_timeseries": {
        "kind": "column",
        "column": "fetched_utc",
        "fallback": "snapshot_utc",
        "labels": ODDS_LABELS,
    },  # fmt: skip
    "meetings": _offset(0),
    "races": _offset(0, RACE_LABELS),
    "runners": _offset(0, RUNNER_LABELS),
    **{table: {"kind": "post + 30 min"} for table in POST_RESULTS},
    "horses": {"kind": "current_value"},
    "horse_pedigree": {"kind": "static", "labels": PEDIGREE_LABELS},
    "horse_runs": _offset(1440),
    # the TYB columns are named tyb_<field> and come from licensed specs (not in schemas.TABLES)
    "jrdb_pre_race": _offset(0, prefixes=("tyb_",)),
    "jrdb_post_race": {"kind": "label_only"},
    # a published value counts from its 発表 time (#90)
    "track_condition": {"kind": "column", "column": "available_utc"},
    # per-race Win sales (#278): each publication from its own time; a final figure's time is its
    # post + 30 min (tests/countries/japan/test_win_sales_store.py pins the view)
    "win_sales": {"kind": "column", "column": "published_utc"},
    # The JRA-VAN-only tables (#376, the owner's request, 2026-11-26). The final odds and the votes
    # are labels or conditioners only, never features at t.
    "final_odds": {"kind": "label_only"},
    "pool_votes": {"kind": "label_only"},
    # The jockey and trainer masters: "the master row as exported is used, with no history", so in
    # every view at any t (tests/countries/japan/test_jravan_masters.py pins the views).
    "jockeys": {"kind": "current_value", "in_view": True},
    "trainers": {"kind": "current_value", "in_view": True},
    # Pedigree: static, known at any t (tests/countries/japan/test_jravan_bloodlines.py).
    "breeding_horses": {"kind": "static"},
    "produce": {"kind": "static"},
    "lineages": {"kind": "static"},
}
VIEW_TABLES = [t for t, o in OWNER.items() if o["kind"] == "offset"]
LABEL_ONLY = [t for t, o in OWNER.items() if o["kind"] == "label_only"]


def _schema(table: str) -> pa.Schema:
    if table == "odds_timeseries":
        return ODDS_SCHEMA
    if table in JRAVAN_SCHEMAS:
        return JRAVAN_SCHEMAS[table]
    spec = schemas.TABLES[table]
    return pa.schema([(c, ARROW_TYPES[k]) for c, k in spec.columns.items()])


@pytest.fixture(scope="module")
def rules():
    return load_availability("japan")


# --- the shipped config is the owner's answer ---------------------------------------------------


def test_the_shipped_config_has_one_rule_per_table_and_nothing_else(rules):
    assert len(schemas.TABLES) == 14  # 12, + track_condition (#90), + win_sales (#278)
    assert len(JRAVAN_SCHEMAS) == 7
    assert not set(JRAVAN_SCHEMAS) & set(schemas.TABLES)  # #376 leaves them out of the store
    assert set(OWNER) == set(schemas.TABLES) | {"odds_timeseries"} | set(JRAVAN_SCHEMAS)
    assert set(rules) == set(OWNER)
    for table, rule in rules.items():
        assert rule.table == table


@pytest.mark.parametrize("table", list(OWNER))
def test_only_the_jockey_and_trainer_masters_are_current_values_in_the_views(rules, table):
    # #376: jockeys and trainers in_view; horses (#80) and every other table without it
    assert rules[table].in_view is OWNER[table].get("in_view", False)


@pytest.mark.parametrize("table", list(OWNER))
def test_each_shipped_rule_is_the_owners(rules, table):
    want, rule = OWNER[table], rules[table]
    if table in POST_RESULTS:  # they enter the views; their post + 30 min rule is pinned in
        # tests/core/store/test_asof.py by behaviour, whatever form the rule takes
        assert rule.kind not in ("label_only", "current_value")
        return
    assert rule.kind == want["kind"]
    assert rule.column == want.get("column")
    assert rule.fallback == want.get("fallback")
    assert rule.anchor == want.get("anchor")
    minutes = want.get("offset_min")
    assert rule.offset == (None if minutes is None else timedelta(minutes=minutes))
    assert rule.tz == want.get("tz")
    assert set(rule.label_columns) == set(want.get("labels", ()))
    assert len(rule.label_columns) == len(set(rule.label_columns))  # no repeats
    assert tuple(rule.label_prefixes) == want.get("prefixes", ())


@pytest.mark.parametrize("table", list(OWNER))
def test_every_column_a_shipped_rule_names_is_in_its_table(rules, table):
    schema = _schema(table)
    rule = rules[table]
    missing = set(rule.label_columns) - set(schema.names)
    assert not missing, f"{table}: label columns not in the schema: {sorted(missing)}"
    # a label prefix never drops a column of the store's contract (the keys, the anchor)
    dropped = [c for c in schema.names if any(c.startswith(p) for p in rule.label_prefixes)]
    assert not dropped, f"{table}: label prefixes drop contract columns {dropped}"
    if table in POST_RESULTS:
        return  # their time is the race's post_utc, not a column of their own table
    for name in (rule.column, rule.fallback):
        if name is not None:
            field = schema.field(name)
            assert pa.types.is_timestamp(field.type) and field.type.tz is not None, name
    if rule.anchor is not None:
        assert pa.types.is_date(schema.field(rule.anchor).type), rule.anchor  # a DATE takes tz


# --- leakage: the shipped rules over synthetic tables ------------------------------------------

RACE_DATES = [date(2026, 11, 21), date(2026, 11, 22), date(2026, 11, 28), date(2026, 11, 29), None]
SCRAPED = datetime(2026, 11, 30, 3, tzinfo=UTC)  # a scrape stamp after every race date
ROW_ID = {"meetings": "meeting_id", "horse_pedigree": "horse_id", "horses": "horse_id"}


def _value(dtype: pa.DataType, column: str, i: int):
    if pa.types.is_integer(dtype):
        return i + 1
    if pa.types.is_floating(dtype):
        return 1.5 * (i + 1)
    if column == "post_time_local":  # "HH:MM" on the JST wall clock, as the results parser writes
        return f"{10 + i}:{5 * i:02d}"
    if pa.types.is_string(dtype):
        return f"{column}-{i}"
    if pa.types.is_boolean(dtype):
        return i % 2 == 0
    if pa.types.is_date(dtype):
        return RACE_DATES[i]
    if pa.types.is_timestamp(dtype):
        return SCRAPED
    if pa.types.is_list(dtype) and pa.types.is_string(dtype.value_type):
        return [f"{column}-{i}-{k}" for k in range(2)]  # produce's pedigree ids (#376)
    if pa.types.is_list(dtype):
        return [i + 1, i + 2] if pa.types.is_integer(dtype.value_type) else [11.5, 12.25]
    raise AssertionError(f"{column}: no synthetic value for {dtype}")


ODDS_ROWS = [  # (fetched_utc, snapshot_utc) of each capture, on 2026-11-28
    (datetime(2026, 11, 28, 9, 0, 5, tzinfo=UTC), datetime(2026, 11, 28, 9, 0, tzinfo=UTC)),
    (None, datetime(2026, 11, 28, 9, 10, tzinfo=UTC)),  # an HKTrack row
    (datetime(2026, 11, 28, 9, 30, tzinfo=UTC), datetime(2026, 11, 28, 9, 20, tzinfo=UTC)),
    (None, None),
    (datetime(2026, 11, 28, 9, 40, tzinfo=UTC), None),
]


def _synthetic(table: str) -> pa.Table:
    schema = _schema(table)
    rows = []
    for i in range(len(RACE_DATES)):
        row = {f.name: _value(f.type, f.name, i) for f in schema}
        if table == "odds_timeseries":
            row["fetched_utc"], row["snapshot_utc"] = ODDS_ROWS[i]
            row["race_id"], row["pool"] = "jra:2026-11-28:TOK:05", "WIN"
            row["combination"] = ["1", "1", "1", "2", "3"][i]
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=schema)


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect(":memory:")
    for table in OWNER:
        connection.register("_arrow", _synthetic(table))
        connection.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _arrow')
        connection.unregister("_arrow")
    yield connection
    connection.close()


def _opens(table: str, i: int) -> datetime | None:
    """When row i of a race-date table is first available under the owner's rule."""
    day = RACE_DATES[i]
    if day is None:
        return None
    midnight = datetime.combine(day, time(0), tzinfo=ZoneInfo(JST)).astimezone(UTC)
    return midnight + timedelta(minutes=OWNER[table]["offset_min"])


@pytest.mark.parametrize("table", VIEW_TABLES)
def test_a_race_date_view_opens_at_the_owners_time_and_never_carries_a_label(rules, con, table):
    key = ROW_ID.get(table, "race_id")
    kept = set(_schema(table).names) - set(OWNER[table]["labels"]) | VIEW_EXTRA.get(table, set())
    opens = {i: _opens(table, i) for i in range(len(RACE_DATES))}
    times = sorted(a for a in opens.values() if a is not None)
    probes = [datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 6, 1, tzinfo=UTC)]
    for t in [*probes, *times, *[a - US for a in times]]:
        view = read_asof(con, table, t, rules=rules)
        assert set(view.columns) == kept, f"{table} at {t.isoformat()}"
        expected = sorted(f"{key}-{i}" for i, a in opens.items() if a is not None and a <= t)
        assert sorted(view[key]) == expected, f"{table} at {t.isoformat()}"


def test_race_day_rows_open_at_midnight_jst_and_past_runs_the_next_day(rules, con):
    # race_date 2026-11-28 is row 2
    jst_midnight = datetime(2026, 11, 27, 15, tzinfo=UTC)
    next_midnight = datetime(2026, 11, 28, 15, tzinfo=UTC)
    for table in ("meetings", "races", "runners", "jrdb_pre_race"):
        key = ROW_ID.get(table, "race_id")
        assert f"{key}-2" not in set(read_asof(con, table, jst_midnight - US, rules=rules)[key])
        assert f"{key}-2" in set(read_asof(con, table, jst_midnight, rules=rules)[key])
    assert "race_id-2" not in set(read_asof(con, "horse_runs", next_midnight - US, rules=rules)[
        "race_id"])  # fmt: skip
    assert "race_id-2" in set(read_asof(con, "horse_runs", next_midnight, rules=rules)["race_id"])


def test_a_runner_or_race_joined_at_t_carries_its_card_and_no_result_label(rules, con):
    jst_midnight = datetime(2026, 11, 27, 15, tzinfo=UTC)
    keys = pd.DataFrame(
        {
            "race_id": ["race_id-2", "race_id-2", "race_id-0"],
            "horse_no": [3, 3, 1],
            "t_utc": pd.to_datetime([jst_midnight - US, jst_midnight, SCRAPED], utc=True),
        }
    )
    got = asof_join(keys, "runners", on=["race_id", "horse_no"], rules=rules, con=con)
    assert not RUNNER_LABELS & set(got.columns)
    assert {"draw", "jockey_key", "actual_wt_kg", "is_scratched"} <= set(got.columns)
    assert [None if pd.isna(v) else int(v) for v in got["draw"]] == [None, 3, 1]
    # the finish (#305): race_id-2 posts 12:10 JST on 2026-11-28, so at 00:00 JST it is null;
    # race_id-0 posted 10:00 JST on 2026-11-21, long before SCRAPED
    assert set(POST_FINISH["runners"]) <= set(got.columns)
    assert [None if pd.isna(v) else int(v) for v in got["finish_pos"]] == [None, None, 1]
    assert [None if pd.isna(v) else float(v) for v in got["finish_time_s"]] == [None, None, 1.5]
    assert [None if pd.isna(v) else bool(v) for v in got["dead_heat"]] == [None, None, True]

    races = asof_join(keys[["race_id", "t_utc"]], "races", on=["race_id"], rules=rules, con=con)
    assert not RACE_LABELS & set(races.columns)
    assert [None if pd.isna(v) else int(v) for v in races["distance_m"]] == [None, 3, 1]
    assert [None if pd.isna(v) else float(v) for v in races["finish_time_s"]] == [None, None, 1.5]


def test_a_race_view_never_carries_its_result_publication_times_but_keeps_the_rail(rules):
    # One race on 2026-11-28 (opens 2026-11-27 15:00 UTC), its result published at 07:10 UTC and
    # withdrawn at 07:40 UTC. Even after both, the view keeps rail_setting and drops both stamps.
    published = datetime(2026, 11, 28, 7, 10, tzinfo=UTC)
    unpublished = datetime(2026, 11, 28, 7, 40, tzinfo=UTC)
    row = {f.name: _value(f.type, f.name, 2) for f in _schema("races")}
    row |= {"race_id": "jra:2026-11-28:TOK:11", "rail_setting": "C",
            "result_published_utc": published, "result_unpublished_utc": unpublished}  # fmt: skip
    one = duckdb.connect(":memory:")
    try:
        one.register("_arrow", pa.Table.from_pylist([row], schema=_schema("races")))
        one.execute('CREATE TABLE "races" AS SELECT * FROM _arrow')
        one.unregister("_arrow")
        opens = datetime(2026, 11, 27, 15, tzinfo=UTC)
        for t in (opens, published - US, published, unpublished, unpublished + timedelta(days=30)):
            view = read_asof(one, "races", t, rules=rules)
            gone = {"result_published_utc", "result_unpublished_utc"} & set(view.columns)
            assert not gone, f"races at {t.isoformat()} carries {sorted(gone)}"
            assert list(view["race_id"]) == ["jra:2026-11-28:TOK:11"], t.isoformat()
            assert list(view["rail_setting"]) == ["C"], t.isoformat()
    finally:
        one.close()


def test_a_runner_view_carries_the_body_weight_in_every_unit_from_post_minus_60(rules):
    # Two runners on 2026-11-28 (opens 2026-11-27 15:00 UTC) in a race posted 15:40 JST (06:40
    # UTC), weighed at 485 kg (1069 lb) and 502 kg (1107 lb). From 00:00 JST the view keeps the
    # card (draw) and holds the body weight, in kg and in lb, null until 05:40 UTC (post - 60 min,
    # the owner's answer on #80, 2026-11-24), then the stored values.
    race_id = "jra:2026-11-28:TOK:11"
    runners = [(3, 485, 1069, 4), (7, 502, 1107, -2)]  # (horse_no, kg, lb, change kg)
    rows = []
    for horse_no, kg, lb, change in runners:
        row = {f.name: _value(f.type, f.name, 2) for f in _schema("runners")}
        row |= {"race_id": race_id, "horse_no": horse_no, "draw": horse_no}
        row |= {"horse_wt_kg": kg, "declared_horse_wt_lb": lb, "horse_wt_change_kg": change}
        rows.append(row)
    race = {f.name: _value(f.type, f.name, 2) for f in _schema("races")}
    race |= {"race_id": race_id, "post_time_local": "15:40"}
    weights = ("horse_wt_kg", "declared_horse_wt_lb", "horse_wt_change_kg")
    one = duckdb.connect(":memory:")
    try:
        for table, table_rows in (("runners", rows), ("races", [race])):
            one.register("_arrow", pa.Table.from_pylist(table_rows, schema=_schema(table)))
            one.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _arrow')
            one.unregister("_arrow")
        opens = datetime(2026, 11, 27, 15, tzinfo=UTC)
        gate = datetime(2026, 11, 28, 5, 40, tzinfo=UTC)
        for t in (
            opens,
            gate - US,
            gate,
            datetime(2026, 11, 28, 7, tzinfo=UTC),
            opens + timedelta(30),
        ):
            want = [r[1:] if t >= gate else (None, None, None) for r in runners]
            view = read_asof(one, "runners", t, rules=rules).sort_values("horse_no")
            assert set(weights) <= set(view.columns), t.isoformat()
            assert [int(v) for v in view["horse_no"]] == [3, 7], t.isoformat()
            assert [int(v) for v in view["draw"]] == [3, 7], t.isoformat()
            got = [
                tuple(None if pd.isna(v) else int(v) for v in values)
                for values in view[list(weights)].itertuples(index=False)
            ]
            assert got == want, f"runners at {t.isoformat()}"
            keys = pd.DataFrame({"race_id": [race_id] * 2, "horse_no": [7, 3],
                                 "t_utc": pd.to_datetime([t, t], utc=True)})  # fmt: skip
            joined = asof_join(keys, "runners", on=["race_id", "horse_no"], rules=rules, con=one)
            assert [int(v) for v in joined["draw"]] == [7, 3], t.isoformat()
            got = [
                tuple(None if pd.isna(v) else int(v) for v in values)
                for values in joined[list(weights)].itertuples(index=False)
            ]
            assert got == want[::-1], f"runners joined at {t.isoformat()}"
        with pytest.raises(ValueError):
            read_label(one, "runners", rules=rules)
    finally:
        one.close()


# --- the finish at post_utc + 30 min (#305; the owner's answer on #96, 2026-11-25) --------------
#
# races.finish_time_s and runners finish_pos, dead_heat and finish_time_s are in the view from
# 00:00 JST of the race date, null until post_utc + 30 min of their race (runners through their
# race's post), then the stored values; a race without a post time, or a runner whose race_id has
# no races row, never shows them. Every other result column stays out of every view. Expected
# instants are derived here with zoneinfo.

FINISH_RACES = [  # (race_id, race_date, post_time_local)
    ("jra:2026-11-28:KYO:01", date(2026, 11, 28), "9:50"),  # 00:50 UTC, finish at 01:20
    ("jra:2026-11-28:TOK:05", date(2026, 11, 28), "12:15"),  # 03:15 UTC, finish at 03:45
    ("jra:2026-11-28:TOK:11", date(2026, 11, 28), "15:40"),  # 06:40 UTC, finish at 07:10
    ("jra:2026-11-29:TOK:12", date(2026, 11, 29), "16:25"),  # 07:25 UTC (11-29), finish at 07:55
    ("jra:2026-11-29:TOK:01", date(2026, 11, 29), None),  # no post time
]
NO_RACE_ROW = ("jra:2026-11-28:NAK:03", date(2026, 11, 28))  # runners, no races row
FINISH_HORSES = (1, 2, 3)
FAR_FUTURE = datetime(2030, 1, 1, tzinfo=UTC)
FINISH_DATES = {rid: day for rid, day, _ in FINISH_RACES} | {NO_RACE_ROW[0]: NO_RACE_ROW[1]}


def _post(day: date, hhmm: str | None) -> datetime | None:
    if hhmm is None:
        return None
    hh, mm = (int(part) for part in hhmm.split(":"))
    return datetime.combine(day, time(hh, mm), tzinfo=ZoneInfo(JST)).astimezone(UTC)


FINISH_POSTS = {rid: _post(day, hhmm) for rid, day, hhmm in FINISH_RACES} | {NO_RACE_ROW[0]: None}
FINISHED_AT = {  # post_utc + 30 min, None without a post time
    rid: post and post + timedelta(minutes=30) for rid, post in FINISH_POSTS.items()
}


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time(0), tzinfo=ZoneInfo(JST)).astimezone(UTC)


def _k(rid: str) -> int:
    return sorted(FINISH_DATES).index(rid)  # a per-race number, so every value differs by race


def runner_finish(rid: str, horse_no: int) -> dict:
    k = _k(rid)
    return {
        "finish_pos": (horse_no + k) % len(FINISH_HORSES) + 1,
        "dead_heat": (horse_no + k) % 2 == 0,
        "finish_time_s": 70.0 + 2 * k + horse_no / 10,
    }


def race_finish(rid: str) -> dict:
    return {"finish_time_s": 69.5 + 2 * _k(rid)}


def _cell(v):
    """A frame cell as a plain Python value, None for null."""
    if v is None or (not isinstance(v, list | np.ndarray) and pd.isna(v)):
        return None
    return v.item() if isinstance(v, np.generic) else v


def _by_race(frame: pd.DataFrame, column: str) -> dict[str, object]:
    return {rid: _cell(v) for rid, v in zip(frame["race_id"], frame[column], strict=True)}


@pytest.fixture(scope="module")
def finish_con():
    races, runners = [], []
    for rid, day, hhmm in FINISH_RACES:
        row = {f.name: _value(f.type, f.name, 2) for f in _schema("races")}  # every column set
        row |= {"race_id": rid, "race_date": day, "post_time_local": hhmm,
                "distance_m": 1200 + 200 * _k(rid), **race_finish(rid)}  # fmt: skip
        races.append(row)
    for rid, day in FINISH_DATES.items():
        for h in FINISH_HORSES:
            row = {f.name: _value(f.type, f.name, 2) for f in _schema("runners")}
            row |= {"race_id": rid, "race_date": day, "horse_no": h, "draw": 10 * _k(rid) + h,
                    **runner_finish(rid, h)}  # fmt: skip
            runners.append(row)
    connection = duckdb.connect(":memory:")
    for table, rows in (("races", races), ("runners", runners)):
        connection.register("_arrow", pa.Table.from_pylist(rows, schema=_schema(table)))
        connection.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _arrow')
        connection.unregister("_arrow")
    yield connection
    connection.close()


def _finish_probes() -> list[datetime]:
    times = {_midnight(day) for day in FINISH_DATES.values()}
    for rid, post in FINISH_POSTS.items():
        if post is not None:  # the weights' gate, then the finish's
            times |= {post - timedelta(minutes=60), FINISHED_AT[rid]}
    edges = [datetime(2026, 11, 20, tzinfo=UTC), FAR_FUTURE]
    return edges + sorted(times) + sorted(t - US for t in times)


def test_the_finish_is_null_until_post_plus_30_then_the_stored_value(rules, finish_con):
    for t in _finish_probes():
        where = f"t={t.isoformat()}"
        open_races = {rid for rid, day in FINISH_DATES.items() if _midnight(day) <= t}
        finished = {rid for rid, at in FINISHED_AT.items() if at is not None and at <= t}

        runners = read_asof(finish_con, "runners", t, rules=rules)
        assert set(POST_FINISH["runners"]) <= set(runners.columns), where
        assert not RUNNER_LABELS & set(runners.columns), where
        keys = list(zip(runners["race_id"], (int(h) for h in runners["horse_no"]), strict=True))
        assert sorted(keys) == sorted((r, h) for r in open_races for h in FINISH_HORSES), where
        for column in POST_FINISH["runners"]:
            got = {key: _cell(v) for key, v in zip(keys, runners[column], strict=True)}
            assert got == {
                (r, h): runner_finish(r, h)[column] if r in finished else None
                for r in open_races for h in FINISH_HORSES
            }, f"runners.{column}, {where}"  # fmt: skip
        draws = {key: int(v) for key, v in zip(keys, runners["draw"], strict=True)}
        assert draws == {(r, h): 10 * _k(r) + h for r, h in keys}, where  # the card, from 00:00

        races = read_asof(finish_con, "races", t, rules=rules)
        assert "finish_time_s" in races.columns, where
        assert not RACE_LABELS & set(races.columns), where
        assert _by_race(races, "finish_time_s") == {
            r: race_finish(r)["finish_time_s"] if r in finished else None
            for r in open_races - {NO_RACE_ROW[0]}
        }, f"races.finish_time_s, {where}"  # fmt: skip

    # the same boundary through asof_join, one key per side of each race's post + 30 min
    timed = [rid for rid, at in FINISHED_AT.items() if at is not None]
    keys = pd.DataFrame(
        {
            "race_id": [rid for rid in timed for _ in (0, 1)],
            "horse_no": [2] * 2 * len(timed),
            "t_utc": pd.to_datetime(
                [t for rid in timed for t in (FINISHED_AT[rid] - US, FINISHED_AT[rid])], utc=True
            ),
        }
    )
    got = asof_join(keys, "runners", on=["race_id", "horse_no"], rules=rules, con=finish_con)
    assert not RUNNER_LABELS & set(got.columns)
    for column in POST_FINISH["runners"]:
        want = [v for rid in timed for v in (None, runner_finish(rid, 2)[column])]
        assert [_cell(v) for v in got[column]] == want, column
    assert [int(v) for v in got["draw"]] == [10 * _k(rid) + 2 for rid in timed for _ in (0, 1)]
    races = asof_join(
        keys[["race_id", "t_utc"]], "races", on=["race_id"], rules=rules, con=finish_con
    )
    assert not RACE_LABELS & set(races.columns)
    want = [v for rid in timed for v in (None, race_finish(rid)["finish_time_s"])]
    assert [_cell(v) for v in races["finish_time_s"]] == want


def test_a_race_without_a_post_time_never_shows_its_finish(rules, finish_con):
    # TOK:01 has a races row without post_time_local; NAK:03 has runners and no races row. Their
    # cards open at 00:00 JST; their finish never does.
    unknown = ["jra:2026-11-29:TOK:01", NO_RACE_ROW[0]]
    for t in (datetime(2026, 11, 29, 12, tzinfo=UTC), FAR_FUTURE):
        runners = read_asof(finish_con, "runners", t, rules=rules)
        assert set(POST_FINISH["runners"]) <= set(runners.columns), t.isoformat()
        for rid in unknown:
            mine = runners[runners["race_id"] == rid]
            assert sorted(int(h) for h in mine["horse_no"]) == list(FINISH_HORSES), rid
            for column in POST_FINISH["runners"]:
                assert [_cell(v) for v in mine[column]] == [None] * len(FINISH_HORSES), (
                    f"{rid} runners.{column} at {t.isoformat()}"
                )
        races = read_asof(finish_con, "races", t, rules=rules)
        row = races[races["race_id"] == unknown[0]]
        assert len(row) == 1 and _cell(row["finish_time_s"].iloc[0]) is None, t.isoformat()

    keys = pd.DataFrame({"race_id": unknown, "horse_no": [1, 3],
                         "t_utc": pd.to_datetime([FAR_FUTURE] * 2, utc=True)})  # fmt: skip
    got = asof_join(keys, "runners", on=["race_id", "horse_no"], rules=rules, con=finish_con)
    assert [int(v) for v in got["draw"]] == [10 * _k(unknown[0]) + 1, 10 * _k(unknown[1]) + 3]
    for column in POST_FINISH["runners"]:
        assert [_cell(v) for v in got[column]] == [None, None], column


# JRDB's day-of TYB columns (tyb_<field>) are not in schemas.TABLES: they come from licensed field
# specs. These stand in for them, with a made-up tyb_ field a future spec could add, beside KYI,
# CYB and JOA columns that stay in the view.
TYB_FIELDS = [
    ("tyb_body_weight", pa.int64()),
    ("tyb_body_weight_change", pa.int64()),
    ("tyb_win_odds", pa.float64()),
    ("tyb_place_odds", pa.float64()),
    ("tyb_odds_time", pa.string()),
    ("tyb_going_code", pa.string()),
    ("tyb_paddock_index", pa.float64()),
    ("tyb_future_field", pa.string()),
]
PRE_RACE_KEPT = [
    ("kyi_idm", pa.float64()),
    ("joa_base_odds", pa.float64()),
    ("cyb_training_index", pa.float64()),
    ("cyb_finish_mark", pa.string()),
]


def test_a_jrdb_pre_race_view_never_carries_the_day_of_tyb_columns(rules):
    # Two runners on 2026-11-28 (opens 2026-11-27 15:00 UTC). From 00:00 JST to long after the race
    # the view keeps the rows and the kyi_/joa_/cyb_ columns and drops every tyb_ column; the
    # stored rows still hold the tyb values.
    runners = {  # horse_no: (the TYB_FIELDS values, then the PRE_RACE_KEPT values)
        3: (486, 4, 3.4, 1.3, "1530", "11", 52.5, "x3", 61.0, 4.1, 55.5, "A"),
        7: (502, -2, 12.8, 3.6, "1532", "12", 47.0, "x7", 48.5, 15.2, 49.0, "B"),
    }
    extra = [*TYB_FIELDS, *PRE_RACE_KEPT]
    schema = _schema("jrdb_pre_race")
    for name, dtype in extra:
        schema = schema.append(pa.field(name, dtype))
    rows = []
    for horse_no, values in runners.items():
        row = {f.name: _value(f.type, f.name, 2) for f in _schema("jrdb_pre_race")}
        row |= {"race_id": "jra:2026-11-28:TOK:11", "horse_no": horse_no}
        row |= dict(zip([name for name, _ in extra], values, strict=True))
        rows.append(row)
    kept = set(schema.names) - {name for name, _ in TYB_FIELDS}
    one = duckdb.connect(":memory:")
    try:
        one.register("_arrow", pa.Table.from_pylist(rows, schema=schema))
        one.execute('CREATE TABLE "jrdb_pre_race" AS SELECT * FROM _arrow')
        one.unregister("_arrow")
        opens = datetime(2026, 11, 27, 15, tzinfo=UTC)
        assert read_asof(one, "jrdb_pre_race", opens - US, rules=rules).empty
        for t in (opens, datetime(2026, 11, 28, 6, 50, tzinfo=UTC), opens + timedelta(days=30)):
            view = read_asof(one, "jrdb_pre_race", t, rules=rules)
            leaked = sorted(c for c in view.columns if c.startswith("tyb_"))
            assert not leaked, f"jrdb_pre_race at {t.isoformat()} carries {leaked}"
            assert set(view.columns) == kept, t.isoformat()
            view = view.sort_values("horse_no")
            assert [int(v) for v in view["horse_no"]] == [3, 7], t.isoformat()
            assert list(view["kyi_idm"]) == [61.0, 48.5], t.isoformat()
            assert list(view["joa_base_odds"]) == [4.1, 15.2], t.isoformat()
            assert list(view["cyb_finish_mark"]) == ["A", "B"], t.isoformat()

            keys = pd.DataFrame({"race_id": ["jra:2026-11-28:TOK:11"] * 2, "horse_no": [7, 3],
                                 "t_utc": pd.to_datetime([t, t], utc=True)})  # fmt: skip
            got = asof_join(keys, "jrdb_pre_race", on=["race_id", "horse_no"], rules=rules, con=one)
            leaked = sorted(c for c in got.columns if c.startswith("tyb_"))
            assert not leaked, f"jrdb_pre_race joined at {t.isoformat()} carries {leaked}"
            assert set(got.columns) == kept | {"t_utc"}, t.isoformat()
            assert list(got["kyi_idm"]) == [48.5, 61.0], t.isoformat()
            assert list(got["cyb_training_index"]) == [49.0, 55.5], t.isoformat()
        stored = one.execute(
            "SELECT horse_no, tyb_body_weight, tyb_win_odds, tyb_going_code, tyb_future_field"
            ' FROM "jrdb_pre_race" ORDER BY horse_no'
        ).fetchall()
        assert stored == [(3, 486, 3.4, "11", "x3"), (7, 502, 12.8, "12", "x7")]
    finally:
        one.close()


@pytest.mark.parametrize("table", LABEL_ONLY)
def test_label_only_tables_never_enter_a_view_and_are_whole_as_labels(rules, con, table):
    keys = pd.DataFrame({"race_id": ["race_id-0"], "t_utc": pd.to_datetime([SCRAPED], utc=True)})
    with pytest.raises(LabelOnlyTable):
        read_asof(con, table, datetime(2100, 1, 1, tzinfo=UTC), rules=rules)
    with pytest.raises(LabelOnlyTable):
        asof_join(keys, table, on=["race_id"], rules=rules, con=con)
    labels = read_label(con, table, rules=rules)
    assert len(labels) == len(RACE_DATES) and set(labels.columns) == set(_schema(table).names)


def test_horses_hold_current_values_and_never_enter_a_view(rules, con):
    keys = pd.DataFrame({"horse_id": ["horse_id-0"], "t_utc": pd.to_datetime([SCRAPED], utc=True)})
    with pytest.raises(CurrentValueTable):
        read_asof(con, "horses", datetime(2100, 1, 1, tzinfo=UTC), rules=rules)
    with pytest.raises(CurrentValueTable):
        asof_join(keys, "horses", on=["horse_id"], rules=rules, con=con)
    with pytest.raises(ValueError):
        read_label(con, "horses", rules=rules)


def test_pedigree_names_are_available_at_any_t_without_the_scrape_columns(rules, con):
    kept = set(_schema("horse_pedigree").names) - PEDIGREE_LABELS
    for t in (datetime(1990, 1, 1, tzinfo=UTC), SCRAPED - US, datetime(2100, 1, 1, tzinfo=UTC)):
        view = read_asof(con, "horse_pedigree", t, rules=rules)
        assert set(view.columns) == kept
        assert sorted(view["horse_id"]) == [f"horse_id-{i}" for i in range(len(RACE_DATES))]
    assert {"sire_name", "dam_name", "dam_sire_name"} <= kept
    with pytest.raises(ValueError):
        read_label(con, "horse_pedigree", rules=rules)


def test_the_board_at_t_under_the_shipped_rule(rules, con):
    kept = set(ODDS_SCHEMA.names) - ODDS_LABELS

    def on_board(t: datetime) -> list[int]:
        view = read_asof(con, "odds_timeseries", t, rules=rules)
        assert set(view.columns) == kept
        return sorted(int(v.rsplit("-", 1)[1]) for v in view["venue"])

    def hm(h, m, s=0):
        return datetime(2026, 11, 28, h, m, s, tzinfo=UTC)

    assert on_board(hm(9, 0, 4)) == []
    assert on_board(hm(9, 0, 5)) == [0]
    assert on_board(hm(9, 10) - US) == [0]
    assert on_board(hm(9, 10)) == [0, 1]  # no fetched_utc: the snapshot counts
    assert on_board(hm(9, 25)) == [0, 1]  # snapshot 09:20 but fetched 09:30: not yet
    assert on_board(hm(9, 30)) == [0, 1, 2]
    assert on_board(hm(23, 59)) == [0, 1, 2, 4]  # both stamps null: never

    keys = pd.DataFrame(
        {
            "race_id": ["jra:2026-11-28:TOK:05"] * 4,
            "pool": ["WIN"] * 4,
            "combination": ["1"] * 4,
            "t_utc": pd.to_datetime([hm(9, 5), hm(9, 15), hm(9, 25), hm(9, 30)], utc=True),
        }
    )
    got = asof_join(
        keys, "odds_timeseries", on=["race_id", "pool", "combination"], rules=rules, con=con
    )
    assert not ODDS_LABELS & set(got.columns)
    assert list(got["venue"]) == ["venue-0", "venue-1", "venue-1", "venue-2"]


def test_a_table_outside_the_shipped_config_is_refused(rules, con):
    con.execute('CREATE TABLE IF NOT EXISTS "final_board" AS SELECT 1 AS row_id')
    with pytest.raises(NoAvailabilityRule):
        read_asof(con, "final_board", SCRAPED, rules=rules)


def _going_at(day: int, h: int, m: int) -> datetime:
    return datetime(2026, 11, day, h, m, tzinfo=UTC)


# (venue, surface, available_utc) of each track_condition row, all for race day 2026-11-28
TRACK_ROWS = [
    ("TOK", "turf", _going_at(27, 6, 0)),  # Tokyo's day-before page, 15:00 JST
    ("TOK", "turf", _going_at(28, 0, 30)),  # its race-morning update, 09:30 JST
    ("TOK", "turf", _going_at(28, 3, 40)),  # a mid-day update, 12:40 JST
    ("KYO", "turf", _going_at(27, 23, 0)),  # Kyoto's race-morning page, 08:00 JST
    ("TOK", "dirt", None),  # no publication time: never
]


def test_track_condition_rows_open_at_their_publication(rules):
    # The rule is on available_utc: each row fetched (ingested) 10 minutes after it was published
    # still opens at its publication. read_asof keeps the whole history; asof_join takes the
    # latest row of the key at t, so a mid-day update replaces the morning's from its 発表 time.
    schema = _schema("track_condition")
    rows = []
    for i, (venue, surface, available) in enumerate(TRACK_ROWS):
        ingested = None if available is None else available + timedelta(minutes=10)
        row = {f.name: _value(f.type, f.name, i) for f in schema}  # going is "going-<i>"
        row |= {"jurisdiction": "jra", "venue": venue, "race_date": date(2026, 11, 28),
                "surface": surface, "published_utc": available, "available_utc": available,
                "ingested_utc": ingested}  # fmt: skip
        rows.append(row)
    one = duckdb.connect(":memory:")
    try:
        one.register("_arrow", pa.Table.from_pylist(rows, schema=schema))
        one.execute('CREATE TABLE "track_condition" AS SELECT * FROM _arrow')
        one.unregister("_arrow")

        def shown(t: datetime) -> list[int]:
            view = read_asof(one, "track_condition", t, rules=rules)
            assert set(view.columns) == set(schema.names), t.isoformat()  # no label columns
            return sorted(int(v.rsplit("-", 1)[1]) for v in view["going"])

        day_before, morning, midday, kyoto = (TRACK_ROWS[i][2] for i in (0, 1, 2, 3))
        assert shown(day_before - US) == []
        assert shown(day_before) == [0]  # published at exactly t: usable
        assert shown(kyoto - US) == [0]
        assert shown(kyoto) == [0, 3]
        assert shown(morning - US) == [0, 3]
        assert shown(morning) == [0, 1, 3]
        assert shown(midday - US) == [0, 1, 3]  # the mid-day update is not out yet
        assert shown(midday) == [0, 1, 2, 3]
        assert shown(datetime(2027, 6, 1, tzinfo=UTC)) == [0, 1, 2, 3]  # a null time never opens

        times = [day_before - US, day_before, morning - US, morning, midday - US, midday]
        keys = pd.DataFrame(
            {
                "venue": ["TOK"] * len(times),
                "race_date": [date(2026, 11, 28)] * len(times),
                "surface": ["turf"] * len(times),
                "t_utc": pd.to_datetime(times, utc=True),
            }
        )
        got = asof_join(
            keys, "track_condition", on=["venue", "race_date", "surface"], rules=rules, con=one
        )
        assert [None if pd.isna(v) else v for v in got["going"]] == [
            None, "going-0", "going-0", "going-1", "going-1", "going-2",
        ]  # fmt: skip
        with pytest.raises(ValueError):
            read_label(one, "track_condition", rules=rules)
    finally:
        one.close()


# --- every view documents its availability rule ------------------------------------------------


def _documented(doc: str, table: str, needles: list[str]) -> bool:
    """A line names the table and each needle (in backticks), or a heading names the table and its
    section names each needle."""
    lines = doc.splitlines()
    name = f"`{table}`"
    if any(name in line and all(n in line for n in needles) for line in lines):
        return True
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") and table in line:
            section = [line]
            for nxt in lines[i + 1 :]:
                if nxt.lstrip().startswith("#"):
                    break
                section.append(nxt)
            if all(n in "\n".join(section) for n in needles):
                return True
    return False


@pytest.mark.parametrize("table", list(OWNER))
def test_the_spec_documents_each_shipped_rule(rules, table):
    doc = SPEC.read_text(encoding="utf-8")
    rule = rules[table]
    needles = [f"`{rule.kind}`"]
    for name in (rule.column, rule.anchor, rule.fallback):
        if name is not None:
            needles.append(f"`{name}`")
    if OWNER[table].get("in_view"):  # jockeys and trainers (#376)
        needles.append("`in_view`")
    assert _documented(doc, table, needles), f"{SPEC.name} does not document {table}: {needles}"
