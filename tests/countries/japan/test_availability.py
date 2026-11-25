"""Japan's shipped availability rules (ticket #80; the owner's answer on #80, 2026-11-24).

`load_availability("japan")` reads `src/horsetrack/countries/japan/config/availability.yaml`. The
owner accepted the proposed defaults, each conservative relative to JRA's publication:

1. odds_timeseries: column on fetched_utc, falling back to snapshot_utc (HKTrack rows have no
   fetched_utc); dividend, is_refund and is_final_capture are label columns.
2. meetings: offset on race_date, 0 (00:00 JST of the race date).
3. races and runners: offset on race_date, 0 (00:00 JST); the result columns, the horse weights
   and the same-day going are label columns (until a post_utc exists). races.result_published_utc
   and races.result_unpublished_utc are label columns too (the owner's answer on #80,
   2026-11-25); races.rail_setting stays in the view.
4. dividends, sectionals, incidents, running_comments: label_only.
5. horses: current_value. horse_pedigree: static (sire, dam and broodmare sire never change),
   with source_race_id and scraped_utc kept out as label columns.
6. horse_runs: offset on race_date + 1 day (00:00 JST of the next day).
7. jrdb_pre_race: offset on race_date, 0 (00:00 JST). 8. jrdb_post_race: label_only.
9. Nothing banned.

The expected rules below are written from that answer, never read back from the config. The
leakage tests run the shipped rules over small synthetic tables built from the store's column
contracts (`core.store.schemas`, `core.store.tables.ARROW_TYPES`) and the odds log's
(`countries/japan/logger/compact.py` SCHEMA).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
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
from horsetrack.countries.japan.logger.compact import SCHEMA as ODDS_SCHEMA

UTC = timezone.utc
US = timedelta(microseconds=1)
JST = "Asia/Tokyo"
SPEC = Path(__file__).resolve().parents[3] / "docs" / "specs" / "as-of.md"

RUNNER_LABELS = frozenset(
    {
        "finish_pos", "placing_raw", "placing_code", "dead_heat", "finish_time_s", "lbw_raw",
        "lbw_lengths", "running_positions", "last_3f_s", "win_odds", "win_fav_rank",
        "horse_wt_kg", "horse_wt_change_kg",
    }
)  # fmt: skip
# result_published_utc and result_unpublished_utc (#225): never in a view at t (the owner's answer
# on #80, 2026-11-25), until #227 may give them a rule of their own. rail_setting stays in.
RACE_LABELS = frozenset(
    {
        "finish_time_s", "split_times_s", "going", "going_raw", "result_published_utc",
        "result_unpublished_utc",
    }
)  # fmt: skip
ODDS_LABELS = frozenset({"dividend", "is_refund", "is_final_capture"})
PEDIGREE_LABELS = frozenset({"source_race_id", "scraped_utc"})


def _offset(minutes: int, labels=frozenset()) -> dict:
    return {"kind": "offset", "anchor": "race_date", "offset_min": minutes, "tz": JST,
            "labels": labels}  # fmt: skip


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
    "dividends": {"kind": "label_only"},
    "sectionals": {"kind": "label_only"},
    "incidents": {"kind": "label_only"},
    "running_comments": {"kind": "label_only"},
    "horses": {"kind": "current_value"},
    "horse_pedigree": {"kind": "static", "labels": PEDIGREE_LABELS},
    "horse_runs": _offset(1440),
    "jrdb_pre_race": _offset(0),
    "jrdb_post_race": {"kind": "label_only"},
}
VIEW_TABLES = [t for t, o in OWNER.items() if o["kind"] == "offset"]
LABEL_ONLY = [t for t, o in OWNER.items() if o["kind"] == "label_only"]


def _schema(table: str) -> pa.Schema:
    if table == "odds_timeseries":
        return ODDS_SCHEMA
    spec = schemas.TABLES[table]
    return pa.schema([(c, ARROW_TYPES[k]) for c, k in spec.columns.items()])


@pytest.fixture(scope="module")
def rules():
    return load_availability("japan")


# --- the shipped config is the owner's answer ---------------------------------------------------


def test_the_shipped_config_has_one_rule_per_table_and_nothing_else(rules):
    assert len(schemas.TABLES) == 12
    assert set(OWNER) == set(schemas.TABLES) | {"odds_timeseries"}
    assert set(rules) == set(OWNER)
    for table, rule in rules.items():
        assert rule.table == table


@pytest.mark.parametrize("table", list(OWNER))
def test_each_shipped_rule_is_the_owners(rules, table):
    want, rule = OWNER[table], rules[table]
    assert rule.kind == want["kind"]
    assert rule.column == want.get("column")
    assert rule.fallback == want.get("fallback")
    assert rule.anchor == want.get("anchor")
    minutes = want.get("offset_min")
    assert rule.offset == (None if minutes is None else timedelta(minutes=minutes))
    assert rule.tz == want.get("tz")
    assert set(rule.label_columns) == set(want.get("labels", ()))
    assert len(rule.label_columns) == len(set(rule.label_columns))  # no repeats


@pytest.mark.parametrize("table", list(OWNER))
def test_every_column_a_shipped_rule_names_is_in_its_table(rules, table):
    schema = _schema(table)
    rule = rules[table]
    missing = set(rule.label_columns) - set(schema.names)
    assert not missing, f"{table}: label columns not in the schema: {sorted(missing)}"
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
    if pa.types.is_string(dtype):
        return f"{column}-{i}"
    if pa.types.is_boolean(dtype):
        return i % 2 == 0
    if pa.types.is_date(dtype):
        return RACE_DATES[i]
    if pa.types.is_timestamp(dtype):
        return SCRAPED
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
    kept = set(_schema(table).names) - set(OWNER[table]["labels"])
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


def test_a_runner_or_race_joined_at_t_carries_its_card_and_never_its_result(rules, con):
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

    races = asof_join(keys[["race_id", "t_utc"]], "races", on=["race_id"], rules=rules, con=con)
    assert not RACE_LABELS & set(races.columns)
    assert [None if pd.isna(v) else int(v) for v in races["distance_m"]] == [None, 3, 1]


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
    assert _documented(doc, table, needles), f"{SPEC.name} does not document {table}: {needles}"
