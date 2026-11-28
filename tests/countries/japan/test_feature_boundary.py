"""The model boundary: only explicitly approved columns enter a feature frame (ticket #417).

Contract pinned here (`horsetrack.countries.japan.feature_boundary`, new in #417):

- `approved_features(table, *, source="jra", rules=None, inputs=None) -> frozenset[str]`: the
  columns of `table` approved as features at t for `source`. A column is approved only when the
  input inventory (`config/model_inputs.yaml`, `research.coverage.load_inputs`; `inputs` is such a
  list, default the shipped inventory) names it in a supply `{source, table, column}` of an input
  whose role is `feature_at_t`, AND the availability rules (`core.store.asof.load_availability`;
  `rules`, default the shipped Japan rules) let the table into the views (not `label_only`, not
  `current_value` without `in_view`) and do not make it a label column (`Availability.is_label`).
  Nothing else is approved: no hand-kept list in code.
- `feature_view(con, table, at_utc, *, features, keys=(), source="jra", rules=None, inputs=None)
  -> pd.DataFrame`: the as-of view of `table` at `at_utc` (`core.store.asof.read_asof`: rows,
  gates and label exclusions unchanged) with exactly the columns `[*keys, *features]`, in that
  order. `keys` are the caller's identifiers (race_id, horse_no, ...), returned as they are; a key
  that is a label column of the rule is refused (a ValueError). Every name in `features` must be
  approved, or nothing is returned:
  - `UnclassifiedColumn` (a `FeatureNotApproved`) when neither config names the column: not in any
    inventory supply of that source and table, and not a label column of the rule. A new column
    added to a known table is in this case until an owner-approved config change classifies it.
  - `FeatureNotApproved` (a `ValueError`) for a classified column that is not a feature at t: a
    label column of the rule, or a column the inventory supplies only as a label or a conditioner.
  The message names every refused column as `table.column`.
- `approximations(table, columns, *, rules=None) -> dict[str, frozenset[str]]`: per column, the
  documented approximations its availability carries, read from the availability rules only:
  `RESULTS_DERIVED` for the accepted leak of docs/specs/as-of.md ("Accepted leak: results-derived
  card columns": runners.is_scratched, races.field_size, races.status, meetings.status, taken from
  the results page and in the views from 00:00 JST), `PUBLICATION_PROXY` for a gated column (in
  the view from an anchor + offset, such as post_utc - 60 min, standing in for its unrecorded
  publication time); an empty set for a column with neither (a card value known from the
  race-date rule, or a column with its own publication time, such as odds_timeseries' fetched_utc
  or track_condition's available_utc).
- `availability_fallback(table, *, rules=None) -> tuple[str, str] | None`: `(column, fallback)` of
  a `column` rule with a `fallback` (a row whose `column` is null is in the views at its
  `fallback`, the `TIMESTAMP_FALLBACK` approximation), read from the rules: under the shipped
  ones odds_timeseries' `("fetched_utc", "snapshot_utc")`; None for any other rule (an offset,
  static, current_value or label_only rule, or a column rule without a fallback).
- Authorized label reads are unchanged: `core.store.asof.read_label` still returns a
  `read_label: true` table whole.

Expected values are computed here from first principles: rows open at 00:00 JST of their race
date, the horse weights at the race's post (Asia/Tokyo wall clock) - 60 min, a board row at its
fetched_utc. Every table is synthetic and in memory; no date is a Japan holdout day (on or after
2026-11-28).
"""

# Ruff reads the module under test as third-party until it exists, then as first-party; the
# block is written in its final order and I001 is waived so lint passes before and after.
from __future__ import annotations  # noqa: I001

import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
import yaml

from horsetrack.core.store import schemas
from horsetrack.core.store.asof import load_availability, read_label
from horsetrack.core.store.tables import ARROW_TYPES
from horsetrack.countries.japan.feature_boundary import (
    PUBLICATION_PROXY,
    RESULTS_DERIVED,
    TIMESTAMP_FALLBACK,
    FeatureNotApproved,
    UnclassifiedColumn,
    approved_features,
    approximations,
    availability_fallback,
    feature_view,
)
from horsetrack.countries.japan.logger.compact import SCHEMA as ODDS_SCHEMA
from horsetrack.countries.japan.research.coverage import load_inputs

UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
US = timedelta(microseconds=1)
AVAILABILITY = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "horsetrack"
    / "countries"
    / "japan"
    / "config"
    / "availability.yaml"
)

# ---------------------------------------------------------------------------------------------
# A synthetic store: three races on two development days, two runners each, a board, dividends
# ---------------------------------------------------------------------------------------------

D1, D2 = date(2026, 6, 6), date(2026, 6, 7)
# race_id -> (race_date, venue, race_no, post_time_local JST, distance_m, field_size, going)
RACES = {
    "jra:2026-06-06:TOK:01": (D1, "TOK", 1, "10:05", 1200, 14, "good"),
    "jra:2026-06-07:TOK:11": (D2, "TOK", 11, "15:40", 2400, 16, "firm"),
    "jra:2026-06-07:KYO:02": (D2, "KYO", 2, "11:00", 1800, 10, "soft"),
}
# (race_id, horse_no) -> (draw, age, horse_wt_kg)
RUNNERS = {
    ("jra:2026-06-06:TOK:01", 1): (3, 3, 480),
    ("jra:2026-06-06:TOK:01", 2): (7, 4, 502),
    ("jra:2026-06-07:TOK:11", 1): (1, 5, 466),
    ("jra:2026-06-07:TOK:11", 2): (9, 3, 510),
    ("jra:2026-06-07:KYO:02", 1): (4, 2, 440),
    ("jra:2026-06-07:KYO:02", 2): (2, 6, 498),
}
BOARD_RACE = "jra:2026-06-07:TOK:11"
# (combination, fetched_utc, odds): WIN captures of BOARD_RACE
BOARD = [
    ("1", datetime(2026, 6, 7, 5, 0, 4, tzinfo=UTC), 3.4),
    ("2", datetime(2026, 6, 7, 5, 0, 4, tzinfo=UTC), 5.1),
    ("1", datetime(2026, 6, 7, 6, 10, 2, tzinfo=UTC), 2.9),
    ("2", datetime(2026, 6, 7, 6, 10, 2, tzinfo=UTC), 6.3),
]
# (race_id, pool, combination, dividend)
DIVIDENDS = [
    ("jra:2026-06-06:TOK:01", "WIN", "2", 510.0),
    ("jra:2026-06-07:TOK:11", "WIN", "1", 290.0),
    ("jra:2026-06-07:KYO:02", "PLACE", "1", 140.0),
]

# Probe times: before the second day opens, the second day's morning, before anything opens.
T_DAY1 = datetime(2026, 6, 6, 15, 0, tzinfo=UTC) - US  # 23:59:59.999999 JST on 2026-06-06
T_DAY2 = datetime(2026, 6, 7, 1, 30, tzinfo=UTC)  # 10:30 JST on 2026-06-07
T_NONE = datetime(2026, 6, 5, 14, 0, tzinfo=UTC)  # 23:00 JST on 2026-06-05
T_BOARD = datetime(2026, 6, 7, 6, 0, tzinfo=UTC)
PROBES = [T_NONE, T_DAY1, T_DAY2, T_BOARD]

# New columns a later schema could add, one per kind of value.
NEW_COLUMNS = [
    ("speed_figure", pa.float64(), lambda i: 80.5 + i),
    ("is_favourite", pa.bool_(), lambda i: i % 2 == 0),
    ("trip_note", pa.string(), lambda i: f"note-{i}"),
]


def _opens(day: date) -> datetime:
    """00:00 JST of `day`, as UTC: when a card row is available (availability.yaml)."""
    return datetime.combine(day, time(0), tzinfo=JST).astimezone(UTC)


def _post(race_id: str) -> datetime:
    day, _, _, clock, *_ = RACES[race_id]
    hh, mm = (int(x) for x in clock.split(":"))
    return datetime.combine(day, time(hh, mm), tzinfo=JST).astimezone(UTC)


def _store_schema(table: str) -> pa.Schema:
    if table == "odds_timeseries":
        return ODDS_SCHEMA
    return pa.schema([(c, ARROW_TYPES[k]) for c, k in schemas.TABLES[table].columns.items()])


def _rows(table: str) -> list[dict]:
    if table == "meetings":
        meetings = sorted({(r[0], r[1]) for r in RACES.values()})
        return [
            {"meeting_id": f"jra:{d.isoformat()}:{v}", "race_date": d, "venue": v, "status": "run"}
            for d, v in meetings
        ]
    if table == "races":
        return [
            {
                "jurisdiction": "jra",
                "race_id": race_id,
                "meeting_id": f"jra:{d.isoformat()}:{venue}",
                "race_date": d,
                "venue": venue,
                "race_no": no,
                "post_time_local": clock,
                "distance_m": dist,
                "surface": "turf",
                "field_size": field,
                "status": "run",
                "going": going,
                "race_name": f"race-{no}",
            }
            for race_id, (d, venue, no, clock, dist, field, going) in RACES.items()
        ]
    if table == "runners":
        return [
            {
                "jurisdiction": "jra",
                "race_id": race_id,
                "race_date": RACES[race_id][0],
                "venue": RACES[race_id][1],
                "race_no": RACES[race_id][2],
                "horse_no": horse_no,
                "horse_id": f"h-{race_id[-6:]}-{horse_no}",
                "horse_name": f"horse-{horse_no}",
                "draw": draw,
                "age": age,
                "horse_wt_kg": wt,
                "placing_raw": str(horse_no),
                "win_odds": 2.5 * horse_no,
                "last_3f_s": 34.0 + horse_no,
                "is_scratched": False,
            }
            for (race_id, horse_no), (draw, age, wt) in RUNNERS.items()
        ]
    if table == "odds_timeseries":
        return [
            {
                "race_id": BOARD_RACE,
                "venue": "TOK",
                "race_no": 11,
                "pool": "WIN",
                "combination": combination,
                "snapshot_utc": fetched - timedelta(seconds=4),
                "fetched_utc": fetched,
                "post_utc": _post(BOARD_RACE),
                "odds": odds,
                "investment": 1000.0,
                "status_raw": "",
                "is_final_capture": False,
                "dividend": 290.0,
                "is_refund": False,
            }
            for combination, fetched, odds in BOARD
        ]
    if table == "dividends":
        return [
            {
                "jurisdiction": "jra",
                "race_id": r,
                "pool": p,
                "combination": c,
                "dividend": d,
                "is_refund": False,
            }
            for r, p, c, d in DIVIDENDS
        ]
    raise AssertionError(table)


def _arrow(table: str, new: tuple | None) -> pa.Table:
    schema = _store_schema(table)
    rows = [{f.name: row.get(f.name) for f in schema} for row in _rows(table)]
    data = pa.Table.from_pylist(rows, schema=schema)
    if new is not None:
        name, dtype, value = new
        data = data.append_column(pa.field(name, dtype), pa.array(
            [value(i) for i in range(len(rows))], type=dtype))  # fmt: skip
    return data


def _con(added: dict[str, tuple] | None = None) -> duckdb.DuckDBPyConnection:
    """The synthetic store; `added` maps a table to a new column (name, type, value of row i)."""
    added = added or {}
    con = duckdb.connect(":memory:")
    for table in ("meetings", "races", "runners", "odds_timeseries", "dividends"):
        con.register("_arrow", _arrow(table, added.get(table)))
        con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _arrow')
        con.unregister("_arrow")
    return con


def _cell(v):
    """A plain, comparable value: None for null, UTC ISO text for a time, int for a whole float."""
    if isinstance(v, list | tuple):
        return v
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, datetime | pd.Timestamp):
        return pd.Timestamp(v).tz_convert("UTC").isoformat()
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _tuples(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    rows = frame[columns].itertuples(index=False)
    return sorted((tuple(_cell(v) for v in row) for row in rows), key=repr)


def _expected(table: str, t: datetime, columns: list[str]) -> list[tuple]:
    """The rows of `table` known at t, from first principles (see the module docstring)."""
    out = []
    if table == "races":
        for race_id, (d, venue, no, clock, dist, field, _going) in RACES.items():
            if _opens(d) <= t:
                row = {"race_id": race_id, "venue": venue, "race_no": no, "distance_m": dist,
                       "field_size": field, "post_time_local": clock}  # fmt: skip
                out.append(tuple(_cell(row[c]) for c in columns))
    elif table == "runners":
        for (race_id, horse_no), (draw, age, wt) in RUNNERS.items():
            if _opens(RACES[race_id][0]) <= t:
                shown = wt if _post(race_id) - timedelta(minutes=60) <= t else None
                row = {"race_id": race_id, "horse_no": horse_no, "draw": draw, "age": age,
                       "horse_wt_kg": shown}  # fmt: skip
                out.append(tuple(_cell(row[c]) for c in columns))
    elif table == "odds_timeseries":
        for combination, fetched, odds in BOARD:
            if fetched <= t:
                row = {"race_id": BOARD_RACE, "pool": "WIN", "combination": combination,
                       "fetched_utc": fetched, "odds": odds}  # fmt: skip
                out.append(tuple(_cell(row[c]) for c in columns))
    else:
        raise AssertionError(table)
    return sorted(out, key=repr)


@pytest.fixture(scope="module")
def rules():
    return load_availability("japan")


# (table, keys, a known feature at t)
KNOWN = [
    ("races", ["race_id"], "distance_m"),
    ("runners", ["race_id", "horse_no"], "draw"),
    ("odds_timeseries", ["race_id", "pool", "combination", "fetched_utc"], "odds"),
]


# ---------------------------------------------------------------------------------------------
# The ticket's test: a synthetic schema addition is rejected; known features and label reads work
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("new", NEW_COLUMNS, ids=[n[0] for n in NEW_COLUMNS])
@pytest.mark.parametrize("table,keys,known", KNOWN, ids=[k[0] for k in KNOWN])
def test_a_schema_addition_is_rejected_while_known_features_and_label_reads_work(
    rules, table, keys, known, new
):
    name = new[0]
    con = _con({table: new, "dividends": new})
    try:
        # the new column is no approved feature, whatever its values
        assert name not in approved_features(table, rules=rules)
        for features in ([name], [known, name], [name, known]):
            with pytest.raises(UnclassifiedColumn, match=re.escape(f"{table}.{name}")):
                feature_view(con, table, T_BOARD, features=features, keys=keys, rules=rules)

        # the known feature still passes, with exactly the columns asked for
        for t in PROBES:
            got = feature_view(con, table, t, features=[known], keys=keys, rules=rules)
            assert list(got.columns) == [*keys, known], t
            assert _tuples(got, [*keys, known]) == _expected(table, t, [*keys, known]), t

        # an authorized label read still returns the label table whole, the new column included
        labels = read_label(con, "dividends", rules=rules)
        assert name in labels.columns
        got = _tuples(labels, ["race_id", "pool", "combination", "dividend"])
        assert got == sorted(DIVIDENDS, key=repr)
    finally:
        con.close()


def test_a_new_meetings_column_is_rejected_too(rules):
    con = _con({"meetings": NEW_COLUMNS[0]})
    try:
        with pytest.raises(UnclassifiedColumn, match=re.escape("meetings.speed_figure")):
            feature_view(
                con, "meetings", T_DAY2, features=["speed_figure"], keys=["meeting_id"], rules=rules
            )
    finally:
        con.close()


# ---------------------------------------------------------------------------------------------
# feature_view: the view at t of the requested columns, and nothing else
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table,keys,features",
    [
        ("runners", ["race_id", "horse_no"], ["draw", "age", "horse_wt_kg"]),
        ("runners", ["race_id"], ["horse_wt_kg"]),
        ("runners", ["horse_no", "race_id"], ["age", "draw"]),
        ("races", ["race_id"], ["distance_m", "field_size", "post_time_local"]),
        ("races", ["race_id", "race_no"], ["venue"]),
        ("odds_timeseries", ["race_id", "pool", "combination", "fetched_utc"], ["odds"]),
    ],
)
def test_feature_view_is_the_view_at_t_of_the_requested_columns(rules, table, keys, features):
    con = _con()
    try:
        for t in PROBES:
            got = feature_view(con, table, t, features=features, keys=keys, rules=rules)
            assert list(got.columns) == [*keys, *features], t
            assert _tuples(got, [*keys, *features]) == _expected(table, t, [*keys, *features]), t
    finally:
        con.close()


def test_one_refused_feature_refuses_the_whole_request_and_names_each_column(rules):
    con = _con({"runners": NEW_COLUMNS[0]})
    try:
        with pytest.raises(FeatureNotApproved) as caught:
            feature_view(
                con,
                "runners",
                T_DAY2,
                features=["draw", "speed_figure", "horse_name"],
                keys=["race_id"],
                rules=rules,
            )
        message = str(caught.value)
        assert "runners.speed_figure" in message and "runners.horse_name" in message
        assert "runners.draw" not in message
    finally:
        con.close()


# Columns the configs classify, but not as features at t: availability label columns, inputs
# supplied only as a label or a conditioner.
NOT_FEATURES = [
    ("runners", "placing_raw"),  # a label column of the rule, in no feature_at_t supply
    ("runners", "last_3f_s"),  # a feature_at_t supply (last_3f), but a label column of the rule
    ("runners", "win_odds"),  # the final price: a conditioner and a label column
    ("runners", "finish_time_s"),  # in the view after post + 30, but the inventory's label
    ("races", "split_times_s"),  # lap_times is feature_at_t, the rule keeps it a label
    ("races", "result_published_utc"),  # a conditioner and a label column
    ("odds_timeseries", "dividend"),
    ("odds_timeseries", "is_final_capture"),
]
# Columns of the shipped store schemas that neither config names.
UNCLASSIFIED = [
    ("runners", "horse_name"),
    ("runners", "jockey_name"),
    ("races", "race_name"),
    ("races", "prize"),
    ("odds_timeseries", "status_raw"),
]


@pytest.mark.parametrize("table,column", NOT_FEATURES)
def test_a_classified_column_that_is_not_a_feature_at_t_is_refused(rules, table, column):
    con = _con()
    try:
        with pytest.raises(FeatureNotApproved, match=re.escape(f"{table}.{column}")) as caught:
            feature_view(con, table, T_BOARD, features=[column], keys=["race_id"], rules=rules)
        assert not isinstance(caught.value, UnclassifiedColumn), f"{table}.{column} is classified"
        assert issubclass(FeatureNotApproved, ValueError)
    finally:
        con.close()


@pytest.mark.parametrize("table,column", UNCLASSIFIED)
def test_a_store_column_no_config_names_is_unclassified(rules, table, column):
    con = _con()
    try:
        with pytest.raises(UnclassifiedColumn, match=re.escape(f"{table}.{column}")):
            feature_view(con, table, T_BOARD, features=[column], keys=["race_id"], rules=rules)
        assert issubclass(UnclassifiedColumn, FeatureNotApproved)
    finally:
        con.close()


def test_a_label_column_cannot_enter_as_a_key(rules):
    con = _con()
    try:
        for keys in (["race_id", "placing_raw"], ["win_odds"]):
            with pytest.raises(ValueError):
                feature_view(con, "runners", T_BOARD, features=["draw"], keys=keys, rules=rules)
    finally:
        con.close()


# ---------------------------------------------------------------------------------------------
# approved_features: the shipped inventory and availability rules, per source and table
# ---------------------------------------------------------------------------------------------

APPROVED = [
    (
        "jra",
        "runners",
        {"horse_no", "horse_id", "jockey_code", "trainer_code", "age", "sex", "actual_wt_kg",
         "horse_wt_kg", "horse_wt_change_kg", "draw", "bracket", "is_scratched", "finish_pos"},
        {"placing_raw", "last_3f_s", "lbw_lengths", "running_positions", "win_odds",
         "finish_time_s", "horse_name", "jockey_name", "trainer_name"},
    ),
    (
        "jra",
        "races",
        {"race_id", "post_time_local", "field_size", "venue", "surface", "distance_m",
         "course_config", "rail_setting", "race_class", "going", "weight_type", "race_no"},
        {"split_times_s", "finish_time_s", "result_published_utc", "result_unpublished_utc",
         "race_name", "prize", "rating_hi", "jurisdiction"},
    ),
    (
        "jra",
        "odds_timeseries",
        {"odds", "odds_hi", "investment", "minutes_to_post"},
        {"dividend", "is_refund", "is_final_capture", "status_raw", "venue", "fetched_utc"},
    ),
    (
        "jra",
        "track_condition",
        {"going", "cushion", "moisture_goal", "moisture_4c", "rail"},
        {"published_utc", "ingested_utc"},
    ),
    ("jravan", "track_condition", set(), {"going", "cushion", "rail"}),  # no jravan supply
    ("jra", "horse_pedigree", {"sire_id", "dam_id", "dam_sire_id"}, {"sire_name", "scraped_utc"}),
    ("jravan", "jockeys", {"jockey_code"}, set()),  # current values with in_view: in the views
    ("jra", "horses", set(), {"horse_id"}),  # current values without in_view: never in a view
    ("jravan", "horses", set(), {"horse_id"}),
    ("jra", "dividends", set(), {"dividend"}),  # in the views after the post, but a label input
    ("jra", "meetings", set(), {"race_date", "meeting_id"}),  # conditioners only
]  # fmt: skip


@pytest.mark.parametrize(
    "source,table,included,excluded", APPROVED, ids=[f"{a[0]}-{a[1]}" for a in APPROVED]
)
def test_approved_features_follow_the_shipped_inventory_and_rules(
    rules, source, table, included, excluded
):
    got = approved_features(table, source=source, rules=rules)
    assert isinstance(got, frozenset)
    assert included <= got, sorted(included - got)
    assert not excluded & got, sorted(excluded & got)
    if not included:
        assert got == frozenset()


def test_approved_features_default_to_the_shipped_configs(rules):
    for source, table, *_ in APPROVED:
        assert approved_features(table, source=source) == approved_features(
            table, source=source, rules=rules, inputs=load_inputs()
        )


def test_every_approved_feature_is_named_by_a_feature_at_t_supply_and_is_no_label(rules):
    inputs = load_inputs()
    for source in ("jra", "jravan"):
        for table, rule in rules.items():
            supplied = {
                s.column
                for i in inputs
                if i.role == "feature_at_t"
                for s in i.supplies
                if s.source == source and s.table == table
            }
            got = approved_features(table, source=source, rules=rules)
            assert got <= supplied, (source, table, sorted(got - supplied))
            assert not any(rule.is_label(c) for c in got), (source, table)
            if not rule.enters_view:
                assert got == frozenset(), (source, table)


# ---------------------------------------------------------------------------------------------
# The approval is read from the configs: change the config, and the boundary follows
# ---------------------------------------------------------------------------------------------


def _inventory(tmp_path: Path, name: str, role: str, supplies: list[dict]) -> list:
    path = tmp_path / f"model_inputs_{name}_{role}.yaml"
    raw = {
        "inputs": {
            name: {"layers": ["C2"], "role": role, "supplies": supplies, "tickets": [417]},
            "distance": {
                "layers": ["C3"],
                "role": "feature_at_t",
                "supplies": [{"source": "jra", "table": "races", "column": "distance_m"}],
                "tickets": [],
            },
        }
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_inputs(path)


def _availability(tmp_path: Path, edit) -> dict:
    raw = yaml.safe_load(AVAILABILITY.read_text(encoding="utf-8"))
    edit(raw["tables"])
    path = tmp_path / "availability.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_availability("japan", path=path)


@pytest.mark.parametrize("new", NEW_COLUMNS, ids=[n[0] for n in NEW_COLUMNS])
def test_an_inventory_entry_approves_a_new_column_and_its_role_and_source_count(
    rules, tmp_path, new
):
    name, _, value = new
    supply = {"source": "jra", "table": "races", "column": name}
    con = _con({"races": new})
    try:
        approving = _inventory(tmp_path, "synthetic_signal", "feature_at_t", [supply])
        assert approved_features("races", rules=rules, inputs=approving) == frozenset(
            {name, "distance_m"}
        )
        got = feature_view(
            con, "races", T_DAY2, features=[name], keys=["race_id"], rules=rules, inputs=approving
        )
        order = list(RACES)
        assert _tuples(got, ["race_id", name]) == sorted(
            ((r, _cell(value(order.index(r)))) for r in order), key=repr
        )
        # the shipped inventory does not name it
        with pytest.raises(UnclassifiedColumn, match=re.escape(f"races.{name}")):
            feature_view(con, "races", T_DAY2, features=[name], keys=["race_id"], rules=rules)

        # supplied as a label: classified, but not a feature at t
        as_label = _inventory(tmp_path, "synthetic_signal", "label", [supply])
        assert approved_features("races", rules=rules, inputs=as_label) == {"distance_m"}
        with pytest.raises(FeatureNotApproved) as caught:
            feature_view(
                con, "races", T_DAY2, features=[name], keys=["race_id"], rules=rules,
                inputs=as_label,
            )  # fmt: skip
        assert not isinstance(caught.value, UnclassifiedColumn)

        # supplied by the other source only: approved for that source, not this one
        jravan = supply | {"source": "jravan"}
        other = _inventory(tmp_path, "synthetic_signal", "feature_at_t", [jravan])
        assert name not in approved_features("races", source="jra", rules=rules, inputs=other)
        assert name in approved_features("races", source="jravan", rules=rules, inputs=other)
    finally:
        con.close()


def test_an_availability_label_column_overrides_an_inventory_feature(tmp_path):
    name = "speed_figure"
    supply = {"source": "jra", "table": "races", "column": name}
    approving = _inventory(tmp_path, "synthetic_signal", "feature_at_t", [supply])

    def label_it(tables):
        tables["races"]["label_columns"] = [*tables["races"]["label_columns"], name]

    labelled = _availability(tmp_path, label_it)
    assert approved_features("races", rules=labelled, inputs=approving) == {"distance_m"}
    con = _con({"races": NEW_COLUMNS[0]})
    try:
        with pytest.raises(FeatureNotApproved, match=re.escape(f"races.{name}")):
            feature_view(
                con, "races", T_DAY2, features=[name], keys=["race_id"], rules=labelled,
                inputs=approving,
            )  # fmt: skip
    finally:
        con.close()

    def label_only(tables):
        tables["races"] = {"kind": "label_only"}

    gone = _availability(tmp_path, label_only)
    assert approved_features("races", rules=gone, inputs=approving) == frozenset()


# ---------------------------------------------------------------------------------------------
# approximations: documented approximations stay identifiable
# ---------------------------------------------------------------------------------------------

# (table, column, the approximation it must carry, or None for none at all)
FLAGS = [
    ("runners", "is_scratched", RESULTS_DERIVED),
    ("races", "field_size", RESULTS_DERIVED),
    ("races", "status", RESULTS_DERIVED),
    ("meetings", "status", RESULTS_DERIVED),
    ("runners", "horse_wt_kg", PUBLICATION_PROXY),
    ("runners", "horse_wt_change_kg", PUBLICATION_PROXY),
    ("runners", "finish_pos", PUBLICATION_PROXY),
    ("races", "going", PUBLICATION_PROXY),
    ("runners", "draw", None),
    ("runners", "age", None),
    ("runners", "actual_wt_kg", None),
    ("runners", "jockey_code", None),
    ("races", "distance_m", None),
    ("races", "surface", None),
    ("races", "venue", None),
    ("meetings", "venue", None),
    ("odds_timeseries", "odds", None),
    ("odds_timeseries", "investment", None),
    ("track_condition", "going", None),
    ("track_condition", "cushion", None),
]


def test_results_derived_and_publication_proxy_are_distinct_tags():
    assert RESULTS_DERIVED != PUBLICATION_PROXY


@pytest.mark.parametrize("table,column,flag", FLAGS, ids=[f"{t}.{c}" for t, c, _ in FLAGS])
def test_approximations_flag_the_documented_ones_under_the_shipped_rules(
    rules, table, column, flag
):
    got = approximations(table, [column], rules=rules)
    assert set(got) == {column}
    assert isinstance(got[column], frozenset)
    if flag is None:
        assert got[column] == frozenset(), f"{table}.{column}: {sorted(got[column])}"
    else:
        assert flag in got[column], f"{table}.{column}: {sorted(got[column])}"


def test_approximations_cover_every_column_asked_for(rules):
    columns = ["draw", "is_scratched", "horse_wt_kg", "age"]
    got = approximations("runners", columns, rules=rules)
    assert set(got) == set(columns)
    assert got["draw"] == got["age"] == frozenset()
    assert RESULTS_DERIVED in got["is_scratched"] and PUBLICATION_PROXY in got["horse_wt_kg"]


def test_approximations_are_read_from_the_availability_rules(tmp_path):
    # A gate on distance_m makes it a publication proxy; going without its gate is none.
    def regate(tables):
        races = tables["races"]
        gates = [g for g in races["gated_columns"] if "going" not in g["columns"]]
        races["gated_columns"] = [
            *gates,
            {"columns": ["distance_m"], "anchor": "post_utc", "offset_min": -60},
        ]

    got = approximations("races", ["distance_m", "going"], rules=_availability(tmp_path, regate))
    assert PUBLICATION_PROXY in got["distance_m"]
    assert PUBLICATION_PROXY not in got["going"]

    # The accepted leak follows the rules' own column names, not a list in code.
    renamed = tmp_path / "renamed.yaml"
    renamed.write_text(
        AVAILABILITY.read_text(encoding="utf-8").replace("is_scratched", "is_withdrawn"),
        encoding="utf-8",
    )
    got = approximations(
        "runners", ["is_withdrawn", "is_scratched"], rules=load_availability("japan", renamed)
    )
    assert RESULTS_DERIVED in got["is_withdrawn"]
    assert RESULTS_DERIVED not in got["is_scratched"]


# ---------------------------------------------------------------------------------------------
# availability_fallback: a column rule's stand-in timestamp stays identifiable
# ---------------------------------------------------------------------------------------------

# (table, its (column, fallback) under the shipped rules, or None): the board rule has the only
# fallback; column rules without one, and offset, static, current_value and label_only rules
SHIPPED_FALLBACKS = [
    ("odds_timeseries", ("fetched_utc", "snapshot_utc")),
    ("track_condition", None),
    ("win_sales", None),
    ("races", None),
    ("runners", None),
    ("meetings", None),
    ("dividends", None),
    ("horse_runs", None),
    ("horse_pedigree", None),
    ("horses", None),
    ("jrdb_post_race", None),
]


def test_timestamp_fallback_is_a_tag_of_its_own():
    assert TIMESTAMP_FALLBACK not in {RESULTS_DERIVED, PUBLICATION_PROXY}


@pytest.mark.parametrize("table,expected", SHIPPED_FALLBACKS, ids=[t for t, _ in SHIPPED_FALLBACKS])
def test_availability_fallback_under_the_shipped_rules(rules, table, expected):
    assert availability_fallback(table, rules=rules) == expected
    assert availability_fallback(table) == expected  # the default is the shipped rules


# (name, edit of the shipped tables, odds_timeseries' and track_condition's expected fallbacks)
FALLBACK_EDITS = [
    (
        "offset-board",
        lambda tables: tables.update(
            odds_timeseries={
                "kind": "offset",
                "anchor": "post_utc",
                "offset_min": -5,
                "label_columns": ["dividend", "is_refund", "is_final_capture"],
            }
        ),
        None,
        None,
    ),
    (
        "no-fallback",
        lambda tables: tables["odds_timeseries"].pop("fallback"),
        None,
        None,
    ),
    (
        "renamed-fallback",
        lambda tables: tables["odds_timeseries"].update(fallback="logged_utc"),
        ("fetched_utc", "logged_utc"),
        None,
    ),
    (
        "renamed-column",
        lambda tables: tables["odds_timeseries"].update(column="captured_utc"),
        ("captured_utc", "snapshot_utc"),
        None,
    ),
    (
        "fallback-added",
        lambda tables: tables["track_condition"].update(fallback="published_utc"),
        ("fetched_utc", "snapshot_utc"),
        ("available_utc", "published_utc"),
    ),
]


@pytest.mark.parametrize(
    "name,edit,board,going", FALLBACK_EDITS, ids=[e[0] for e in FALLBACK_EDITS]
)
def test_availability_fallback_is_read_from_the_availability_rules(
    tmp_path, name, edit, board, going
):
    edited = _availability(tmp_path, edit)
    assert availability_fallback("odds_timeseries", rules=edited) == board
    assert availability_fallback("track_condition", rules=edited) == going
