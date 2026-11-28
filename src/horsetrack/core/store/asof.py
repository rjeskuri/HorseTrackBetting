"""As-of views over the store's tables (docs/specs/as-of.md): what was known at a decision time t.

Every table that enters a view needs an availability rule: a `column` rule (a row is available at
`row[column]`, or at `row[fallback]` where the column is null), an `offset` rule (at
`row[anchor] + offset`; a DATE anchor is read as midnight of that date in `tz`), `static` (every
row at any t), `label_only` (never in a view at t, only through `read_label`) or `current_value`
(never in a view, never as a label). A `current_value` rule with `in_view` (the values as
exported, with no history, their look-ahead accepted) enters the views as a `static` one does,
and is still never a label. A view rule may list `label_columns`, the table's result or
look-ahead columns, and `label_prefixes`, name prefixes of such columns, which a view never
returns. A row whose availability time is null is never available. A table without a rule is
refused. The per-table rules are owner decisions, kept in
`countries/<country>/config/availability.yaml`.

A view rule may also list `derived` columns (a UTC instant from a local DATE and a wall-clock
"H:MM" string, such as a race's `post_utc`) and `gates` (columns kept in the frame but null until
an anchor time + offset); an `offset` rule's anchor, and a gate's, may be read through `via`, from
another table's row matching on keys. A `via` time counts only once that row is available itself.

A view rule may also list `results_derived` columns: view columns whose values come from the
results page yet are in the view under the table's own rule (the accepted leak of
docs/specs/as-of.md). The views return them unchanged; the list only makes the leak
machine-readable, for the model boundary (`countries/japan/feature_boundary.py`, #417).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import duckdb
import pandas as pd
import yaml

from horsetrack.core.config import load_country_yaml

KINDS: tuple[str, ...] = ("column", "offset", "label_only", "current_value", "static")
# the kinds that enter a view and take label columns (and current_value with in_view, as static)
_VIEW_KINDS = ("column", "offset", "static")
# the fields each kind takes: (required, optional)
_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "column": (frozenset({"column"}), frozenset({"fallback"})),
    "offset": (frozenset({"anchor", "offset"}), frozenset({"tz", "via"})),
    "label_only": (frozenset(), frozenset()),
    "current_value": (frozenset(), frozenset()),
    "static": (frozenset(), frozenset()),
}
# the YAML keys each kind takes besides `kind` and the view keys (_VIEW_KEYS): (required, optional)
_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "column": (frozenset({"column"}), frozenset({"fallback"})),
    "offset": (frozenset({"anchor", "offset_min"}), frozenset({"tz", "via"})),
    "label_only": (frozenset(), frozenset()),
    "current_value": (frozenset(), frozenset({"in_view"})),
    "static": (frozenset(), frozenset()),
}
# the YAML keys only the view kinds take (and current_value with in_view, all but read_label)
_VIEW_KEYS = (
    "label_columns",
    "label_prefixes",
    "derived",
    "gated_columns",
    "read_label",
    "results_derived",
)
_TZ = "TIMESTAMP WITH TIME ZONE"
_DATE = "DATE"
_TEXT = "VARCHAR"
_CLOCK = "[0-9]{1,2}:[0-9]{2}"  # a wall-clock time, "H:MM" or "HH:MM"
_INTERNAL = "__asof_"  # the prefix of every internal column name
_POS = "__asof_key_pos"  # internal: the key row's position, to keep the keys' order
_AVAIL = "__asof_available_utc"  # internal: the row's availability time
_GATE = "__asof_gate_"  # internal: + the gate's index, the time its columns are shown from


class LabelOnlyTable(ValueError):
    """The table may be read only as a label (read_label), never into a view at t."""


class CurrentValueTable(ValueError):
    """The table holds values as of the scrape, not as of t: it never enters a view."""


class NoAvailabilityRule(KeyError):
    """The table has no availability rule, so nothing says when its rows were known."""


def _names(owner: str, key: str, value: Any, *, empty: bool = True) -> None:
    """Refuse a `value` that is not a tuple of column names (or an empty one, unless `empty`)."""
    if not isinstance(value, tuple) or not all(isinstance(c, str) and c for c in value):
        raise ValueError(f"{owner}: {key} must be a tuple of column names: {value!r}")
    if not value and not empty:
        raise ValueError(f"{owner}: {key} must name at least one column")


@dataclass(frozen=True)
class LocalTime:
    """A derived column `name`: the UTC instant of the DATE column `date` at the wall-clock time
    of the string column `time` ("H:MM" or "HH:MM") in the IANA zone `tz`. Null where either is
    null or `time` is malformed."""

    name: str
    date: str
    time: str
    tz: str

    def __post_init__(self) -> None:
        for key in ("name", "date", "time"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"derived column: {key} must be a column name: {self!r}")
        _zone(self.name, self.tz)


@dataclass(frozen=True)
class Via:
    """A time read from `table`: from its rows whose `on` columns equal the row's."""

    table: str
    on: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.table, str) or not self.table:
            raise ValueError(f"via: table must be a table name, got {self.table!r}")
        _names(f"via {self.table}", "on", self.on, empty=False)


@dataclass(frozen=True)
class Gate:
    """`columns` stay in the view, null until `anchor + offset`: the row's own anchor (a
    timezone-aware or derived column), or the anchor of the `via` table's matching row."""

    columns: tuple[str, ...]
    anchor: str
    offset: timedelta
    via: Via | None = None

    def __post_init__(self) -> None:
        _names("gate", "columns", self.columns, empty=False)
        if not isinstance(self.anchor, str) or not self.anchor:
            raise ValueError(f"gate of {self.columns}: anchor must be a column name")
        if not isinstance(self.offset, timedelta):
            raise ValueError(f"gate of {self.columns}: offset must be a timedelta")
        if self.via is not None and not isinstance(self.via, Via):
            raise ValueError(f"gate of {self.columns}: via must be a Via, got {self.via!r}")


@dataclass(frozen=True)
class Availability:
    table: str
    kind: str
    column: str | None = None
    anchor: str | None = None
    offset: timedelta | None = None
    # result or look-ahead columns: never returned by a view (column, offset and static rules)
    label_columns: tuple[str, ...] = ()
    # the IANA zone of a DATE anchor (offset rules only): the row opens at its local midnight
    tz: str | None = None
    # a second UTC column, used where `column` is null (column rules only)
    fallback: str | None = None
    # every column whose name starts with one of these is a label column too (column, offset and
    # static rules): for a family of columns the store's schema does not name
    label_prefixes: tuple[str, ...] = ()
    # columns a view adds to the table's own (column, offset and static rules)
    derived: tuple[LocalTime, ...] = ()
    # the anchor is read from another table's matching row, not the row's own (offset rules only)
    via: Via | None = None
    # columns in the view but null until a time of their own (column, offset and static rules)
    gates: tuple[Gate, ...] = ()
    # read_label also returns the table whole (column, offset and static rules)
    label_readable: bool = False
    # the current values enter every view at any t, as a static table's rows do, with no history:
    # their look-ahead is accepted (current_value rules only)
    in_view: bool = False
    # view columns taken from the results page, in the view under the table's rule: the accepted
    # leak of docs/specs/as-of.md, listed so it stays identifiable (column, offset and static
    # rules, and current_value with in_view)
    results_derived: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.table}: unknown availability kind {self.kind!r} ({KINDS})")
        if not isinstance(self.in_view, bool):
            raise ValueError(f"{self.table}: in_view must be a bool, got {self.in_view!r}")
        if self.in_view and self.kind != "current_value":
            raise ValueError(f"{self.table}: a {self.kind!r} rule takes no in_view")
        given = {
            name
            for name, value in (
                ("column", self.column),
                ("anchor", self.anchor),
                ("offset", self.offset),
                ("tz", self.tz),
                ("fallback", self.fallback),
                ("via", self.via),
            )
            if value is not None
        }
        required, optional = _FIELDS[self.kind]
        if not required <= given <= required | optional:
            raise ValueError(
                f"{self.table}: a {self.kind!r} rule takes {sorted(required) or 'no fields'}"
                f"{f' (optional {sorted(optional)})' if optional else ''}, "
                f"got {sorted(given) or 'none'}"
            )
        if self.offset is not None and not isinstance(self.offset, timedelta):
            raise ValueError(f"{self.table}: offset must be a timedelta, got {self.offset!r}")
        if self.tz is not None:
            _zone(self.table, self.tz)
        if self.fallback is not None and (not isinstance(self.fallback, str) or not self.fallback):
            raise ValueError(f"{self.table}: fallback must be a column name: {self.fallback!r}")
        labels = self.label_columns
        if not isinstance(labels, tuple) or not all(isinstance(c, str) and c for c in labels):
            raise ValueError(f"{self.table}: label_columns must be a tuple of names: {labels!r}")
        if labels and not self.enters_view:
            raise ValueError(f"{self.table}: a {self.kind!r} rule takes no label_columns")
        prefixes = self.label_prefixes
        if not isinstance(prefixes, tuple) or not all(isinstance(p, str) and p for p in prefixes):
            raise ValueError(
                f"{self.table}: label_prefixes must be a tuple of non-empty strings: {prefixes!r}"
            )
        if prefixes and not self.enters_view:
            raise ValueError(f"{self.table}: a {self.kind!r} rule takes no label_prefixes")
        for stamp in (self.column, self.anchor, self.fallback):
            if stamp is not None and stamp in labels:
                raise ValueError(
                    f"{self.table}: label_columns list the rule's time column {stamp!r}"
                )
            if stamp is not None and stamp.startswith(prefixes):
                raise ValueError(
                    f"{self.table}: label_prefixes match the rule's time column {stamp!r}"
                )
        self._check_view_extras()

    def _check_view_extras(self) -> None:
        """Refuse malformed `derived`, `via`, `gates` and `label_readable` fields."""
        if self.via is not None and not isinstance(self.via, Via):
            raise ValueError(f"{self.table}: via must be a Via, got {self.via!r}")
        if self.via is not None and self.tz is not None:
            raise ValueError(f"{self.table}: a via anchor is a timestamp: its rule takes no tz")
        if not isinstance(self.label_readable, bool):
            raise ValueError(f"{self.table}: label_readable must be a bool")
        derived, gates = self.derived, self.gates
        if not isinstance(derived, tuple) or not all(isinstance(d, LocalTime) for d in derived):
            raise ValueError(f"{self.table}: derived must be a tuple of LocalTime: {derived!r}")
        if not isinstance(gates, tuple) or not all(isinstance(g, Gate) for g in gates):
            raise ValueError(f"{self.table}: gates must be a tuple of Gate: {gates!r}")
        if (derived or gates) and not self.enters_view:
            raise ValueError(f"{self.table}: a {self.kind!r} rule takes no derived or gates")
        leaked = self.results_derived
        _names(self.table, "results_derived", leaked)
        if leaked and not self.enters_view:
            raise ValueError(f"{self.table}: a {self.kind!r} rule takes no results_derived")
        if len(set(leaked)) != len(leaked):
            raise ValueError(f"{self.table}: a results_derived column is listed twice: {leaked}")
        bad = [c for c in leaked if self.is_label(c)]
        if bad:
            raise ValueError(f"{self.table}: results_derived columns {bad} are label columns")
        if self.label_readable and self.kind not in _VIEW_KINDS:  # in_view is never a label
            raise ValueError(f"{self.table}: a {self.kind!r} rule takes no label_readable")
        names = [d.name for d in derived]
        gated = [c for g in gates for c in g.columns]
        stamps = {self.column, self.anchor, self.fallback} - {None}
        for kind, listed in (("derived", names), ("gated", gated)):
            if len(set(listed)) != len(listed):
                raise ValueError(f"{self.table}: a {kind} column is listed twice: {listed}")
            bad = [c for c in listed if self.is_label(c) or c in stamps]
            if bad:
                raise ValueError(f"{self.table}: {kind} columns {bad} are labels or time columns")
        if set(names) & set(gated):
            raise ValueError(f"{self.table}: derived columns cannot be gated")

    @property
    def enters_view(self) -> bool:
        """Whether the rows enter the views: a column, offset or static rule, or a current_value
        rule with in_view (read as a static one)."""
        return self.kind in _VIEW_KINDS or self.in_view

    @property
    def gated(self) -> dict[str, int]:
        """Each gated column, with the index of its gate."""
        return {c: i for i, gate in enumerate(self.gates) for c in gate.columns}

    def is_label(self, name: str) -> bool:
        """Whether a view leaves the column out: a label column, or one a label prefix starts."""
        return name in self.label_columns or name.startswith(self.label_prefixes)


def _zone(table: str, tz: Any) -> None:
    """Refuse a `tz` that is not an IANA zone name."""
    if not isinstance(tz, str) or not tz:
        raise ValueError(f"{table}: tz must be an IANA zone name, got {tz!r}")
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"{table}: unknown time zone {tz!r}") from exc


def _entry(table: str, raw: Any) -> Availability:
    if not isinstance(raw, Mapping):
        raise ValueError(f"availability of {table!r}: an entry must be a mapping, got {raw!r}")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise ValueError(f"availability of {table!r}: kind must be one of {KINDS}, got {kind!r}")
    keys = set(raw) - {"kind", *_VIEW_KEYS}
    required, optional = _KEYS[kind]
    if not required <= keys <= required | optional:
        raise ValueError(
            f"availability of {table!r}: kind {kind!r} takes {sorted(required)}"
            f"{f' (optional {sorted(optional)})' if optional else ''}, got {sorted(keys)}"
        )
    in_view = raw.get("in_view", False)
    if not isinstance(in_view, bool):
        raise ValueError(f"availability of {table!r}: in_view must be true or false")
    for key in _VIEW_KEYS:
        if key in raw and kind not in _VIEW_KINDS and not (in_view and key != "read_label"):
            rule = f"kind {kind!r}{' with in_view' if in_view else ''}"
            raise ValueError(f"availability of {table!r}: {rule} takes no {key}")
    readable = raw.get("read_label", False)
    if not isinstance(readable, bool):
        raise ValueError(f"availability of {table!r}: read_label must be true or false")
    extras: dict[str, Any] = {
        "label_columns": _labels(table, "label_columns", raw.get("label_columns", [])),
        "label_prefixes": _labels(table, "label_prefixes", raw.get("label_prefixes", [])),
        "derived": _derived(table, raw.get("derived", {})),
        "gates": _gates(table, raw.get("gated_columns", [])),
        "label_readable": readable,
        "results_derived": _labels(table, "results_derived", raw.get("results_derived", [])),
    }
    if kind == "column":
        column = _name(table, "column", raw["column"])
        fallback = _name(table, "fallback", raw["fallback"]) if "fallback" in raw else None
        return Availability(table, kind, column=column, fallback=fallback, **extras)
    if kind == "offset":
        tz = raw.get("tz")
        if "tz" in raw:
            _zone(table, tz)
        return Availability(
            table,
            kind,
            anchor=_name(table, "anchor", raw["anchor"]),
            offset=_minutes(table, raw["offset_min"]),
            tz=tz,
            via=_via(table, raw["via"]) if "via" in raw else None,
            **extras,
        )
    return Availability(table, kind, in_view=in_view, **extras)


def _minutes(table: str, minutes: Any) -> timedelta:
    if isinstance(minutes, bool) or not isinstance(minutes, int | float):
        raise ValueError(f"availability of {table!r}: offset_min must be a number of minutes")
    return timedelta(minutes=minutes)


def _mapping(table: str, key: str, raw: Any, keys: set[str]) -> Mapping:
    """A YAML mapping with exactly `keys`."""
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise ValueError(f"availability of {table!r}: {key} takes {sorted(keys)}, got {raw!r}")
    return raw


def _via(table: str, raw: Any) -> Via:
    # `match`, not `on`: YAML 1.1 reads a bare `on` key as true
    raw = _mapping(table, "via", raw, {"table", "match"})
    return Via(_name(table, "via table", raw["table"]), _labels(table, "via match", raw["match"]))


def _derived(table: str, raw: Any) -> tuple[LocalTime, ...]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"availability of {table!r}: derived must be a mapping, got {raw!r}")
    out = []
    for name, spec in raw.items():
        spec = _mapping(table, f"derived {name}", spec, {"date", "time", "tz"})
        _zone(table, spec["tz"])
        out.append(
            LocalTime(
                _name(table, "derived", name),
                _name(table, "derived date", spec["date"]),
                _name(table, "derived time", spec["time"]),
                spec["tz"],
            )
        )
    return tuple(out)


def _gates(table: str, raw: Any) -> tuple[Gate, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"availability of {table!r}: gated_columns must be a list, got {raw!r}")
    out = []
    for spec in raw:
        if not isinstance(spec, Mapping) or not {"columns", "anchor", "offset_min"} <= set(
            spec
        ) <= {"columns", "anchor", "offset_min", "via"}:
            raise ValueError(
                f"availability of {table!r}: a gate takes columns, anchor, offset_min "
                f"(optional via), got {spec!r}"
            )
        out.append(
            Gate(
                _labels(table, "gate columns", spec["columns"]),
                _name(table, "gate anchor", spec["anchor"]),
                _minutes(table, spec["offset_min"]),
                _via(table, spec["via"]) if "via" in spec else None,
            )
        )
    return tuple(out)


def _labels(table: str, key: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"availability of {table!r}: {key} must be a list, got {value!r}")
    return tuple(_name(table, key, item) for item in value)


def _name(table: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"availability of {table!r}: {key} must be a column name, got {value!r}")
    return value


def load_availability(country: str, path: Path | None = None) -> dict[str, Availability]:
    """The availability rules of a country: `countries/<country>/config/availability.yaml`, or
    `path`. A missing file is a FileNotFoundError, a malformed one a ValueError."""
    if path is None:
        raw = load_country_yaml(country, "availability")
    else:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    tables = raw.get("tables") if isinstance(raw, Mapping) else None
    if not isinstance(tables, Mapping):
        raise ValueError(f"availability of {country!r}: expected a top-level `tables:` mapping")
    return {str(table): _entry(str(table), entry) for table, entry in tables.items()}


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _rule(table: str, rules: Mapping[str, Availability]) -> Availability:
    rule = rules.get(table)
    if rule is None:
        raise NoAvailabilityRule(f"table {table!r} has no availability rule")
    return rule


def _view_rule(table: str, rules: Mapping[str, Availability]) -> Availability:
    rule = _rule(table, rules)
    if rule.kind == "label_only":
        raise LabelOnlyTable(f"table {table!r} is label-only: read it with read_label")
    if rule.kind == "current_value" and not rule.in_view:
        raise CurrentValueTable(f"table {table!r} holds current values: it never enters a view")
    return rule


def _type(con, table: str, column: str) -> str:
    return str(con.sql(f"SELECT {_ident(column)} FROM {_ident(table)} LIMIT 0").types[0])


def _utc_column(con, table: str, column: str) -> str:
    """The quoted column, which must be timezone-aware: a naive one would be read in the session's
    time zone."""
    kind = _type(con, table, column)
    if kind != _TZ:
        raise ValueError(f"{table}.{column} is {kind}, not a UTC timestamp ({_TZ})")
    return _ident(column)


def _available_sql(con: duckdb.DuckDBPyConnection, rule: Availability) -> tuple[str, list[Any]]:
    """The SQL expression of a row's availability time, with its parameters."""
    if rule.kind == "static" or rule.in_view:
        return "CAST('-infinity' AS TIMESTAMPTZ)", []
    if rule.kind == "column":
        assert rule.column is not None
        column = _utc_column(con, rule.table, rule.column)
        if rule.fallback is None:
            return column, []
        return f"coalesce({column}, {_utc_column(con, rule.table, rule.fallback)})", []
    assert rule.anchor is not None and rule.offset is not None
    # the offset as microseconds only: an interval with a day part would be added on the
    # session's calendar, 23 or 25 hours across its DST change
    micros = rule.offset // timedelta(microseconds=1)
    if _type(con, rule.table, rule.anchor) == _DATE:
        if rule.tz is None:
            raise ValueError(f"{rule.table}.{rule.anchor} is a DATE: its rule needs a tz")
        _duck_zone(con, rule.table, rule.tz)
        local = f"timezone(?, CAST({_ident(rule.anchor)} AS TIMESTAMP))"
        return f"({local} + to_microseconds(?))", [rule.tz, micros]
    if rule.tz is not None:
        raise ValueError(f"{rule.table}.{rule.anchor} is not a DATE: its rule takes no tz")
    return f"({_utc_column(con, rule.table, rule.anchor)} + to_microseconds(?))", [micros]


def _duck_zone(con, table: str, tz: str) -> None:
    """Refuse a zone DuckDB's zone database, which may differ from Python's, does not know."""
    try:  # fetched, so no unfinished result blocks the connection's next statement (DuckDB 2)
        con.execute("SELECT timezone(?, TIMESTAMP '2000-01-01')", [tz]).fetchall()
    except duckdb.Error as exc:
        raise ValueError(f"{table}: DuckDB does not know the zone {tz!r}") from exc


def _base(con, rule: Availability) -> tuple[str, list[Any]]:
    """The table with its derived columns, as SQL and parameters."""
    if not rule.derived:
        return f"SELECT * FROM {_ident(rule.table)}", []
    columns = set(con.sql(f"SELECT * FROM {_ident(rule.table)} LIMIT 0").columns)
    exprs = []
    for d in rule.derived:
        if d.name in columns:
            raise ValueError(f"table {rule.table!r} already has the derived column {d.name!r}")
        for name, want in ((d.date, _DATE), (d.time, _TEXT)):
            if (kind := _type(con, rule.table, name)) != want:
                raise ValueError(f"{rule.table}.{name} is {kind}, not {want}")
        _duck_zone(con, rule.table, d.tz)
        clock, day = _ident(d.time), _ident(d.date)
        exprs.append(
            f"CASE WHEN regexp_full_match({clock}, '{_CLOCK}') THEN timezone(?, try_strptime("
            f"strftime({day}, '%Y-%m-%d') || ' ' || {clock}, '%Y-%m-%d %H:%M')) END "
            f"AS {_ident(d.name)}"
        )
    return f"SELECT *, {', '.join(exprs)} FROM {_ident(rule.table)}", [d.tz for d in rule.derived]


def _instant(con, rule: Availability, name: str) -> str:
    """The quoted time column `name` of the rule's table: a derived or timezone-aware column."""
    if name in {d.name for d in rule.derived}:
        return _ident(name)
    return _utc_column(con, rule.table, name)


def _reached(
    con, rule: Availability, anchor: str, offset: timedelta, via: Via | None, rules, alias: str
) -> tuple[str, list[Any], str, list[Any]]:
    """The time `anchor + offset` of a row of `rule`'s table: (its SQL, its parameters, the join
    it needs, the join's parameters). Through `via`, it is the anchor of the via table's rows
    matching on `via.on` (the latest, null if any is null or none matches), and never before those
    rows are available themselves."""
    micros = offset // timedelta(microseconds=1)  # microseconds: see _available_sql
    if via is None:
        return f"({_instant(con, rule, anchor)} + to_microseconds(?))", [micros], "", []
    target = _view_rule(via.table, rules)
    if target.via is not None:
        raise ValueError(f"{rule.table}: via {via.table!r}, whose own rule reads through a via")
    if target.is_label(anchor) or anchor in target.gated:
        raise ValueError(f"{rule.table}: via {via.table}.{anchor}, a label or gated column")
    base, base_params = _base(con, target)
    available, available_params = _available_sql(con, target)
    keys = [_ident(f"{_INTERNAL}on_{j}") for j in range(len(via.on))]
    inner = (
        f"SELECT {', '.join(f'{_ident(c)} AS {k}' for c, k in zip(via.on, keys, strict=True))}, "
        f"({_instant(con, target, anchor)} + to_microseconds(?)) AS __asof_g, "
        f"{available} AS __asof_a FROM ({base}) AS b"
    )
    lookup = (
        f"SELECT {', '.join(keys)}, CASE WHEN count(*) = count(__asof_g) "
        "AND count(*) = count(__asof_a) THEN greatest(max(__asof_g), max(__asof_a)) END "
        f"AS __asof_v FROM ({inner}) GROUP BY {', '.join(keys)}"
    )
    match = " AND ".join(f"s.{_ident(c)} = {alias}.{k}" for c, k in zip(via.on, keys, strict=True))
    join = f" LEFT JOIN ({lookup}) AS {alias} ON {match}"
    return f"{alias}.__asof_v", [], join, [micros, *available_params, *base_params]


def _source(con, rule: Availability, rules) -> tuple[str, list[Any]]:
    """The table as SQL and parameters: its columns, its derived columns, each row's availability
    time (_AVAIL) and each gate's time (_GATE + the gate's index)."""
    joins: list[str] = []
    join_params: list[Any] = []

    def reached(anchor: str, offset: timedelta, via: Via | None) -> tuple[str, list[Any]]:
        expr, params, join, more = _reached(
            con, rule, anchor, offset, via, rules, f"{_INTERNAL}l{len(joins)}"
        )
        if join:
            joins.append(join)
            join_params.extend(more)
        return expr, params

    if rule.via is not None:
        assert rule.anchor is not None and rule.offset is not None
        available, params = reached(rule.anchor, rule.offset, rule.via)
    else:
        available, params = _available_sql(con, rule)
    picked = [f"{available} AS {_ident(_AVAIL)}"]
    for i, gate in enumerate(rule.gates):
        expr, more = reached(gate.anchor, gate.offset, gate.via)
        picked.append(f"{expr} AS {_ident(f'{_GATE}{i}')}")
        params += more
    base, base_params = _base(con, rule)
    sql = f"SELECT s.*, {', '.join(picked)} FROM ({base}) AS s{''.join(joins)}"
    return sql, [*params, *base_params, *join_params]


def _view_columns(con, rule: Availability) -> list[str]:
    """The table's columns a view returns: all but the rule's label columns, each of which the
    table must have (a misspelt one would otherwise let the real column through), and all but
    those a label prefix starts (a prefix that starts none is fine); then the derived columns.
    Each gated column must be in the table too."""
    columns = list(con.sql(f"SELECT * FROM {_ident(rule.table)} LIMIT 0").columns)
    missing = [c for c in rule.label_columns if c not in columns]
    if missing:
        raise ValueError(f"table {rule.table!r} has no label columns {missing}")
    missing = [c for c in rule.gated if c not in columns]
    if missing:
        raise ValueError(f"table {rule.table!r} has no gated columns {missing}")
    return [c for c in columns if not rule.is_label(c)] + [d.name for d in rule.derived]


def _shown(rule: Availability, column: str, row: str, gate_open: str) -> str:
    """A view column's SQL: a gated column is null unless `gate_open` holds for its gate time."""
    i = rule.gated.get(column)
    if i is None:
        return f"{row}{_ident(column)}"
    when = gate_open.replace("{gate}", f"{row}{_ident(f'{_GATE}{i}')}")
    return f"CASE WHEN {when} THEN {row}{_ident(column)} END AS {_ident(column)}"


def _fetch_utc(con, sql: str, params: list[Any]) -> pd.DataFrame:
    """The query's frame, with timezone-aware columns in UTC (DuckDB hands them back in the
    session's time zone)."""
    frame = con.execute(sql, params).df()
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.DatetimeTZDtype):
            frame[column] = frame[column].dt.tz_convert("UTC")
    return frame


def _aware(at_utc: datetime) -> None:
    if at_utc.tzinfo is None or at_utc.utcoffset() is None:
        raise ValueError(f"at_utc must be timezone-aware, got {at_utc!r}")


def read_asof(con, table: str, at_utc: datetime, *, rules) -> pd.DataFrame:
    """Every row of `table` available at or before `at_utc` (the whole history, not the latest)."""
    rule = _view_rule(table, rules)
    _aware(at_utc)
    columns = _view_columns(con, rule)
    source, params = _source(con, rule, rules)
    picked = ", ".join(_shown(rule, c, "", "{gate} <= ?") for c in columns)
    gated = [at_utc for c in columns if c in rule.gated]
    sql = f"SELECT {picked} FROM ({source}) AS v WHERE {_ident(_AVAIL)} <= ?"
    return _fetch_utc(con, sql, [*gated, *params, at_utc])


def asof_join(
    keys: pd.DataFrame,
    table: str,
    *,
    on: list[str],
    rules,
    con,
    t_col: str = "t_utc",
) -> pd.DataFrame:
    """One row per key row, in the keys' order, with the keys' columns unchanged plus the columns
    of the latest `table` row matching on `on` and available at or before the key's `t_col`
    (an ASOF LEFT JOIN: a key with no such row keeps null table columns). The rule's label
    columns and the columns its label prefixes start are left out, and joining on one is a
    ValueError."""
    rule = _view_rule(table, rules)
    if not isinstance(keys[t_col].dtype, pd.DatetimeTZDtype):
        raise ValueError(f"keys[{t_col!r}] must be timezone-aware, got {keys[t_col].dtype}")
    joined = [c for c in on if rule.is_label(c) or c in rule.gated]
    if joined:
        raise ValueError(f"table {table!r}: cannot join on label or gated columns {joined}")
    columns = _view_columns(con, rule)
    added = [c for c in columns if c not in on]
    clash = [c for c in added if c in keys.columns or c.startswith(_INTERNAL)]
    if clash:
        raise ValueError(f"table {table!r} and the keys both have columns {clash}")
    source, params = _source(con, rule, rules)

    probe = keys[[*on, t_col]].reset_index(drop=True)
    probe[_POS] = range(len(probe))
    matches = " AND ".join(f"k.{_ident(c)} = r.{_ident(c)}" for c in on)
    gate_open = f"k.{_ident(t_col)} >= {{gate}}"
    picked = ", ".join(_shown(rule, c, "r.", gate_open) for c in added)
    sql = (
        f"SELECT k.{_ident(_POS)}{', ' + picked if picked else ''} "
        f"FROM {_ident(_POS)} AS k ASOF LEFT JOIN ("
        f"SELECT * FROM ({source}) AS v WHERE {_ident(_AVAIL)} IS NOT NULL) AS r "
        f"ON {matches + ' AND ' if matches else ''}k.{_ident(t_col)} >= r.{_ident(_AVAIL)} "
        f"ORDER BY k.{_ident(_POS)}"
    )
    con.register(_POS, probe)
    try:
        found = _fetch_utc(con, sql, params)
    finally:
        con.unregister(_POS)
    found = found.drop(columns=[_POS])
    found.index = keys.index
    return pd.concat([keys, found], axis=1)


def read_label(con, table: str, *, rules) -> pd.DataFrame:
    """A label-only table whole (final odds, results used as labels): never a feature at t, or a
    view table whose rule is `label_readable` (results that also enter the views after the post).
    Every other kind, `static` and `current_value` with `in_view` included, is a ValueError."""
    rule = _rule(table, rules)
    if rule.kind != "label_only" and not rule.label_readable:
        raise ValueError(f"table {table!r} is a {rule.kind!r} table, not label-only")
    return _fetch_utc(con, f"SELECT * FROM {_ident(table)}", [])
