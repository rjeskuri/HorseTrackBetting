"""As-of views over the store's tables (docs/specs/as-of.md): what was known at a decision time t.

Every table that enters a view needs an availability rule: a `column` rule (a row is available at
`row[column]`, or at `row[fallback]` where the column is null), an `offset` rule (at
`row[anchor] + offset`; a DATE anchor is read as midnight of that date in `tz`), `static` (every
row at any t), `label_only` (never in a view at t, only through `read_label`) or `current_value`
(never in a view, never as a label). A `column`, `offset` or `static` rule may list
`label_columns`, the table's result or look-ahead columns, and `label_prefixes`, name prefixes of
such columns, which a view never returns. A row whose availability time is null is never
available. A table without a rule is refused. The per-table rules are owner decisions,
kept in `countries/<country>/config/availability.yaml`.
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
_VIEW_KINDS = ("column", "offset", "static")  # the kinds that enter a view and take label columns
# the fields each kind takes: (required, optional)
_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "column": (frozenset({"column"}), frozenset({"fallback"})),
    "offset": (frozenset({"anchor", "offset"}), frozenset({"tz"})),
    "label_only": (frozenset(), frozenset()),
    "current_value": (frozenset(), frozenset()),
    "static": (frozenset(), frozenset()),
}
# the YAML keys each kind takes besides `kind`, `label_columns` and `label_prefixes`:
# (required, optional)
_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "column": (frozenset({"column"}), frozenset({"fallback"})),
    "offset": (frozenset({"anchor", "offset_min"}), frozenset({"tz"})),
    "label_only": (frozenset(), frozenset()),
    "current_value": (frozenset(), frozenset()),
    "static": (frozenset(), frozenset()),
}
_TZ = "TIMESTAMP WITH TIME ZONE"
_DATE = "DATE"
_POS = "__asof_key_pos"  # internal: the key row's position, to keep the keys' order
_AVAIL = "__asof_available_utc"  # internal: the row's availability time


class LabelOnlyTable(ValueError):
    """The table may be read only as a label (read_label), never into a view at t."""


class CurrentValueTable(ValueError):
    """The table holds values as of the scrape, not as of t: it never enters a view."""


class NoAvailabilityRule(KeyError):
    """The table has no availability rule, so nothing says when its rows were known."""


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

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.table}: unknown availability kind {self.kind!r} ({KINDS})")
        given = {
            name
            for name, value in (
                ("column", self.column),
                ("anchor", self.anchor),
                ("offset", self.offset),
                ("tz", self.tz),
                ("fallback", self.fallback),
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
        if labels and self.kind not in _VIEW_KINDS:
            raise ValueError(f"{self.table}: a {self.kind!r} rule takes no label_columns")
        prefixes = self.label_prefixes
        if not isinstance(prefixes, tuple) or not all(isinstance(p, str) and p for p in prefixes):
            raise ValueError(
                f"{self.table}: label_prefixes must be a tuple of non-empty strings: {prefixes!r}"
            )
        if prefixes and self.kind not in _VIEW_KINDS:
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
    keys = set(raw) - {"kind", "label_columns", "label_prefixes"}
    required, optional = _KEYS[kind]
    if not required <= keys <= required | optional:
        raise ValueError(
            f"availability of {table!r}: kind {kind!r} takes {sorted(required)}"
            f"{f' (optional {sorted(optional)})' if optional else ''}, got {sorted(keys)}"
        )
    for key in ("label_columns", "label_prefixes"):
        if key in raw and kind not in _VIEW_KINDS:
            raise ValueError(f"availability of {table!r}: kind {kind!r} takes no {key}")
    labels = _labels(table, "label_columns", raw.get("label_columns", []))
    prefixes = _labels(table, "label_prefixes", raw.get("label_prefixes", []))
    if kind == "column":
        column = _name(table, "column", raw["column"])
        fallback = _name(table, "fallback", raw["fallback"]) if "fallback" in raw else None
        return Availability(
            table,
            kind,
            column=column,
            fallback=fallback,
            label_columns=labels,
            label_prefixes=prefixes,
        )
    if kind == "offset":
        minutes = raw["offset_min"]
        if isinstance(minutes, bool) or not isinstance(minutes, int | float):
            raise ValueError(f"availability of {table!r}: offset_min must be a number of minutes")
        tz = raw.get("tz")
        if "tz" in raw:
            _zone(table, tz)
        return Availability(
            table,
            kind,
            anchor=_name(table, "anchor", raw["anchor"]),
            offset=timedelta(minutes=minutes),
            label_columns=labels,
            label_prefixes=prefixes,
            tz=tz,
        )
    return Availability(table, kind, label_columns=labels, label_prefixes=prefixes)


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
    if rule.kind == "current_value":
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
    if rule.kind == "static":
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
        try:  # DuckDB's zone database, which may differ from Python's
            # fetched, so no unfinished result blocks the connection's next statement (DuckDB 2)
            con.execute("SELECT timezone(?, TIMESTAMP '2000-01-01')", [rule.tz]).fetchall()
        except duckdb.Error as exc:
            raise ValueError(f"{rule.table}: DuckDB does not know the zone {rule.tz!r}") from exc
        local = f"timezone(?, CAST({_ident(rule.anchor)} AS TIMESTAMP))"
        return f"({local} + to_microseconds(?))", [rule.tz, micros]
    if rule.tz is not None:
        raise ValueError(f"{rule.table}.{rule.anchor} is not a DATE: its rule takes no tz")
    return f"({_utc_column(con, rule.table, rule.anchor)} + to_microseconds(?))", [micros]


def _view_columns(con, rule: Availability) -> list[str]:
    """The table's columns a view returns: all but the rule's label columns, each of which the
    table must have (a misspelt one would otherwise let the real column through), and all but
    those a label prefix starts (a prefix that starts none is fine)."""
    columns = list(con.sql(f"SELECT * FROM {_ident(rule.table)} LIMIT 0").columns)
    missing = [c for c in rule.label_columns if c not in columns]
    if missing:
        raise ValueError(f"table {rule.table!r} has no label columns {missing}")
    return [c for c in columns if not rule.is_label(c)]


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
    available, params = _available_sql(con, rule)
    picked = ", ".join(_ident(c) for c in columns)
    if rule.kind == "static":
        return _fetch_utc(con, f"SELECT {picked} FROM {_ident(table)}", [])
    sql = f"SELECT {picked} FROM {_ident(table)} WHERE {available} <= ?"
    return _fetch_utc(con, sql, [*params, at_utc])


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
    joined = [c for c in on if rule.is_label(c)]
    if joined:
        raise ValueError(f"table {table!r}: cannot join on label columns {joined}")
    columns = _view_columns(con, rule)
    added = [c for c in columns if c not in on]
    clash = [c for c in added if c in keys.columns or c in (_POS, _AVAIL)]
    if clash:
        raise ValueError(f"table {table!r} and the keys both have columns {clash}")
    available, params = _available_sql(con, rule)

    probe = keys[[*on, t_col]].reset_index(drop=True)
    probe[_POS] = range(len(probe))
    matches = " AND ".join(f"k.{_ident(c)} = r.{_ident(c)}" for c in on)
    picked = ", ".join(f"r.{_ident(c)}" for c in added)
    sql = (
        f"SELECT k.{_ident(_POS)}{', ' + picked if picked else ''} "
        f"FROM {_ident(_POS)} AS k ASOF LEFT JOIN ("
        f"SELECT *, {available} AS {_ident(_AVAIL)} FROM {_ident(table)} "
        f"WHERE {available} IS NOT NULL) AS r "
        f"ON {matches + ' AND ' if matches else ''}k.{_ident(t_col)} >= r.{_ident(_AVAIL)} "
        f"ORDER BY k.{_ident(_POS)}"
    )
    con.register(_POS, probe)
    try:
        found = _fetch_utc(con, sql, params + params)
    finally:
        con.unregister(_POS)
    found = found.drop(columns=[_POS])
    found.index = keys.index
    return pd.concat([keys, found], axis=1)


def read_label(con, table: str, *, rules) -> pd.DataFrame:
    """A label-only table whole (final odds, results used as labels): never a feature at t.
    Every other kind, `static` included, is a ValueError."""
    rule = _rule(table, rules)
    if rule.kind != "label_only":
        raise ValueError(f"table {table!r} is a {rule.kind!r} table, not label-only")
    return _fetch_utc(con, f"SELECT * FROM {_ident(table)}", [])
