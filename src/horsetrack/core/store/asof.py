"""As-of views over the store's tables (docs/specs/as-of.md): what was known at a decision time t.

Every table that enters a view needs an availability rule: a `column` rule (a row is available at
`row[column]`), an `offset` rule (at `row[anchor] + offset`), `label_only` (never in a view at t,
only through `read_label`) or `current_value` (never in a view, never as a label). A `column`
or `offset` rule may list `label_columns`, the table's result or look-ahead columns, which a view
never returns. A row whose availability time is null is never available. A table without a rule is
refused. The per-table rules are owner decisions, kept in
`countries/<country>/config/availability.yaml`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

from horsetrack.core.config import load_country_yaml

KINDS: tuple[str, ...] = ("column", "offset", "label_only", "current_value")
_VIEW_KINDS = ("column", "offset")  # the kinds that enter a view, and may list label columns
# the YAML keys each kind takes besides `kind`; all of them are required
_KEYS: dict[str, frozenset[str]] = {
    "column": frozenset({"column"}),
    "offset": frozenset({"anchor", "offset_min"}),
    "label_only": frozenset(),
    "current_value": frozenset(),
}
_TZ = "TIMESTAMP WITH TIME ZONE"
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
    # result or look-ahead columns: never returned by a view (column and offset rules only)
    label_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.table}: unknown availability kind {self.kind!r} ({KINDS})")
        given = {
            name
            for name, value in (
                ("column", self.column),
                ("anchor", self.anchor),
                ("offset", self.offset),
            )
            if value is not None
        }
        wanted = {"column": {"column"}, "offset": {"anchor", "offset"}}.get(self.kind, set())
        if given != wanted:
            raise ValueError(
                f"{self.table}: a {self.kind!r} rule takes {sorted(wanted) or 'no fields'}, "
                f"got {sorted(given) or 'none'}"
            )
        if self.offset is not None and not isinstance(self.offset, timedelta):
            raise ValueError(f"{self.table}: offset must be a timedelta, got {self.offset!r}")
        labels = self.label_columns
        if not isinstance(labels, tuple) or not all(isinstance(c, str) and c for c in labels):
            raise ValueError(f"{self.table}: label_columns must be a tuple of names: {labels!r}")
        if labels and self.kind not in _VIEW_KINDS:
            raise ValueError(f"{self.table}: a {self.kind!r} rule takes no label_columns")
        stamp = self.column if self.kind == "column" else self.anchor
        if stamp is not None and stamp in labels:
            raise ValueError(f"{self.table}: label_columns list the rule's time column {stamp!r}")


def _entry(table: str, raw: Any) -> Availability:
    if not isinstance(raw, Mapping):
        raise ValueError(f"availability of {table!r}: an entry must be a mapping, got {raw!r}")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise ValueError(f"availability of {table!r}: kind must be one of {KINDS}, got {kind!r}")
    keys = set(raw) - {"kind", "label_columns"}
    if keys != _KEYS[kind]:
        raise ValueError(
            f"availability of {table!r}: kind {kind!r} takes {sorted(_KEYS[kind])}, "
            f"got {sorted(keys)}"
        )
    if "label_columns" in raw and kind not in _VIEW_KINDS:
        raise ValueError(f"availability of {table!r}: kind {kind!r} takes no label_columns")
    labels = _labels(table, raw.get("label_columns", []))
    if kind == "column":
        column = _name(table, "column", raw["column"])
        return Availability(table, kind, column=column, label_columns=labels)
    if kind == "offset":
        minutes = raw["offset_min"]
        if isinstance(minutes, bool) or not isinstance(minutes, int | float):
            raise ValueError(f"availability of {table!r}: offset_min must be a number of minutes")
        return Availability(
            table,
            kind,
            anchor=_name(table, "anchor", raw["anchor"]),
            offset=timedelta(minutes=minutes),
            label_columns=labels,
        )
    return Availability(table, kind)


def _labels(table: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"availability of {table!r}: label_columns must be a list, got {value!r}")
    return tuple(_name(table, "label_columns", item) for item in value)


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


def _available_sql(con: duckdb.DuckDBPyConnection, rule: Availability) -> tuple[str, list[Any]]:
    """The SQL expression of a row's availability time, with its parameters. The time column must
    be timezone-aware: a naive one would be read in the session's time zone."""
    stamp = rule.column if rule.kind == "column" else rule.anchor
    assert stamp is not None
    kind = str(con.sql(f"SELECT {_ident(stamp)} FROM {_ident(rule.table)} LIMIT 0").types[0])
    if kind != _TZ:
        raise ValueError(f"{rule.table}.{stamp} is {kind}, not a UTC timestamp ({_TZ})")
    if rule.kind == "column":
        return _ident(stamp), []
    return f"({_ident(stamp)} + ?)", [rule.offset]


def _view_columns(con, rule: Availability) -> list[str]:
    """The table's columns a view returns: all but the rule's label columns, each of which the
    table must have (a misspelt one would otherwise let the real column through)."""
    columns = list(con.sql(f"SELECT * FROM {_ident(rule.table)} LIMIT 0").columns)
    missing = [c for c in rule.label_columns if c not in columns]
    if missing:
        raise ValueError(f"table {rule.table!r} has no label columns {missing}")
    return [c for c in columns if c not in rule.label_columns]


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
    columns are left out, and joining on one is a ValueError."""
    rule = _view_rule(table, rules)
    if not isinstance(keys[t_col].dtype, pd.DatetimeTZDtype):
        raise ValueError(f"keys[{t_col!r}] must be timezone-aware, got {keys[t_col].dtype}")
    joined = [c for c in on if c in rule.label_columns]
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
    """A label-only table whole (final odds, results used as labels): never a feature at t."""
    rule = _rule(table, rules)
    if rule.kind != "label_only":
        raise ValueError(f"table {table!r} is a {rule.kind!r} table, not label-only")
    return _fetch_utc(con, f"SELECT * FROM {_ident(table)}", [])
