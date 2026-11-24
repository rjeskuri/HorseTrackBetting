# As-of views: what was known at a decision time

Code: `src/horsetrack/core/store/asof.py`. Input: a DuckDB connection holding the tables (e.g.
`core.store.tables.connect(source)`) and the availability rules of a country. Every feature
frame at (race, horse, t) is built through these views, so nothing available after t reaches it.

## Rule kinds

Every table that is read needs a rule. A table without one raises `NoAvailabilityRule` (a
`KeyError`) from `read_asof`, `asof_join` and `read_label`; there is no default and no fallback
column.

| kind | a row is available at | `read_asof` / `asof_join` | `read_label` |
| --- | --- | --- | --- |
| `column` | `row[column]` | yes | refused (`ValueError`) |
| `offset` | `row[anchor] + offset` | yes | refused (`ValueError`) |
| `label_only` | never, at any t | `LabelOnlyTable` | the whole table |
| `current_value` | never: the values are as of the scrape, not as of t | `CurrentValueTable` | refused (`ValueError`) |

- A row whose availability time is null is never available.
- The time column (`column`, or `anchor`) must be a timezone-aware timestamp
  (`TIMESTAMP WITH TIME ZONE`); any other type is a `ValueError`, since a naive one would be read
  in the session's time zone.
- `LabelOnlyTable` and `CurrentValueTable` are `ValueError`s, and neither is the other.

## Label columns

The rule kinds above apply per row, so a `column` or `offset` rule alone makes every column of an
available row available. A table that carries result or look-ahead columns on rows that are
otherwise known before the race needs those columns named: a `column` or `offset` rule takes
`label_columns`, a tuple of column names (default `()`).

- `read_asof` and `asof_join` drop the label columns from the result: they are not in the frame
  at all, not present and null.
- A listed label column the table does not have is a `ValueError` at query time, from both, so a
  misspelt name cannot let the real column through. Joining `on` a label column is a
  `ValueError`.
- A `label_only` or `current_value` rule with label columns, or label columns that list the rule's
  own `column` or `anchor`, is a `ValueError`.
- `read_label` still refuses every table that is not `label_only`: label columns of a `column` or
  `offset` table are not readable as labels through it.

## Views

- `read_asof(con, table, at_utc, *, rules)`: every row available at or before `at_utc`, the whole
  history rather than only the latest row, without the rule's label columns. `at_utc` must be timezone-aware (a naive time is a
  `ValueError`); any zone is read as the instant it names.
- `asof_join(keys, table, *, on, rules, con, t_col="t_utc")`: an ASOF LEFT JOIN. One row per key
  row, in the keys' order and with the keys' index and columns unchanged, plus the table's columns
  (other than `on` and the rule's label columns) from the latest row whose `on` columns equal the key's and whose availability
  time is at or before the key's `t_col`. A key with no such row is kept with those columns null.
  `t_col` must be timezone-aware. A table column that the keys also carry (other than `on`) is a
  `ValueError`. At equal availability times the row taken is unspecified.
- `read_label(con, table, *, rules)`: a `label_only` table whole, for labels and conditioners
  only.

Timezone-aware columns of the table come back in UTC, whatever the connection's time zone.
Table and column names come from config: the SQL quotes them as identifiers, and times and
offsets are passed as parameters.

## The board at t

`odds_timeseries` follows the board-at-t rule (`docs/specs/board-at-t.md`): for each (race, pool,
combination) the board at t is the latest capture with `fetched_utc <= t`. A capture whose
`snapshot_utc` is before t but which was fetched after t is not on the board. As a view this is a
`column` rule on `fetched_utc` joined on `race_id`, `pool` and `combination`.

A plain `fetched_utc` rule is not safe for `odds_timeseries` on its own. The compacted log
(`countries/japan/logger/compact.py` `SCHEMA`) puts the race's result on every capture: `dividend`
and `is_refund` are the payout of each combination, and `is_final_capture` is worked out from the
whole day's log. Without listing them in `label_columns`, the board at t carries the payout. Which
columns the shipped rule lists is an owner decision (see Rules per table).

## Config

`load_availability(country, path=None)` reads `src/horsetrack/countries/<country>/config/
availability.yaml` (through `core.config.load_country_yaml`), or `path`:

```yaml
tables:
  <table>:
    kind: column | offset | label_only | current_value
    column: <name>        # kind column only, required
    anchor: <name>        # kind offset only, required
    offset_min: <number>  # kind offset only, required; minutes, int or float
    label_columns: [<name>, ...]  # kinds column and offset only, optional
```

The result maps each table to an `Availability(table, kind, column, anchor, offset,
label_columns)` with `offset = timedelta(minutes=offset_min)` and `label_columns` a tuple (`()`
when the key is absent; flow or block lists both work). A key that does not belong to the entry's
kind, an unknown key, an unknown kind, a missing required key, a non-numeric `offset_min`, a
`label_columns` that is not a list of strings (a string, null, a mapping, nested lists) or that
lists the entry's `column` or `anchor`, `label_columns` on `label_only` or `current_value`, a file
without a
top-level `tables:` mapping or an entry that is not a mapping is a `ValueError`; a missing file is
a `FileNotFoundError`. With no file, every view is refused: the safe default.

## Rules per table

The per-table values are owner decisions (08 §9.18), pending on the needs-input of #80; they are
never implementation rulings. Until they are filled, no `japan/config/availability.yaml` ships.
The tables that must get a rule:

- every store table in `schemas.TABLES`: `meetings`, `races`, `runners`, `dividends`,
  `sectionals`, `incidents`, `running_comments`, `horses`, `horse_runs`, `horse_pedigree`,
  `jrdb_pre_race`, `jrdb_post_race`;
- `odds_timeseries` (the compacted odds log, #32).

What the store schema implies for those decisions:

- Only three tables carry a UTC time: `horses` (`profile_scraped_utc`), `horse_pedigree`
  (`scraped_utc`) and `odds_timeseries` (`fetched_utc`). A `column` or `offset` rule on any other
  table needs a UTC column the store does not have yet.
- `races` has `post_time_local` (a string) and no `post_utc`, so an offset rule anchored on the
  post time needs that column derived first.
- `runners` and `races` mix pre-race columns (draw, weights, jockey) and result columns
  (`finish_pos`, `finish_time_s`, `win_odds`) under one table rule: one rule cannot make the
  pre-race columns available before the post and the results only after it. Instead of making the
  whole table `label_only`, a `column` or `offset` rule can list the result columns in
  `label_columns`, so the pre-race columns enter a view and the results never do.
- `odds_timeseries` carries result and look-ahead columns on every capture (`dividend`,
  `is_refund`, `is_final_capture`; see The board at t): a rule that does not list them in
  `label_columns` puts the payout on the board at t.

Which columns go in `label_columns` for each shipped table is also an owner decision, pending on
the same needs-input of #80; no `availability.yaml` ships until it is answered.

## Known traps (07 §6)

- Profile pages show current values (career record, retirement, current rating), not the values
  at t: a scrape time says when the page was read, not when its values held.
- A same-day track variant or bias comes only from races already resulted by t, under the result
  rule, never from the whole day.
- JRDB data is revised, and each file has a release time: a rule must use when a version was
  released, not the race date, and a later revision must not overwrite what was known at t.
- Final odds are a label or a conditioner only, never a feature at t.
- Bets processed after the start move the 'final' price, so the final board is not a board any
  bettor saw before the off.
- Scratches: a runner scratched after t was on the board at t, and one scratched before t was
  not; a view must not read the scratch list as of the scrape.
