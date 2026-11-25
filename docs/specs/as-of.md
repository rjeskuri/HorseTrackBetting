# As-of views: what was known at a decision time

Code: `src/horsetrack/core/store/asof.py`. Input: a DuckDB connection holding the tables (e.g.
`core.store.tables.connect(source)`) and the availability rules of a country. Every feature
frame at (race, horse, t) is built through these views, so nothing available after t reaches it.

## Rule kinds

Every table that is read needs a rule. A table without one raises `NoAvailabilityRule` (a
`KeyError`) from `read_asof`, `asof_join` and `read_label`; there is no default rule.

| kind | a row is available at | `read_asof` / `asof_join` | `read_label` |
| --- | --- | --- | --- |
| `column` | `row[column]`, or `coalesce(row[column], row[fallback])` with a `fallback` | yes | refused (`ValueError`) |
| `offset` | `row[anchor] + offset`; on a DATE anchor, 00:00 of the date in `tz` + offset | yes | refused (`ValueError`) |
| `static` | always, at any t | yes, every row | refused (`ValueError`) |
| `label_only` | never, at any t | `LabelOnlyTable` | the whole table |
| `current_value` | never: the values are as of the scrape, not as of t | `CurrentValueTable` | refused (`ValueError`) |

- A row whose availability time is null is never available.
- The time columns (`column`, `fallback`, or an `anchor` without `tz`) must be timezone-aware
  timestamps (`TIMESTAMP WITH TIME ZONE`); any other type is a `ValueError`, since a naive one
  would be read in the session's time zone. The session's time zone never changes a view.
- The offset is an absolute duration: 1440 min is 24 hours even across a DST change.
- `LabelOnlyTable` and `CurrentValueTable` are `ValueError`s, and neither is the other.

### Fallback (column rules)

`fallback` names a second timezone-aware column, used only where `column` is null: a non-null
`column` wins even when the fallback is earlier, and a row with both null is never available. A
`fallback` of another type is a `ValueError` at query time; a `fallback` listed in
`label_columns`, or a `fallback` on any kind but `column`, is a `ValueError` at construction.

### Local dates (offset rules)

An `anchor` of type `DATE` is a local date (the race date), not an instant. Its rule takes `tz`,
an IANA zone name: the row is available at 00:00 of the date on the wall clock of `tz`, plus the
offset (for `Asia/Tokyo`, 15:00 UTC of the day before). A DATE anchor without `tz`, a `tz` on a
timezone-aware anchor and an anchor of any other type are `ValueError`s at query time; an unknown
zone is a `ValueError` at construction (or at query time, if DuckDB's zone database lacks it), and
so is a `tz` on any kind but `offset`.

### Static tables

`static` is for values that never change once known (a horse's sire, dam and broodmare sire):
every row is in every view at any t, and `asof_join` matches on `on` alone. It takes no time
field (`column`, `anchor`, `offset`, `tz`, `fallback`), only `label_columns`. It is not a label
table: `read_label` refuses it.

## Label columns

The rule kinds above apply per row, so a `column` or `offset` rule alone makes every column of an
available row available. A table that carries result or look-ahead columns on rows that are
otherwise known before the race needs those columns named: a `column`, `offset` or `static` rule
takes `label_columns`, a tuple of column names (default `()`).

- `read_asof` and `asof_join` drop the label columns from the result: they are not in the frame
  at all, not present and null.
- A listed label column the table does not have is a `ValueError` at query time, from both, so a
  misspelt name cannot let the real column through. Joining `on` a label column is a
  `ValueError`.
- A `label_only` or `current_value` rule with label columns, or label columns that list the rule's
  own `column`, `fallback` or `anchor`, is a `ValueError`.
- `read_label` still refuses every table that is not `label_only`: label columns of a `column`,
  `offset` or `static` table are not readable as labels through it.

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
`column` rule on `fetched_utc` joined on `race_id`, `pool` and `combination`. Rows ported from
HKTrack have no `fetched_utc`; the shipped rule's `fallback` on `snapshot_utc` counts those at
their snapshot time, and a row with neither stamp is never on the board.

A plain `fetched_utc` rule is not safe for `odds_timeseries` on its own. The compacted log
(`countries/japan/logger/compact.py` `SCHEMA`) puts the race's result on every capture: `dividend`
and `is_refund` are the payout of each combination, and `is_final_capture` is worked out from the
whole day's log. Without listing them in `label_columns`, the board at t carries the payout; the
shipped rule lists all three (see Rules per table).

## Config

`load_availability(country, path=None)` reads `src/horsetrack/countries/<country>/config/
availability.yaml` (through `core.config.load_country_yaml`), or `path`:

```yaml
tables:
  <table>:
    kind: column | offset | static | label_only | current_value
    column: <name>        # kind column only, required
    fallback: <name>      # kind column only, optional
    anchor: <name>        # kind offset only, required
    offset_min: <number>  # kind offset only, required; minutes, int or float
    tz: <IANA zone>       # kind offset only, optional; required at query time on a DATE anchor
    label_columns: [<name>, ...]  # kinds column, offset and static only, optional
```

The result maps each table to an `Availability(table, kind, column, anchor, offset,
label_columns, tz, fallback)` with `offset = timedelta(minutes=offset_min)` and `label_columns` a
tuple (`()` when the key is absent; flow or block lists both work). A key that does not belong to
the entry's kind, an unknown key, an unknown kind, a missing required key, a non-numeric
`offset_min`, a `fallback` or `tz` that is not a non-empty string, an unknown zone, a
`label_columns` that is not a list of strings (a string, null, a mapping, nested lists) or that
lists the entry's `column`, `fallback` or `anchor`, `label_columns` on `label_only` or
`current_value`, a file without a top-level `tables:` mapping or an entry that is not a mapping is
a `ValueError`; a missing file is a `FileNotFoundError`. With no file, every view is refused: the
safe default.

## Rules per table

The per-table values are owner decisions (08 §9.18), never implementation rulings: the owner's
answer on #80 (2026-11-24) accepted the proposed defaults, each conservative relative to JRA's
actual publication. They ship in `src/horsetrack/countries/japan/config/availability.yaml`, one
entry per store table in `schemas.TABLES` and for `odds_timeseries` (the compacted odds log,
#32); `tests/countries/japan/test_availability.py` checks the file against that answer.

| table | kind | available at | label columns (never in a view) |
| --- | --- | --- | --- |
| `odds_timeseries` | `column` | `fetched_utc`, else `fallback` `snapshot_utc` (HKTrack rows) | `dividend`, `is_refund`, `is_final_capture` |
| `meetings` | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 0 min | none |
| `races` | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 0 min | `finish_time_s`, `split_times_s`, `going`, `going_raw`, `result_published_utc`, `result_unpublished_utc` |
| `runners` | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 0 min | `finish_pos`, `placing_raw`, `placing_code`, `dead_heat`, `finish_time_s`, `lbw_raw`, `lbw_lengths`, `running_positions`, `last_3f_s`, `win_odds`, `win_fav_rank`, `horse_wt_kg`, `horse_wt_change_kg`, `declared_horse_wt_lb` |
| `dividends` | `label_only` | never; `read_label` only | n/a |
| `sectionals` | `label_only` | never; `read_label` only | n/a |
| `incidents` | `label_only` | never; `read_label` only | n/a |
| `running_comments` | `label_only` | never; `read_label` only | n/a |
| `horses` | `current_value` | never: profile pages show today's values | n/a |
| `horse_pedigree` | `static` | always: sire, dam and broodmare sire never change | `source_race_id`, `scraped_utc` |
| `horse_runs` | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 1440 min (the next day) | none |
| `jrdb_pre_race` | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 0 min | none |
| `jrdb_post_race` | `label_only` | never; `read_label` only | n/a |

- Nothing else is banned. JRDB tables are inputs to private models only (D21: no JRDB-derived
  figure in the public repository).
- Same-day results may be used once JRA has officially published them, but the store has no UTC
  post time yet: `races` has `post_time_local` (a string) and no `post_utc`. Until follow-up #227
  derives `post_utc`, the rules stay on the conservative side:
  - `dividends`, `sectionals`, `incidents` and `running_comments` are `label_only`; the owner's
    rule once `post_utc` exists is post + 30 min.
  - `horse_wt_kg`, `horse_wt_change_kg` and `declared_horse_wt_lb` (runners; the last is
    `horse_wt_kg` in pounds) and the same-day `going` and `going_raw`
    (races) are label columns; the owner's rule once `post_utc` exists is post − 60 min.
  - The result columns of `races` and `runners` stay label columns.
  - `result_published_utc` and `result_unpublished_utc` (races, #225: the odds logger's sighting
    of the published result and its lower bound) are label columns, never in a view at t: the
    owner's answer on #80 (2026-11-25), an owner decision, not an implementation ruling. Follow-up
    #227 may give them a rule that makes each available at its own value. `rail_setting` (races,
    #222) stays in the view: JRA announces the rail position in advance.
- A race-date rule opens a whole day's card at 00:00 JST of the race date: a column JRA changes
  later that day (a late scratch, a jockey change) is a trap the rule does not catch (see Known
  traps).
- A table outside the file is refused with `NoAvailabilityRule`.

### Accepted leak: results-derived card columns (owner decision, 2026-11-25)

Four columns that are not label columns are derived from the results page
(`countries/japan/jra/netkeiba/parse_results.py`), not from the card as it stood before the race:

| table | column | derived from |
| --- | --- | --- |
| `runners` | `is_scratched` | `placing_code` in (`WV`, `WX`) |
| `races` | `field_size` | the count of runners not scratched on the results page |
| `races` | `status` | `run` or `abandoned`, from the results page |
| `meetings` | `status` | whether any race of the meeting ran |

The owner's answer on #80 (2026-11-25) keeps all four in the views under the `race_date` 00:00
JST rule of their table, as an owner choice, not an implementation ruling. So a scratch or an
abandonment that happens on race day is visible in every view from 00:00 JST of the race date,
before it happened: a known, accepted leak. Follow-up #227 is to give these columns a post-based
rule once `races.post_utc` exists; until then a feature or model that reads them carries this
leak.

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
  not; in general a view must not read the scratch list as of the scrape. One exception is
  accepted: `runners.is_scratched`, `races.field_size`, `races.status` and `meetings.status` come
  from the results page and are in the views from 00:00 JST of the race date, so a race-day
  scratch or abandonment shows before it happened (owner decision, 2026-11-25; see Accepted leak:
  results-derived card columns). #227 is to give them a post-based rule.
