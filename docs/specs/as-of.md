# As-of views: what was known at a decision time

Code: `src/horsetrack/core/store/asof.py`. Input: a DuckDB connection holding the tables (e.g.
`core.store.tables.connect(source)`) and the availability rules of a country. Every feature
frame at (race, horse, t) is built through these views, so nothing available after t reaches it.

## Rule kinds

Every table that is read needs a rule. A table without one raises `NoAvailabilityRule` (a
`KeyError`) from `read_asof`, `asof_join` and `read_label`; there is no default rule.

| kind | a row is available at | `read_asof` / `asof_join` | `read_label` |
| --- | --- | --- | --- |
| `column` | `row[column]`, or `coalesce(row[column], row[fallback])` with a `fallback` | yes | refused (`ValueError`), unless `label_readable` |
| `offset` | `row[anchor] + offset`; on a DATE anchor, 00:00 of the date in `tz` + offset; with `via`, the anchor of another table's matching row + offset | yes | refused (`ValueError`), unless `label_readable` |
| `static` | always, at any t | yes, every row | refused (`ValueError`), unless `label_readable` |
| `label_only` | never, at any t | `LabelOnlyTable` | the whole table |
| `current_value` | never: the values are as of the scrape, not as of t | `CurrentValueTable` | refused (`ValueError`) |
| `current_value` with `in_view` | always, at any t: the values as exported, look-ahead accepted | yes, every row, as `static` | refused (`ValueError`) |

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

### Current values in the views (`in_view`)

A `current_value` table holds values as of its scrape or export, with no history, so by default
it never enters a view. A `current_value` rule with `in_view` (`Availability(..., in_view=True)`,
YAML `in_view: true`; default false) is the owner accepting that look-ahead for one table: the
table behaves as `static` in every view. Every row is in `read_asof` at any t, `asof_join`
matches on `on` alone, and `read_label` still refuses it (`ValueError`). The rule keeps kind
`current_value` and takes no time field.

- With `in_view` the rule takes the view fields `static` takes (`label_columns`,
  `label_prefixes`, `derived`, `gates`), but not `label_readable`: it is never a label table.
- `in_view` on any other kind, an `in_view` that is not a bool, or a `current_value` rule
  without it that has view fields is a `ValueError`.
- A view of such a table shows each value as of the export, not as of t: see the jockey and
  trainer masters under Rules per table.

### Via (offset rules)

A table without a time column of its own (the result tables carry `race_id`, not the post) reads
its anchor from another table: `via = Via(table, on)` takes the `anchor` of the `via.table` rows
whose `on` columns equal the row's. The anchor may be a timezone-aware or a derived column of that
table (see Derived columns), never one of its label or gated columns; the via table needs a view
rule of its own, without a `via`.

- The row is available at `max(anchor + offset, the via row's own availability)`: a time read
  from a row counts only once that row is known.
- A row with no matching via row, or whose matching row has a null anchor or is never available,
  is never available. Several matching rows: the latest time counts, and a null among them makes
  it null.
- A `via` rule takes no `tz`; `via` on any kind but `offset` is a `ValueError`.

## Derived columns

A `column`, `offset` or `static` rule (or `current_value` with `in_view`) takes `derived`, a
tuple of `LocalTime(name, date, time, tz)`: a column the view adds to the table's own, the UTC
instant of the `DATE` column `date` at the wall-clock time in the string column `time` (`H:MM`
or `HH:MM`, a one-digit hour too) in the IANA zone `tz`. A null date or time, or a time of any
other form (`25:00`, `9:5`, an ISO string), gives null: the conservative side, since a null time
never opens anything.

- `read_asof` and `asof_join` return it, timezone-aware, in UTC, from the row's own availability.
- It can be a `via` anchor or a gate's anchor (`races.post_utc`).
- A derived name the table already has, a `date` that is not a `DATE`, a `time` that is not a
  string, or a derived name listed as a label or gated column is a `ValueError`.

## Gated columns

A `column`, `offset` or `static` rule (or `current_value` with `in_view`) takes `gates`, a
tuple of `Gate(columns, anchor, offset, via=None)`: those columns stay in the frame of an
available row, null until `anchor + offset`, then the stored value. The anchor is the row's own
timezone-aware or derived column, or, with `via`, the anchor of another table's matching row, as
for an offset rule's `via` (including the via row's own availability).

- In `read_asof` a gated column is shown where its gate time is at or before `at_utc`; in
  `asof_join`, where it is at or before the key's `t_col`. A null gate time never shows it.
- A gated column the table does not have is a `ValueError` at query time. A gated column that is
  also a label column, the rule's time column, a derived column, or in two gates is a
  `ValueError` at construction. Joining `on` a gated column is a `ValueError`.

## Label columns

The rule kinds above apply per row, so a `column` or `offset` rule alone makes every column of an
available row available. A table that carries result or look-ahead columns on rows that are
otherwise known before the race needs those columns named: a `column`, `offset` or `static` rule
(or `current_value` with `in_view`) takes `label_columns`, a tuple of column names (default `()`).

- `read_asof` and `asof_join` drop the label columns from the result: they are not in the frame
  at all, not present and null.
- A listed label column the table does not have is a `ValueError` at query time, from both, so a
  misspelt name cannot let the real column through. Joining `on` a label column is a
  `ValueError`.
- A `label_only` rule, or a `current_value` rule without `in_view`, with label columns, or label
  columns that list the rule's own `column`, `fallback` or `anchor`, is a `ValueError`.
- `read_label` refuses every table that is neither `label_only` nor `label_readable`: label
  columns of a `column`, `offset` or `static` table are not readable as labels through it.
- `label_readable` (YAML `read_label: true`) on a `column`, `offset` or `static` rule lets
  `read_label` return the table whole as well: for result tables that enter the views after the
  post and are also read as labels.

### Label prefixes

A family of columns the store's schema does not name (JRDB's `tyb_` columns come from licensed
field specs outside the repository) cannot be listed column by column. A `column`, `offset` or
`static` rule (or `current_value` with `in_view`) also takes `label_prefixes`, a tuple of
non-empty strings (default `()`): every column whose name starts with one of them is a label
column too.

- `read_asof` and `asof_join` drop every such column, beside the `label_columns`. It is a prefix,
  not a substring: `tyb_` drops `tyb_body_weight`, not `kyi_tyb_ref` or `tyb`.
- A prefix that matches no column of the table is fine (a family the source has not sent yet).
  Joining `on` a column a prefix matches is a `ValueError`.
- A `label_only` rule, or a `current_value` rule without `in_view`, with label prefixes, a
  prefix that is empty or not a
  string, a `label_prefixes` that is not a tuple, or a prefix that matches the rule's own
  `column`, `fallback` or `anchor` is a `ValueError` at construction.

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
- `read_label(con, table, *, rules)`: a `label_only` or `label_readable` table whole, for labels
  and conditioners only.

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
    via: {table: <table>, match: [<name>, ...]}  # kind offset only, optional; no tz
    in_view: true | false  # kind current_value only, optional (default false)
    label_columns: [<name>, ...]  # kinds column, offset, static (and in_view) only, optional
    label_prefixes: [<prefix>, ...]  # kinds column, offset, static (and in_view) only, optional
    derived:              # kinds column, offset, static (and in_view) only, optional
      <name>: {date: <name>, time: <name>, tz: <IANA zone>}
    gated_columns:        # kinds column, offset, static (and in_view) only, optional
      - columns: [<name>, ...]
        anchor: <name>
        offset_min: <number>
        via: {table: <table>, match: [<name>, ...]}  # optional
    read_label: true | false  # kinds column, offset and static only, optional (default false)
    results_derived: [<name>, ...]  # kinds column, offset, static (and in_view) only, optional
```

`via` names its key columns `match`, not `on`: YAML 1.1 reads a bare `on` key as `true`.

The result maps each table to an `Availability(table, kind, column, anchor, offset,
label_columns, tz, fallback, label_prefixes)` with `offset = timedelta(minutes=offset_min)` and
`label_columns` and `label_prefixes` tuples (`()` when the key is absent; flow or block lists both
work), `via = Via(table, match)`, `derived` a tuple of `LocalTime`, `gates` a tuple of `Gate`,
`label_readable = read_label`, `in_view` (false when absent) and `results_derived` a tuple
(`()` when absent). `results_derived` lists the view columns taken from the results page (see
Accepted leak: results-derived card columns): the views return them unchanged, and the list only
makes the leak machine-readable for the model boundary; a name listed twice or a label column
there is a `ValueError` too. A key that does not belong to
the entry's kind, an unknown key, an unknown kind, a missing required key, a non-numeric
`offset_min`, a `fallback` or `tz` that is not a non-empty string, an unknown zone, a
`label_columns` that is not a list of strings (a string, null, a mapping, nested lists) or that
lists the entry's `column`, `fallback` or `anchor`, `label_columns` on `label_only` or
`current_value` without `in_view`, a `label_prefixes` that is not a list of non-empty strings or
that has a prefix matching the entry's `column`, `fallback` or `anchor`, `label_prefixes` on
`label_only` or `current_value` without `in_view`, an `in_view` that is not `true` or `false` or
on any kind but `current_value` (even `false`), `read_label` with `in_view`, a `via`, `derived`,
`gated_columns` or `read_label` entry of the wrong shape or on a kind that does not take it, a
file without a top-level `tables:` mapping or an entry that is not a mapping is
a `ValueError`; a missing file is a `FileNotFoundError`. With no file, every view is refused: the
safe default.

## Rules per table

The per-table values are owner decisions (08 §9.18), never implementation rulings: the owner's
answer on #80 (2026-11-24) accepted the proposed defaults, each conservative relative to JRA's
actual publication. They ship in `src/horsetrack/countries/japan/config/availability.yaml`, one
entry per store table in `schemas.TABLES`, for `odds_timeseries` (the compacted odds log, #32)
and for each JRA-VAN-only table, read from `processed/jravan/` parquet and not in
`schemas.TABLES` (the owner's request on #376, 2026-11-26; schemas in
`countries/japan/jravan/`, docs/specs/jravan.md); `tests/countries/japan/test_availability.py`
checks the file against those answers.

| table | kind | available at | label columns (never in a view) |
| --- | --- | --- | --- |
| `odds_timeseries` | `column` | `fetched_utc`, else `fallback` `snapshot_utc` (HKTrack rows) | `dividend`, `is_refund`, `is_final_capture` |
| `meetings` | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 0 min | none |
| `races` | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 0 min; derives `post_utc`; `going`, `going_raw` null until `post_utc` − 60 min; `finish_time_s` null until `post_utc` + 30 min | `split_times_s`, `result_published_utc`, `result_unpublished_utc` |
| `runners` | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 0 min; `horse_wt_kg`, `horse_wt_change_kg`, `declared_horse_wt_lb` null until the race's `post_utc` − 60 min, `finish_pos`, `dead_heat`, `finish_time_s` null until the race's `post_utc` + 30 min (both `via` races on `race_id`) | `placing_raw`, `placing_code`, `lbw_raw`, `lbw_lengths`, `running_positions`, `last_3f_s`, `win_odds`, `win_fav_rank` |
| `dividends` | `offset` | the race's `post_utc` + 30 min (`via` races on `race_id`); also `read_label` | none |
| `sectionals` | `offset` | the race's `post_utc` + 30 min (`via` races on `race_id`); also `read_label` | none |
| `incidents` | `offset` | the race's `post_utc` + 30 min (`via` races on `race_id`); also `read_label` | none |
| `running_comments` | `offset` | the race's `post_utc` + 30 min (`via` races on `race_id`); also `read_label` | none |
| `horses` | `current_value` | never: profile pages show today's values | n/a |
| `horse_pedigree` | `static` | always: sire, dam and broodmare sire never change | `source_race_id`, `scraped_utc` |
| `horse_runs` | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 1440 min (the next day) | none |
| `jrdb_pre_race` | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 0 min | every column starting `tyb_` (`label_prefixes: [tyb_]`) |
| `jrdb_post_race` | `label_only` | never; `read_label` only | n/a |
| `track_condition` | `column` | `available_utc` (= `published_utc`, the 発表 time printed on the going page or the yearly record; #90) | none |
| `win_sales` | `column` | `published_utc`: a live page's clock, a final (最終オッズ) page's printed post + 30 min, its fetch when it prints none (the owner's answer on #278, 2026-11-25) | none |
| `final_odds` (jravan) | `label_only` | never: a label or a conditioner, never a feature at t; `read_label` only (#376) | n/a |
| `pool_votes` (jravan) | `label_only` | never: a label or a conditioner, never a feature at t; `read_label` only (#376) | n/a |
| `jockeys` (jravan) | `current_value` with `in_view` | always: the master row as exported (JV-Data KS), no history; look-ahead accepted (#376, see below) | none |
| `trainers` (jravan) | `current_value` with `in_view` | always: the master row as exported (JV-Data CH), no history; look-ahead accepted (#376, see below) | none |
| `breeding_horses` (jravan) | `static` | always: pedigree is known at any t (#376) | none |
| `produce` (jravan) | `static` | always: pedigree is known at any t (#376) | none |
| `lineages` (jravan) | `static` | always: pedigree is known at any t (#376) | none |
| `career_counts` (jravan) | `offset` | `race_date` 00:00 JST (`tz` Asia/Tokyo) + 0 min, the card's rule (the owner's answer B on #407, 2026-11-28); the build drops a key whose kept version's データ作成年月日 is after its race date, or blank, and counts it in `created_after_race` | none |

- Nothing else is banned. JRDB tables are inputs to private models only (D21: no JRDB-derived
  figure in the public repository).
- The post-anchored rules (#227, the owner's answer on #80, 2026-11-24, points 3 and 4) hang on
  `races.post_utc`, which the races view derives (`derived`): `post_time_local` (`H:MM` or
  `HH:MM`, as the results parser writes it) on the Asia/Tokyo wall clock of `race_date`, so
  15:40 on 2026-11-28 is 2026-11-28 06:40 UTC and 9:50 is 00:50 UTC. The store has no `post_utc`
  column: `read_table` and the table names are unchanged. A null or malformed `post_time_local`
  (the official card's ISO string included) gives a null `post_utc`.
  - Horse weights (`horse_wt_kg`, `horse_wt_change_kg`, `declared_horse_wt_lb`; the last is
    `horse_wt_kg` in pounds) and the same-day `going` and `going_raw` are gated columns: in the
    frame from 00:00 JST, null before `post_utc` − 60 min of their race (absent at post − 60 min
    − 1 µs, present at post − 60 min).
  - `dividends`, `sectionals`, `incidents` and `running_comments` rows enter the views at the
    race's `post_utc` + 30 min (absent at + 30 min − 1 µs, present at + 30 min); `read_label`
    still returns them whole.
  - The finish (`races.finish_time_s`; `runners` `finish_pos`, `dead_heat` and `finish_time_s`)
    is a gated column on the same rule as same-day dividends (#305; the owner's answer on #96,
    2026-11-25): in the frame from 00:00 JST, null before the race's `post_utc` + 30 min
    (absent at + 30 min − 1 µs, present at + 30 min); runners read their race's `post_utc` by
    `race_id`. A same-day variant at t can read the finishes of the races resulted by t, never
    the current race's before its post + 30 min. `read_label` still refuses races and runners
    (#80).
  - A race without a post time, and a row whose `race_id` has no races row, never shows its
    weights, going, finish or results.
  - The other result columns of `races` (`split_times_s`) and `runners` (`placing_raw`,
    `placing_code`, `lbw_raw`, `lbw_lengths`, `running_positions`, `last_3f_s`, `win_odds`,
    `win_fav_rank`) stay label columns.
  - `result_published_utc` and `result_unpublished_utc` (races, #225: the odds logger's sighting
    of the published result and its lower bound) are label columns, never in a view at t: the
    owner's answer on #80 (2026-11-25), an owner decision, not an implementation ruling; #227
    left them labels. `rail_setting` (races, #222) stays in the view: JRA announces the rail
    position in advance.
- `jrdb_pre_race` merges JRDB's TYB family, the day-of file: race-day body weight, the board
  (`tyb_win_odds`, `tyb_odds_time`), same-day going, the paddock. Under the race-date rule those
  would be in every view from 00:00 JST, before they exist. The owner's answer on #80 (2026-11-25),
  an owner decision, makes every `tyb_` column a label (`label_prefixes: [tyb_]`) until a JRDB
  release-time rule exists; the `kyi_`, `cyb_` and `joa_` columns stay in the view. The TYB names
  come from licensed specs outside the repository, so the rule matches the prefix rather than
  listing columns.
- A race-date rule opens a whole day's card at 00:00 JST of the race date: a column JRA changes
  later that day (a late scratch, a jockey change) is a trap the rule does not catch (see Known
  traps).
- `track_condition` (#90) follows the owner's rule, not a ruling: his #136 request ("Ingest them
  as of their publication time", 2026-11-24), #228's outline step 4 ("available_utc =
  published_utc as in #136") and his answer on #87 (usable once JRA has officially published it,
  with its publication timestamp). A yearly record's rows count from the record's own 発表 time,
  so a race-day view never sees them; `asof_join` on (venue, race_date, surface) takes the latest
  publication at t.
- `horses` stays `current_value` without `in_view` (the owner's answer on #80): #376 puts only
  the jockey and trainer masters in the views, so a view of `horses` still raises
  `CurrentValueTable`.
- A table outside the file is refused with `NoAvailabilityRule`.

### Accepted look-ahead: the jockey and trainer masters (owner decision, #376)

The owner's request on #376 (2026-11-26): "the master row as exported is used, with no history".
The `jockeys` and `trainers` views return each master row as exported (JV-Data KS and CH, the
latest version per code), the same at every t, so a view at t shows the values as of the
export, not as of t. The columns that carry it:

- `is_retired` and `licence_revoked`: a retirement or a licence revocation after t shows at t (a
  jockey retired in 2014 shows as retired in a 2012 view).
- `affiliation` (`miho`, `ritto`, `nar`, `overseas`) and the jockey's `trainer_code` (his
  stable): a later transfer shows at t.
- `apprentice_code`: the apprentice status as of the export, not at t.
- `name` and `name_short`: a later name change shows at t.

A study that reads these columns at t must treat them as accepted look-ahead (the owner's
decision, #376, not an implementation ruling), not as values known at t.

### Accepted leak: results-derived card columns (owner decision, 2026-11-25)

Four columns that are neither label nor gated columns are derived from the results page
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
before it happened: a known, accepted leak. The JRA-VAN export cannot lift it: it holds each
race's and runner's final version only, no pre-race card (#401; docs/specs/jravan.md, "Card
versions (#401)").

The forward feed is the card log of race-day 出馬表 captures (#426; docs/specs/odds-log.md, "Card
log (ticket #426)"). The owner's decision on #426 (2026-11-28, an owner decision, not an
implementation ruling) sets its rule. Once a view reads the card log, `runners.is_scratched` and
`races.field_size` at t come from the latest capture with `fetched_utc <= t`, each capture
counting from its own fetch time with no extra lag; where there is no capture at or before t (a
day before the logger, a missed grid point), they fall back to the results-derived values above
under the race-date rule. `races.status` and `meetings.status` stay on the accepted leak, since
the card does not show an abandonment reliably. No view reads the card log yet (a follow-up ticket
wires it), so until then all four columns stay under the race-date rule, and a feature or model
that reads them carries this leak.

Since #417 each table's rule lists these columns under `results_derived` (`meetings: [status]`,
`races: [field_size, status]`, `runners: [is_scratched]`), so the model boundary can tag them
(`approximations`, below) from the rules rather than from a list in code.

## The model boundary (#417)

`read_asof` returns every column of a table outside its label exclusions, so a column a later
schema adds to a known table would reach a model by default. Japan's model boundary,
`countries/japan/feature_boundary.py`, sits on top of it (`read_asof` itself is unchanged):

- `approved_features(table, *, source="jra", rules=None, inputs=None)`: the columns approved as
  features at t. A column is approved only when an input of the inventory
  (`countries/japan/config/model_inputs.yaml`) whose role is `feature_at_t` supplies it
  (`{source, table, column}`; a supply's `filter` does not narrow it), the table's rule enters
  the views and the column is no label column of the rule. Nothing else is approved.
- `feature_view(con, table, at_utc, *, features, keys=(), ...)`: `read_asof` at `at_utc` with
  exactly `[*keys, *features]`. Features are always named; one refused feature refuses the whole
  request. A column neither config names is `UnclassifiedColumn`; a classified one that is not a
  feature at t (a label column, or supplied only as a label or a conditioner) is
  `FeatureNotApproved`, a `ValueError`. Keys are the caller's identifiers and pass unchecked,
  except that a label column is refused as a key.
- `approximations(table, columns)`: per column, the documented approximations its availability
  carries, from the rules only: `RESULTS_DERIVED` for a `results_derived` column,
  `PUBLICATION_PROXY` for a gated column (its anchor + offset stands in for an unrecorded
  publication time). A report built on the boundary names the predictors that carry either.
- `availability_fallback(table)`: `(column, fallback)` of a `column` rule with a `fallback`, else
  None. A row whose `column` is null is in the views at its `fallback` timestamp instead
  (`TIMESTAMP_FALLBACK`; for `odds_timeseries`, a capture without `fetched_utc` is on the board at
  its `snapshot_utc`, before it was fetched). It approximates rows, not a column, so
  `approximations` leaves the table's columns untagged; a report built on the boundary discloses
  the fallback for the values it reads from such a table, with the rows or races that used it.
- Authorized label reads (`read_label`) are unchanged.

A new column becomes a feature only through a config change: a `feature_at_t` supply in the
inventory, under the availability rules.

### Adopting the boundary

The avoidance study (`studies/avoidance.py`, #81) is the migrated caller: `view_at` reads the
board through `feature_view` (identifiers `race_id`, `pool`, `combination`, and those of
`snapshot_utc`, `fetched_utc`, `post_utc` the table has; feature `odds`), and its report lists
`runners.is_scratched` and `races.status`, the race and starter filters it reads from the store's
card, under "Documented approximations" with their `RESULTS_DERIVED` tag, and `odds` with the
tags and the `TIMESTAMP_FALLBACK` of the board rule its board reader uses (`StudyData.board_rules`,
the shipped rules when unset) and the count of races whose board at t holds a capture on that
fallback. `compute_study` carries the fallback (column and stand-in), the tags and the count in
its result, and the report renders the disclosure from the result alone: a board rule without a
fallback (a column rule without one, or an offset rule) gives no fallback line, and a fallback
whose primary column the board view drops is a `ValueError` before any variant is scored. The two filters are read as before from the store's card, outside
`feature_view`: `runners.is_scratched` is an approved feature at t (the inventory's
`scratches_at_t`), `races.status` is no feature at all, and the inventory gains no entry for it
(feature inclusion is the owner's).

Every other caller of `read_asof` or `asof_join` that builds a model or study frame (today
`studies/board_quality.py`; `studies/speed_run.py` and the other studies read the store
directly, and adopt it when they move to views) adopts it the same way, one ticket each:

1. Replace `read_asof(con, table, t, rules=...)` with `feature_view(con, table, t, features=[...],
   keys=[...], rules=...)`, naming the identifiers as keys and every value the study reads as a
   feature. Run its tests: an `UnclassifiedColumn` or `FeatureNotApproved` names a column the
   study read without an approval.
2. Such a column is either dropped from the frame, read through an explicit, documented
   non-feature path (a filter or a conditioner, as above), or proposed to the owner as an
   inventory entry; the coder never adds one.
3. The study's report lists its predictors' `approximations`.

An `asof_join` caller selects the joined columns through `approved_features` until the boundary
has a join of its own.

## Known traps (07 §6)

- Profile pages show current values (career record, retirement, current rating), not the values
  at t: a scrape time says when the page was read, not when its values held. The JRA-VAN jockey
  and trainer masters are the same, and are in the views by the owner's decision (see Accepted
  look-ahead: the jockey and trainer masters).
- A same-day track variant or bias comes only from races already resulted by t, under the result
  rule, never from the whole day.
- JRDB data is revised, and each file has a release time: a rule must use when a version was
  released, not the race date, and a later revision must not overwrite what was known at t.
  Until such a rule exists, `jrdb_pre_race` opens at 00:00 JST of the race date and its day-of
  TYB columns (`tyb_`) are kept out as labels (see Rules per table).
- Final odds are a label or a conditioner only, never a feature at t: JRA-VAN's `final_odds` and
  `pool_votes` are `label_only` (#376).
- Bets processed after the start move the 'final' price, so the final board is not a board any
  bettor saw before the off.
- Scratches: a runner scratched after t was on the board at t, and one scratched before t was
  not; in general a view must not read the scratch list as of the scrape. One exception is
  accepted: `runners.is_scratched`, `races.field_size`, `races.status` and `meetings.status` come
  from the results page and are in the views from 00:00 JST of the race date, so a race-day
  scratch or abandonment shows before it happened (owner decision, 2026-11-25; see Accepted leak:
  results-derived card columns). Once a view reads the card log, `is_scratched` and `field_size`
  come from the latest race-day card capture at or before t, falling back to this leak where
  there is none, and the two status columns keep the leak (owner decision on #426, 2026-11-28);
  no view reads the card log yet.
