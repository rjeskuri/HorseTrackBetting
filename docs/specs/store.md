# Store: tables, keys and read access

Code: `src/horsetrack/core/store/schemas.py` (the column contracts) and `tables.py` (read access).
The store is jurisdiction-agnostic: the source is always an argument, never a constant, and every
table carries a `jurisdiction` column. The table and column set starts from HKTrack's normalised
tables; anything one source alone needs is an optional column, null for the other sources.

## Layout

One Parquet file per table and source:
`$HORSETRACK_DATA_DIR/processed/<source>/<table>.parquet`
(`core.paths.processed_dir(source) / f"{table}.parquet"`), where `<source>` is the data-source key
(`jra`, `nar`, `hkjc`). The data directory is resolved on every call. A file of any other name in
that folder is not a store table.

## Column kinds

| kind | Arrow type | pandas dtype from `read_table` |
| --- | --- | --- |
| `int` | int64 | `Int64` (nullable) |
| `float` | float64 | `float64` (NaN for missing) |
| `str` | string | `object` holding `str` (None for missing) |
| `bool` | bool | `boolean` (nullable) |
| `date` | date32 | `object` holding `datetime.date` (the local race date) |
| `utc` | timestamp[us, UTC] | `datetime64[us, UTC]` |
| `int_list` | list<int32> | `object` holding numpy arrays |
| `float_list` | list<float64> | `object` holding numpy arrays |

Missing values are null, never sentinel numbers. Where parsing can lose information the page's
text is kept next to the parsed value (`*_raw`). Timestamps are UTC and only appear in `*_utc`
columns that come from the pages' fetch times.

## Tables

Each table's full column list and order is `schemas.TABLES[name].columns`; this section gives the
grain, the key and the meaning. The sort key is the order rows are stored in.

| table | grain | sort key |
| --- | --- | --- |
| `meetings` | one race meeting | `race_date, venue` |
| `races` | one race | `race_id` |
| `runners` | one declared runner in a race (the modelling grain) | `race_id, horse_no, horse_id` |
| `dividends` | one paying combination of a pool | `race_id` (page order kept within a race) |
| `sectionals` | one runner's section of a race | `race_id, horse_no, section_no` |
| `incidents` | one incident text about a runner | `race_id` |
| `running_comments` | one runner's running comment | `race_id, horse_no` |
| `horses` | one horse, static attributes | `horse_id` |
| `horse_runs` | one run on a horse's form table | `horse_id, race_date, race_index` |
| `horse_pedigree` | one horse's pedigree | `horse_id` |
| `jrdb_pre_race` | one JRDB pre-race row matched to a runner | `race_id, horse_no` |
| `jrdb_post_race` | one JRDB post-race row matched to a runner | `race_id, horse_no` |

- **meetings**: `meeting_id, race_date, venue, season, day_night, surface_mix, status` (`run` or
  `abandoned`).
- **races**: `race_id, race_index, race_name, race_class, rating_hi, rating_lo, distance_m, surface`
  (`turf`, `dirt`, `awt`, `jump`; jump races are kept and flagged so), `course_config, going, prize,
  post_time_local, field_size, finish_time_s, split_times_s` (cumulative leader times), `status`.
  Denormalised meeting keys: `meeting_id, race_date, venue, race_no, season`. Optional:
  `weight_type, race_class_raw, going_raw, source_race_key` (the key of the results page the race
  came from, for sources whose other race pages use it).
- **runners**: the meeting keys as on races, then `horse_id, horse_no, horse_name, brand_raw`,
  jockey and trainer `*_key, *_code, *_name`, `actual_wt_lb, declared_horse_wt_lb, draw,
  placing_raw, finish_pos` (null when not placed), `placing_code, dead_heat, lbw_raw, lbw_lengths`
  (0 for the winner), `running_positions, finish_time_s, win_odds` (final, decimal including the
  stake), `is_scratched`, and `rating, gear_raw` (the rating and gear at that run, where a source
  prints them). Optional: `bracket, actual_wt_kg, horse_wt_kg, horse_wt_change_kg, last_3f_s,
  win_fav_rank, sex, age`. A source that weighs in kg keeps the exact values in the kg columns; how
  it fills the lb columns is that source's own rule, stated in its adapter.
- **dividends**: `race_id, pool` (a normalised pool code), `combination` (`"4"`, `"4,7"`,
  `"4,7,9"`), `dividend` (per the source's unit stake, from its config), `is_refund`.
- **sectionals**: `race_id, horse_id, horse_no, section_no, section_pos, section_margin_lengths,
  section_time_s, section_margin_raw, sub_split_times_s`.
- **incidents**: `race_id, horse_id` (null when unmatched), `horse_no, text`.
- **running_comments**: `race_id, horse_id, horse_no, gear_raw, comment`.
- **horses**: `horse_id, name, origin, colour, sex, import_type, sire, dam, dam_sire, is_retired,
  profile_scraped_utc`. Static attributes only: anything time-varying on a profile page is as of the
  scrape and is not a feature.
- **horse_runs**: a horse's form table, `horse_id, race_date, race_index, venue, surface,
  course_config, distance_m, going, race_class, draw, rating, trainer_name, jockey_name, lbw_raw,
  win_odds, actual_wt_lb, running_positions, finish_time_s, declared_horse_wt_lb, gear_raw,
  placing_raw, race_id`.
- **horse_pedigree**: written only for sources with pedigree pages. `sire_name, dam_name,
  dam_sire_name` as printed, their `*_norm` forms (NFKC and casefold), `sire_id, dam_id,
  dam_sire_id` (null where the page does not link them), `source_race_id, scraped_utc`.
- **jrdb_pre_race**, **jrdb_post_race**: JRDB rows matched to the store's races and runners. Declared
  here: the key columns `race_id, race_date, horse_no, horse_id, race_key, jrdb_horse_id,
  matched_by`. The per-family data columns (`<family>_<column>`) are declared by the ticket that
  builds these tables.

Withdrawals and placing codes differ by site: each source adapter maps its own codes before rows
reach the store, and documents the mapping.

## Read access

- `tables.read_table(source, name, columns=None) -> pandas.DataFrame`: the declared columns (or the
  requested subset), in declared or requested order whatever the file's order, cast to their
  declared types. A name that is not in `schemas.TABLES` (case-sensitive) raises `ValueError`, as
  does a column the table does not declare. A missing file raises `FileNotFoundError` naming the
  table and the source.
- `tables.connect(source) -> duckdb.DuckDBPyConnection`: an in-memory DuckDB with one view per
  declared table whose file exists for that source, named after the table. File paths are quoted
  as SQL strings, so a data directory with a `'` in it works.

## Not here yet

The build that writes these tables (parsers, incremental rebuilds, the build report), the
reconciliation checks, upcoming race cards and the odds snapshots
(`docs/specs/odds-log.md`) are outside this spec.
