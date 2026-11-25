# Store: tables, keys and read access

Code: `src/horsetrack/core/store/schemas.py` (the column contracts), `tables.py` (read access)
and `build.py` (the build that writes the tables).
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
columns that come from the pages' fetch times (or, for `races.result_published_utc` and
`races.result_unpublished_utc`, the odds logger's poll times).

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
  came from, for sources whose other race pages use it), `rail_setting` (JRA's turf rail position,
  `A`, `B`, `C` or `D`, from the going page's 使用コース; null when not published and for every
  non-turf race, never a guess; filled by `jra.going.attach_rail_setting`, not by the results
  parsers), `result_published_utc` (when the race's official result was first seen published,
  ticket #225; null for history without a logged time), `result_unpublished_utc` (its lower bound,
  the last logger poll that did not show the result; null when unknown: a late sighting, shown on a
  run's first poll, or no logged time). All seven are `schemas.OPTIONAL_COLUMNS['races']`:
  `scripts/check_store_rebuild.py` accepts one that a copied table lacks and the rebuild holds all
  null (HKTrack's copy has neither `rail_setting` nor the two `result_*_utc` columns).
- **runners**: the meeting keys as on races, then `horse_id, horse_no, horse_name, brand_raw`,
  jockey and trainer `*_key, *_code, *_name`, `actual_wt_lb, declared_horse_wt_lb, draw,
  placing_raw, finish_pos` (null when not placed), `placing_code, dead_heat, lbw_raw, lbw_lengths`
  (0 for the winner), `running_positions, finish_time_s, win_odds` (final, decimal including the
  stake), `is_scratched`, and `rating, gear_raw` (the rating and gear at that run, where a source
  prints them). Optional: `bracket, actual_wt_kg, horse_wt_kg, horse_wt_change_kg, last_3f_s,
  win_fav_rank, sex, age` (`schemas.OPTIONAL_COLUMNS`; `scripts/check_store_rebuild.py` accepts
  one that a copied table lacks and the rebuild holds all null). A source that weighs in kg keeps
  the exact values in the kg columns; how it fills the lb columns is that source's own rule,
  stated in its adapter.
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
  dam_sire_id` (null where the page does not link them), `source_race_id, scraped_utc`. For jra,
  `horsetrack japan jrdb build` rewrites it when JRDB UKC is delivered: UKC's rows, plus the
  netkeiba rows for horses UKC lacks, in `jrdb.pedigree.OUTPUT_COLUMNS` (the netkeiba-only
  columns dropped; see `countries/japan/jrdb/tables.py`).
- **jrdb_pre_race**, **jrdb_post_race**: JRDB rows matched to the store's races and runners. Declared
  here: the key columns `race_id, race_date, horse_no, horse_id, race_key, jrdb_horse_id,
  matched_by`. The per-family data columns (`<family>_<column>`, from each family's spec under
  `<data dir>/jrdb/specs`) are written by `horsetrack japan jrdb build`
  (`countries/japan/jrdb/tables.py`), which also writes `jurisdiction`; a family not delivered
  still has its columns, all null.

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

## Build

`build.build_tables(source, *, page_types=None, workers=None, limit=None) -> BuildSummary` turns
the ingest manifest and the raw page cache into the tables above (`core/store/build.py`, ported
from HKTrack). It does no HTTP and only ever reads the manifest (opened read-only, so a live
backfill can keep writing).

- **Input**: the manifest rows with status `fetched` or `abandoned` and a sha256
  (`build.read_manifest(source)`), in `(page_type, key)` order. A page whose raw file is missing,
  is not valid gzip or is not the body the manifest recorded is listed as unreadable.
- **Parsing**: a page is parsed again only when its sha256 or its parser version
  (`parse_worker.parser_version`) differs from the parse cache's (`interim_dir(source)/
  parse_cache.sqlite`). Failures are never cached. With `workers >= 2` and at least 64 stale pages
  the pages go to spawned worker processes; a worker that dies is a `RuntimeError`.
  `page_types` and `limit` (pages per type, newest keys first) restrict what a run parses, never
  what it assembles: the tables always hold every cached page that still matches the manifest, so
  a partial run never drops rows an earlier run parsed.
- **Page types feed tables** as `schemas.FED_BY` says (`fixture` and `results` feed `meetings`;
  `results` races, runners, dividends, incidents; `sectional` sectionals; `corunning`
  running_comments; `horse` horses and horse_runs; `pedigree_card` horse_pedigree). Any other
  page type, or one whose adapter has no parser version, is skipped and counted.
- **Assembly**: pages whose parser flags a key mismatch are dropped, with every page sharing that
  key. Two pages claiming one race: the smaller page key wins; one horse: the latest
  `profile_scraped_utc` wins; the loser's rows leave every table its page type feeds. Races and
  runners get `meeting_id, race_date, venue, race_no` from the race id and a `season` label
  (`YYYY/YY`) starting in the month of the adapter's optional `season_start_month()` hook
  (default 1). Races take `result_published_utc` from the adapter's optional
  `result_published_utc()` hook (race_id -> aware datetime, stored as that instant in UTC; JRA's
  reads the odds logger's `<VENUE>.results.json` sidecars, `docs/specs/odds-log.md`): a race it
  does not name is null, a race_id no page holds adds no row, and an adapter without the hook gets
  an all-null column. `result_unpublished_utc` comes the same way, independently, from the
  optional `result_unpublished_utc()` hook (JRA's: the bounded races of the same sidecars only).
  Meetings start from the calendar pages (the newer page wins) and are upgraded by
  the results: `run`, `abandoned`, or `scheduled` without results. Runners take `rating` and
  `gear_raw` from `horse_runs` on `(horse_id, race_date)`, withdrawals excluded and one run per
  key. `horse_pedigree` keeps one row per horse, each column's first non-null value, newest
  card first. A bad cell becomes null, never a broken build.
- **Output**: the nine tables fed by `fixture` .. `horse` on every build, `horse_pedigree` only
  when the manifest holds pedigree pages (`schemas.SOURCE_TABLES`), never the JRDB tables; each
  written to a tmp file and moved into place. `processed_dir(source)/_build.json` holds the
  summary's fields (page counts, failures, rows per table, enrichment stats, `snapshot_id`: sha1
  over the included pages' `(page_type, key, sha256)`) plus `built_utc`; a report goes to
  `reports_dir()/store-build-<source>.md`.
- **Byte-stable**: two builds of the same inputs write identical Parquet bytes; `_build.json`
  differs only in `built_utc` and `seconds`.

## Not here yet

Upcoming race cards and the odds snapshots (`docs/specs/odds-log.md`) are outside this spec.
The one reader of the cards here is `horsetrack japan jrdb build` (#186): it joins JRDB rows to
every `processed_dir("jra")/upcoming/<meeting>/` holding both `races.parquet` and
`racecard_runners.parquet` as well as to history, so not-yet-run races reach `jrdb_pre_race`.

The post-build reconciliation checks live in `core/store/checks.py` (`run_checks(source, config=...)`,
ported from HKTrack with its tolerances); the report goes to `reports_dir()/store-check-<source>.md`.
