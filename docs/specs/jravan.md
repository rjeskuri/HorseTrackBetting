# JRA-VAN DataLab (JV-Data)

Code: `src/horsetrack/countries/japan/jravan/`. Source key: `jravan` (the store's source, the
registry source, `paths.raw_dir('jravan')`). Epic #359.

## Where the data and the specification live

Both are licensed (D21) and stay under `$HORSETRACK_DATA_DIR`, never in git, a ticket or a PR;
outputs built on them carry counts, dates and rates only and go through `write_report`.

- The export (#196): `raw/jravan/`, with its `README.md` and `manifest.json`.
  `stored/<DATASPEC>/<RT>/<YYYY>.jvd` (or `all.jvd`) and
  `timeseries/0B41|0B42/<YYYY>/<YYYYMMDD>.jvd` (0B41 holds O1 records, 0B42 O2). One record per
  CRLF line, cp932, the first 2 bytes the record type. `probe/`, `stored_shift/` and `_incoming/`
  are not part of the export.
- The JV-Data 4.9.0.1 specification workbook: `specs/jravan/JV-Data4901.xlsx`, placed by the
  owner. It is a precondition, not an owner step: `horsetrack japan jravan check` fails, naming the
  path, when it is absent.

## What is built

Every command below is `horsetrack japan jravan <command>`, reads `raw/jravan/` and, apart from
`check`, `dump`, `reconcile` and `card-versions`, writes the `jravan` source's tables under
`processed/jravan/` (how they sit beside the store's: docs/specs/store.md, "The jravan source").
None of it runs on its own: there is no scheduler, no manifest and no fetcher, and a rebuild is a
command run by hand. The same inputs give the same bytes.

| command | module | writes | table kind | availability rule (docs/specs/as-of.md) |
|---|---|---|---|---|
| `check` | `check.py` | nothing | the workbook against the export | n/a |
| `races` | `races.py` | `races`, `meetings` | store tables | those of `races`, `meetings` |
| `runners` | `runners.py` | `runners`; fills `races.finish_time_s` | store table | that of `runners` |
| `dividends` | `dividends.py` | `dividends` | store table | that of `dividends` |
| `final-odds` | `odds.py` | `final_odds` | jravan-only | `label_only` |
| `votes` | `votes.py` | `pool_votes` | jravan-only | `label_only` |
| `timeseries` | `timeseries.py` | `odds_timeseries/<YYYY-MM-DD>.parquet` | the odds log's schema | that of `odds_timeseries` |
| `masters` | `masters.py` | `horses`, `horse_pedigree`; `jockeys`, `trainers` | store tables; jravan-only | `horses` refused, `horse_pedigree` `static`; `current_value` with `in_view` |
| `bloodlines` | `bloodlines.py` | `breeding_horses`, `produce`, `lineages` | jravan-only | `static` |
| `workouts` | `handicap/workouts_jravan.py` | `workouts` | jravan-only | none: its readers apply the module's own publication rule (#229), and the as-of views refuse it |
| `dump` | `dump.py` | `interim/jravan/records/` | not tables | none |
| `reconcile` | `reconcile.py` | `reports/jravan-reconcile.md` | not a table | n/a |
| `card-versions` | `card_versions.py` | `reports/jravan-card-versions.json` only | not a table | n/a |

Order: `races` before `runners` (which rewrites the races file's `finish_time_s`); `races` and
`dividends` before `timeseries`, whose `post_utc`, `minutes_to_post`, `dividend` and `is_refund`
are null without them. The helper modules
are `spec.py` (the workbook), `records.py` (decoding and the output helpers) and `keys.py` (race
keys and times). Every command's report goes through `write_report` and carries counts only.
The jravan-only tables are outside `schemas.TABLES`: `read_table` refuses them, and a reader opens
their Parquet file itself.

## Layouts (#361)

`horsetrack.countries.japan.jravan.spec` reads the workbook at run time, so the repository holds
no field offset, length, repeat count, description or code-table text:

- `spec_path()`: `<data dir>/specs/jravan/JV-Data4901.xlsx`, resolved on every call. Every reader
  raises `FileNotFoundError` naming it when the file is missing.
- `record_types(path=None) -> dict[str, int]`: record type ID to record length in bytes (CRLF
  included), for every section, without validating the layouts.
- `record_spec(record_type, path=None) -> RecordSpec`: the section's `Item`s (name, 1-based
  start, bytes of one occurrence, repeat, key flag, initial value, description, sub-items; a
  sub-item's start is relative to its group). It validates only its own section, so a section no
  parser uses (the real-time records) never stops a read: the items cover bytes 1 to the length
  (a repeated item spans repeat x length), each group's sub-items cover its width, with no gap or
  overlap, and the last item is the 2-byte record separator; otherwise a `ValueError` naming the
  record type and the item. An unknown type is a `LookupError`.
- `code_table(key, path=None) -> CodeTable`: by 4-digit number (`'2010'`) or name; `values` maps
  each code to its first content column, `extra` to the other columns in order.

The workbook is read with `zipfile` and `xml.etree` (shared strings without their phonetic runs,
inline strings, numbers as their cached values), the sheets `フォーマット` and `コード表` found by
name through `xl/workbook.xml` and its rels, and cached per (path, size, mtime_ns). Names lose
their phonetic text, every half- and full-width space and a surrounding `< >`. On the format sheet
a section runs from a `レコード長` title row to the next; rows whose column B is not an item number
and that are not sub-items (the header, labels, notes, blanks) are skipped, and a sub-item with no
position follows on from the one before it. The record type ID is the two characters quoted at the
start of item 1's description. On the code sheet a table runs from a `NNNN．name` title row to the
next; a code longer than the table's width is a note, not a code.

`horsetrack japan jravan check` loads the workbook, then for every `stored/<DATASPEC>/<RT>` folder
and `timeseries/0B41` and `0B42` (when present) checks that the layout validates and that its
length equals the byte width of the first record of the folder's first `.jvd` file. It prints one
line per folder (`stored/RACE/RA: ok, 62 files`, or the failure with both widths) and a summary,
and exits 1 on any failure, a missing workbook or a missing export. It reads only each first line
and prints no record content.

## Decoding (#362)

`horsetrack.countries.japan.jravan.records` turns a record type's bytes into named byte arrays,
so each family maps fields instead of slicing bytes:

- `decode(data, spec) -> dict[str, ndarray]`: the records become one uint8 matrix and every leaf
  item a fixed-width view of its columns, an `S<length>` array. Keys are a top-level item's name
  (`開催年`) or `group/sub` (`単勝オッズ/オッズ`); a name that occurs again at the same level is
  `name#2`, `name#3` (inside a group too). Shapes are (n,), (n, repeat) or
  (n, repeat, sub-repeat), a repeat of 1 adding no axis. Items named `予備` (at any level) and
  the record separator (the layout's last item) are left out. Records split on LF (a CR before it
  is dropped), empty lines are skipped, and the last record may lack its separator. A record that
  is not `spec.length - 2` bytes is a `RecordLengthError` (a `ValueError` with `record_type`,
  `record` (1-based), `length` and `expected`); one of the right width that does not start with
  the record type is a `RecordTypeError` (a `ValueError` with `record_type`, `record` and
  `found`). `leaf_axes(spec)` gives each key, in layout order, with its number of repeat axes.
- `iter_file(path, spec, *, chunk_records=20_000)`: the same dicts, lazily, for at most
  `chunk_records` records each and at most `CHUNK_BYTES` (16 MiB) of records, so a 7 GB H6 or CK
  file streams in constant memory (about 300 MB above the interpreter on a 2.4 GB synthetic file
  of 100 KB records). Record numbers in errors count the whole file.
- `text(raw)` (cp932, half- and full-width spaces trimmed at both ends, blank is None),
  `integer(raw)` (digits with an optional sign and space padding as float64, anything else such
  as `----`, `****`, blank or `12A4` is NaN) and `scaled(raw, places)` (`integer` with implied
  decimals: `b'0123'`, 1 is 12.3). Each converts every distinct value once and keeps the input's
  shape.
- `latest_versions(frame, key, *, created='データ作成年月日', kubun='データ区分')`: one row per
  key, the latest creation date (a tie goes to the later row); a key whose kept row is a deletion
  (`kubun` `'0'`) is dropped. Kept rows stay in input order with their index labels.
- Output helpers every family uses: `write_parquet(table, path)` writes through a tmp file in the
  same folder and `os.replace` (zstd, no wall-clock metadata, so the same table gives the same
  bytes); `BuildReport(name, counts, by_year, unmapped, failures)` with `to_json()` (sorted keys,
  counts and codes only, no record content); `write_build_report(report)` writes
  `<reports dir>/jravan-<name>.json` through `core.eval.reports.write_report` with
  `sources=('jravan',)`, which refuses a path inside the repository.

`horsetrack.countries.japan.jravan.keys` maps race keys and times to the store's:

- `race_id(year, mmdd, venue_code, race_no)`: the jra store's id, `jra:2026-11-28:KYO:11`
  (`jra.netkeiba.common.store_race_id`), the venue's short code looked up by number in
  `config/jra.yaml`; `meeting_id(year, mmdd, venue_code)`: `jra:2026-11-28:KYO`. None for a venue
  outside JRA, an invalid date or race number.
- `source_race_key(year, venue_code, kai, nichi, race_no)`: the 12-digit netkeiba race id
  (`202608050711`) the jra store keeps in `races.source_race_key`; None for a non-JRA venue or a
  malformed part.
- `jurisdiction(venue_code)`: `'jra'` for 01-10, `'nar'` for any other two-digit number,
  `'overseas'` for any other non-blank code, None for `'00'` or blank.
- `announced_utc(year, race_mmdd, mmddhhmm)`: a JST minute (UTC+9, no DST) as an aware UTC
  datetime; a month after the race's month is in the year before (a December announcement for a
  January race). None for `00000000`, blank or an invalid time.
- `post_time_local(hhmm)`: `'HH:MM'`, zero-padded, JST wall clock (the netkeiba results parser
  keeps the page's text, which may read `9:50`); None for `0000`, blank or an invalid time.

## RA: races and meetings (#364)

`horsetrack.countries.japan.jravan.races`; `horsetrack japan jravan races [--year YYYY ...]`.

- `build_races(*, years=None, raw=None, out=None)` reads `stored/RACE/RA/<YYYY>.jvd` one year at
  a time (only the items it maps), keeps one version per race (`records.latest_versions` on the
  layout's key items), keeps the JRA venues (01-10) and writes `processed/jravan/races.parquet`
  (sorted by `race_id`) and `meetings.parquet` (by `race_date`, `venue`) in the store's columns
  and types. The same inputs give the same bytes. The report, `<reports dir>/jravan-races.json`,
  holds per year `records`, `versions_dropped`, `deletions`, `races_written`, `non_jra_nar`,
  `non_jra_overseas`, `non_jra_unknown` and `unmapped` (a count), and `unmapped` as store column
  (`race_class`, `weight_type`, `surface`, `going`) -> code -> count over the races written.
- Columns use the jra store's vocabulary (`jra/netkeiba/parse_results.py`): ids and keys from
  `keys.py`; `season` the calendar-year label; `race_class` from the youngest-age condition code,
  then the grade for an open or unknown class (jump grades as their flat code); `surface` and
  `course_config` from the track code; `going` from the going code of the race's own surface
  (turf for a jump race); `prize` the winner's prize in yen; `split_times_s` cumulative leader
  times (blank and zero laps dropped); `status` `abandoned` for data kubun 9, else `run`;
  `rail_setting` the course letter A-D of a turf race, else null. `race_index`, `rating_*`,
  `finish_time_s` and the `result_*_utc` columns are null. Each mapping's code values are
  constants in `races.py`; a code a mapping lacks is null and counted. A blank or all-zero
  code is JV-Data's "unset": null (its `*_raw` column too), counted only on a race that was run.
- Meetings: one row per `meeting_id`, `run` when any race ran, else `abandoned`; `day_night` and
  `surface_mix` null.

The run on the export (2026-11-26; the export ends 2026-11-23), counts only:

| | records | versions dropped | deletions | races written | NAR | overseas | unmapped |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1954-1985 (21 year files) | 156 | 0 | 0 | 156 | 0 | 0 | 132 |
| 1986-2009 | 136,041 | 474 | 0 | 81,985 | 51,559 | 2,023 | 12,741 |
| 2010-2026 | 107,385 | 79 | 0 | 58,750 | 46,933 | 1,623 | 0 |
| all 62 year files | 243,582 | 553 | 0 | 140,891 | 98,492 | 3,646 | 12,873 |

140,891 races (422 abandoned) on 11,970 meetings (30 abandoned), race ids unique. Every race has
a surface. Unmapped codes, all before 2002 (the last in 2001):

- `race_class` (condition codes the mapping lacks): `009` 6,742, `004` 4,542, `015` 867, `014`
  373, `003` 132, `007` 63, `008` 19, `006` 12, `001` 1 and the unset `000` 1 (12,752 races
  without a class).
- `weight_type`: the unset `0` 121 (races before 1986).
- `going`: none counted; the 273 abandoned races with an unset going are null.
- Not counted (`rail_setting` is not a counted column): the turf course letters `A1` (2,277)
  and `A2` (321), all before 2010, give a null `rail_setting`. 76 races before 1986 have no post
  time.

## SE: runners (#365)

`horsetrack.countries.japan.jravan.runners`; `horsetrack japan jravan runners [--year YYYY ...]`.

- `build_runners(*, years=None, raw=None, out=None)` reads `stored/RACE/SE/<YYYY>.jvd` one year
  at a time (only the items it maps), keeps one version per runner (`records.latest_versions` on
  the layout's key items: the race key, 馬番 and 血統登録番号), keeps the JRA venues (01-10) and
  writes `processed/jravan/runners.parquet` in the store's columns and types, sorted by
  (`race_id`, `horse_no`, `horse_id`). It then sets `finish_time_s` in `<out>/races.parquet` to
  the time of each race's `finish_pos` 1 runner (the lower horse number on a dead heat, null for a
  race without one) and rewrites that file whole, every other column unchanged; a missing races
  file is a `FileNotFoundError` before anything is read (build the races first). The same inputs
  give the same bytes, and a second fill changes nothing. The report,
  `<reports dir>/jravan-runners.json`, holds per year `records`, `versions_dropped`, `deletions`,
  `runners_written`, `non_jra_nar`, `non_jra_overseas`, `non_jra_unknown` and `unmapped` (a
  count), in `counts` the totals and `races_timed`, and `unmapped` as field (`placing`, `sex`,
  `dead_heat`, `margin`) -> code -> count over the runners written.
- Columns use the jra store's vocabulary (`jra/netkeiba/parse_results.py`, whose placing codes,
  scratched codes, margin names, beaten-lengths sum and pound conversion the mapping reuses): the
  race keys and `season` as on the jravan races; `horse_id` the top-level 血統登録番号 as printed;
  `draw` = `horse_no`; `jockey_key` / `trainer_key` `jockey:<code>` / `trainer:<code>` from the
  codes, the names the short names; `actual_wt_lb` and `declared_horse_wt_lb` `round(kg *
  2.20462)`; `horse_wt_kg` null when not weighed (999) or zero, `horse_wt_change_kg` signed by
  増減符号, null for 999 or blank. 異常区分コード gives `placing_code` WV (取消), WX (発走除外,
  競走除外), PU (競走中止) and DISQ (失格) with no `finish_pos` and `placing_raw` `取`, `除`, `中`
  or `失`; a demotion keeps the official place with `placing_raw` `<n>(降)`; a remount after a fall
  is a plain place. 同着区分 gives `dead_heat` and `placing_code` DH on a placed runner;
  `is_scratched` is true for WV and WX. `lbw_raw` is 着差コード as printed (blank or all zero
  null); `lbw_lengths` the running sum down the placed order (by place, then horse number), 0 for
  the winner and null from a blank or unreadable margin on. `running_positions` the four corner
  positions without the unused ones; `finish_time_s`, `last_3f_s` (999 and zero null), `win_odds`
  and `win_fav_rank` (zero null) in store units; `sex` H, M, G; `age` as printed. `brand_raw`,
  `rating` and `gear_raw` are null. Each mapping's code values are constants in `runners.py`; a
  code a mapping lacks is null and counted.
- Runners of abandoned races (RA data kubun 9) are written, without a place; their undrawn
  runners carry 馬番 99 as printed.

The run on the export (2026-11-26; the export ends 2026-11-23, about 60 s, 2 GB peak), counts only:

| | records | versions dropped | deletions | runners written | NAR | overseas | unmapped |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1954-1985 (21 year files) | 182 | 0 | 0 | 182 | 0 | 0 | 0 |
| 1986-2009 | 1,591,680 | 831 | 0 | 1,056,283 | 530,377 | 4,189 | 0 |
| 2010-2026 | 1,322,806 | 2,827 | 26 | 829,297 | 484,891 | 5,765 | 0 |
| all 62 year files | 2,914,668 | 3,658 | 26 | 1,885,762 | 1,015,268 | 9,954 | 0 |

Every runner's race is in `races`. Zero unmapped codes. Placing codes: 9,281 PU, 7,247 DH, 4,479
WV, 2,891 WX, 331 DISQ. 140,429 of the 140,469 run races carry the winner's time; the other 40
(all before 1990) have runner records but none placed first. 153 run races, all before 1990, have
a starter count other than their field size (the early years hold only part of each field). 1,960
placed runners have no beaten lengths: 305 placed non-winners have a blank margin (304 of them
demoted, 1991-2023, most in 2004-2012), which leaves them and every runner behind them unknown. 164 runners of abandoned races carry 馬番 99.

## HR: payouts (#366)

`horsetrack.countries.japan.jravan.dividends` writes the store's `dividends` table for `jravan`
(`processed/jravan/dividends.parquet`) in the jra store's shape, so the reconciliation compares
row for row:

- `build_dividends(*, years=None, raw=None, out=None) -> BuildReport` reads
  `<raw>/stored/RACE/HR/<YYYY>.jvd` (`raw` the export root, default `raw/jravan`; `years` None
  reads every year file, a named year without a file is a `FileNotFoundError`), keeps JRA venues
  (競馬場コード 01-10) and one version per race (`latest_versions` on 開催年, 開催月日,
  競馬場コード, 開催回, 開催日目, レース番号), and writes `<out>/dividends.parquet` (default
  `processed/jravan`) through `write_parquet`, stable-sorted by `race_id`.
- `dividends_from_records(cols)` maps decoded HR arrays to rows. Each paid combination is one row:
  `jurisdiction` `jra`, `race_id` (`keys.race_id`), `pool` (`WIN`, `PLACE`, `BRACKET_QUINELLA`,
  `QUINELLA`, `QUINELLA_PLACE`, `FORECAST`, `TRIO`, `TIERCE`, from the groups 単勝払戻,
  複勝払戻, 枠連払戻, 馬連払戻, ワイド払戻, 馬単払戻, 3連複払戻, 3連単払戻), `combination`
  (the group's 馬番 or 組番 split into its 1, 2 or 3 equal-width legs, joined with `,`, leading
  zeros dropped; FORECAST and TIERCE in finishing order, the others ascending), `dividend` (払戻金,
  yen per 100 as printed) and `is_refund` false. Within a race the pools come in that order, then
  the record's slot order (dead heats give several slots). 人気順 is not used.
- A slot is a row only when every leg and the payout are positive integers: blank slots, zeros
  (an unsold pool prints zeros and a blank payout) and zero or blank payouts write nothing, so a
  pool not yet sold in a year has no rows, never zeros.

The pool flags, each `1` when set, per pool (単勝 複勝 枠連 馬連 ワイド 馬単 3連複 3連単):

- 不成立フラグ (not established): the pool was not formed (no valid sales or no result to pay
  on). The pool writes no row; its slots hold zeros or the amount paid back.
- 特払フラグ (special payout): no ticket on the winning combination was sold, so every ticket is
  paid a fixed amount back. The pool writes no row.
- 返還フラグ (refund): some tickets were refunded, typically a horse scratched after sales opened.
  It writes no row of its own; the pool's paid combinations are still rows.
- 返還馬番情報 (one `1` per refunded horse number), 返還枠番情報 and 返還同枠情報 (brackets) write
  no row.

`horsetrack japan jravan dividends [--year YYYY ...]` runs the build and writes
`reports/jravan-dividends.json` through `write_build_report`: per 開催年 (`by_year`) and in total
(`counts`), over the kept JRA records, `not_established`, `special_payout` and `refund` (races x
pools flagged), `refunded_horses` (horse numbers set), `races` (records kept) and `rows`.

The build on the export (2026-11-26, 1986 to 2026-11-23, 41 year files, about 5 s): 140,313
records, all JRA venues, one version each (データ区分 2 throughout, no deletion), 140,313 races and
1,329,908 rows. Flags: 0 not established and 0 special payout in every year (the export never sets
either), 23,989 refund (races x pools) and 3,205 refunded horses, all from 2002 on (none before).
Rows per pool, with the first year that has any (the pools' start of sale):

| pool | rows | races | first year |
|---|---|---|---|
| WIN | 140,490 | 140,313 | 1986 |
| PLACE | 417,340 | 140,304 | 1986 |
| BRACKET_QUINELLA | 137,918 | 137,673 | 1986 |
| QUINELLA | 118,069 | 117,841 | 1991 |
| QUINELLA_PLACE | 278,566 | 92,704 | 1999 |
| FORECAST | 84,627 | 84,363 | 2002 |
| TRIO | 84,592 | 84,387 | 2002 |
| TIERCE | 68,306 | 67,917 | 2004 |

## O1-O6: final odds (#367)

`horsetrack.countries.japan.jravan.odds` writes `processed/jravan/final_odds.parquet`: the final
price of every combination of every sold pool, the pool's own baseline for Place, Wide and the
exotics (D5, C5). It is a jravan-only table, not in `schemas.TABLES`.

- `build_final_odds(*, years=None, raw=None, out=None) -> BuildReport` reads
  `<raw>/stored/RACE/<O1..O6>/<YYYY>.jvd` one year at a time (`years` None reads every year file
  of every type; a type with no file for a year, named or not, adds nothing), streaming each file
  with `records.iter_file` in chunks of at most 2,000 records (fewer for O5 and O6, capped at
  `CHUNK_BYTES`). It keeps JRA venues (競馬場コード 01-10) and one version per race and record
  type (`latest_versions` on 開催年, 開催月日, 競馬場コード, 開催回, 開催日目, レース番号; a race whose
  latest record of a type is a deletion has no rows of that type). Each year is sorted by
  `race_id, pool, combination` (plain string order, so a pool row sorts after its slots) and
  written as one row group through a `ParquetWriter` (zstd) into a tmp file, then `os.replace`;
  the same inputs give the same bytes. A year file holding a race of another year is a
  `ValueError`.
- `odds_rows(record_type, cols)` maps decoded O1-O6 arrays to long rows (the 0B41/0B42
  time series, #369, reuses it for O1 and O2): one per record and filled slot of each sold pool,
  plus one pool row per sold pool.

| column | type | from |
|---|---|---|
| `jurisdiction` | string | `jra` |
| `race_id`, `race_date`, `venue`, `race_no` | string, date32, string, int64 | `keys.race_id(開催年, 開催月日, 競馬場コード, レース番号)` and its parts |
| `pool` | string | the record type and group (below) |
| `combination` | string | the group's 馬番 or 組番; `__pool__` on a pool row |
| `odds` | float64 | オッズ, or 最低オッズ for PLACE and QUINELLA_PLACE, one implied decimal place (stake included) |
| `odds_hi` | float64 | 最高オッズ for PLACE and QUINELLA_PLACE, else null |
| `fav_rank` | int64 | 人気順, null when not a positive integer (`--`, `**`, blank) |
| `investment` | float64 | on a pool row, the pool's 票数合計 x 100 (yen); null on slot rows |
| `status_raw` | string | the low price's text as printed when `odds` is null, else null |
| `published_utc` | timestamp[us, UTC] | `keys.announced_utc(開催年, 開催月日, 発表月日時分)` |
| `data_kubun` | string | データ区分 of the kept record |

Pools, per record type and sale flag: O1 発売フラグ単勝 `WIN`, 発売フラグ複勝 `PLACE`, 発売フラグ枠連
`BRACKET_QUINELLA` (複勝着払キー unused); O2 `QUINELLA`; O3 `QUINELLA_PLACE`; O4 `FORECAST`; O5
`TRIO`; O6 `TIERCE`. A pool is sold when its flag is `7`. Flags `0` (not sold), `1` (cancelled
before sale) and `3` (cancelled after sale, the tickets refunded) write no row, pool row
included: none of them has a final price that pays.

- The combination: the number item split into its 1, 2 or 3 equal-width legs (a bracket one
  digit, a horse two), leading zeros dropped, joined with `,`; FORECAST and TIERCE in the printed
  order, the others ascending (`0704` is `4,7` on O2 and `7,4` on O4, `070401` is `7,4,1` on O6).
  A slot whose number is blank, or not all positive integers, writes no row.
- A price that is not a positive number (`0000` no votes, `----` cancelled before sale, `****`
  cancelled after sale, blank) is null with its text in `status_raw`, untrimmed. The low and high
  ends of a ranged price are read independently: a bad low with a valid high keeps `odds_hi`, and
  `status_raw` carries the low's text. A capped price (`9999`, 999.9 or more; lower caps before
  2004-08-13) is kept as the cap.
- A sold pool with a blank vote total still has its pool row, with null `investment`.
- `published_utc`: in the stored export 発表月日時分 is `00000000` on every record (the workbook
  sets it for intermediate odds only), so it is null throughout `final_odds`; it is set on the
  time series (#369).
- データ区分 in the export: 5 (確定, Monday) on almost every record, 9 (race cancelled) on 102
  races in each of O1-O5 (O6 not surveyed), whose prices print `****`.

Availability rule `label_only` (the owner's request on #376, 2026-11-26; docs/specs/as-of.md):
`final_odds` is a label or a conditioner, never a feature at t, so `read_asof` and `asof_join`
on a view of the table raise `LabelOnlyTable` and `read_label` returns it whole.

`horsetrack japan jravan final-odds [--year YYYY ...]` runs the build and writes
`reports/jravan-final-odds.json` through `write_build_report`, counts only: `records` (kept JRA
records over the six types), `rows` and rows per pool, in total (`counts`) and per race year
(`by_year`, the years that have rows).

The build on the export (2026-11-26, O1 and O2 from 1993, O3 from 1999, O4 and O5 from 2002, O6
from 2004, to 2026-11-23): 558,526 records, 251,200,511 rows in 34 row groups (1.4 GB), 318 s
wall, peak RSS 2.96 GB (`/usr/bin/time -l`). Rows per pool, slots and pool rows together:

| pool | rows | first year |
|---|---|---|
| WIN | 1,713,230 | 1993 |
| PLACE | 1,713,201 | 1993 |
| BRACKET_QUINELLA | 3,926,923 | 1993 |
| QUINELLA | 10,716,653 | 1993 |
| QUINELLA_PLACE | 9,043,121 | 1999 |
| FORECAST | 16,385,001 | 2002 |
| TRIO | 35,576,717 | 2002 |
| TIERCE | 172,125,665 | 2004 |

## H1 and H6: pool votes (#368)

`horsetrack.countries.japan.jravan.votes` writes `processed/jravan/pool_votes.parquet`: the final
votes on every combination of every sold pool, and each pool's total and refunds, for pool sizes
and own-stake impact (C5, C6, #283). It is not in `schemas.TABLES`, so `read_table` refuses it;
its availability rule is `label_only` (the owner's request on #376, 2026-11-26): the as-of views
raise `LabelOnlyTable` and `read_label` returns it whole.

- `build_pool_votes(*, years=None, raw=None, out=None, chunk_records=500) -> BuildReport` reads
  `<raw>/stored/RACE/H1/<YYYY>.jvd` and `H6/<YYYY>.jvd` (`raw` the export root, default
  `raw/jravan`; `years` None reads every year file of either folder, a year without an H6 file,
  every year before 2004, reads H1 alone; a named year with neither file is a
  `FileNotFoundError`). Each file is streamed twice through `records.iter_file`, `chunk_records`
  records at a time: the first pass reads the version keys and keeps one version per race
  (`latest_versions` on 開催年, 開催月日, 競馬場コード, 開催回, 開催日目, レース番号; a race whose
  latest record of a type is a deletion has no rows of that type), the second turns the kept
  records of JRA venues into rows, so no more than one chunk is decoded at a time. Each year's
  rows, H1 and H6 together, are sorted by `race_id, pool, combination` (plain string order) and
  written as exactly one row group through a `ParquetWriter` (zstd) into a tmp file, then
  `os.replace`; the same inputs give the same bytes, whatever the chunk size.
- `votes_rows(record_type, cols)` maps decoded H1 or H6 arrays to rows (`POOL_VOTES_SCHEMA`):
  `jurisdiction` `jra`, `race_id` (`keys.race_id`), `race_date`, `venue` and `race_no` from the
  race id, `pool` and `combination`, `votes`, `refund_votes`, `fav_rank` (int64) and
  `data_kubun` (データ区分 of the record kept: 2 前日売最終, 4 確定, 5 確定(月曜), 9 レース中止).
- Pools, from the votes groups: H1 単勝票数 `WIN`, 複勝票数 `PLACE`, 枠連票数 `BRACKET_QUINELLA`,
  馬連票数 `QUINELLA`, ワイド票数 `QUINELLA_PLACE`, 馬単票数 `FORECAST`, 3連複票数 `TRIO`; H6
  3連単票数 `TIERCE`. A pool is sold when its 発売フラグ is `7`; a pool flagged `0` (not sold) or
  `1` (cancelled before sale) writes no row.
- A combination row per registered slot of a sold pool: the slot's 馬番 or 組番 is digits only and
  every one of its 1, 2 or 3 equal-width legs is positive (blank slots, zeros and partly blank
  combinations write nothing). `combination` is the legs joined with `,`, leading zeros dropped
  (the form of `docs/specs/odds-log.md`): FORECAST and TIERCE in finishing order, the others
  ascending (`1,1` is a same-bracket pair). `votes` is 票数 (100-yen tickets, as printed),
  `fav_rank` 人気順 (`--`, `**` and blank are null), `refund_votes` null. A registered slot with
  zero votes is a row.
- A `__pool__` row per sold pool: `votes` its 票数合計 (refunded votes included), `refund_votes`
  its 返還票数合計 (null when blank), `fav_rank` null. A count or rank that is not a number is
  null. 登録頭数, 出走頭数, 複勝着払キー and the refunded horse and bracket items write no row.

`horsetrack japan jravan votes [--year YYYY ...]` runs the build and writes
`reports/jravan-votes.json` through `write_build_report`: per year (`by_year`) and in total
(`counts`), `records` (H1 and H6 records read), `versions_dropped`, `deletions`, `non_jra`, `kept`,
`races` (with a row), `rows`, and per pool `combinations_<POOL>`, `pools_<POOL>` and
`refunds_<POOL>` (pool rows with refunded votes).

The build on the export (2026-11-26, 1986 to 2026-11-23, 41 H1 and 23 H6 year files): 257 s
wall clock, peak RSS 2.58 GB (`/usr/bin/time -l`, 2,578,612,224 bytes), a 1.1 GiB file of 41 row
groups. 208,368 records, all JRA venues, one version each (no version dropped, no deletion),
140,383 races and 252,897,444 rows (10.8 to 12.7 million a year from 2009 on). データ区分 is 5 on
every race but the abandoned ones (9: 70 races in H1, 64 in H6); those 69 H1 and 63 H6 races
(2005-2025) print no number on their combination votes, so 247,468 combination rows have a null
`votes` (their pool totals are numbers). 1,488,188 combination rows hold 0 votes and 2,551,950 a
null `fav_rank`. Rows per pool, with the first year sold:

| pool | combination rows | sold pools (`__pool__` rows) | with refunds | first year |
|---|---:|---:|---:|---|
| WIN | 1,880,558 | 140,383 | 3,086 | 1986 |
| PLACE | 1,880,514 | 140,374 | 3,081 | 1986 |
| BRACKET_QUINELLA | 4,572,766 | 137,743 | 2,878 | 1986 |
| QUINELLA | 10,979,725 | 117,911 | 3,084 | 1991 |
| QUINELLA_PLACE | 8,947,799 | 92,774 | 3,084 | 1999 |
| FORECAST | 16,295,500 | 84,433 | 3,083 | 2002 |
| TRIO | 35,481,466 | 84,457 | 3,084 | 2002 |
| TIERCE | 171,993,060 | 67,981 | 2,571 | 2004 |

## The 0B41/0B42 time series (#369)

`horsetrack.countries.japan.jravan.timeseries` writes the export's time-series odds into the
odds log's own table, `processed/jravan/odds_timeseries/<YYYY-MM-DD>.parquet`, one file per race
day in exactly `logger.compact.SCHEMA` (docs/specs/odds-log.md), so the readers of the logger's
days (#102, #85, #358) read years of boards the logger never saw.

- Layout: `raw/jravan/timeseries/0B41/<YYYY>/<YYYYMMDD>.jvd` holds the race day's O1 records (WIN,
  PLACE, BRACKET_QUINELLA), `0B42/...` its O2 records (QUINELLA); either may be missing. The day
  file is the file name's race day, so the evening-before boards are in the race day's file.
  The folder holds the day files and `_build.json` only.
- `build_day(race_date, *, raw=None, out=None) -> BuildReport` streams the two files
  (`records.iter_file`), decodes each record with the O1-O6 mapping (`odds.odds_rows`, above) and
  writes `<out>/odds_timeseries/<date>.parquet` (`out` default `processed/jravan`) through
  `write_parquet`. `build_all(*, years=None, raw=None, out=None)` runs every day of the years
  that have a 0B41 or 0B42 file, one day at a time in one process. A day that fails (a cut
  record, a record of another race day, a dividend key twice) is in `failures` and the others are
  written; a previous file of a failed day is left as it was.
- The skip rule: `_build.json` records, per day, both inputs' size and mtime, the workbook's
  sha256, `PARSER_VERSION` and a sha256 of the day's races and dividends rows (the joined
  columns). A day whose record matches and whose file exists is not rebuilt; a changed input,
  workbook, parser version or joined row rebuilds it, and so does a missing day file.

| column | from |
|---|---|
| `race_id`, `venue`, `race_no`, `pool`, `combination` | `odds_rows`; pool rows `__pool__` |
| `snapshot_utc` | the record's 発表月日時分 (JST) as UTC, `keys.announced_utc` (a month after the race's month is the year before) |
| `fetched_utc` | null: the export has no fetch time |
| `post_utc` | `races.post_time_local` (JST, on the race date) from `processed/jravan/races.parquet`; null without a race row or a post time |
| `minutes_to_post` | (`post_utc` - `snapshot_utc`) in minutes, negative after the scheduled post |
| `odds`, `odds_hi`, `investment`, `status_raw` | `odds_rows` (investment: the pool row's 票数合計 x 100 yen) |
| `is_final_capture` | the latest `snapshot_utc` per (race_id, pool, combination), pool rows included |
| `dividend`, `is_refund` | left join on (race_id, pool, combination) to `processed/jravan/dividends.parquet`; null without a match or without the table |

The same (race_id, pool, combination, snapshot_utc) twice is one row, the later record's (O1
records before O2, each file in record order). Rows are sorted on (race_id, pool, combination,
snapshot_utc), unique after that dedupe, so the same inputs give the same bytes whatever their
record order. A record without an announcement time (`00000000`, blank or invalid) writes no row
and is counted (`no_snapshot_records`); the export has none.

The snapshot-time rule: `snapshot_utc` is the minute JRA-VAN stamps as announced (発表), written
as it is. Under the shipped `odds_timeseries` rule (`fetched_utc`, fallback `snapshot_utc`,
docs/specs/as-of.md) the board at t from a jravan day holds the snapshots announced at or before
t. **Open (owner decision, the epic's open question):** whether a 発表 minute is a fair
availability time for a bettor, i.e. how long after it the board was actually in hand. The data
does not answer it; until the owner rules, nothing here adds a lag.

データ区分 in the export: 1 (中間, intermediate) on 98.5% of records, 2 (前日売最終, the
day-before close) on 4,439, 3 (最終) and 4 (確定) on about one record per race and family each.
The 確定 board is announced after the race, so `is_final_capture` is almost always a post-race
row (12,937,159 of 12,939,954 final rows have `minutes_to_post` < 0); a reader that wants the last
board before the off filters on `snapshot_utc`, not on `is_final_capture`.

`horsetrack japan jravan timeseries [--year YYYY ...] [--day YYYY-MM-DD ...]` runs the build and
writes `reports/jravan-timeseries.json` through `write_build_report`, counts only, even when a day
fails (then it exits 1): days written, unchanged and failed, records per データ区分, rows and rows
per pool, repeated rows, races and races without a post time, and per year the days, races,
records, rows and snapshots per race (median, min, max).

The build on the export (2026-11-26, 2003-10-04 to 2026-11-23, 2,493 race days, no failure):
21,273,043 records, 1,727,457,117 rows (WIN 161,614,223, PLACE 161,613,056, BRACKET_QUINELLA
361,867,663, QUINELLA 1,042,362,175), 79,965 races, every one with a jravan post time; 500,376
repeated rows dropped. 1,918 s wall, peak RSS 1.04 GB (`/usr/bin/time -l`), 3.6 GB on disk, with
21 GB free before. The rerun rewrote nothing (2,493 days unchanged, 12 s). Snapshots per race
(distinct announcement minutes over both families), by year:

| year | days | races | median | min | max |
|---|---|---|---|---|---|
| 2003 | 27 | 836 | 44 | 10 | 236 |
| 2004 | 106 | 3,452 | 63 | 11 | 373 |
| 2005 | 105 | 3,450 | 65 | 12 | 380 |
| 2006 | 106 | 3,453 | 67 | 11 | 439 |
| 2007 | 105 | 3,453 | 74 | 23 | 439 |
| 2008 | 108 | 3,462 | 73 | 14 | 435 |
| 2009 | 105 | 3,453 | 72 | 23 | 447 |
| 2010 | 107 | 3,454 | 130 | 19 | 515 |
| 2011 | 106 | 3,453 | 136 | 17 | 501 |
| 2012 | 107 | 3,454 | 142 | 25 | 520 |
| 2013 | 108 | 3,454 | 143 | 24 | 522 |
| 2014 | 113 | 3,451 | 145 | 21 | 523 |
| 2015 | 108 | 3,454 | 144 | 13 | 520 |
| 2016 | 107 | 3,454 | 144 | 26 | 718 |
| 2017 | 111 | 3,455 | 145 | 12 | 532 |
| 2018 | 109 | 3,454 | 143 | 13 | 532 |
| 2019 | 110 | 3,452 | 144 | 25 | 531 |
| 2020 | 109 | 3,456 | 146 | 12 | 521 |
| 2021 | 106 | 3,456 | 143 | 14 | 520 |
| 2022 | 108 | 3,456 | 146 | 26 | 489 |
| 2023 | 107 | 3,456 | 151 | 12 | 503 |
| 2024 | 106 | 3,454 | 150 | 13 | 513 |
| 2025 | 109 | 3,455 | 154 | 29 | 502 |
| 2026 | 100 | 3,138 | 153 | 96 | 504 |

D14: the export ends 2026-11-23, before the 2026-11-28 holdout start; readers drop holdout days
by file name before reading. D10: pre-2014-06-07 days are kept; whether a model reads them is the
model's business.

## UM, KS, CH: horse, jockey and trainer masters (#370)

`horsetrack.countries.japan.jravan.masters`; `horsetrack japan jravan masters`.

The masters are one current snapshot: the export of 2026-11-25/26 (latest データ作成年月日
2026-11-24), not a history. Anything time-varying in them (a horse's or a person's retirement,
the east/west affiliation, a jockey's stable and apprentice code, licence dates) is the state at
export time, not a value at t (docs/specs/as-of.md, Known traps).

- `build_masters(*, raw=None, out=None)` reads `<raw>/stored/DIFN/{UM,KS,CH}/all.jvd` chunk by
  chunk (only the items it maps; 3代血統情報 cut to its first five occurrences), keeps one version
  per key (`records.latest_versions` on 血統登録番号, 騎手コード, 調教師コード; a key whose latest
  record is a deletion, データ区分 0, has no row; UM データ区分 9, a deregistration, is kept) and
  writes `horses.parquet` and `horse_pedigree.parquet` (`schemas.TABLES`), `jockeys.parquet`
  (`JOCKEYS_SCHEMA`) and `trainers.parquet` (`TRAINERS_SCHEMA`), sorted by key; the same inputs
  give the same bytes. The report, `<reports dir>/jravan-masters.json`, holds per record type
  `records`, `versions_dropped` and `deletions`, the rows written per table, and `unmapped` as
  field (`colour`, `sex`, `mark`, `retired`, `affiliation`, `date`) -> code -> count (an impossible
  date counts under `invalid`, never by value).
- `horses`: `horse_id` 血統登録番号 as printed, `name` 馬名, `origin` 産地名; `colour` and
  `import_type` are the meanings of 毛色コード and 馬記号コード in code tables 2203 and 2204, read
  from the workbook at run time (null for the unset `00`, a blank and a code the table lacks;
  `import_type` is null for a horse without a mark); `sex` H, M, G from 性別コード; `sire`, `dam`,
  `dam_sire` the names of 3代血統情報 occurrences 1, 2 and 5; `is_retired` 競走馬抹消区分;
  `jurisdiction` `jra`; `profile_scraped_utc` null. Under the shipped rules `horses` is
  `current_value`: never in a view.
- `horse_pedigree`: the same three names, their `*_name_norm` (NFKC then casefold, as the netkeiba
  pedigree card) and `sire_id`, `dam_id`, `dam_sire_id` (繁殖登録番号, null when blank or all
  zeros); `source_race_id` and `scraped_utc` null. `static`: every horse in every view.
- `jockeys` and `trainers` (jravan-only, not in `schemas.TABLES`): `jurisdiction` `jra`, the code,
  `jockey_key` / `trainer_key` (`jockey:<code>` / `trainer:<code>`, as on runners), `name`,
  `name_short`, `birth_date`, `licence_issued`, `licence_revoked` (date32, null when blank or all
  zeros), `is_retired`, `affiliation` (`miho`, `ritto`, `nar`, `overseas` from the east/west code,
  code table 2301); jockeys also carry `trainer_code` (所属調教師コード, null for `00000`, a
  freelance jockey) and `apprentice_code` as printed. Availability rule `current_value` with
  `in_view` (the owner's request on #376, 2026-11-26): every master row is in every view at any
  t, as exported, with the look-ahead docs/specs/as-of.md documents; `read_label` refuses them.

The run on the export (2026-11-26, about 5 s, 0.7 GB peak), counts only:

| type | records | versions dropped | deletions | rows written |
|---|---:|---:|---:|---:|
| UM | 223,626 | 7,640 | 0 | 215,986 horses (and pedigree rows) |
| KS | 1,923 | 357 | 0 | 1,566 jockeys |
| CH | 2,038 | 559 | 0 | 1,479 trainers |

Zero unmapped codes. Kept UM versions by データ区分: 105,413 `4`, 56,073 `1`, 54,481 `9`, 18 `3`,
1 `2`. Horses: 109,668 H, 98,547 M, 7,771 G; 207,062 retired; every horse has a colour, 64,219 a
mark; 24 have no 産地名. Pedigree: 1 horse has no sire, dam or dam-sire name, and about 400 per
ancestor have no breeding id (401 sire, 407 dam, 408 dam sire). Jockeys: 1,436 retired, 47 with a
stable (every one a trainer code in `trainers`); trainers: 1,291 retired. Every one of the
1,885,762 jravan runners has its horse in `horses`, and every `jockey_key` and `trainer_key` joins
but the unset `00000` (71 jockey, 27 trainer runners).

## HN, SK, BT: bloodlines (#371)

`horsetrack.countries.japan.jravan.bloodlines` writes `processed/jravan/breeding_horses.parquet`,
`produce.parquet` and `lineages.parquet`. None is in `schemas.TABLES`, so `read_table` refuses
them; their availability rule is `static` (the owner's request on #376, 2026-11-26: pedigree is
known at any t), so every row is in every view at any t and `read_label` refuses them.

- `build_bloodlines(*, raw=None, out=None) -> BuildReport` reads `<raw>/stored/BLDN/HN/all.jvd`,
  `SK/all.jvd` and `BT/all.jvd` (`raw` the export root, default `raw/jravan`; a missing file is a
  `FileNotFoundError`) through `records.iter_file`, keeps one version per key
  (`latest_versions` on 繁殖登録番号 for HN and BT, 血統登録番号 for SK; a key whose latest record
  is a deletion has no row) and writes each table sorted by its key (plain string order) through
  `records.write_parquet`; the same inputs give the same bytes.
- `breeding_horses_from_records`, `produce_from_records` and `lineages_from_records` map decoded
  HN, SK and BT arrays to rows (one per record), exported schemas `BREEDING_HORSES_SCHEMA`,
  `PRODUCE_SCHEMA` and `LINEAGES_SCHEMA`.
- Values: text is `records.text` (trimmed at both ends, blank null). An id (a breeding id, a
  血統登録番号, 生産者コード) is the trimmed text as printed (leading zeros kept), null when blank
  or all zeros. A year is an int64, null when blank or zero; a date `YYYYMMDD`, null when blank or
  `00000000`. `sex` is 性別コード `1` H, `2` M, `3` G, anything else null. `colour` is the first
  content column of code table 2203 for 毛色コード, read from the workbook at run time (the
  meanings stay in the licensed workbook), null for the unset `00`, a blank code or a code the
  table lacks. 品種コード and 馬名半角ｶﾅ are not used.

`breeding_horses` (HN, key 繁殖登録番号):

| column | type | from |
|---|---|---|
| `breeding_id` | string | 繁殖登録番号 |
| `horse_id` | string | 血統登録番号 (null for a horse without one) |
| `name`, `name_en` | string | 馬名, 馬名欧字 |
| `birth_year` | int64 | 生年 |
| `sex` | string | 性別コード |
| `colour` | string | 毛色コード |
| `import_type_code` | string | 繁殖馬持込区分 as printed (`0` stays `0`) |
| `import_year` | int64 | 輸入年 |
| `origin` | string | 産地名 |
| `sire_breeding_id`, `dam_breeding_id` | string | 父馬繁殖登録番号, 母馬繁殖登録番号 |

`produce` (SK, key 血統登録番号):

| column | type | from |
|---|---|---|
| `horse_id` | string | 血統登録番号 |
| `birth_date` | date32 | 生年月日 |
| `sex`, `colour` | string | 性別コード, 毛色コード, as on HN |
| `import_type_code` | string | 産駒持込区分 as printed |
| `import_year` | int64 | 輸入年 |
| `breeder_code` | string | 生産者コード |
| `origin` | string | 産地名 |
| `pedigree_breeding_ids` | list<string> | 3代血統繁殖登録番号, the 14 repeats in the record's order, each null in place (always 14 long) |

`lineages` (BT, key 繁殖登録番号):

| column | type | from |
|---|---|---|
| `breeding_id` | string | 繁殖登録番号 |
| `lineage_id` | string | 系統ID |
| `lineage_name` | string | 系統名 |
| `lineage_description` | string | 系統説明 |

`horsetrack japan jravan bloodlines` runs the build and writes `reports/jravan-bloodlines.json`
through `write_build_report`, counts only: per record type `<RT>_records`,
`<RT>_versions_dropped` and `<RT>_deletions`; the rows written (`breeding_horses`, `produce`,
`lineages`); `sire_ids` (non-null `sire_breeding_id`) and `sire_ids_resolved` (equal to a written
`breeding_id`); `pedigree_ids` (non-null pedigree values, repeats counted) and
`pedigree_ids_resolved`; `lineages_resolved` (lineages rows whose `breeding_id` is a written
breeding id).

The build on the export (2026-11-26): 10 s wall, peak RSS 1.10 GB (`/usr/bin/time -l`).

| type | records | versions dropped | deletions | rows written |
|---|---:|---:|---:|---:|
| HN | 162,551 | 70 | 0 | 162,481 |
| SK | 429,886 | 342 | 0 | 429,544 |
| BT | 92 | 0 | 0 | 92 |

152,434 breeding horses have a sire id, 152,411 of them a written breeding horse (23 do not);
all 5,892,823 non-null pedigree ids and all 92 lineages resolve to a breeding horse.

## The dump of unmapped families (#363)

`horsetrack japan jravan dump [--type RT ...] [--year YYYY ...]`
(`horsetrack.countries.japan.jravan.dump.run`) decodes every stored family that no ticket of this
epic maps into a store table, so nothing in the export goes unread and the coverage audit can see
what each family holds.

- What is read: `raw/jravan/stored/<DATASPEC>/<RT>/*.jvd`, the files directly in the folder, of
  every folder not in `MAPPED` (RACE RA, SE, HR, H1, H6, O1-O6; DIFN UM, KS, CH; BLDN HN, SK, BT).
  `--type` names record types instead, in every dataspec, mapped or not. `--year` keeps
  `<YYYY>.jvd` of those years; `all.jvd` is always read. `timeseries/` and the folders outside
  `stored/` are never dumped.
- The layout: `interim/jravan/records/<DATASPEC>/<RT>/<stem>.parquet` (zstd), one row per record
  and one column per `records.decode` key holding `records.text` (cp932, trimmed, blank is null): a
  plain leaf is `string`, a repeat `list<string>`, a nested repeat `list<list<string>>`, typed
  from the layout so even an all-null column keeps its type. `_file` is the path relative to
  `raw/jravan` and `_record` the 1-based record number (int64). Each `records.iter_file` chunk is a
  row group written through a `pyarrow.ParquetWriter` into a tmp file, then `os.replace`, so a
  multi-GB file (SNPN CK) streams in bounded memory.
- `interim/jravan/records/_dump.json` keeps, per file written, its size, mtime and the workbook's
  sha256; a file that matches and whose parquet exists is skipped, anything else is rewritten. Runs
  limited by `--type` or `--year` keep the other entries.
- A file that fails (a record of the wrong width or type, or a record type whose layout the
  workbook lacks or cannot validate) leaves no parquet and no tmp file, loses its manifest entry so
  the next run tries it again, and is a failure `{file, record, reason}` (`record` 1-based, null
  for a layout failure); the other files are written and the command exits 1.
- The report, `reports/jravan-dump.json` through `records.write_build_report`, holds counts only:
  `counts` the totals of the files written this run (`files`, `records`, `bytes` read, and
  `skipped`, `failed`), `unmapped["<DATASPEC>/<RT>"]` the same `files`, `records`, `bytes` and
  `seconds` per folder with a file written, and `failures`. An output folder inside the repository
  is refused before anything is written.

Dumps are not store tables: no reader in `core/store` serves them and they have no availability
rule (no as-of time), so no model or feature may read them until a ticket maps the family into a
table with one.

## Reconciliation with the netkeiba store (#373)

`horsetrack.countries.japan.jravan.reconcile`; `horsetrack japan jravan reconcile [--start
YYYY-MM-DD] [--end YYYY-MM-DD]` (default 2010-01-01 to 2026-11-23). It compares the `races`,
`runners` and `dividends` of `processed/jravan/` with those of `processed/jra/`, one calendar year
at a time, and changes neither store:

- Reads go through `read_store_table(..., source=..., touches_price=False)`, so holdout days are
  never returned; dividends are read only for the race ids those reads returned. No D10 cut
  applies, because the comparison is a data check that no model or fit uses.
- Races match by `race_id`, runners by (`race_id`, `horse_no`), dividends by (`race_id`, `pool`,
  `combination`). Over the matched rows it counts the equal fields (`RACE_FIELDS`,
  `RUNNER_FIELDS`; post times as minutes of the day, `finish_time_s` and `win_odds` at the
  tenth), the races whose Win odds all agree and those whose payouts all agree, and the rows whose
  key repeats within a store. One-store and repeated-key counts are broken down by the race's
  `status` in that store.
- The command prints the overall counts and rates; the detail by year, with up to 20 failing race
  ids per check, goes to `reports/jravan-reconcile.md` through `write_report`
  (`sources=('jravan', 'jra')`), which refuses the repository. Its figures come from a licensed
  source (D21) and are not in git.

## The model-input coverage audit (#374)

`horsetrack japan coverage [--write-doc]` (`countries/japan/research/coverage.py`) states, for
every input of the C0-C6 stack (docs/modeling.md) and the evaluation's own, which source and
table supplies it, over which dates and where the gaps are. The inputs, their supplies and the
tickets that cover each gap are `countries/japan/config/model_inputs.yaml`. A supply may be a
jravan table, a jravan-only table, the time-series day files or a dumped family
(`records/<DATASPEC>/<RT>`), and it is read column-scoped with holdout days dropped. The private
report goes to `reports/data-coverage.md`; `--write-doc` writes the committed
[docs/data-coverage.md](../data-coverage.md), where every JRA-VAN and JRDB supply shows
`withheld (D21)` and no figure.

## HC and WC: workouts (#229)

`horsetrack.countries.japan.handicap.workouts_jravan` (beside #137's JRDB `workouts`);
`horsetrack japan jravan workouts`. JRA-VAN's 坂路 (HC) and ウッド (WC) automatic workout times as a
second, first-party workout source with #137's columns:

- `build_workout_table(raw=None, out=None)` reads every `stored/SLOP/HC/<YYYY>.jvd` and
  `stored/WOOD/WC/<YYYY>.jvd` (a missing folder is no rows; neither is a `FileNotFoundError`),
  keeps one version per key (`records.latest_versions` on the layout's key items, across all the
  year files of a record type; a kept deletion has no row) and writes `workouts.parquet`, sorted
  by workout date, centre, course, time and horse, through `records.write_parquet`. An `out`
  inside the repository is `paths.DataDirInsideRepo`.
- Columns: `horse_id` (血統登録番号 as printed, the store's id; blank or all zeros null, the row
  kept), `workout_date` and `race_date` (both 調教年月日; `race_date` lets `read_store_table` scope
  and drop holdout days by workout date), `workout_time` (調教時刻, `HH:MM`), `centre` (トレセン区分
  0 `miho`, 1 `ritto`, else `other`), `course` (`hill` for HC, `wood` for WC), `furlongs` (the
  largest n whose n-furlong total is measured), `split_first_s` (that total), `split_mid_s` (the
  3-furlong total), `final_furlong_s` (the 200 m to 0 m lap). Times are tenths of a second; all
  zeros (no measure) or all nines (over the limit) is null. The totals are found by the distance
  in the item name, since one name is spelled differently from the others. `effort` and `partner`
  are null: the source has neither.
- As of t (owner decisions on #229, 2026-11-27): a workout counts from `published_at`, (workout
  date + 1 day) 00:00 Asia/Tokyo, strictly before t, and only for a race on a later day;
  データ作成年月日 is ignored, and the latest stored version is used at every t (accepted
  look-ahead, as for the masters). `workout_table` and `workout_features` are #137's with these
  rules: `days_since_hard` is always null, and no JRDB table or `HORSETRACK_JRDB_FEATURES` is read.
- The command writes the build counts to `reports/jravan-workouts.json` and the coverage per
  centre and course (first and last workout date, rows, horses; holdout days not read) to
  `reports/jravan-workouts-coverage.json`, both through `write_report`.

## Card versions (#401)

`horsetrack.countries.japan.jravan.card_versions`; `horsetrack japan jravan card-versions
[--as-of YYYY-MM-DD]` (default today in Tokyo). It builds no table: it states which versions of
the race card the export holds.

JV-Data publishes a race (RA) and each runner (SE) in versions told apart by データ区分, each
stamped with データ作成年月日: `1` entries (出走馬名表) and `2` the card (出馬表) before the race,
`3`-`6` flash results (速報成績) on race day, `7` the final result (成績); besides these, `9` an
abandoned race (レース中止), `A` a NAR race (地方競馬), `B` an overseas race (海外国際レース) and
`0` a deletion (削除). The builders keep each key's latest version (`records.latest_versions`).

- `census(*, raw=None, as_of)` reads `stored/<DATASPEC>/RA|SE/*.jvd` of every DATASPEC folder,
  streamed, and counts per record type the records per データ区分 and per race year, the keys
  (the layout's key items) holding more than one version with their codes in データ作成年月日
  order (a same-day tie by each version's last occurrence in read order: folders, then files,
  sorted), and the pre-race versions (`1`, `2`) of races run before
  `as_of`. A version is a key's (データ区分, データ作成年月日): a record repeating one already read
  (the same record in `stored/RACE` and `stored/DIFN`) counts once, the repeats as `duplicates`.
  Each record's two-byte record type is checked first, whatever its width or day: one that is not
  the folder's type (a file of another type misfiled there) stops the census (`RecordTypeError`).
  Its race date is read next, from its 開催年 and 開催月日 bytes alone: a record on a holdout day
  is read no further, whatever its width, and counted only as `holdout_skipped` (D14); one
  without a valid race date only as `undated`. The others are then checked: one of the wrong
  width, or too short to hold its date, stops the census (`RecordLengthError`). Either error
  names its file (`stored/<DATASPEC>/<RA|SE>/<file>`). Combos pool every folder, while the
  builders read `stored/RACE` one year file at a time: a combo's last version is the one
  `latest_versions` keeps only for the same records in the same order.
- `final_only`: true when no version `1`-`6` is held; null (inconclusive) when the record type
  has no record to count (no file, only empty files, every record skipped), never true. The
  command fails, writing no report, when RA or SE has no file under any `stored/<DATASPEC>/`
  (an incomplete copy of the export), and prints `inconclusive` for a null verdict.
- The report, `<reports dir>/jravan-card-versions.json` through `records.write_build_report`,
  holds counts, codes and years only; its figures are withheld (D21).

The export holds the final version only (the run of 2026-11-27): RA holds `7`, `9`, `A` and `B`,
SE `7`, `9`, `A`, `B` and `0`, and neither holds a record of データ区分 `1`-`6`. A key with more
than one version holds re-issues of one code (and, in SE, an overseas runner later deleted), never
a card before a result. So no pre-race card survives for a race already run, `latest_versions`
always keeps the result, and `runners.is_scratched`, `races.field_size`, `races.status` and
`meetings.status` keep the accepted leak (docs/specs/as-of.md, "Accepted leak: results-derived
card columns"): the JRA-VAN export cannot lift it.

The card versions can only be carried forward, by reading them before the race, each with the time
it was read; nothing does today:

- The JRA official 出馬表 on race day is the feed for it. `jra/official_card.py` parses a race's
  出馬表 page (JRADB `pw01dde`) into `races` and `racecard_runners` rows, the scratches as printed
  (`is_scratched`) and each runner row stamped `card_scraped_utc`, the page's fetch time. The
  logger does not fetch that page: `log-day` reads the 出馬表 menu and the card race list for the
  post times only (`jra/capture.py`, `read_post_times`), nothing calls `official_card.parse`, and
  no store table holds its rows. The logger would need to fetch each race's 出馬表 page from that
  race list before the off, as often as a late scratch matters, and write the parsed rows with
  their fetch time into a table whose availability rule is the owner's.
- JV-Link's regular (non-setup) RACE reads during race week deliver the `1` and `2` versions while
  they are current. The export is a bulk read of the stored data (#196), and the Windows VM logs
  nothing between exports (no repository code runs there); keeping them would need a race-week
  reader on the VM and a folder of its own under `raw/jravan/`.
