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

## SE: runners (planned, #365)

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

## O1-O6: final odds (planned, #367)

## H1 and H6: pool votes (planned, #368)

## The 0B41/0B42 time series (planned, #369)

## UM, KS, CH: horse, jockey and trainer masters (planned, #370)

## HN, SK, BT: bloodlines (planned, #371)

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
