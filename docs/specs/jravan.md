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
  the record type is a `ValueError`.
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

## RA: races and meetings (planned, #364)

## SE: runners (planned, #365)

## HR: payouts (planned, #366)

## O1-O6: final odds (planned, #367)

## H1 and H6: pool votes (planned, #368)

## The 0B41/0B42 time series (planned, #369)

## UM, KS, CH: horse, jockey and trainer masters (planned, #370)

## HN, SK, BT: bloodlines (planned, #371)

## The dump of unmapped families (planned, #363)
