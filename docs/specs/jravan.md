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

## Decoding (planned, #362)

## RA: races and meetings (planned, #364)

## SE: runners (planned, #365)

## HR: payouts (planned, #366)

## O1-O6: final odds (planned, #367)

## H1 and H6: pool votes (planned, #368)

## The 0B41/0B42 time series (planned, #369)

## UM, KS, CH: horse, jockey and trainer masters (planned, #370)

## HN, SK, BT: bloodlines (planned, #371)

## The dump of unmapped families (planned, #363)
