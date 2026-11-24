# Shadow log

Code: `src/horsetrack/countries/japan/shadow/log.py` (#119). Every shadow pick of a JRA race day is
logged once, for later calibration, in a hash-chained file that is sealed when the day is done.

## The log

Path: `$HORSETRACK_DATA_DIR/shadow/japan/<YYYY-MM-DD>.jsonl` (`shadow_log_path(day)`), one file per
race day, named by its local (JST) date. `ShadowLog(path, day=...)` refuses a file whose name is not
`<day>.jsonl`, so records are never labelled under another day's status.

- One JSON object per line, compact separators, ASCII-escaped (so UTF-8), each line ending in `\n`.
- The file is opened only for append (`ab`), never truncated or rewritten. Each `append` and the
  `seal` is one write, flushed and `os.fsync`ed before it returns. A batch is written whole or not
  at all: every record is checked before any byte is written.
- Every call (`append`, `seal`, `logged_keys`, `read`, `verify`) reads the whole file and checks the
  chain first. A broken chain raises `ShadowLogTampered` (a `ValueError`), and nothing is appended
  onto it. An append also refuses a file whose size changed after it was checked.

### Lines

```json
{"seq":1,"prev_sha256":"0000…0000","record":{…}}
{"seq":2,"prev_sha256":"<sha256 of line 1>","record":{…}}
{"seq":3,"prev_sha256":"<sha256 of line 2>","seal":{"n_records":2,"sealed_utc":"2026-11-28T09:00:00+00:00"}}
```

- `seq`: 1 on the first line, one more on each line after it.
- `prev_sha256`: the hex sha256 of the previous line's bytes without its newline; 64 zeros on the
  first line.
- A line is exactly the bytes the log writes for its JSON: the same JSON with other whitespace or
  escapes is a changed line.

### Record fields

| key | type | notes |
| --- | --- | --- |
| `race_id` | text | `jra:<YYYY-MM-DD>:<VENUE>:<NN>` |
| `pool` | text | `WIN`, `PLACE`, `QUINELLA`, ... |
| `combination` | text | runner or combination |
| `tau_min` | integer | minutes before post the pick was made for |
| `decision_utc` | ISO UTC text | when the pick was decided; `Z` or `+00:00` |
| `board_capture_utc` | ISO UTC text or null | the board the pick used; null only when `status` is `no_board`; never after `decision_utc` |
| `written_utc` | ISO UTC text | when the logger wrote the record |
| `mode` | text | `live` or `replay` |
| `status` | text | `ok`, or `no_board` when no board was captured in time |
| `models` | mapping or null | the fit ids the pick used (`win`, `ordering`, `nowcast`, ...) |
| `git_sha` | text | the commit the logger ran |
| `day_status` | text | `holdout` or `development`; set by the log, never by the caller |

The key of a record is `(race_id, pool, combination, tau_min)`. Other keys (`board_odds`,
`value_mean`, `value_q05`, `value_q95`, `flags`, ...) are kept as given. Records are stored as given:
no keys dropped, no values coerced, no timestamps normalised; the log adds `day_status` only.

Example, synthetic:

```json
{"race_id": "jra:2026-11-28:TOK:11", "pool": "WIN", "combination": "7", "tau_min": 2,
 "decision_utc": "2026-11-28T06:38:00Z", "board_capture_utc": "2026-11-28T06:37:44Z",
 "written_utc": "2026-11-28T06:38:16Z", "mode": "live", "status": "ok", "board_odds": 3.4,
 "value_mean": 0.81, "value_q05": 0.74, "value_q95": 0.88, "flags": ["tail"],
 "models": {"win": "a1", "ordering": "b2", "nowcast": "c3"}, "git_sha": "0123abc"}
```

### Formation records

`betting.formations.log_choice` (#139) writes one record per formation search: `combination` is
`formation:<legs>` (legs joined by `/`, horses by `,`, e.g. `formation:1/2,3/2,3,4`;
`formation:none` when nothing qualified), so it never collides with a single-combination pick. It
adds `kind` (`formation`), `shadow` (true), `legs`, `n_tickets`, `combinations` (texts like
`"1,2,3"`), `cost`, `expected_return`, `expected_profit` and `ev_per_yen` (null when empty).

### Refused records

`append(records)` raises `RecordRefused` (a `ValueError`) and writes nothing when any record in the
batch:

- lacks a field of the table, or has one of the wrong type or value (timestamps are ISO 8601 text
  with a UTC offset);
- has a `board_capture_utc` later than its `decision_utc`, compared as instants (look-ahead);
- has a key containing `stake`, `bet`, `kelly`, `amount` or `wager`, in any case, at any depth,
  inside lists too;
- repeats a key already in the file or earlier in the batch;
- sets `day_status`, whatever its value;
- is not plain JSON: NaN or infinity, a value JSON has no type for, or keys that collide once
  written as text.

## The seal

`seal(sealed_utc=...)` appends `{"seq", "prev_sha256", "seal": {"n_records", "sealed_utc"}}` and
returns that line's sha256. `sealed_utc` is a timezone-aware datetime, written in UTC with
`isoformat()`; `n_records` counts the record lines before it. After the seal, `append` and `seal`
raise `LogSealed` (a `ValueError`), through any `ShadowLog` on the file. A day with no records can
be sealed (`n_records` 0).

## Holdout days

- `day_status(day)` is `holdout` when `is_holdout_day("japan", day)` holds (the committed rule in
  `countries/japan/config/holdout.yaml`, through `core.eval.holdout`, #71), else `development`.
- The log sets each record's `day_status`. Once any record in a file is logged as `holdout`, the file
  counts as holdout: later records are labelled `holdout` and `verify()` reports `holdout`.
- A holdout day's records are written prospectively and sealed, never read back through this module:
  `read()` raises `HoldoutSealed` (a `PermissionError`) when the rule says the day is holdout at the
  time of reading, or when any record in the file was logged as `holdout`.
- Reading a holdout day's values for settlement or calibration is #101's, through
  `LockedHoldout.open(ticket=..., purpose=..., family=...)` (D15, #126). This module gives no other path.

## Reading

- `read()`: the records of a development day, in order, each with its `day_status`.
- `logged_keys()`: the set of `(race_id, pool, combination, tau_min)` in the file, on any day, so a
  restarted logger skips what it already wrote; empty when the file does not exist.
- `verify()`: `LogSummary(n_records, by_tau, no_board, sealed, day_status)`, counts only. `by_tau`
  counts every record by `tau_min`, so its values sum to `n_records`.

## What the chain detects

Detected, as `ShadowLogTampered`:

- an edited, inserted, reordered or deleted line that has a later line after it;
- on a sealed log, any change to the records (the seal chains onto the last record and counts them);
- a line after the seal, a cut-off last line (no newline), a line not in the log's own encoding.

Not detected without an outside anchor:

- on an unsealed log, a change to the last line, or the removal of the last whole lines;
- the removal of the seal line, alone or with the last records before it: the log then reads as
  unsealed;
- a rewrite of the whole file with recomputed hashes (the chain has no secret).

The sha256 that `seal()` returns is that anchor if it is kept elsewhere; this module keeps none.
