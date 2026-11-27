# Bet log

Code: `src/horsetrack/countries/japan/bets/log.py` (#453). Every bet the owner places by hand on a JRA
race day is recorded once, in a hash-chained file, and counted against the race-day cap. No code
logs in to a betting account or places a bet; the `bets` package imports no HTTP client, browser
driver or `horsetrack.core.ingest`.

The bet log is the owner's financial record: it stays under `$HORSETRACK_DATA_DIR`, never in git, a
PR, a ticket or a comment. It is separate from the shadow log (docs/specs/shadow-log.md), which
refuses stake keys by design.

## The log

Path: `$HORSETRACK_DATA_DIR/bets/japan/<YYYY-MM-DD>.jsonl` (`bet_log_path(day)`, the data dir read on
every call), one file per race day, named by its local (JST) date. `BetLog(path, day=..., limits=...)`
refuses a file whose name is not `<day>.jsonl` (`ValueError`) and a path inside the repository
(`BetLogInRepo`, a `RuntimeError`; `core.eval.reports.is_inside` against `core.env.repo_root()`, so
`..`, symlinks and relative paths count).

- One JSON object per line, compact separators, ASCII-escaped (so UTF-8), each line ending in `\n`.
- The file is opened only for append, never truncated or rewritten. Each `record` is one line,
  written, flushed and `os.fsync`ed before it returns (and the directory fsynced when the file is
  new). The writer holds an exclusive `flock` from reading the file to the end of its write, so two
  writers cannot chain onto the same line; readers take a shared lock.
- Every call (`record`, `check`, `read`, `staked_yen`) reads the whole file and checks the chain
  first. A broken chain raises `BetLogTampered` (a `ValueError`), and nothing is appended onto it.

### Lines

```json
{"seq":1,"prev_sha256":"0000…0000","record":{…,"seq":1,…}}
{"seq":2,"prev_sha256":"<sha256 of line 1>","record":{…,"seq":2,…}}
```

- `seq`: 1 on the first line, one more on each line after it; the record carries the same `seq`.
- `prev_sha256`: the hex sha256 of the previous line's bytes without its newline; 64 zeros on the
  first line.
- A line is exactly the bytes the log writes for its JSON: the same JSON with other whitespace or
  escapes is a changed line.

### Record fields

Given by the caller:

| key | type | notes |
| --- | --- | --- |
| `race_id` | text | `jra:<YYYY-MM-DD>:<VENUE>:<NN>`, the log's own day, a venue code of `config/jra.yaml`, `NN` from `01` |
| `pool` | text | one of `POOLS`, the jra config's pool codes: `WIN`, `PLACE`, `BRACKET_QUINELLA`, `QUINELLA`, `QUINELLA_PLACE`, `FORECAST`, `TRIO`, `TIERCE`, `WIN5` |
| `combination` | text | horse numbers joined by `,` (`"4"`, `"4,7"`), as the store's dividends hold them; as many numbers as the pool's `legs` |
| `stake_yen` | integer | at least 100 and a multiple of 100 (the jra config's `unit_stake`) |
| `placed_utc` | ISO text with a UTC offset | when the bet was placed; `Z`, `+00:00` or another offset |
| `option_id` | text or null | optional: the leaderboard option bet on; null for the owner's own pick |
| `note` | text or null | optional |

Set by the log, never by the caller:

| key | type | notes |
| --- | --- | --- |
| `seq` | integer | the line's `seq` |
| `written_utc` | ISO UTC text | when the log wrote the record |
| `day_status` | text | `holdout` or `development` (see Holdout days) |
| `over_cap` | boolean | the day's staked total including this bet exceeds the day cap |
| `off_flat` | boolean | a flat stake is set and this stake differs from it |

Records are stored as given: no keys dropped, no values coerced, no timestamps normalised. `record`
returns the stored record and does not change the caller's dict.

Example, synthetic:

```json
{"race_id": "jra:2026-11-22:TOK:11", "pool": "QUINELLA", "combination": "4,7", "stake_yen": 1000,
 "placed_utc": "2026-11-22T05:40:00Z", "option_id": null, "note": "own pick", "seq": 3,
 "written_utc": "2026-11-22T05:41:12.512034+00:00", "day_status": "development",
 "over_cap": false, "off_flat": false}
```

## The day cap and the flat stake

`Limits(day_cap_yen, flat_stake_yen)`; `load_limits()` reads `countries/japan/config/betting.yaml`
(owner's answer on #453: `day_cap_yen: 30000`, `flat_stake_yen: 1000`; `flat_stake_yen` may be
null for no flat stake). Changing either is an owner decision.

- `check(stake_yen)` returns `CapCheck(fits, staked_yen, remaining_yen, off_flat)` and writes
  nothing: `staked_yen` is the day's total in the file, `remaining_yen` is the cap minus that total
  before this stake (negative once the day is past the cap), `fits` is whether the total plus this
  stake is at most the cap. A stake that is not whole hundreds raises `BetRefused`. `check` is the
  gate before betting.
- A placed bet is a fact: a bet that takes the day past the cap, or that differs from the flat
  stake, is recorded and flagged `over_cap` or `off_flat`, never refused.
- `staked_yen()` is the day's total, every bet in the file counted, whatever the limits; the cap and
  the flat stake are the `BetLog`'s limits.

## Refused bets

`record(bet)` raises `BetRefused` (a `ValueError`) and writes nothing (no file is created for a
refused first bet, and a refusal uses up no `seq`) when the bet:

- is not a dict, or lacks a caller field of the table;
- has any other key, including the five the log sets (a stake under another name, a caller-set
  `seq`, `over_cap`, ...);
- has a field of the wrong type or form: a `race_id` of another source, form, day or venue; a pool
  outside `POOLS`; a blank or non-numeric combination, or one of the wrong number of horses; a
  stake that is not an integer number of hundreds of yen (`100.0`, `"100"`, `150`, `0`); a
  `placed_utc` that is not ISO text with a UTC offset (a date, a naive time, a datetime object);
  an `option_id` that is blank or not text; a `note` that is not text.

## Holdout days

- `day_status` is `holdout` when `is_holdout_day("japan", day)` holds (the committed rule in
  `countries/japan/config/holdout.yaml`, #71), else `development`. Once any bet in a file is logged
  as `holdout`, later bets are labelled `holdout` too (fail closed).
- Stakes are not outcomes: a holdout day's bets are written, read back (`read`, `staked_yen`,
  `check`) and counted toward the cap like any other day's. The outcomes of holdout days stay
  locked (D14), behind `LockedHoldout.open`; this module reads no outcome.

## Reading

- `read()`: the day's bets in order, each as stored; empty when the file does not exist.
- `staked_yen()`: the sum of `stake_yen` over the file.

## What the chain detects

Detected, as `BetLogTampered`:

- an edited, inserted, reordered, replayed or deleted line that has a later line after it;
- a cut-off last line (no newline), a line not in the log's own encoding, a line or record not of
  the shape this log writes (keys, `seq`, a stake, the flags, `day_status`).

Not detected without an outside anchor:

- a change to the last line that keeps it well formed, or the removal of the last whole lines;
- a rewrite of the whole file with recomputed hashes (the chain has no secret).
