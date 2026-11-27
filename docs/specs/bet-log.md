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
every call), one file per race day, named by its local (JST) date. A day has one log, at one place,
so `check` and `record` always count the same stakes. `BetLog(path, day=..., limits=...)` takes only
`bet_log_path(day)`, compared after resolving (`..`, symlinks and a path relative to the working
directory that name the same file are the same log):

- a path inside the repository raises `BetLogInRepo` (a `RuntimeError`;
  `core.eval.reports.is_inside` against `core.env.repo_root()`, so `..`, symlinks and relative
  paths count); this check runs first;
- any other path raises `ValueError`, even one outside the repository named `<day>.jsonl`, and once
  `HORSETRACK_DATA_DIR` moves, a file under the old data dir is no longer the day's log;
- a refused path creates nothing.

- One JSON object per line, compact separators, ASCII-escaped (so UTF-8), each line ending in `\n`.
- The file is opened only for append, never truncated or rewritten. Each `record` is one line,
  written, flushed and `os.fsync`ed before it returns (and the directory fsynced when the file is
  new). The writer holds an exclusive `flock` from reading the file to the end of its write, so two
  writers cannot chain onto the same line; readers take a shared lock.
- Every call (`record`, `check`, `read`, `staked_yen`) reads the whole file and checks the chain,
  each line's checksum and each record first. A broken chain or checksum, or a record that is not a
  bet of the log's day, raises `BetLogTampered` (a `ValueError`), and nothing is appended onto it.

### Lines

```json
{"seq":1,"prev_sha256":"0000…0000","record":{…,"seq":1,…},"sha256":"<sha256 of this line without sha256>"}
{"seq":2,"prev_sha256":"<sha256 of line 1>","record":{…,"seq":2,…},"sha256":"<…>"}
```

- The keys are `seq`, `prev_sha256`, `record` and `sha256`, in that order.
- `seq`: 1 on the first line, one more on each line after it; the record carries the same `seq`.
- `prev_sha256`: the hex sha256 of the previous whole line's bytes (its `sha256` included) without
  its newline; 64 zeros on the first line.
- `sha256`: the hex sha256 of the line's own encoding without the `sha256` member, the other keys
  in the line's order (`{"seq":…,"prev_sha256":…,"record":…}`). A wrong or missing `sha256` is a
  changed line, so an edit to the last line is caught too.
- A line is exactly the bytes the log writes for its JSON: the same JSON with other whitespace or
  escapes is a changed line.
- Every stored record must be a bet the log would take on its day: a record whose `race_id` is
  dated another day (a log copied under another day's name, or one record edited) is tampering, as
  is a combination not in the store's order.

### Record fields

Given by the caller:

| key | type | notes |
| --- | --- | --- |
| `race_id` | text | `jra:<YYYY-MM-DD>:<VENUE>:<NN>`, the log's own day, a venue code of `config/jra.yaml`, `NN` from `01`; a WIN5 bet, and only a WIN5 bet, takes the day-level key `jra:<YYYY-MM-DD>:WIN5` (as `research/win5.py` keys the pool), dated the log's own day |
| `pool` | text | one of `POOLS`, the jra config's pool codes: `WIN`, `PLACE`, `BRACKET_QUINELLA`, `QUINELLA`, `QUINELLA_PLACE`, `FORECAST`, `TRIO`, `TIERCE`, `WIN5` |
| `combination` | text | horse numbers joined by `,` (`"4"`, `"4,7"`), no leading zeros, as many numbers as the pool's `legs`; stored in the store's order (see Combinations) |
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

Records are stored as given, but for the combination's order: no keys dropped, no values coerced,
no timestamps normalised. `record` returns the stored record and does not change the caller's dict.

### Combinations

A combination is stored as the store's dividends hold it (`jravan/dividends.py`), so a bet joins
its dividend on `race_id`, `pool` and `combination`:

- the legs of a pool whose jra config `kind` is `ordered_*` (FORECAST, TIERCE: finishing order) or
  `multi_race_*` (WIN5: race order) are stored as given;
- the others (BRACKET_QUINELLA, QUINELLA, QUINELLA_PLACE, TRIO) ascending by number, not as text:
  `"7,4"` is stored `"4,7"`, `"12,3"` is stored `"3,12"`.

A number given twice is refused, except where the legs are not horses of one race: BRACKET_QUINELLA
(`kind` `*_bracket`; `"3,3"` is two horses of one bracket) and WIN5 (one race per leg;
`"1,1,1,1,1"`).

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
  outside `POOLS`; a WIN5 bet keyed by one race, or another pool on the WIN5 key; a blank or
  non-numeric combination, one with a leading zero, one of the wrong number of horses, or one with
  a number twice where the pool takes no repeat; a stake that is not an integer number of hundreds
  of yen (`100.0`, `"100"`, `150`, `0`); a `placed_utc` that is not ISO text with a UTC offset (a
  date, a naive time, a datetime object); an `option_id` that is blank or not text; a `note` that
  is not text.

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
- an edit to any line, the last one included, that leaves its `sha256` as it was, and a wrong or
  missing `sha256`;
- a record dated another day than the log's, whether one record was edited (hashes recomputed) or
  another day's whole log was copied under this day's name;
- a cut-off last line (no newline), a line not in the log's own encoding, a line or record not of
  the shape this log writes (keys, `seq`, a bet the log would refuse, a combination not in the
  store's order, the flags, `day_status`).

Not detected without an outside anchor:

- the removal of the last whole lines (what remains is a valid shorter chain);
- a rewrite of the file with every `sha256` and `prev_sha256` recomputed (the chain has no secret).
