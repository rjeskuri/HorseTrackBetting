# Bet log

Code: `src/horsetrack/countries/japan/bets/log.py` (#453), `settle.py` and `cli.py` (#454). Every bet
the owner places by hand on a JRA race day is recorded once, in a hash-chained file, counted against
the race-day cap and settled against the final dividends (see Settlement and Commands). No code
logs in to a betting account or places a bet; the `bets` package imports no HTTP client, browser
driver or `horsetrack.core.ingest`.

The bet log is the owner's financial record: it stays under `$HORSETRACK_DATA_DIR`, never in git, a
PR, a ticket or a comment. It is separate from the shadow log (docs/specs/shadow-log.md), which
refuses stake keys by design.

## The log

Path: `$HORSETRACK_DATA_DIR/bets/japan/<YYYY-MM-DD>.jsonl` (`bet_log_path(day)`, the data dir read on
every call), one file per race day, named by its local (JST) date. A day has one log, at one place,
so `check` and `record` always count the same stakes. `BetLog(path, day=..., limits=...)` takes only
`bet_log_path(day)`, compared by filesystem identity: two paths are the same log when they are
equal once resolved (`..`, symlinks and a path relative to the working directory), or else when the
nearest existing ancestor of each (the path itself included) is the same file by device and inode
and the not-yet-existing names below it are identical. So on a filesystem that ignores letter case
(macOS's default APFS), a data dir spelled in another case is the same log, before the day's first
bet too; on a case-sensitive one it is another directory.

- a path inside the repository raises `BetLogInRepo` (a `RuntimeError`;
  `core.eval.reports.is_inside` against `core.env.repo_root()`, so `..`, symlinks and relative
  paths count); this check runs first;
- any other path raises `ValueError`, even one outside the repository named `<day>.jsonl`, and once
  `HORSETRACK_DATA_DIR` moves, a file under the old data dir is no longer the day's log;
- a refused path creates nothing.

An open `BetLog` re-resolves `bet_log_path(day)` at the start of every call (`record`, `read`,
`check`, `staked_yen`). Once that is no longer its path (`HORSETRACK_DATA_DIR` moved in the
environment or `.env`, or became blank), the call raises `ValueError` (not `BetLogTampered`,
`BetRefused` or `DataDirNotSet`) and touches neither the old file nor the new data dir, so a
process holding a log cannot stake into the old file while a fresh `check` at the new place sees
an empty day. Moved back, the same object works again.

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

- The keys are `seq`, `prev_sha256`, `record` and `sha256`, in that order (format 2; for the
  format-1 prefix without `sha256`, see Format version).
- `seq`: 1 on the first line, one more on each line after it; the record carries the same `seq`.
- `prev_sha256`: the hex sha256 of the previous whole line's bytes (its `sha256` included) without
  its newline; 64 zeros on the first line.
- `sha256`: the hex sha256 of the line's own encoding without the `sha256` member, the other keys
  in the line's order (`{"seq":…,"prev_sha256":…,"record":…}`). A wrong `sha256` is a changed
  line, so an edit to the last line is caught too. A line without `sha256` is format 1, taken only
  as the leading run the Format version section describes.
- A line is exactly the bytes the log writes for its JSON: the same JSON with other whitespace or
  escapes is a changed line.
- Every stored record must be a bet the log would take on its day: a record whose `race_id` is
  dated another day (a log copied under another day's name, or one record edited) is tampering, as
  is a combination not in the store's order (a format-1 record is checked as a565ac0 checked it).

### Format version

`record` writes format 2, `{seq, prev_sha256, record, sha256}`. Format 1, `{seq, prev_sha256,
record}` with no `sha256`, is what the first revision of #453 (a565ac0) wrote. A day begun in
format 1 is read and continued, append-only; nothing is rewritten:

- A file may start with a run of format-1 lines. Once a format-2 line appears, every later line is
  format 2: a format-1 line after a format-2 line is `BetLogTampered`.
- Both formats chain alike: line 1's `prev_sha256` is 64 zeros, each later line's is the sha256 of
  the previous line's bytes, whatever its format; `seq` runs in order; every line is in the log's
  own compact encoding.
- A format-1 record is checked as a565ac0 checked it: the record shape (keys, `seq`, flags,
  `day_status`) and the bet checks a565ac0 made when writing (day, venue, pool, the number of legs,
  the stake, `placed_utc`, `option_id`, `note`). The three checks a565ac0 did not make are skipped:
  the store's combination order, the no-repeated-number rule, and the WIN5 day key with its
  pairing rule. a565ac0's `race_id` is always `jra:<day>:<VENUE>:<NN>` at a known venue, a WIN5 bet
  included, so a format-1 record on `jra:<day>:WIN5` is tampered.
- A format-1 record's `written_utc` must be ISO text with an offset, earlier than
  `FORMAT_2_SINCE`, 2026-11-27T09:31:47Z (the commit time of f366f20, when format 2 replaced
  format 1). A format-1 line stamped at or after it is a format-2 line with its `sha256` stripped,
  and is tampered.
- `read`, `staked_yen` and `check` count format-1 records exactly as stored. `record` appends a
  format-2 line chained onto the last line, whatever its format. A `holdout` label on any record
  carries forward (see Holdout days).
- `read` returns a format-1 record as stored, so its combination may not be in the store's order
  and a format-1 WIN5 bet carries a venue `race_id`. A reader that joins dividends (`settle_bets`,
  #454) first puts a format-1 combination in the store's order (see Combinations) and keys a
  format-1 WIN5 bet by its day (`jra:<day>:WIN5`); the stored record itself is never changed.

Limits: the chain has no secret, so an editor who strips a format-2 line's `sha256` and also
rewrites its `written_utc` to before `FORMAT_2_SINCE` is not detected, the same limit as a full
recompute. An all-format-1 file's last line carries no checksum of itself, so an edit to it is
caught only once a format-2 line follows it.

A future change to the line or record format adds another explicit, append-only upgrade, in a
ticket of its own, rather than read old lines loosely.

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

A combination is stored as the store's dividends hold it (`jravan/dividends.py`), so a bet of one
of the eight single-race pools that table covers (WIN to TIERCE) joins its dividend on `race_id`,
`pool` and `combination`. WIN5 is not in that table, and the WIN5 history (`research/win5.py`)
stores day-level payouts and the number of winning combinations, not the combinations, so a WIN5
bet stays `unsettled` until combination-level WIN5 results reach `dividends` (see Settlement). The
order:

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

## Settlement

Code: `bets/settle.py` (#454). It reads the dividends and writes no file.

`settle_bets(bets, dividends)` takes `read()` rows and dividend rows (`race_id` in the log's form,
`pool`, `combination`, `dividend` in yen per ¥100, `is_refund`) and returns one row per bet, in
order, its columns kept, plus `outcome`, `paid_yen` and `return_per_100`:

| outcome | when | `paid_yen` | `return_per_100` |
| --- | --- | --- | --- |
| `holdout` | its `day_status` is `holdout`, or `is_holdout_day("japan", <race day>)` holds, whatever dividends are passed | null | null |
| `unsettled` | its race has no dividend row of its pool yet (rows of other pools only are not a loss) | null | null |
| `refunded` | its combination has an `is_refund` row (`day_dividends` adds one for a runner scratched before the start, see Scratched runners) | its stake | null |
| `won` | its combination is a paying row of its race and pool | stake / 100 × that row's dividend | paid / stake × 100 |
| `lost` | its race and pool have rows, none of them its combination | 0 | 0 |

- Matching: both sides are put in the store's order (see Combinations), so PLACE, BRACKET_QUINELLA,
  QUINELLA, QUINELLA_PLACE and TRIO match in any order, WIN, FORECAST and TIERCE only as given, and
  WIN5 as text in race order. A dead heat, and Place and Wide on every result, pay each paying
  combination its own dividend; each bet is paid by the one row of its combination.
- WIN5 is keyed by its day, `jra:<day>:WIN5`, a format-1 record on a venue race included. The
  store's `dividends` holds no WIN5 rows, so a real WIN5 bet stays `unsettled`.
- A race abandoned, or a pool not established (不成立), writes no dividend rows, so its bets stay
  `unsettled`.
- A combination with two different rows in one race and pool, a dividend row whose combination is
  not horse numbers, or a paying row without a dividend raises `ValueError`, for the races and
  pools of the bets only: settlement never guesses which row paid.
- Holdout outcomes are read only through `LockedHoldout.open` (D14, D15); this module never opens
  it, so a holdout-day bet stays `holdout` here.

`played_by_option(settled)`: one row per `option_id`, sorted (`off-board` for a null or absent
one, the owner's own pick), with `n_bets`, `staked_yen`, `paid_yen` and `return_per_100` (paid /
staked × 100, null when nothing was staked) over the won and lost bets, and `n_refunded` and
`n_pending` (`unsettled` and `holdout` bets) beside them. `totals(settled)` gives the same figures
over all the rows, the day's line of `bets settle`.

`day_dividends(day)`: the jra store's dividend rows of the day's races, and the refund rows of its
scratched runners (below). A holdout day (`is_holdout_day("japan", day)`) gives an empty frame of
the five columns without opening the store, so it needs no store on disk. Otherwise `races`
(`race_id, race_date, venue, race_no`) is read through `models/v0/store.read_store_table`, which
drops holdout days, then the `dividends` rows of those race ids; each race is keyed
`jra:<date>:<VENUE>:<NN>` from its race_date, venue (the jra config's venue code; a venue number
or name maps to it) and race_no. A day without races gives the empty frame too.

### Scratched runners

JRA returns the stake of a ticket on a runner scratched before the start, but the store's
`dividends` holds no refund row for it (the results parser and jravan write `is_refund` false).
`day_dividends` adds them from `runners` (`race_id, horse_no, bracket, is_scratched`, read through
`read_store_table`): a refund row (`is_refund` true, no dividend) for each combination below, in
each race with a scratched runner and each pool the race has dividend rows of.

- Refunded: a runner with `is_scratched` true, which is exactly 出走取消 (`placing_code` WV) and
  発走除外/競走除外 (WX). Every combination of the race's entered horse numbers that holds one is
  refunded, in every pool from WIN to TIERCE.
- Not refunded: a runner pulled up (競走中止, PU) or disqualified (失格, DISQ) started; a bet on it
  is a loss like any other.
- BRACKET_QUINELLA: a pair of two brackets is refunded only when one of them has no runner left; a
  pair of one bracket (`"7,7"`, ゾロ目) when that bracket, entered with two or more, has fewer than
  two left.
- WIN5: `day_dividends`, and so `bets settle`, adds no refund rows for a scratched leg horse, and
  a WIN5 bet stays `unsettled` as above, because the store holds no WIN5 combination dividends to
  refund against. JRA's WIN5 rule for a scratched horse is not established here: `docs/ledger.md`
  (site, rules and pricing) carries the reading of netkeiba's WIN5 shortfall as refunds of
  combinations with a scratched horse as an assumption, unverified from a JRA page (#324).
- A combination the store already has a row of keeps that row, and a pool without rows (not sold,
  不成立, or no results yet) gets no refund rows, so its bets stay `unsettled`. A store without a
  `runners` table, which the store build never writes beside `races` (both come from the results
  pages), adds none.

## Commands

`horsetrack japan bets` (`bets/cli.py`), through the day's `BetLog`:

- `check --date YYYY-MM-DD --stake N`: prints the day's staked total, the cap and what is left
  before this stake (`BetLog.check`), writes nothing; exit 0 when it fits, 1 when it does not.
- `record --date ... --race-id ... --pool ... --combination ... --stake N [--option-id ...]
  [--placed-utc ...] [--note ...]`: appends the bet (`placed_utc` defaults to now, UTC, only when
  the option is left out: an empty one is refused like any other bad timestamp). A bet over
  the cap is kept: `over_cap` is printed and it exits 1; `off_flat` is printed too.
- `settle --date ...`: settles the day's bets against `day_dividends(day)`, writes
  `$HORSETRACK_DATA_DIR/bets/japan/<day>.settled.json` beside the log (a JSON array, one object per
  bet: its record and `outcome`, `paid_yen`, `return_per_100`), replaced whole on every run, and
  prints each bet and the day's staked, paid and return per ¥100 over won and lost bets. The bet log
  is not changed. A day with no bets prints so, writes nothing and exits 0. A holdout day settles
  without reading the store (`day_dividends` does not open it, so a fresh data dir with no store
  settles too, and no `processed/` is made): its file holds every bet as `holdout`, with no paid
  figure, and it exits 0.
- A refused stake or bet, a tampered log, an unset data dir or a missing store prints the reason
  and exits 2, writing nothing.

The settled file and the printed figures stay in the data dir and the terminal, never the
repository.

## What the chain detects

Detected, as `BetLogTampered`:

- an edited, inserted, reordered, replayed or deleted line that has a later line after it;
- an edit to any format-2 line, the last one included, that leaves its `sha256` as it was, and a
  wrong `sha256`;
- a missing `sha256`: a format-1 line after a format-2 line, or one whose `written_utc` is not
  before `FORMAT_2_SINCE` (see Format version);
- a record dated another day than the log's, whether one record was edited (hashes recomputed) or
  another day's whole log was copied under this day's name;
- a cut-off last line (no newline), a line not in the log's own encoding, a line or record not of
  the shape this log writes (keys, `seq`, a bet the log would refuse, a combination not in the
  store's order, the flags, `day_status`; for a format-1 record, the checks a565ac0 made).

Not detected without an outside anchor:

- the removal of the last whole lines (what remains is a valid shorter chain);
- a rewrite of the file with every `sha256` and `prev_sha256` recomputed (the chain has no secret);
- a format-2 line stripped of its `sha256` with its `written_utc` also rewritten to before
  `FORMAT_2_SINCE`, and an edit to the last line of a file that is all format 1 (no line after it
  pins it).
