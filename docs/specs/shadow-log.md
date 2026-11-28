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
| `record_version` | integer, optional | `1` or `2` (`RECORD_VERSION`, #564); absent means 1 |
| `p_model` | number in [0, 1] or null | version 2 only, and required there: the model's probability of the combination; null without one |
| `value_model` | number >= 0 or null, optional | `p_model` x `board_odds`, on the pools with no price forecast |

The key of a record is `(race_id, pool, combination, tau_min)`. Other keys (`board_odds`,
`value_mean`, `value_q05`, `value_q95`, `flags`, ...) are kept as given. Records are stored as given:
no keys dropped, no values coerced, no timestamps normalised; the log adds `day_status` only.

### Versioned values (#566)

Every `ok` record of a point valued with models (`models` not null) also carries:

| key | type | notes |
| --- | --- | --- |
| `values_version` | integer | `2` from #566 on |
| `starters` | integer | the runners at the start of sales the PLACE and Wide values used: the runners priced on the board plus the ones scratched on it |

JRA fixes the places Place pays (and whether Wide is sold) by the runners at the start of sales, so
a scratch after sales opened leaves them unchanged: an 8-drawn race that loses a runner still pays
three places. The shadow day takes a scratch on the board as one after sales opened (an assumption,
`docs/ledger.md`), and passes `starters` to `shadow_values`, which reads the places from it. Only
the places come from `starters`; WIN values, shocks and the rows are those of the runners priced.

A record without the two keys has values version 1: its places came from the runners priced at the
time. `values_version` is its own axis, apart from `record_version` (#564, "Record versions"
below): `record_version` says whether the record carries a `p_model`; `values_version` says how
its PLACE and Wide values took their places. A shadow-day record has both (a point valued with
models: `record_version` 2 and `values_version` 2), and the `p_model` of the same record forms its
places from `shadow_feed._starters` (the log's opening of sales), not from the `starters` above.
The log keeps both keys as given (like any other key, covered by the chain), does not require
them, and old files keep verifying and take new records. A `no_board` record, and a point logged
without models, carry neither key.

Example, synthetic:

```json
{"race_id": "jra:2026-11-28:TOK:11", "pool": "WIN", "combination": "7", "tau_min": 2,
 "decision_utc": "2026-11-28T06:38:00Z", "board_capture_utc": "2026-11-28T06:37:44Z",
 "written_utc": "2026-11-28T06:38:16Z", "mode": "live", "status": "ok", "board_odds": 3.4,
 "value_mean": 0.81, "value_q05": 0.74, "value_q95": 0.88, "flags": ["tail"],
 "models": {"win": "a1", "ordering": "b2", "nowcast": "c3"}, "git_sha": "0123abc"}
```

### Record versions

`record_version(record)` is the record's `record_version`, or 1 when it has none: every record
written before #564, and the formation records. A version-1 record has no `p_model` (it reads as
`record.get("p_model") is None`); nothing is added to an old record, so a log written before #564
verifies and reads as it did, and a new logger carries on onto it (a day can mix both versions).

### Shadow-day records (version 2, #564)

`shadow.day.run_shadow_day` writes every record as version 2 with a `p_model`, the no_board record
(null) included. `p_model` is the shadow feed's of the point's decision board
(docs/specs/board-feed.md, "Model probabilities", through `shadow_feed._p_model` and
`_starters`): WIN `models.win.predict(q, board.context())` on the board's 1/O_WIN, every other pool
`model_probabilities(p, models.ordering, brackets=..., starters=...)`. Null without models, on a
record logged without models (its values failed for every pool: a p_model is kept only beside the
fit ids it came from), for every pool but WIN when the ordering model is not an `OrderingFit`, and
on every record when the computation fails (logged, the point is still written).

The WIN, PLACE and Wide records keep their #120 fields and add `record_version` and `p_model`.
The pools with no price forecast (QUINELLA, BRACKET_QUINELLA, FORECAST, TRIO, TIERCE) get records
of their own: `race_id`, `pool`, `combination` (the odds log's text), `tau_min`, `decision_utc`,
`written_utc`, `mode`, `git_sha`, `record_version`, `status` `ok`, `models` (the fit ids of the
p_model), `p_model`, `board_odds` (the capture's odds), `value_model` = p_model x board_odds (gross
return per 1 yen if the odds stood), `board_capture_utc` and `board_age_s` (decision time minus
that capture).

- The capture: per pool, the rows sharing the latest `snapshot_utc` among its rows available
  (fetched, else captured) by the decision time; `board_capture_utc` is the earliest availability
  among them, so never after the decision. Priced rows only (odds present, finite and positive,
  not SCR). A pool with no capture by then writes no record. No `max_board_age_s` here: the
  logger captures FORECAST, TRIO and TIERCE once per grid point, at or after it (logger.yaml
  `tier2`), so the capture a point can use is usually the previous point's, minutes old; the
  record says how old in `board_age_s` and a reader filters on it.
- The stored subset: only combinations with a p_model (a scratched runner has none), and of
  those every QUINELLA and BRACKET_QUINELLA combination (at most 153 and 36 a race) but only the
  `EXOTIC_TOP_N` = 15 with the largest p_model of FORECAST, TRIO and TIERCE (all when fewer; ties
  by combination text), the rows the shadow page shows. An 18-runner race has 306 Exacta, 816
  Trio and 4,896 Trifecta combinations: logging all of them at six grid points would put some
  37,000 lines a race (6,207 a point) in the log, almost all of long shots whose p_model x odds scoring can only
  measure in aggregate. The p_model is still computed over every ordered triple, one race at a
  time, never kept.
- Without a p_model for a pool (no models, no ordering fit, a failed computation), no record of
  that pool is written; the WIN, PLACE and Wide records are written as before.

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
- has a `record_version` other than the integer 1 or 2 (text, a float, a bool, null refused);
- is version 2 without a `p_model` key (null is allowed), or version 1 with one;
- has a `p_model` that is neither null nor a number in [0, 1], or a `value_model` that is neither
  null nor a finite number >= 0 (a bool is not a number here);
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

## Scoring

Code: `src/horsetrack/countries/japan/shadow/score.py` (#564).
`score_shadow(records, dividends) -> {(pool, tau_min): PoolScore}` puts one notional flat 100 yen
on every scored record. `records` are `read()`'s (a development day's); `dividends` a frame with
`bets.settle.DIVIDEND_COLUMNS` (`race_id`, `pool`, `combination`, `dividend`, `is_refund`), the
dividend in yen per 100 yen, matched as `bets.settle` matches a bet (both sides in the store's
order: FORECAST and TIERCE in finishing order, the other pools in any order).

- Scored: version 2, `p_model` not null, `status` `ok`, `board_odds` not null, `day_status`
  `development`, and its race and pool have dividend rows.
- Left out: version-1 records, a null `p_model`, no_board records, holdout-labelled records, a race
  and pool with no dividend row yet (unsettled), and a combination with a refund row (its stake
  comes back; it is not measured).
- `PoolScore(pool, tau_min, n, hits, expected_per_100, realised_per_100)`: `n` the records scored,
  `hits` those whose combination has a paying row, `expected_per_100` = 100 x mean(p_model x
  board_odds) (the return per 100 yen the model claims at the decision-time odds),
  `realised_per_100` = the dividends paid summed over the scored records / n (0 for a loss).
- A (pool, tau) with nothing scored has no entry. The module reads no file and opens no holdout:
  a holdout day stays behind `LockedHoldout.open`.

## The ladder log

Code: `src/horsetrack/countries/japan/shadow/ladder_log.py` (#628), written by `shadow-day`
(`run_shadow_day(..., ladder_log=...)`; the command always writes it). It keeps the shadow page's
odds ladder (#499, docs/specs/board-feed.md, "The odds ladder") so the live nowcast can be scored
on the day's file alone. It is display data: it never feeds a value, a flag or the shadow log, and
the shadow log's records are the same with or without it.

Path: `$HORSETRACK_DATA_DIR/shadow/japan/<YYYY-MM-DD>.ladder.jsonl` (`ladder_log_path(day)`), beside
the shadow log. `LadderLog(path, day=...)` shares `ShadowLog`'s lines, chain, seal, `day_status`,
`read()` (`HoldoutSealed` on a holdout day), `logged_keys()` and `verify()` (`by_tau` counts by
`checkpoint_min`) through `shadow.log.ChainedLog`; it is a separate file because the shadow log's
key already uses tau 5 and 2 for the decision grid. The key of a record is
`(race_id, pool, combination, checkpoint_min)`.

What `shadow-day` writes, for each race with a post time:

- At each checkpoint k in (20, 10, 5, 2), once it is due at post - k + `grace_s`: one record per
  WIN runner, PLACE runner and Wide pair on the board of the rows captured at or before post - k
  (the latest row per key by `snapshot_utc`, `_DayRows.refs`' rule, of the rows read so far).
  `odds` is that row's low figure, `fetched_utc` its fetch time, and `projected` maps every later
  checkpoint j < k and `"0"` (the final) to `shadow_feed._projector(models)(pool, odds, tau, j)`,
  tau the minutes from that fetch to the post (null without a nowcast or a cell it needs).
- A checkpoint whose board is missing, or older than its limit, is one record with `pool` and
  `combination` `*` and `status` `no_board`. The age is post - k minus the board's
  `capture_utc` (the earliest fetch of its rows). The limits are shadow-day's `max_board_age_s`
  (#625): a tau's own limit when it has one, else linear in tau between the nearest limited taus
  below and above (the shipped limits give T-20 about 214 s and T-10 about 115 s).
- Once the race's final board is in (`snapshot.final_board` of its captures showing a final): one
  record per key of it with `checkpoint_min` 0, `status` `ok`, `odds` the final low figure,
  `fetched_utc` the final board's `at_utc` and `projected` `{}`. A pool whose final capture comes
  later gets its records then.
- A restart writes only the checkpoints and final keys not yet in the file, also after the shadow
  log is sealed. The run seals the ladder log once the shadow log is sealed and every race has its
  four checkpoints and a final for WIN and every pool its checkpoints showed; when a final never
  comes it is sealed at 17:30 Asia/Tokyo without it. A rerun of a sealed day writes nothing. A
  failure of the ladder log is logged and the shadow log goes on without it.

| key | type | notes |
| --- | --- | --- |
| `race_id`, `pool`, `combination` | text | as the shadow log; `*` on a no_board record |
| `checkpoint_min` | integer | 20, 10, 5, 2, or 0 for a final |
| `status` | text | `ok` or `no_board` |
| `odds` | number > 0 or null | the observed (or final) low figure; null only on no_board |
| `fetched_utc` | ISO UTC text or null | the observed row's fetch time; the final board's `at_utc` |
| `board_age_s` | number or null | the checkpoint board's age; null on a final and a missing board |
| `projected` | mapping | `{"<j>": odds or null}` for each later checkpoint and `"0"`; `{}` on no_board and final |
| `nowcast_id` | text or null | `models.ids["nowcast"]` of the run; null without models |
| `mode` | text | `live` when written within `max_lag_s` of post - k (of the final's `at_utc`), else `replay` |
| `post_utc`, `written_utc` | ISO UTC text | the race's post; when the record was written |
| `git_sha` | text | the commit shadow-day ran |
| `day_status` | text | set by the log, as the shadow log |

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
