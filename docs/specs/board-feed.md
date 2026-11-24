# The live-board feed

Code: `src/horsetrack/countries/japan/board/feed.py`. Input: a race day's odds snapshot logs
(docs/specs/odds-log.md), read with `odds_log.read_rows`, and the post times of
`snapshot.post_times_for`. The feed never requests anything over the network.

`build_board(race_date, *, venues=None, now=utc_now, morning_from='07:00') -> dict`

## Shape

```
{race_date, generated_utc,
 venues: [{venue,
           races: [{race_no, race_id, post_utc, last_capture_utc,
                    pools: {POOL: {investment,
                                   rows: [{combination, odds, odds_hi, status,
                                           morning_odds, move_morning_pct,
                                           t20_odds, move_t20_pct}]}}}]}]}
```

- The feed is JSON-ready: `race_date` is ISO date text, every timestamp is ISO-8601 UTC text,
  and a missing figure is null.
- `venues`: every `<VENUE>.jsonl.gz` under `odds_snapshots/jra/<date>/`, sorted by name, or only
  those named in `venues`. No log for the day gives an empty list.
- `races`: every race id `jra:<date>:<VENUE>:<NN>` in the venue's log, in race-number order.
  `last_capture_utc` is the latest `snapshot_utc` among the race's rows; `post_utc` the race's
  post time, or null.
- `pools`: every pool the race's rows name, sorted by name. `investment` is the `investment` of
  the latest `__pool__` row of that pool that carries one, else null; `__pool__` is never a row.
- `rows`: one per logged `combination` text, in numeric order (`'2'` before `'10'`, `'1,7'` before
  `'3,7'`). `odds` and `odds_hi` are the current row's (`odds_hi` null for single-figure pools),
  `status` its `status_raw` as logged.
- No key names a probability, an expected value, a fair price, a bet or a stake: the feed shows
  prices and their moves only.

## As of now

A row's capture time is its `snapshot_utc`; a row without a readable one is not used. Rows
captured after `now()` are ignored everywhere (current odds, both references and
`last_capture_utc`), so replaying a past day with an earlier `now` gives the board as it stood at
that moment. `generated_utc` is `now()`.

## Current odds and references

- Current: per (race, pool, combination), the row with the latest capture time; at the same
  instant the later row in the log wins. Log order otherwise does not matter.
- Morning reference (`morning_odds`): the earliest priced row of the combination captured on race
  day at or after `morning_from` Japan time (+09:00; 07:00 JST is 22:00 UTC the day before) and
  before midnight Japan time. Rows from before `morning_from` or from another day never count.
  When the only such row is the current one, the reference is that row and the move is 0.0.
- T-20 reference (`t20_odds`): the latest row of the combination captured at or before
  post - 20 min, when that row is priced. Null while `now` is before post - 20 min, without a post
  time, and when the combination has no row by then.
- Move: `current / reference - 1`, in percent rounded to one decimal, on `odds` (the low figure
  of a ranged pool such as PLACE or QUINELLA_PLACE). Null when either figure is missing.

## Scratches

A row whose current `status_raw` is `SCR` carries status `SCR`, null `odds` and `odds_hi`, and
null moves. Its references still show what the runner was priced at, for the record.

## Post times

`snapshot.post_times_for(race_date, venue)` (docs/specs/board-at-t.md): the `<VENUE>.races.json`
sidecar beside the log; else HKTrack's stored card,
`processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet` column `post_time_local` (the copied
HKTrack days, 2026-11-21 and 22 at Kyoto, have no sidecar); else null for the race.
