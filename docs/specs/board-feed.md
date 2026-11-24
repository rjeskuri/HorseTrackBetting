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

A scratch carries onto every combination containing the runner. When a runner's current WIN or
PLACE row is `SCR`, every row in every pool of that race whose combination contains that runner
number (whole numbers of the comma-separated pair: runner 1 matches `'1,10'`, not `'10,11'`) is
reported the same way: status `SCR`, null `odds`, `odds_hi` and moves, references kept. This
holds whether or not the source still lists the combination: after a scratch it stops listing the
runner's pairs, whose last row would otherwise stand as current. The carry stays within the race
and starts only once the `SCR` row is captured at or before `now()`.

## Post times

`snapshot.post_times_for(race_date, venue)` (docs/specs/board-at-t.md): the `<VENUE>.races.json`
sidecar beside the log; else HKTrack's stored card,
`processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet` column `post_time_local` (the copied
HKTrack days, 2026-11-21 and 22 at Kyoto, have no sidecar); else null for the race.

## The shadow feed

Code: `src/horsetrack/countries/japan/board/shadow_feed.py` (ticket #121), served at
`/shadow-feed` beside `/feed` and read by the shadow page at `/shadow` (#122). The plain feed
above is unchanged and stays free of probabilities and values; the shadow values travel only here.

`build_shadow(race_date, *, venues=None, now=utc_now, models, flag_rules, n_draws=400, seed=0,
cache=None) -> dict`

```
{race_date, generated_utc,
 models: {win, ordering, nowcast} | null,
 races: [{race_id, venue, race_no, post_utc, minutes_to_post, tau_bucket_min, board_capture_utc,
          pools: {WIN:            {"7":   {value_mean, value_q05, value_q95, flags}},
                  PLACE:          {"7":   {value_at_low, value_at_high, flags}},
                  QUINELLA_PLACE: {"3,7": {value_at_low, value_at_high, flags}}}}]}
```

- Input: the same logs as the plain feed, read on with `odds_log.LogTail`, and the post times of
  `post_times_for`. It never reads the shadow log (docs/specs/shadow-log.md), so nothing from a
  sealed holdout record reaches the page.
- `races`: every race with a board as of `now()` (`board_at`: rows available after `now()` are not
  on it), venues sorted by name and races in number order. A race with no row available yet is
  not listed.
- `minutes_to_post`: `(post - now)` in minutes, two decimals. `tau_bucket_min`:
  `bucket_for(minutes_to_post, decision grid)`, the smallest grid horizon at or above the minutes
  to post (the largest for anything earlier) and 0 at or after the post, where the board is the
  latest one. Without a post time both are null, and so are the values.
- Values: `shadow_values(board, tau_min=tau_bucket_min, ...)` with the draws seeded by
  `point_seed(seed, race_id, tau)` as in shadow-day. A value is the gross return per 1 yen at the
  final price (1.0 is break-even). A value that cannot be formed (no models, no post time, no
  ordering fit, or a failed computation) is null; NaN and infinity are null.
- `flags`: the sorted `avoid_flags(board, rules=flag_rules, now_utc=now())` of the key, `[]` when
  none. Every WIN runner, PLACE runner and Wide pair on the board has an entry.
- `models`: the ids of the fits in use, or null when `load_shadow_models` found none; then every
  value is null and the flags still show.
- No key names a stake, a bet, Kelly, an amount or a wager.

Recompute: `cache` is a mapping the caller keeps between builds (the server keeps one). A race's
values are recomputed only when its board (`board_capture_utc` and its figures) or its bucket
changed; flags read the clock on every build. The same mapping holds the log readers, so a build
parses only the members appended since the last one; a clock that goes back reads the logs again.

Serving: `/shadow-feed` has the headers, gzip, ETag and 304 of `/feed`. It is rebuilt at most
every `min_rebuild_s`, whether or not a log changed (buckets and the stale flag follow the clock),
and its body and ETag change only when something other than `generated_utc` and
`minutes_to_post` did, so the page's poll costs a 304 between captures; the page counts down to
the post from `post_utc`. `board-serve` loads the models once at start with `load_shadow_models`.
