# The live-board feed

Code: `src/horsetrack/countries/japan/board/feed.py`. Input: a race day's odds snapshot logs
(docs/specs/odds-log.md), streamed member by member with `odds_log.iter_members`, and the post
times of `snapshot.post_times_for`. The feed never requests anything over the network.

The feed carries the tier-1 pools `WIN`, `PLACE`, `QUINELLA`, `QUINELLA_PLACE` (`BOARD_POOLS`)
with their histories and references, as below, and, since ticket #506, the tier-2 pools
`BRACKET_QUINELLA`, `FORECAST`, `TRIO`, `TIERCE` (`TIER2_POOLS`) as their latest capture only
("Tier-2 pools"). Tier-2 rows put no race on the board (a race with only such rows is not listed)
and do not move `last_capture_utc`: the board over a log, less its tier-2 pools, is exactly the
board over its tier-1 rows. `WIN5` rows are dropped as each member is read.

`build_board(race_date, *, venues=None, now=utc_now, morning_from='07:00') -> dict`

## Shape

```
{race_date, generated_utc,
 venues: [{venue,
           races: [{race_no, race_id, post_utc, last_capture_utc,
                    pools: {POOL: {investment,
                                   rows: [{combination, odds, odds_hi, status,
                                           morning_odds, move_morning_pct,
                                           t20_odds, move_t20_pct,
                                           t10_odds, t5_odds,
                                           t2_odds, move_t20_t2_pct}]},
                           TIER2_POOL: {investment, capture_utc,
                                        rows: [{combination, odds, odds_hi, status}]}}}]}]}
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
  `'3,7'`); the board and shadow pages redraw them favourite first (#498), the feed's order
  stays. `odds` and `odds_hi` are the current row's (`odds_hi` null for single-figure pools),
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
- T-2 reference (`t2_odds`, ticket #498): picked exactly as `t20_odds` is, at post - 2 min: the
  latest row captured at or before post - 2 min, when that row is priced; null while `now` is
  before post - 2 min, without a post time, and when the combination has no row by then.
- T-10 and T-5 references (`t10_odds`, `t5_odds`, ticket #499): picked exactly as `t20_odds` is,
  at post - 10 min and post - 5 min. The shadow page's odds ladder falls back to these four
  references when a row has no shadow ladder, and, as the two feeds refresh apart, shows a
  non-null reference as observed in place of a projected or null ladder point (an observed
  ladder point stays), but only when the plain race's `post_utc` equals the shadow race's: on a
  delayed post one feed may still carry the old post time, its checkpoints are for another
  schedule, and the ladder shows as it is. The race header (post time and minutes to post)
  shows the later of the two feeds' post times. `board-serve` rebuilds `/feed` when any of post - 20,
  10, 5 or 2 min passes, so each reference fills at its cutoff without a log write.
- Move: `current / reference - 1`, in percent rounded to one decimal, on `odds` (the low figure
  of a ranged pool such as PLACE or QUINELLA_PLACE). Null when either figure is missing.
- T-20 -> T-2 move (`move_t20_t2_pct`): `t2_odds / t20_odds - 1`, rounded the same way; null
  without both references. The shadow page no longer shows it (#499 replaced its T-20→T-2 column
  with the odds ladder); the field stays for the record.

## Scratches

A row whose current `status_raw` is `SCR` carries status `SCR`, null `odds` and `odds_hi`, and
null moves. Its references still show what the runner was priced at, for the record.

A scratch carries onto every combination containing the runner. When a runner's current WIN or
PLACE row is `SCR`, every row in every pool of that race whose combination contains that runner
number (whole numbers of the comma-separated pair: runner 1 matches `'1,10'`, not `'10,11'`) is
reported the same way: status `SCR`, null `odds`, `odds_hi` and moves, references kept. This
holds whether or not the source still lists the combination: after a scratch it stops listing the
runner's pairs, whose last row would otherwise stand as current. The carry stays within the race
and starts only once the `SCR` row is captured at or before `now()`. In the tier-2 pools it
reaches `FORECAST`, `TRIO` and `TIERCE` rows of the latest capture (status `SCR`, null `odds` and
`odds_hi`), never `BRACKET_QUINELLA`, whose combinations are bracket numbers, not runners.

## Tier-2 pools

Ticket #506 (the shadow page's exotic boards). The logger captures `BRACKET_QUINELLA` (枠連),
`FORECAST` (馬単, exacta), `TRIO` (3連複) and `TIERCE` (3連単, trifecta) at the decision grid's
points, each capture one gzip member of the same log (an 18-runner race: 36 + 306 + 816 + 4,896
prices). For a race on the board:

- `pools[POOL]` is `{investment, capture_utc, rows}`, once the pool has a capture at or before
  `now()`; a pool not yet captured is absent. `capture_utc` is that latest capture's
  `snapshot_utc`; `rows` are the rows sharing it, one per combination (at one instant the later
  row in the log wins), in the numeric order of tier 1 (`'12,3'` sorts as 12 then 3), each with
  `combination`, `odds`, `odds_hi` (null: single-figure pools) and `status` (its `status_raw`) and
  nothing else: no morning or checkpoint references, no moves. A capture holding part of a pool
  shows that part; the earlier capture's other rows are not merged in.
- `investment` is the latest `__pool__` figure of the pool at or before `now()`, as for tier 1.
- Combination texts are the odds log's: `FORECAST` and `TIERCE` in finishing order, `TRIO`
  ascending, `BRACKET_QUINELLA` the two bracket numbers ascending (`'3,3'` when one bracket holds
  both), so they join the shadow feed's `p_model`.
- Memory: the members are streamed and only the latest capture per (race, pool) is kept as they
  are read (an older one is dropped when a newer arrives), never a day's trifecta boards
  (`tests/countries/japan/test_tier2_log_readers.py` bounds the peak).
- The board page (`board.js`) draws the tier-1 pools only; the tier-2 pools are the shadow page's.

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
          pools: {WIN:            {"7":   {value_mean, value_q05, value_q95, flags, ladder}},
                  PLACE:          {"7":   {value_at_low, value_at_high, flags, ladder}},
                  QUINELLA_PLACE: {"3,7": {value_at_low, value_at_high, flags, ladder}}},
          p_model: {POOL: {combination: probability}} | null}],
 win5: {race_date, capture_utc, investment, carryover, pool_ev} | null}
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
- `win5` (ticket #135): on a WIN5 day, `research.win5.live_pool_ev` over the log's WIN5 `__pool__`
  rows (race_id `jra:<date>:WIN5`, docs/specs/odds-log.md) available as of `now()`: `capture_utc`
  is the latest such row's `snapshot_utc`, `investment` and `carryover` the latest figures any such
  row carries (yen), and `pool_ev` = `(investment * (1 - 0.30) + carryover) / investment`, the
  whole pool's gross return per 1 yen with the carryover (a carryover not printed counts as 0 and
  `carryover` stays null; null without an `investment`). Null on a day without WIN5 rows. WIN5 rows
  never become a race in `races`. The shadow page draws it as one line (`data-win5`) above the
  board; the plain feed and board page stay free of it.
- `p_model` (ticket #506): see "Model probabilities" below.
- No key names a stake, a bet, Kelly, an amount or a wager.

### Model probabilities

`p_model` (ticket #506): per race, `{pool: {combination: probability}}` of the board as of `now()`,
keyed by the odds log's combination texts, so it joins the plain feed's rows (tier 1 and tier 2).
Display only: it is not in `pools`, whose entries keep exactly their value fields, and it never
reaches a value, a flag, the shadow log or `verify()`.

- WIN: the calibrated Win model on the board's current WIN odds, `q = (1/O_WIN) / sum(1/O_WIN)`
  over the board's runners and `p = models.win.predict(q, board.context())` (not an average over
  nowcast draws).
- Every other pool: `shadow_values.model_probabilities(p, models.ordering, brackets=...,
  starters=...)`, exact sums over the v0 ordering model's ordered first three (discounted Harville,
  P(i, j, k) = p_i * p_j^l2 / sum_{l != i} p_l^l2 * p_k^l3 / sum_{l not in {i, j}} p_l^l3): PLACE
  `"7"` P(in the first `places`, from the jra config for the starters), QUINELLA `"a,b"`
  (a < b) P({first two} = {a, b}), QUINELLA_PLACE `"a,b"` P(both in the first three), FORECAST
  `"a,b"` P(a first, b second), TRIO `"a,b,c"` (ascending) P({first three} = {a, b, c}), TIERCE
  `"a,b,c"` P(a, b, c in that order), BRACKET_QUINELLA `"x,y"` (x <= y) P({bracket of the first,
  bracket of the second} = {x, y}). Totals: WIN, QUINELLA, FORECAST, TRIO, TIERCE and
  BRACKET_QUINELLA 1, PLACE the places paid, Wide 3. Computed per race (an 18-runner race has
  4,896 ordered triples), never per day at once.
- Scratches: a runner with a current SCR WIN or PLACE row is off the board, so it has no key and
  the others renormalise, as the shadow values do.
- Places: JRA fixes the places Place and Wide pay, and whether each is sold, at the start of sales
  (jra config `places_paid`), so a scratch after sales opened leaves them unchanged. The starters
  are read from the race's earliest WIN capture in the logs the feed reads (not the board at t,
  which keeps only the latest row per combination), plus any runner priced on the board (it was
  in the field when sales opened). The open the logger found (`DaySummary.sales_open_utc`, #358)
  is not in the logs, so coverage is read off that capture's time:
  - Covered: it is before logger.yaml's `day_before.probe_until` (Tokyo) on the day before the
    race date, so the logger was running when the day-before sales opened. The runners not SCR
    on it count: a runner priced there and scratched later counts; one already SCR there was
    withdrawn before sales opened and does not.
  - Not covered: it is at or after `probe_until` (a logger first run in the race date's 07:00
    window, or late on the eve), or logger.yaml has no `probe_until`. Every runner the log shows
    counts: those on that capture, SCR ones included, and the board's SCR runners, each scratch
    taken as after the opening.

  Before any WIN capture is read the starters are the board's priced runners. 8 drawn and one out
  after the opening, Place pays three places over the seven left; 8 drawn and one out before it,
  two; 5 drawn and one out after, Place is still sold with two; 5 drawn and one out before, Place
  is not sold; 4 drawn and one out after, Wide is still sold and Place is not. A late start: 8
  drawn, first logged at 07:00 on the race date with one SCR, 8 starters, so Place pays three
  places over the seven left. The ordered finishes are over the runners left. The shadow values
  (`shadow_values()`, the shadow log) keep the board's priced runners (#566).
- The bracket draw: N is the largest horse number on the board, scratched runners included (the
  draw was made before the scratch). N <= 8: no bracket quinella. 9 to 16: brackets 1..16-N take
  one horse each and the rest two, in number order; 17: bracket 8 takes three; 18: brackets 7 and
  8 take three (N // 8 horses per bracket, one more in the last N % 8). A bracket left with two
  runners or more has its `"x,x"` key; a bracket all of whose runners are out has no key.
- Without an ordering fit (or with an ordering model that has no discounts) only WIN; without
  models `p_model` is null; a failed computation is logged and shows null, tried again next build.
- Recompute: cached with the values (`cache`), recomputed only when the board (capture time, WIN
  figures, largest number, starters) or the model ids changed.

### The shadow page

`board/web/shadow.js` (#122, #506) draws one race at a time, with one section (`data-pool`) per
bet type whatever the plain feed holds, in this order: Win, Place, Quinella, Wide, Exacta
(`FORECAST`), Trio, Trifecta (`TIERCE`), Bracket quinella.

- Rows: Win, Place, Quinella, Wide and Bracket quinella list every row of the plain feed's pool,
  favourite first (#498). Exacta, Trio and Trifecta list the 15 combinations with the largest
  `p_model` (all of them when fewer exist), most likely first, each joined to the plain feed's
  latest-capture row for its odds (`-` without one); without a shadow feed they list none.
- Scratches first: the two feeds are polled apart, so /feed can show a scratch the shadow feed's
  `p_model` has not caught up with. Before the 15 are taken, Exacta, Trio and Trifecta leave out
  every combination with a runner whose current plain WIN or PLACE row is `SCR` (either is
  enough, with or without a tier-2 row for the combination) and every combination whose own
  tier-2 row is `SCR`; the next most likely eligible combinations move up. An eligible combination
  missing from the capture, or in a race with no tier-2 capture, still shows odds `-`.
- Model p: every row has a `Model p` cell (`data-col="p-model"`), `p_model[pool][combination]` as
  a percent with one decimal, `-` without one.
- Value: Win, Place and Wide keep the shadow values. Quinella, Bracket quinella, Exacta, Trio and
  Trifecta have no price forecast (no nowcast, #400): value = Model p x the plain row's current
  odds (gross return per 1 yen if the odds stood), two decimals, `-` without both, and the section
  says `no price forecast`.
- Capture age: each tier-2 section states its capture's age at now in whole minutes
  (`data-capture-age`), or that nothing is captured yet.
- Never coloured or highlighted by value or probability.

### The odds ladder

`ladder` (ticket #499): the entry's odds at the checkpoints 20, 10, 5 and 2 minutes before the
post and at the final, keyed `20`, `10`, `5`, `2`, `0` (text once JSON), each
`{"odds": x, "observed": true}`, `{"odds": x, "observed": false}` or null. Display only: it feeds
no value, flag, row order, shadow log record or `verify()`.

- Observed, k > 0: once `now` is at or after post - k, the latest row of the combination captured
  (`snapshot_utc`) at or before post - k among the rows available by `now`, the `t20_odds` rule of
  the plain feed; null when that row is unpriced or there is none, and always without a post time.
- Observed, the final (0): once the final board exists (`final_board` over the captures that show
  a final price, available by `now`), that board's figure. Until then the final is projected.
- Projected: a checkpoint still ahead, from O_now (the board's current figure) with the nowcast
  the values use (`models.nowcast`, the loader pointer's fit; read, never refitted):
  `ln O_k = ln O_now + mu(tau_now, b) - mu(k, b)`, `b = band_of(O_now, nowcast.bands)` for both
  terms. `mu(pool, tau, b)` is the fitted cell's mean, the expected `ln O_final - ln O_tau`, on the
  nowcast's grid; between neighbouring grid taus (T-20, T-10, an off-grid tau_now) it is linear in
  tau, below the smallest grid tau linear toward `mu(0) = 0`, and beyond the largest (T-60) it is
  that cell's. mu reads only the cells it needs (the exact grid cell, or the two bracketing tau);
  a checkpoint whose projection needs a cell the fit lacks (for tau_now or for k) is null, the
  others still project. `tau_now` is the minutes from the fetch of the row O_now comes from to
  the post, 0 after the post: the row's availability (`fetched_utc`, else `snapshot_utc`;
  `snapshot._available`), the row being the latest by availability (then log order) among those
  available by `now`. The native logger stamps a capture before it fetches the odds pages, so a
  row snapped at post - 12 and fetched at post - 11 projects from tau 11. With staggered captures
  each pool projects from its own row's fetch, not from the race's newest capture of another
  pool. The observed checkpoints keep the capture rule above (`snapshot_utc` at or before
  post - k, among the rows available by `now`). The figure shown is `exp(ln O_k)`, the median odds under the nowcast's
  log-normal drift (the mean would add `sigma^2 / 2`; the median is the forecast a reader can hold
  half above and half below), floored at 1.0 and rounded to two decimals. It is a forecast of the
  crowd's drift, not a value and not a signal.
- Pools: the low figure throughout (Place and Wide). Projections exist for the nowcast's pools,
  WIN, PLACE and QUINELLA_PLACE; a pool the nowcast has no cells for, a nowcast without cells
  (a fake), no models or no post time gives null for every checkpoint ahead, the observed ones
  staying.

Recompute: `cache` is a mapping the caller keeps between builds (the server keeps one). A race's
values are recomputed only when its board (`board_capture_utc` and its figures) or its bucket
changed; flags read the clock on every build. The same mapping holds the log readers, so a build
parses only the members appended since the last one; a clock that goes back, or a change of the
day's post times, reads the logs again. The ladder is formed on every build (it follows the clock)
from per-checkpoint references kept as the rows are read, so no capture history is held.

Serving: `/shadow-feed` has the headers, gzip, ETag and 304 of `/feed`. It is rebuilt at most
every `min_rebuild_s`, whether or not a log changed (buckets and the stale flag follow the clock),
and its body and ETag change only when something other than `generated_utc` and
`minutes_to_post` did, so the page's poll costs a 304 between captures; the page counts down to
the post from `post_utc`. `board-serve` loads the models once at start with `load_shadow_models`.
