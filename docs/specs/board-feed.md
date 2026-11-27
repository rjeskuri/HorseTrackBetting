# The live-board feed

Code: `src/horsetrack/countries/japan/board/feed.py`. Input: a race day's odds snapshot logs
(docs/specs/odds-log.md), streamed member by member with `odds_log.iter_members`, and the post
times of `snapshot.post_times_for`. The feed never requests anything over the network.

The feed carries tier-1 pools only: `WIN`, `PLACE`, `QUINELLA`, `QUINELLA_PLACE` (`BOARD_POOLS`).
Tier-2 (`BRACKET_QUINELLA`, `FORECAST`, `TRIO`, `TIERCE`) and `WIN5` rows in the same log are
dropped row by row as each member is read, before anything below: they never reach a pool, a race
with only such rows is not on the board, and they do not move `last_capture_utc`. The board over a
log is exactly the board over its tier-1 rows, and a day's trifecta boards are never held in memory.

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
                  QUINELLA_PLACE: {"3,7": {value_at_low, value_at_high, flags}}}}],
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

## Access

Code: `src/horsetrack/countries/japan/board/server.py` (the gate, `write_access_token`) and
`src/horsetrack/countries/japan/board/tunnel.py` (`quick_tunnel`), ticket #505.

- Without `--tunnel`, `board-serve` is as above: bound to 127.0.0.1, no token, no cookie.
- Tunnel: `board-serve --tunnel` serves on 127.0.0.1 only (another `--host` is refused) and starts
  `cloudflared tunnel --url http://127.0.0.1:<port> --no-autoupdate` as a child, cloudflared found
  on PATH (`brew install cloudflared`). It reads the `https://<name>.trycloudflare.com` URL from
  cloudflared's output (never `api.` or `login.trycloudflare.com`), then waits for the first
  `Registered tunnel connection` line (the banner comes before the tunnel can serve), and drains
  the rest unread. cloudflared missing, exiting before the URL is handed over (after a registered
  connection included), or not printing both within 30 s stops board-serve with an error naming
  it. The child is terminated, and killed 5 s later if it still runs, whenever the server stops
  (Ctrl-C, SIGTERM, an error), so no tunnel outlives it; if cloudflared dies first, a warning says
  so.
- Token: a quick-tunnel URL is public, so every path needs an access token, a fresh
  `secrets.token_urlsafe(24)` per run, held in memory until the server is bound and the tunnel is
  up, then written to `$HORSETRACK_DATA_DIR/board/access-token` (mode 0600, never in the
  repository, a symlinked `board/` included). A start that fails (the port taken, cloudflared
  missing or exiting before its URL, a connection or the handoff) exits non-zero and leaves that
  file as it was, so a tunnel still serving keeps its token there. It is printed once, in the
  phone URL line `open on the phone: https://<name>.trycloudflare.com/shadow?t=<token>`; the
  request log blanks query strings and the token.
- Cookie: a request with the right `?t=` is served and gets `Set-Cookie: board_access=<token>;
  Path=/; HttpOnly; SameSite=Strict`, with `Secure` when it came over https (`X-Forwarded-Proto:
  https`), so the page's relative fetches (`feed`, `shadow-feed`, `board.js`, `shadow.js`) carry
  it. Either is compared with `hmac.compare_digest` on bytes.
- 404: a missing or wrong token gets a 404 with an empty body, the same response as an unknown
  path, and before any 503. The check holds for requests from 127.0.0.1 too, since the tunnel
  connects from there.
- Headers: every response, in both modes, carries Cache-Control no-store, X-Robots-Tag noindex
  nofollow, Referrer-Policy no-referrer, X-Content-Type-Options nosniff and X-Frame-Options DENY.
  The pages carry no data beyond the feeds above.

Race day (#124): `horsetrack japan board-serve --date 2026-11-28 --from-vm htb-vm --tunnel`, then
open the printed URL on the phone; `/shadow` without the token is a 404.
