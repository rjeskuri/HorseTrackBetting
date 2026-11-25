# The board at a decision time

Code: `src/horsetrack/countries/japan/board/snapshot.py`. Input: rows of the odds snapshot log
(docs/specs/odds-log.md), as `odds_log.read_rows` returns them.

## Availability rule

- A row is available at its `fetched_utc`; when that key is absent or null (HKTrack logs), at its
  `snapshot_utc`. A row with neither readable is never used.
- `board_at(rows, race_id, t)` uses only rows of that race available at or before t. For each
  (pool, combination) it takes the latest such row; at the same instant the later row in the log
  wins. Nothing available after t is used, in any pool or in the totals.
- Pools read: WIN (`win[runner]`), PLACE (`place[runner] = (odds, odds_hi)`) and QUINELLA_PLACE
  (`wide[(a, b)] = (odds, odds_hi)` with a < b, read from `'a,b'` in either order, so `'2,1'` and
  `'1,2'` are one pair). `odds_hi` is None when the row has none. Other pools are not read.
- A runner whose latest WIN or PLACE row has `status_raw` `SCR` is in `scratched` and has no price
  in any pool, Wide pairs included. Any row with null odds is unpriced.
- `totals[pool]` is the `investment` of the latest `__pool__` row of that pool that carries one.
- `capture_utc` is the oldest availability time among the price rows used: the rows that reach
  `win`, `place` or `wide`, plus the `SCR` rows that put a runner in `scratched`. Totals do not
  count, and neither does a row dropped for naming a scratched runner (a Wide pair the logger
  stopped refreshing after a scratch, or a stale priced PLACE row), so staleness is neither
  understated nor pinned to rows the board does not show. When no row is used, it is the oldest
  among all the latest price rows.
- No price row (WIN, PLACE or Wide) of the race available by t gives None, even when pool totals
  are.
- `starters` is the number of runners with a WIN price; `context()` gives venue and race number
  from the race id `jra:<date>:<VENUE>:<NN>`.

## Final-state rule

- A row is final when `source_update_utc` is null and `最終オッズ` appears in `page_state` (rows
  from #21) or in `status_raw` (HKTrack rows).
- A capture is the set of rows of the race sharing one `snapshot_utc`. Its WIN rows are final when
  it has at least one priced WIN row and every priced WIN row is final; `SCR` rows do not count
  either way, and a capture with no priced WIN row is never final.
- `final_board` takes each pool's final capture (owner sign-off on #234, 2026-11-25), because
  HKTrack-format logs split a round into captures (WIN+PLACE, QUINELLA ~1 s on, Wide ~2 s on) and
  a pool can turn final a round after WIN. Log order does not matter.
  1. The anchor is the first capture by `snapshot_utc` whose WIN rows are final; none gives None.
  2. A capture is final for a pool when it holds at least one priced row of that pool and every
     priced row of that pool is final (`SCR` rows do not count): the WIN test applied per pool.
  3. The board starts from the anchor's rows, so WIN and the other pools' totals (QUINELLA's
     `__pool__`) are the anchor's. PLACE and QUINELLA_PLACE, each with its prices, `SCR` rows and
     `__pool__` total, are replaced by that pool's rows from its first final capture at or after
     the anchor, with no round bound; when there is none, from its latest final capture before the
     anchor; when the pool has no final capture at all, it is empty and has no total. A pool final
     in the anchor comes from the anchor.
  4. The rules above apply unchanged (scratched runners from WIN and PLACE rows; Wide pairs naming
     a scratched runner dropped). `at_utc` is the latest availability time among the rows taken,
     `capture_utc` the oldest among its price rows, and `final` is True.

  A one-capture round whose pools are all final (horsetrack's own logs) gives that capture's board
  alone, as before #234.
- `board_at(...).final` is True when the WIN rows it uses meet the same rule.

## Post times

`post_times_for(race_date, venue)` returns race number to aware UTC post time:

1. the `<VENUE>.races.json` sidecar beside the odds log (`odds_log.read_post_times`), when it
   holds any time;
2. else HKTrack's stored card,
   `$HORSETRACK_DATA_DIR/processed/jra/upcoming/jra_<date>_<VENUE>/races.parquet`, columns
   `race_no` and `post_time_local` (ISO with `+09:00`; a time without an offset is read as Japan
   time), read column-scoped;
3. else empty.

## Decision grid and buckets

- `config/decision.yaml` ships D2's grid (docs/problem-statement.md): `grid_min: [60, 30, 16, 5,
  2, 1]`, `primary_min: 2`. `load_decision_grid(path)` reads it, or another file.
- `bucket_for(minutes_to_post, grid)`: the smallest grid horizon at or above the minutes to post;
  the largest horizon for anything earlier; 0 at or after the post. Exactly 0 minutes to post is
  the post itself, when sales close, so it gives 0, not the smallest horizon.
