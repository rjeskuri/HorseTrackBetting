# Pool depth and stake capacity from the final votes: N3 (ticket #432)

Status: **FROZEN on 2026-11-27**, before any run. Family id: **`japan-pool-depth-v1`**, spec
sha256 **`1545d788a9ac6997365900f48e1abc48177a24b9451d985f4d8b754b1857f34c`**. The spec is
committed at `src/horsetrack/countries/japan/config/pool_depth.yaml`; `horsetrack japan
pool-depth freeze` froze it, as loaded, with `Registry.freeze_family` in
`$HORSETRACK_DATA_DIR/registry/japan.jsonl` (append-only: a freeze cannot be undone). Under the
owner's reframe on #432 (2026-11-27), Claude pre-registers the spec and freezes it before the run,
with no owner sign-off. `horsetrack japan pool-depth run` and `research.pool_depth.run_depth`,
`run_coplacer` and `run_exacta_impact` refuse a spec that is not the one frozen for the family
(`NotPreRegistered`); any change after the freeze is a new family, frozen again before it runs.

## Question

How much money sits on a combination, by pool x class group x odds band, and what stake can we
put on it before our own money eats half of a 5, 10 or 20% edge? In particular: is betting big
possible in maiden and newcomer pools? Per leaderboard option's pool x class x odds band, the
capacity stake stands beside the flat stake (staking is flat, capped at 30,000 yen per race day,
owner 2026-11-27). Nothing here tests whether an edge exists: the capacity is a property of the
pool, given an edge. It also measures #350's question (the own impact of 100 yen on one exacta
combination, by pool size) on history, and gives #349's pool totals (final and two minutes before
the post).

## Data and days

- The jravan store under `processed_dir("jravan")`: `pool_votes` (each combination's final votes
  and each pool's `__pool__` total and refunds), `final_odds` (the final price, for the odds band),
  `races` (`race_class` for the class group, `status`), `dividends` (the paying Place and Wide
  combinations, co-placer variant only) and the time-series day files
  `odds_timeseries/<YYYY-MM-DD>.parquet` (the totals at t).
- Race days from 2014-06-07 (D10: no price row before it) to the last race day in the store.
  Development days only: every scan applies `core.eval.holdout.file_development_filter`, a
  time-series day file of a holdout day is skipped by its name before it is opened, and
  `LockedHoldout` is never opened.
- Races whose `status` is `abandoned` give no row: every ticket on them is refunded.
- Memory: the tables are streamed a batch at a time, a year at a time, with only the columns
  needed.

## Definitions

- **W**, the money on a combination: its final votes x 100 (yen). A combination counts when the
  final board prices it (non-null `odds`) and it has a votes row.
- **V**, the pool: the `__pool__` row's (votes - refund_votes) x 100, the money that pays. The
  store's `__pool__` votes include refunded votes and JRA pays (sales - refunds) x the payout rate.
  The run reports, privately, how the gross and net totals compare with `final_odds` investment.
- **Class group**: `research/pool_scan.CLASS_GROUPS` of `race_class` (newcomer, maiden,
  conditions, open, graded).
- **Odds band**: `pool_scan.odds_band` of the final odds on the pool's edges (the spec's
  `odds_bands`); Place and Wide band on the low end of the final range.
- **Capacity stake** (single-winner pools: WIN, QUINELLA, FORECAST, TRIO): the continuous stake s
  (no breakage, no floor) that solves (1 + e)(V + s)W / (V(W + s)) = 1 + e(1 - share_lost):
  s = (e - k)VW / ((1 + k)V - (1 + e)W), k = e(1 - share_lost). The ticket's form
  kVW / ((1 + k)V - (1 + e)W) is the same solution at share_lost 1/2 only, the value frozen here
  (`share_lost: 0.5`). Where the denominator is not positive the stake never eats that share of
  the edge, and the capacity is unbounded (`inf`).
- **Pool totals at t** (WIN, PLACE, QUINELLA, the pools of the 0B41/0B42 time series): the
  investment of the latest time-series `__pool__` capture with minutes_to_post >= 2.0; the final
  total beside it is the gross `__pool__` votes x 100, like the time series' investment.

## Cells and what is reported

- Edges e: 0.05, 0.10 and 0.20. Share lost: one half.
- Quantiles: 0.10, 0.25, 0.50, 0.75 and 0.90 (numpy's linear interpolation; an unbounded
  capacity sorts last).
- Cell set 1, pool x class group x odds band (WIN, PLACE, QUINELLA, QUINELLA_PLACE, FORECAST,
  TRIO): per cell, its combinations and races, the quantiles of W, of V (each race once) and of the
  capacity at each edge, and the share of its combinations whose capacity reaches each flat stake
  of 1,000, 10,000 and 30,000 yen. Newcomer and maiden rows are in every pool.
- Cell set 2, pool x pool-size tertile x odds band: the tertiles of V per pool over the run's
  races (low, mid, high).
- Pool totals at t per pool x class group: the races, those with a total at t, and the quantiles
  of the final total, the total at t and their ratio.

## Place and Wide: two registered variants

Place and Wide pay by the stake-back-share rule with co-placers, so the single-winner identity
does not hold for them.

1. **`depth-only`** (in the `depth` run): W and V quantiles, no capacity.
2. **`co-placer`** (its own run): for each paying combination of each race, given the realised
   co-placers (the race's paid Place or Wide combinations in `dividends`), JRA's rule pays per yen
   d = 1 + (rV - sum W) / (P W_i), the sum over the P paying combinations and r the pool's base
   payout rate (`config/jra.yaml`). Our stake s on combination i adds s to V, to W_i and to the
   sum; the capacity solves (1 + e) d(s) / d(0) = 1 + k, which with c = (1 + k) / (1 + e) and
   A = rV - sum W is s = (1 - c)(A + P W_i) / ((c d(0) - 1) P + 1 - r); at P = 1 it is the
   single-winner identity. A combination with A <= 0 (the rule pays no more than the stake back)
   has no capacity and is counted. The rule is checked against the paid dividends (broken to 10
   yen, at least 110); the run is recorded **failed** when it reproduces fewer than 95% of them
   (`min_rule_match: 0.95`). The private report also gives how often the rule of
   `core.pricing.stake_back_share_dividend`, r(1 + (V - sum W) / (P W_i)), reproduces them. Boosted
   takeout days are not adjusted.

## #350's measure: the 100-yen exacta own impact

For each priced FORECAST combination, |own_impact| of 100 yen (`research/exacta_impact.own_impact`,
takeout 0.25, its odds (1 - t)V / W so that it sees the combination's final money W), grouped by V
in `exacta_impact.POOL_BIN_EDGES`: per bin, the races, combinations, median, maximum and the
quantiles above. It replaces #283's measure from the logged odds, which had no logged exacta day.

## Records and reports

- Three registry runs per invocation, each one RunRecord in the family (hypothesis `P432`, ticket
  432, sources `('jravan',)`, fold scheme `development-days`), failures included: `depth`,
  `co-placer` and `exacta-own-impact`. The same run again (same store, same commit) adds nothing.
  Metrics: per cell the median capacity at e = 0.10 (Place and Wide in `depth`: the median W);
  the co-placer run adds the two rules' match shares; the exacta run, per bin, the median absolute
  impact with the maximum as its upper value.
- Private reports through `core.eval.reports.write_report` under `$HORSETRACK_DATA_DIR/reports/`:
  `pool-depth-japan-<hash12>.md`, `pool-depth-coplacer-japan-<hash12>.md` and
  `exacta-impact-history-japan-<hash12>.md`.
- Public: `docs/experiments/pool-depth-japan.md` (`render_public_report`), holding the identity and
  worked arithmetic on invented numbers; every figure from the store is withheld (D21). The
  committed registry summary is regenerated by `write_summary`.

## Not in the cells

TIERCE (too many combinations for exact quantiles in memory on this machine; a streaming exact
quantile is a follow-up) and BRACKET_QUINELLA (needs the horse-to-bracket map).

Note (2026-11-28, not part of the frozen spec): #490 measures both pools in their own family,
`japan-pool-depth-exotics-v1` ("Exotics" below).

## After the run (2026-11-27; not part of the frozen spec)

Every run is listed, with its status and registry id, in `docs/experiments/pool-depth-japan.md`;
the failures are kept.

- First invocation, at `f21e0aa`: `depth` and `exacta-own-impact` scored, `co-placer` **failed** by
  its rule check.
- Review round 1 corrected how the frozen spec is applied, not the spec: the scans filter holdout
  and pre-2014-06-07 rows inside the scan; `pool_votes` and `final_odds` keep confirmed records
  only (データ区分 4 and 5: a day-before close, 2, is no final money); the co-placer check counts
  every eligible paid combination, one without its votes or final price as not reproduced, and
  gates each pool apart (both must reach `min_rule_match`); the exacta run is failed when every
  pool-size bin is empty; a run's identity is a digest of what it read, and its private report is
  `<variant stem>-<first 16 hex of its run id>.md`, not the spec-hash name above.
- Second invocation, at `3ff70c5`, under the same frozen spec: `depth` and `exacta-own-impact`
  scored, `co-placer` **failed** again by its rule check, on the rule itself rather than on
  missing votes or prices: the frozen rule is not JRA's (see "Co-placer v2" below).
- `japan-pool-depth-coplacer-v2`, frozen before its run, at `42b46cc`: `co-placer` **scored** by
  its rule check in both pools (the shares: withheld (D21)).
- Review round 2 again corrected how both frozen specs are applied, not the specs: the open span
  ends on the store's last development day (never a holdout day) and the run records it; a run's
  identity digests the rows it read, day files included, never a file's bytes; the co-placer
  check takes sum W from the confirmed votes of the race's paying combinations, so a paid
  combination without its final price is one not reproduced rather than its whole race; the
  totals at t keep a race whose final record is missing; a `depth` run with no cell is recorded
  **failed**; the public report keeps every run it already lists.
- Third invocation, at `3de5687`, under both frozen specs: v1's `depth` and `exacta-own-impact`
  scored and its `co-placer` **failed** by its rule check (the frozen rule, as before); v2's
  `co-placer` **scored** in both pools (the shares: withheld (D21)). The Place and Wide capacity
  of this study is that v2 run's (registry run
  `aaeaaccd5dc4f9cfed503d5ba7d0167ddef5d9f87e3f401400941e5ac2f1d371`).

## Co-placer v2: the co-placer variant re-registered (2026-11-27)

Status: **FROZEN on 2026-11-27**, before its run. Family id: **`japan-pool-depth-coplacer-v2`**,
spec sha256 **`82a471b219bfee2e65f53c5c12e2f66514757169d82184b6be45b35b2454e89a`**, committed at
`src/horsetrack/countries/japan/config/pool_depth_coplacer_v2.yaml` and frozen with `horsetrack
japan pool-depth freeze --spec pool_depth_coplacer_v2`.

Why: `japan-pool-depth-v1`'s co-placer runs were recorded **failed** by their rule check, and not
for want of votes or prices. The frozen rule is applied as registered; it is the rule that does not
reproduce JRA's payouts. It returns the paying
combinations' own money untaxed, while JRA takes the payout rate on the whole net pool, those
stakes included. This registration was written after v1's private report, whose pre-registered
diagnostic gives how often the rule below reproduces the payouts (withheld, D21); it changes the
rule and nothing else.

- Rule (`core-stake-back-share`): the rule of `core.pricing.stake_back_share_dividend`, the
  stake-back-share rule as `config/jra.yaml` states it: per yen d = r(1 + B / (P W_i)), with
  B = V - sum W the losers' money, broken to 10 yen and at least 110; V net of refunds, P the
  race's paying combinations in `dividends`, r the pool's base payout rate (boosted days not
  adjusted, as in v1).
- Capacity: own stake s on combination i adds s to V and to sum W, so B stays and
  d(s) = r(1 + B / (P(W_i + s))). (1 + e) d(s) / d(0) = 1 + k with c = (1 + k) / (1 + e) gives
  s = (1 - c)(B + P W_i) / (P(c - 1) + c B / W_i), whatever r is; unbounded (`inf`) where the
  denominator is not positive, none where W_i <= 0. At P = 1 it is the single-winner capacity
  stake.
- Everything else is v1's: the days, cells, edges, share lost, quantiles, odds bands, flat
  stakes, and the gate (each pool's share of the eligible paid combinations the rule reproduces
  >= `min_rule_match` 0.95, a paid combination without its votes or final price counting as not
  reproduced; recorded **failed** below it).
- One run, the co-placer variant alone (`horsetrack japan pool-depth run --spec
  pool_depth_coplacer_v2`), one RunRecord in the family (hypothesis `P432`, ticket 432, sources
  `('jravan',)`), failures included; its private report is
  `pool-depth-coplacer-japan-<first 16 hex of its run id>.md` under `$HORSETRACK_DATA_DIR/reports/`.

## Exotics: TIERCE and BRACKET_QUINELLA (#490, 2026-11-28)

Status: **FROZEN on 2026-11-28**, before its run. Family id: **`japan-pool-depth-exotics-v1`**,
spec sha256 **`531658f78e2513daaa4a5fd61cee9924dec3eb529afcd343216ec71d56a0ec8c`**, committed at
`src/horsetrack/countries/japan/config/pool_depth_exotics.yaml` and frozen with `horsetrack japan
pool-depth freeze --spec pool_depth_exotics`.

Why: `japan-pool-depth-v1` left both pools out of its cells ("Not in the cells" above). This
family registers the `depth` variant for them alone and changes nothing in v1 or v2.

- Pools: TIERCE (the trifecta) and BRACKET_QUINELLA, both single-winner pools: every combination
  has the capacity stake of "Definitions" above, s = (e - k)VW / ((1 + k)V - (1 + e)W).
- Everything else is v1's: the days (2014-06-07 to the store's last development day, development
  days only, confirmed records only), W and V, class groups, the abandoned races left out, the
  edges 0.05, 0.10 and 0.20, share lost one half, the quantiles 0.10 to 0.90 (numpy's linear
  interpolation; an unbounded capacity sorts last) and the flat stakes 1,000, 10,000 and 30,000
  yen.
- Odds bands (chosen by analogy with v1's, before any read of these pools): TIERCE edges 30, 100,
  300, 1,000, 3,000, 10,000 and 30,000; BRACKET_QUINELLA edges 5, 10, 20, 50, 100 and 300 (v1's
  quinella edges).
- Cells: pool x class group x odds band, newcomer and maiden included: per cell its combinations
  and races, the quantiles of W, of V (each race once) and of the capacity at each edge, and the
  share of its combinations whose capacity reaches each flat stake. Not registered: the
  pool-size tertile cells and the totals at t.
- The quantile is exact, not a sketch: `research.pool_depth.StreamingQuantile` keeps each distinct
  value once with its count and reads numpy's linear quantile from the cumulative counts. The run
  reads the store a year at a time and keeps, per cell, the distinct (V, W) vote pairs of each
  month with their counts (a race's combinations share V, and W repeats within a race); W, V and
  every capacity quantile are computed from those pairs, so memory follows the distinct pairs, not
  the combinations. Peak resident memory must stay under 2.5 GB.
- The bracket map: per race, each starter's 枠番 from the jravan `runners` table (`bracket`,
  development days only); a runner marked `is_scratched` (取消 WV or 除外 WX) or without a bracket
  is in no bracket, and a race giving one horse two brackets fails the run. A bracket pair "a,b"
  (a <= b, as the store writes 組番) is sold when both brackets hold a starter, a same-bracket pair
  when its bracket holds two or more; a pair not sold gives no row even where the final board
  prices it. The private report counts the priced pairs, the sold ones, and the races priced
  without a bracket map.
- One run, the depth variant (`horsetrack japan pool-depth run --spec pool_depth_exotics`), one
  RunRecord in the family (hypothesis `P490`, ticket 490, sources `('jravan',)`, fold scheme
  `development-days`), failures included; recorded **failed** when no combination has a final
  price. Metrics: per cell the median capacity at e = 0.10. Its private report is
  `pool-depth-exotics-japan-<first 16 hex of its run id>.md` under `$HORSETRACK_DATA_DIR/reports/`;
  the public report (`docs/experiments/pool-depth-japan.md`) lists the run and withholds every
  figure (D21).

After the run (2026-11-28; not part of the frozen spec): frozen at `5af150b` before any run; the
first run, at `5af150b`, **scored** (registry run
`7363def84fd94c5482d56a1a404bb43ea6956ffdbba2461b5eff931ed22537ec`, private report
`pool-depth-exotics-japan-7363def84fd94c54.md`), with a peak resident set of about 1.65 GB
(`/usr/bin/time -l`), under the 2.5 GB bound. Every figure: withheld (D21).

## The depth table for the leaderboard (#491, 2026-11-28; not part of the frozen spec)

Each scored depth run, of `pool_depth` and of `pool_depth_exotics`, also writes its long cell
table (`depth_table`'s columns) as CSV beside its private report, replacing the spec's previous
one: `pool-depth-japan-table.csv` and `pool-depth-exotics-japan-table.csv` under the run's
out_dir (`$HORSETRACK_DATA_DIR/reports/` by default), through `write_report` with sources
`('jravan',)` (`write_depth_table`). A failed run leaves the previous table. The leaderboard reads
both (`read_depth_tables`) and shows each row's cell capacity beside it (`docs/specs/leaderboard.md`,
"Capacity"); every figure stays private (D21).
