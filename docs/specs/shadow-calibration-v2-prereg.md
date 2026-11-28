# C6 in shadow, v2: calibration on more meetings, an out-of-sample nowcast and E20 per pair, pre-registration (ticket #617)

Status: **FROZEN** on 2026-11-28 at commit `f7a41c7`, before any scored read (it was PROPOSED
until then). Spec sha256 `c00ae1caa6179359e927727117c9786badbb726f6fc5551ef921f7e14a2c6056`.

Family: `japan-shadow-calibration-v2`, the rerun of #101's `japan-shadow-calibration-v1`
([shadow-calibration-prereg.md](shadow-calibration-prereg.md)). The spec is
`horsetrack.countries.japan.studies.shadow_run.build_spec()`, plain JSON;
`uv run python -m horsetrack.countries.japan.studies.shadow_run --freeze` freezes it in
`$HORSETRACK_DATA_DIR/registry/japan.jsonl` (append-only: a freeze cannot be undone), and `--run`
reads no shadow record, dividend, board or bet unless v2 is frozen with exactly that spec (v1's
freeze does not unlock it). A change after the freeze is a new family version, frozen again
before it runs. Under the owner's reframe of 2026-11-27 nothing here is a gate: every figure is
reported, and every option is a ranked row on the leaderboard (#447). Code:
`countries/japan/models/shadow.py` (the functions), `countries/japan/studies/shadow_run.py` (the
run).

What v2 changes from v1, and why:

- **New meetings.** v1 scored one meeting (Kyoto, 2026-11-21), so D7's meeting interval of the
  primary T-2 slope was not estimable. v2 names the development days logged with values from
  2026-11-28 on.
- **An out-of-sample nowcast.** v1's claims used nowcast fit `ec0a8794fe55`, fitted on the boards
  of its own two meetings. v2's claims name fit `61413e8a77a6`, fitted before any v2 day.
- **E20 per (tau, lag) pair.** v1 named lags only, and lag 14 pooled T-2 from T-16 with T-16 from
  T-30. v2 names each pair and scores it alone.
- **E20 with the odds level partialled out.** v1's score correlated -0.81 with ln O_t (read
  descriptively, after its run), so its drift slopes could not separate an overlay effect from
  the drift of favourites and longshots. v2 scores each pair plain and with ln O_t as a covariate.

Read before the freeze, and only this: the claims, never an outcome. The 2026-11-28 shadow log
(sealed, development) holds 13,429 records over two meetings (Tokyo, Kyoto) and 24 races; 1,512
WIN records carry a value: 337 at each of T-1, T-2, T-5 and T-16, 72 at T-30 (5 races) and 92 at
T-60 (7 races). At T-2, 168 are Tokyo's and 169 Kyoto's. Every record with a value names nowcast
fit `61413e8a77a6` (36 `no_board` records name none). No T-2 WIN record has `value_q05` above 1,
so the pick rule is expected to select nothing. The nowcast's manifest (metadata only) was read
for its training days. No dividend, final board or result was joined to a claim before the
freeze. The leaderboard was read for its row counts only (8,392 rows, 294 measured).

## Hypotheses

- **P5, calibration (the joint check, problem statement section 3; 08 S1).** Regress the realised
  net return per yen at the final dividend on the claimed net EV per yen, OLS with an intercept:
  a calibrated claim has slope 1. Shown wrong by an interval that excludes 1. The starting point
  is the null: the claim carries nothing beyond the takeout (slope 0).
- **E20, selection drift (08 E20).** For one (tau, lag) pair at a time, the fall in log odds from
  t = tau to the final, ln `board_odds`(t) - ln `final_odds`, regressed on the claimed EV of the
  record of the same race, pool and combination at t + lag, never on the board at t, whose noise
  the fall contains. Plain: OLS on [1, score]. Partialled: OLS on [1, score, ln O_t], the score's
  coefficient reported, where ln O_t is ln `board_odds` of the record at t (the record whose fall
  is regressed). E20 expects a positive slope (a claimed overlay is shortened by the late money);
  the null is 0. A plain slope that vanishes once ln O_t is in reads as favourite-longshot drift,
  not an overlay effect.

## Expected result, stated now

Two meetings. D7's bootstrap resamples whole meetings, so each replicate draws two meetings with
replacement: half the replicates are one meeting twice. With two clusters the 95% one-sided
bounds sit near the slopes of the single meetings, so the interval of every slope is very wide
and is expected to cover both 0 and 1: estimable, but not informative. That is still what v1
could not give, and the family is rerun as later development days are logged (a v3 that names
them). The pick rule is expected to select nothing (no T-2 `value_q05` above 1), so its return
is unmeasured and its log growth is the 'no picks' row.

## Data

- Days: 2026-11-28, the only development day logged with values from 2026-11-28 on at the
  freeze. 2026-11-29 is a holdout day by the committed rule (A2, `holdout.yaml`) and is never
  named; `read_shadow` reads development days only, and the holdout is never opened. v1's days
  (2026-11-21, 2026-11-22) are not v2's.
- Pool: WIN. v0 logs a point claim (`value_mean`, `value_q05`, `value_q95`: the gross return per
  yen at the nowcast's final price) for WIN alone; PLACE and Wide records carry a bracket.
- Claimed EV: `value_mean - 1`. Realised: the dividend / 100 - 1 for a winner, -1 for a loser, at
  the jra store's final dividends with the scratch refunds of `bets.settle.day_dividends`.
  Refunded, unsettled, `no_board` and holdout records are left out of every regression.
- Final odds (E20): the WIN figure of each race's final board, `board.snapshot.final_board` on the
  raw odds log.
- The nowcast: fit `61413e8a77a6` (manifest `models/japan/v0/manifest-nowcast-61413e8a77a6.json`
  under the data dir, created 2026-11-27T12:55Z), its WIN and PLACE cells fitted on
  `processed/jravan/odds_timeseries`, 1,353 development days from 2014-06-07 to 2026-11-23:
  `fitted_through` 2026-11-23, before 2026-11-28, so every v2 claim's price half is out of sample.
  Its Wide cells come from the earlier fit `ec0a8794fe55` on the logged boards of 2026-11-21 and
  2026-11-22, which v2's WIN claims do not use. The win calibration (`aac501f927b0`, to
  2026-11-15) and the ordering fit (`2b822d643cf7`, created 2026-11-24) are out of sample too.
- Claims naming another nowcast fit: scored like the rest, and counted by fit id in the report
  (`counts.claims_by_nowcast`). Every 2026-11-28 claim names `61413e8a77a6`, so this choice does
  not move v2's figures; leaving such claims out was considered and not used, since the count by
  id states the same fact without a second definition of the sample.
- Sources: `jra` and `jravan`. The claims rest on a nowcast fitted on JRA-VAN data, a licensed
  source, so v2's figures are licensed-derived (D21): the registry records and the report name
  `jravan`, the report is written through `write_report` under the data dir, and the public
  experiment report gives counts and non-licensed facts only.

## Variants (22 registry records per run)

| Variant | E id | What |
| --- | --- | --- |
| `calibration-win-t2-meeting` | P5 | primary: WIN at T-2 (D2's decision time), meeting clusters (D7) |
| `calibration-win-t{1,5,16,30,60}-meeting` | P5 | the other grid points, one regression each (a runner's records at several taus share one outcome, so taus are never pooled) |
| `calibration-win-t2-race` | P5 | sensitivity: T-2 with race clusters, which understate the dependence within a meeting; beside the primary, never in its place |
| `pick-rule-q05-t2` | P5 | the 2026-11-24 pick rule as one leaderboard option (below) |
| `drift-win-t1-lag1`, `drift-win-t2-lag3` (primary), `drift-win-t5-lag11`, `drift-win-t16-lag14`, `drift-win-t30-lag30` | E20 | each adjacent step of the grid: the fall from T-tau scored from T-(tau + lag) |
| `drift-win-t2-lag14` | E20 | v1's other half of lag 14: the fall from T-2 scored from T-16 |
| each `drift-...` again with `-partial` | E20 | the same pair with ln O_t as a covariate (`partial_log_odds`: true) |
| `hand-played-bets` | P5 | the owner's bets by day and option, beside each option's development rank |

Expected pair sizes from the claims: 337 rows over 24 races for (1, 1), (2, 3), (5, 11) and
(2, 14); 72 rows over 5 races for (16, 14); 28 rows over 2 races for (30, 30), before the join to
the final boards. A variant whose interval cannot be estimated (fewer than two meetings, or a
replicate with no spread) is recorded as failed with its point slope when one exists and its
`n_clusters[meeting]`; a variant with no input is recorded with its counts.

## Options shadowed

- **The leaderboard's top options: K = 0.** v2 reruns v1's design on new meetings and scores the
  v0 shadow claims only; settling another family's ranked option needs that family's rule. No
  leaderboard option is named; one enters through a family version that names it, and the
  hand-played bets carry each played option's rank.
- **The pick rule** (owner, 2026-11-24), option `japan-shadow-calibration-v2/pick-rule-q05-t2`:
  a WIN record at T-2 whose `value_q05` is above 1 and whose own `flags` are empty, never a
  holdout record. Picks take the flat stake (`config/betting.yaml`: ¥1,000) under the race-day
  cap (¥30,000, every venue together) in order of decision time, then `value_q05` (highest
  first), through `models.shadow.flat_stakes`. Its leaderboard row: return per ¥100 at the final
  dividend with a meeting bootstrap (`core.eval.bootstrap.paired_bootstrap`), `n` in bets, and
  `n_clusters[meeting]`; on one meeting the return has no interval and the variant says 'not
  estimable' (never a zero-width interval); unmeasured when nothing is picked.
- **Log growth by day** of the pick rule's paper stakes, `models.shadow.log_growth_by_day`, the
  day starting with the day cap (¥30,000), one row per study day. A day that ends at ¥0 keeps its
  row ('settled', its stakes and ¥0), its log growth -inf written as null with that reason, since
  strict JSON has no infinity.

## The hand-played bets

Every bet log on disk (`$HORSETRACK_DATA_DIR/bets/japan/*.jsonl`, `models.shadow.read_bets`),
settled at the final dividends by #454's `settle_bets`, by day and option beside the option's
rank on the development leaderboard (`models.shadow.played_by_day`). Each day's staked total is
shown against the ¥30,000 cap and a day above it is flagged (`over_cap`), never refused. A
holdout day's bets stay 'holdout', and its dividends are never read (`day_dividends` gives a
holdout day none).

## Bootstrap and outputs

- D7 from `config/scoring.yaml`: meeting clusters, 95% one-sided bounds (each of `lo`, `hi` is a
  one-sided quantile), n_boot 10,000; seed 617.
- One registry record per variant per run, failures included, each naming the sources `jra` and
  `jravan`. The run's full output (every variant, the counts, the pick rule's growth, the bets by
  day) is one JSON report and a settlement file through `core.eval.reports.write_report` under
  `$HORSETRACK_DATA_DIR/reports/`, named as v1's are (spec hash, commit, artifact id).
- Experiment report: `docs/experiments/shadow-calibration-japan.md`, a v2 section: counts and
  non-licensed facts only, every licensed-derived figure withheld (D21).
