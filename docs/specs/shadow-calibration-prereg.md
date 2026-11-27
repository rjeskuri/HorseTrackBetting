# C6 in shadow: calibration, the pick rule, E20 and the hand-played bets, pre-registration (ticket #101)

Status: **PROPOSED**, to be frozen with `Registry.freeze_family` before any scored read.

Family: `japan-shadow-calibration-v1`. The spec is
`horsetrack.countries.japan.studies.shadow_run.build_spec()`, plain JSON;
`uv run python -m horsetrack.countries.japan.studies.shadow_run --freeze` freezes it in
`$HORSETRACK_DATA_DIR/registry/japan.jsonl` (append-only: a freeze cannot be undone), and `--run`
reads no shadow record, dividend or board unless the family is frozen with exactly that spec. A
change to the spec after the freeze is a new family version, frozen again before it runs. Under
the owner's reframe of 2026-11-27 nothing here is a gate: the calibration slope is reported, and
every option is a ranked row on the leaderboard (#447). Code: `countries/japan/models/shadow.py`
(the functions), `countries/japan/studies/shadow_run.py` (the run).

Read before the freeze, and only this: the claims, never an outcome. On 2026-11-21 (Kyoto, the
one meeting) 560 WIN records carry a value, 117 of them at T-2 over 9 races; on 2026-11-22 no
record carries one (its records name no fits: `models` is null). No T-2 WIN record has
`value_q05` above 1, so the pick rule is expected to select nothing. No dividend, final board or
result was joined to a claim before the freeze.

## Hypotheses

- **P5, calibration (the joint check, problem statement section 3; 08 S1).** Regress the realised
  net return per yen at the final dividend on the claimed net EV per yen, OLS with an intercept:
  a calibrated claim has slope 1. Shown wrong by an interval that excludes 1. The starting point
  is the null: the claim carries nothing beyond the takeout (slope 0). With one meeting and about
  a hundred bets at a decision time, an interval covering both 0 and 1, or none at all, is the
  expected result.
- **E20, selection drift (08 E20).** The fall in log odds from t to the final,
  ln `board_odds`(t) - ln `final_odds`, regressed on the claimed EV of the record of the same
  race, pool and combination `lag` minutes earlier (t + lag), never on the board at t, whose noise
  the fall contains. E20 expects a positive slope (a claimed overlay is shortened by the late
  money); the null is 0.

## Data

- Days: 2026-11-21 and 2026-11-22, every logged shadow day before the holdout start
  (2026-11-28); both are development days by the committed rule. `read_shadow` reads development
  days only; the holdout is never opened.
- Pool: WIN. v0 logs a point claim (`value_mean`, `value_q05`, `value_q95`: the gross return per
  yen at the nowcast's final price) for WIN alone; PLACE and Wide records carry a bracket
  (`value_at_low`, `value_at_high`; owner, 2026-11-24), which has no point claim to calibrate.
- Claimed EV: `value_mean - 1`. Realised: the dividend / 100 - 1 for a winner, -1 for a loser, at
  the jra store's final dividends with the scratch refunds of `bets.settle.day_dividends`.
  Refunded, unsettled, `no_board` and holdout records are left out of every regression.
- Final odds (E20): the WIN figure of each race's final board, `board.snapshot.final_board` on the
  raw odds log.
- In-sample caveat, stated now: the nowcast fit the logged values used (`ec0a8794fe55`) was
  fitted on the boards of these same two meetings, so the price half of each claim is in-sample.
  Both the calibration and E20 read in-sample claims; the win calibration (`aac501f927b0`, to
  2026-11-15) and the ordering fit are out of sample.

## Variants (12 registry records per run)

| Variant | E id | What |
| --- | --- | --- |
| `calibration-win-t2-meeting` | P5 | primary: WIN at T-2 (D2's decision time), meeting clusters (D7) |
| `calibration-win-t{1,5,16,30,60}-meeting` | P5 | the other grid points, one regression each (a runner's records at several taus share one outcome, so taus are never pooled) |
| `calibration-win-t2-race` | P5 | sensitivity: T-2 with race clusters, which understate the dependence within a meeting; reported beside the primary, never in its place |
| `pick-rule-q05-t2` | P5 | the 2026-11-24 pick rule as one leaderboard option (below) |
| `drift-win-lag3` | E20 | primary: the fall from T-2, scored from T-5 |
| `drift-win-lag11`, `drift-win-lag14` | E20 | the fall from T-5 scored from T-16; from T-16 scored from T-30 |
| `hand-played-bets` | P5 | the owner's bets by day and option, beside each option's development rank |

A variant whose interval cannot be estimated (fewer than two meetings) is recorded as failed,
with its point slope when one exists; a variant with no input is recorded with its counts.

## Options shadowed

- **The leaderboard's top options: K = 0.** At the freeze no row of the Japan leaderboard
  (`rows_from_registry` on the japan registry) has a measured return, so no option has a
  development rank to shadow by. A ranked option enters through a new family version that names
  it; every other option keeps its row.
- **The pick rule** (owner, 2026-11-24), option `japan-shadow-calibration-v1/pick-rule-q05-t2`:
  a WIN record at T-2 whose `value_q05` is above 1 and whose own `flags` are empty (the avoid
  flags are set per key; "the race has no avoid flag" read as every key of the race would exclude
  almost every race, since nearly all have a tail runner). Picks take the flat stake
  (`config/betting.yaml`: ¥1,000) under the race-day cap (¥30,000, every venue together) in order
  of decision time, then `value_q05` (highest first), through `models.shadow.flat_stakes`. Its
  leaderboard row: return per ¥100 at the final dividend with a meeting bootstrap
  (`core.eval.bootstrap.paired_bootstrap`), `n` in bets; unmeasured when nothing is picked.
- **Log growth by day** of the pick rule's paper stakes, `models.shadow.log_growth_by_day`, the
  day starting with the day cap (¥30,000) as its money: the one committed money figure, so
  ln(ending / starting) reads as the share of the day's allowance that came back. A notional
  bankroll was considered and not used: no owner figure grounds one.

## The hand-played bets

Every bet log on disk (`$HORSETRACK_DATA_DIR/bets/japan/*.jsonl`, read through
`models.shadow.read_bets`), settled at the final dividends by #454's `settle_bets`, by day and
option beside the option's rank on the development leaderboard (`models.shadow.played_by_day`).
Each day's staked total, every bet of the day counted, is shown against the ¥30,000 cap and a day
over it is flagged: #453 records an over-cap bet as a fact, so the cap is reported here, not
enforced. A holdout day's bets stay 'holdout'.

## Bootstrap and outputs

- D7 from `config/scoring.yaml`: meeting clusters, 95% one-sided bounds (each of `lo`, `hi` is a
  one-sided quantile), n_boot 10,000; seed 101.
- One registry record per variant per run, failures included. The run's full output (every
  variant, the counts, the pick rule's growth, the bets by day) is one JSON report through
  `core.eval.reports.write_report` under `$HORSETRACK_DATA_DIR/reports/`; it is built on JRA's
  public pages (source `jra`), with no JRDB or JRA-VAN input, so its aggregate figures may appear
  in the public experiment report (D21 does not bar them); per-race rows stay out of the tree.
- Experiment report: `docs/experiments/shadow-calibration-japan.md`.
