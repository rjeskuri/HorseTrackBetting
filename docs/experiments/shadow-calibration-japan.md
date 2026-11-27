# Experiment report: C6 in shadow, calibration, the pick rule, E20 and the hand-played bets (Japan, #101)

Ticket #101 (epic #73; 08 B8 under the owner's reframe of 2026-11-27). Nothing is adopted and no
bet is placed: every figure is paper, and the calibration slope is reported, not a gate. Every
figure is an aggregate. The claims come from the v0 shadow logs; the outcomes from the `jra`
store's dividends (scraped results pages, with #454's scratch refunds); the final boards from
JRA's own odds pages, as logged. No JRDB or JRA-VAN data is used, and no runner or race row
appears here.

- Registry family `japan-shadow-calibration-v1`, spec sha256 `8b2239a4f6cc...` (frozen
  2026-11-28 at `66574dd`, before any scored read: `docs/specs/shadow-calibration-prereg.md`).
  Twelve records, one per variant, under git `66574dd`; their summary is
  `docs/experiments/registry/japan/japan-shadow-calibration-v1.md`.
- Code: `countries/japan/models/shadow.py` (the functions), `countries/japan/studies/shadow_run.py`
  (the run). Reproduce with `uv run python -m horsetrack.countries.japan.studies.shadow_run --run`
  (the family is frozen; `--freeze` again is a no-op). The full output is under the data
  directory at `reports/shadow-calibration-japan-8b2239a4f6cc.json`.
- Data: the shadow logs of 2026-11-21 and 2026-11-22 (Kyoto), both development days, sealed:
  10,810 records. Only 2026-11-21 carries claims: 560 WIN records over the grid, 116 settled at
  T-2 over 9 races, one meeting. 2026-11-22's records name no fits and carry no value. Every race
  of both days has a final WIN board (24) and stored dividends. No bet log exists on disk.

## Owner decisions used

D2 sets the grid (T-60, T-30, T-16, T-5, T-2, T-1) and T-2 as the decision time. D7 sets the
bootstrap: meeting clusters, 95% one-sided bounds, 10,000 replicates. D14/D15: holdout days are
never read, and the holdout is not opened. The owner's 2026-11-24 pick rule (the 5th-percentile
shadow value at T-2 above 1, no avoid flag) and 2026-11-27 reframe (selection by rank, flat
stakes, ¥30,000 per race day, no Kelly, no EV threshold, no gate) set the options and staking.

## Hypothesis

- P5, calibration: the realised net return per yen at the final dividend, regressed (OLS with an
  intercept) on v0's claimed net EV per yen (`value_mean - 1`), has slope 1. An interval excluding
  1 would show it wrong. The starting point is the null, slope 0 (the claim carries nothing beyond
  the takeout); with one meeting, no interval at all was the expected result.
- E20, selection drift: the fall ln O_t - ln O_final rises with the claimed EV of the record `lag`
  minutes earlier (a claimed overlay is shortened by the late money). Null: slope 0.

## Variants tried

All on the development days above; WIN only (v0 logs no point claim for Place or Wide). "Not
estimable" means D7's meeting bootstrap has one cluster, so no interval exists; the point slope
is still recorded.

| Variant | n | Races | Mean claimed EV | Mean realised | Slope | Interval (95% one-sided each) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `calibration-win-t2-meeting` (primary) | 116 | 9 | -0.289 | -0.778 | 0.975 | not estimable (one meeting) | failed |
| `calibration-win-t1-meeting` | 116 | 9 | -0.288 | -0.778 | 0.983 | not estimable | failed |
| `calibration-win-t5-meeting` | 116 | 9 | -0.281 | -0.778 | 0.980 | not estimable | failed |
| `calibration-win-t16-meeting` | 116 | 9 | -0.275 | -0.778 | 1.111 | not estimable | failed |
| `calibration-win-t30-meeting` | 38 | 3 | -0.259 | -0.768 | 1.416 | not estimable | failed |
| `calibration-win-t60-meeting` | 51 | 4 | -0.240 | -0.722 | 1.356 | not estimable | failed |
| `calibration-win-t2-race` (sensitivity) | 116 | 9 | -0.289 | -0.778 | 0.975 | [0.784, 1.204], race clusters | scored |
| `pick-rule-q05-t2` | 0 picks | | | | | no return: nothing picked | scored |
| `drift-win-lag3` (primary): T-2 from T-5 | 116 | 9 | | | 0.592 | not estimable | failed |
| `drift-win-lag11`: T-5 from T-16 | 116 | 9 | | | 0.935 | not estimable | failed |
| `drift-win-lag14`: T-2 from T-16 and T-16 from T-30 | 154 | 9 | | | 0.854 | not estimable | failed |
| `hand-played-bets` | 0 bets | | | | | no bet log on disk | scored |

- Mean realised -0.778 at T-2 is a ¥100 ticket on every runner of the 9 races returning about
  ¥22 per ¥100: the day's 9 winners, against the claimed -0.289. The levels differ by the
  intercept; the slope is the scale of the claim.
- The race-clustered interval treats the 9 races of one meeting as independent, which D7 does not
  allow; it understates the uncertainty and is reported beside the primary, never in its place.
- Lag 14 pools two pairs of decision times (T-2 from T-16, 116; T-16 from T-30, 38): the frozen
  spec names only the lags, and on the grid 2 + 14 = 16 (the spec's note).
- Log growth by day: no figure. 2026-11-21 picked nothing and 2026-11-22 has no claims, so neither
  day staked anything (bankroll for the figure: the ¥30,000 day cap).
- The hand-played bets: no bet log exists (`bets/japan` is absent), so there is no realised
  return by day, no option to rank and no day stake to set against the cap. No day is over it.

## Alternatives rejected and why

- Pooling the taus in one calibration: a runner's six records share one outcome; each tau is its
  own regression.
- Calibrating Place and Wide: v0 logs a bracket for them (`value_at_low`, `value_at_high`), not a
  point claim.
- Race clusters as the primary: D7 signs the meeting. Kept as a labelled sensitivity only.
- A notional bankroll for log growth: no owner figure grounds one; the day cap is the committed
  money figure.
- The literal reading "the race has no avoid flag" (every key of the race): nearly every race has
  a tail runner, so it selects almost nothing. On these claims the q05 condition alone selects
  nothing, so the reading does not change this run.
- Scoring E20 from the board at t: its noise is in the fall, so the slope would be mechanical
  (`test_selection_drift_scores_from_a_lagged_board`).
- Kelly, shrunk Kelly or a quantile gate (E46): out by the owner's reframe of 2026-11-27.
- Running #100's `ShadowDecider` on the logged days: its `meta[r].predict(q, view)` needs an
  adapter to C4's `MetaFit.predict(frame, q)` and C1/C5 refits, and the two days' logs are sealed.
  `log_decisions` is built and tested for its records; the run is a follow-up.

## What this rules out

Nothing, at D7's standard. One meeting carries claims, so no meeting-clustered interval exists
for any slope, and the pick rule selected nothing, so its return per ¥100 at the final price is
unmeasured. Measurements, with their limits:

- Calibration, point slopes 0.98 at T-2, T-1 and T-5 and 1.11 at T-16 (116 runners, 9 races),
  and 1.42 and 1.36 at T-30 and T-60 (3 and 4 races). The race-clustered interval at T-2,
  [0.784, 1.204], covers 1 and excludes 0, but understates the dependence within the meeting. The
  price half of each claim is in-sample: the nowcast fit the values used (`ec0a8794fe55`) was
  fitted on these two meetings' boards. So these slopes neither confirm nor rule out calibration
  out of sample; they show the claim's scale is not wildly off on this one day.
- E20, point slopes 0.59, 0.93 and 0.85: positive, as E20 expects, with no interval. Read
  descriptively (not pre-registered), the score at T-5 is strongly tied to the odds level at T-2
  (correlation -0.81 with ln O_t), so these slopes cannot separate an overlay effect from the
  drift of favourites and longshots, and the in-sample nowcast was fitted on this drift.
- The pick rule: no T-2 WIN record of the day has `value_q05` above 1 (the highest claims are
  below break-even), so the rule placed no paper ticket.

## Leaderboard rows

The family's twelve options, `japan-shadow-calibration-v1/<variant>`, enter the leaderboard
through `rows_from_registry`, all **unranked**: none records a development `return_per_100`.
`japan-shadow-calibration-v1/pick-rule-q05-t2` carries the leaderboard block (rule, WIN, decision
time, payout rate 0.80, bets) and is unmeasured until a day picks something. At the freeze and at
the run no row of the Japan leaderboard was measured (0 of 5,183), so K = 0 top-ranked options
were shadowed beside the pick rule.

## Follow-ups filed

Proposed for the orchestrator to file (`status:proposed`, epic #73):

- Rerun calibration, the pick rule and E20 as `japan-shadow-calibration-v2` on the development
  days logged with fits from 2026-11-28 on (the A2 holdout leaves about half the race days), once
  enough meetings exist for D7 intervals, with a nowcast fitted out of sample of those days.
- The C4 `MetaFit` adapter for `ShadowDecider` and a live development day logged through
  `log_decisions`, so the claim is #100's net EV with the flat stake inside the dividend.
- E20 with the score's odds level partialled out (the claimed EV's residual on ln O at the score's
  time), pre-registered, to separate an overlay effect from favourite drift.
- Leaderboard rows for the existing studies (`params['leaderboard']` blocks or #450 adapters):
  with no measured row, no option can be shadowed or played by rank.
