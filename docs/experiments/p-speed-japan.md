# Experiment report: E39, #97's p_speed standalone and marginal at the final price (Japan, #347)

Ticket #347 (epic #73, the C0–C6 testing scaffold; #97's definition-of-done item 2). Does C3's per-horse finish-speed
forecast, turned into a win probability, add anything to the final Win price? **Nothing is adopted (D16).** Every
figure is an aggregate over the JRA results store (netkeiba race results; no JRDB, no JRA-VAN, so nothing here is
withheld under D21). No runner or race row appears.

- Registry family `japan-c3-p-speed-final-win`, spec sha256
  `d084b95959c3d999711b566db2db22fb06e8c230b3c796a86a5ded9bf9ad22ba`, frozen in the registry before the run read any
  store row. The timing and smoke runs below came first and read no score. Sixteen records (all scored, none failed),
  git `8d09b62`; the run ids are in the summary,
  `docs/experiments/registry/japan/japan-c3-p-speed-final-win.md`.
- Code: `countries/japan/studies/p_speed.py`. Reproduce with
  `uv run python -m horsetrack.countries.japan.studies.p_speed --run`.
- Store: `processed/jra/runners.parquet@6f3cec95347e0132`, `processed/jra/races.parquet@9cce511831074b64`. Runs and
  races from 2010-01-05 train the speed models (no price is read before 2014-06-07, D10); every fold trains on every
  earlier run (788,150 at the last fold).
- Windows, pre-registered in the spec (`predict_from`) on runtime alone. Before the freeze, one fold's fits were
  timed (no score read): on 788,150 training runs the normal `HorseSpeed` fit took 77 s and the Student-t fit 900 s
  (18 reweighting rounds).
  - The normal points are predicted on #70 folds 16 to 44, from 2019-Q4 to 2026-11-23; 2019-Q4 only trains the scale.
    The standalone comparison is scored on D4's whole window, 2020-Q1 to 2026-11-23: 22,968 races in 1,991 meetings.
    The marginal is scored from 2020-Q2: 22,088 races in 1,914 meetings. Jump races are not in C3 (the owner on #96),
    and 50 races whose cell had no par in their fold's fit were not predicted.
  - The Student-t points are predicted on 2026-Q2 to Q4 only. Q2 trains the scale; the standalone is scored on Q3 and
    Q4, 1,368 races in 118 meetings; the marginal on Q4, 508 races in 44 meetings. The full-window rerun is #651.
- Before the freeze, two smoke runs (2026-Q3 to Q4 and 2026-Q2 to Q4, one normal point) checked the code path on
  the store and printed each variant's status only. No figure was read.
- Holdout: development days only. The scoped readers drop holdout days, the store ends 2026-11-23, and
  `LockedHoldout.open` is not called.
- Full report (per-fold tables, fitted terms): `$HORSETRACK_DATA_DIR/reports/p-speed-japan-d084b95959c3.md`. The
  run took 2 h 58 min with a peak resident set of 1.6 GB, on a machine shared with other sessions (load average 10
  to 68).

## Hypothesis

E39 (08 §4) holds that p_speed, the win probability from #97's `HorseSpeed` expected finish times, adds to the final
Win price. 08 §4 expects at most +0.0005 pseudo-R², plausibly 0. **Null:** p_speed's coefficient alpha is 0 in #87's
stage-2 meta u = alpha · ln p_speed + beta · ln q (q the final Win price normalised per race; alpha = 0 is the market
alone and the shrinkage target), and the out-of-fold ΔLL of that meta against the market-only meta u = beta · ln q is
at most 0. A ΔLL whose D7 one-sided 95% lower bound is at or below 0 does not reject the null. The standalone
comparison (p_speed alone against D4's calibrated market) is a measurement, not a claim.

How p_speed is built, per #70 fold (D9: quarterly, expanding, 1-week embargo):

1. #96's `SpeedModel` (par per track × surface × distance × layout × rail × class cell, day variant per (day, track,
   surface), Student-t error with 5 df, #305's primary) is fitted on the fold's training races and observes its test
   races. A fit day's in-sample variant is used only once that day is fully resulted before t (the owner on #97, Q3
   (b); inside `HorseSpeed`).
2. #97's `HorseSpeed(drift_var_per_day, t_df)` is fitted on the training races' runs (`models.speed_runs.build_runs`:
   every non-scratched runner of a kept flat race from 2010, #129's style and early position as of the run's own
   decision time) and observes the test races' runs in one batch (the owner on #97: per fold, test days in batches).
   The global terms stay the training runs'.
3. Each test race is predicted at the off (the ticket's outline), reading every run and race-level result published
   strictly before it (results at post + 30 min, availability.yaml). A race whose cell has no par in the fold's fit is
   not predicted.
4. p_speed per race: Plackett-Luce, softmax(−mu / s) with s = 1 / the conditional-logit coefficient of the winner on
   −mu; or Henery, P(fastest) of independent N(mu, s²) times with s its own one-parameter maximum likelihood. Fold k's
   s is fitted on the earlier folds' predicted rows dated on or before fold k's train_end, so the first predicted fold
   only trains.

Scored rows are the priced starters (`studies.meta.store_frame`) of races predicted whole, where D4's calibrated
market exists out of fold (the market's own #70 folds over the price era, from 2020-Q1).

## Variants tried

All sixteen variants are registry records, and none failed: the pre-registered grid drift_var_per_day {0, 0.004}
(s² per day) × t_df {normal, 5}, each with PL and Henery, each scored standalone and marginal. ΔLL is in nats per race
of winner log loss (D6), baseline minus model, so positive means p_speed (or its meta) is better. Its bounds are D7's
one-sided 95% bounds from a meeting bootstrap with 10,000 replicates and seed 347. α's interval is two-sided 95%, α ±
1.96 times the meeting-clustered sandwich se.

**Standalone: p_speed alone against D4's calibrated market (out of fold)**

| Point | Method | Races (meetings) | ΔLL | Lower, upper | Pseudo-R² vs equal chances (D4: 0.258) |
| --- | --- | --- | --- | --- | --- |
| drift 0, normal | PL | 22,968 (1,991) | −0.4671 | −0.4764, −0.4578 | 0.079 |
| drift 0, normal | Henery | 22,968 (1,991) | −0.4727 | −0.4822, −0.4632 | 0.077 |
| drift 0.004, normal | PL | 22,968 (1,991) | −0.4360 | −0.4452, −0.4269 | 0.091 |
| drift 0.004, normal | Henery | 22,968 (1,991) | −0.4426 | −0.4521, −0.4333 | 0.088 |
| drift 0, t 5 df | PL | 1,368 (118) | −0.4740 | −0.5138, −0.4336 | 0.074 |
| drift 0, t 5 df | Henery | 1,368 (118) | −0.4758 | −0.5156, −0.4354 | 0.073 |
| drift 0.004, t 5 df | PL | 1,368 (118) | −0.4449 | −0.4840, −0.4059 | 0.085 |
| drift 0.004, t 5 df | Henery | 1,368 (118) | −0.4475 | −0.4863, −0.4086 | 0.084 |

D4's pseudo-R² on the Student-t points' races is 0.259. The fitted PL scale per fold runs from 0.63 to 0.67 s (drift 0)
and 0.76 to 0.83 s (drift 0.004), and Henery's from 1.15 to 1.26 s and 1.41 to 1.54 s. Every fold's ΔLL is negative
(−0.54 to −0.36).

**Marginal: p_speed in #87's stage-2 meta next to ln q**

| Point | Method | α (pooled) | α 95% CI | β | ΔLL vs market only | Lower, upper | Races (meetings) | Δ pseudo-R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drift 0, normal | PL | −0.0099 | −0.0348 to +0.0150 | 1.032 | −0.000044 | −0.000110, +0.000022 | 22,088 (1,914) | −0.000017 |
| drift 0, normal | Henery | −0.0058 | −0.0122 to +0.0005 | 1.031 | −0.000044 | −0.000126, +0.000039 | 22,088 (1,914) | −0.000017 |
| drift 0.004, normal | PL | −0.0306 | −0.0532 to −0.0080 | 1.040 | +0.000029 | −0.000150, +0.000206 | 22,088 (1,914) | +0.000011 |
| drift 0.004, normal | Henery | −0.0085 | −0.0136 to −0.0035 | 1.032 | −0.000184 | −0.000617, +0.000133 | 22,088 (1,914) | −0.000071 |
| drift 0, t 5 df | PL | +0.0468 | −0.0616 to +0.1553 | 1.029 | +0.000384 | −0.001330, +0.002070 | 508 (44) | +0.000148 |
| drift 0, t 5 df | Henery | +0.0419 | −0.0667 to +0.1505 | 1.030 | +0.000120 | −0.001894, +0.002098 | 508 (44) | +0.000046 |
| drift 0.004, t 5 df | PL | +0.0252 | −0.0693 to +0.1197 | 1.033 | +0.000060 | −0.000071, +0.000189 | 508 (44) | +0.000023 |
| drift 0.004, t 5 df | Henery | +0.0148 | −0.0813 to +0.1109 | 1.036 | +0.000029 | −0.000084, +0.000138 | 508 (44) | +0.000011 |

α is pooled over the standalone's rows (22,968 races for the normal points, 1,368 for the Student-t points). Henery's
α is on another scale from PL's, since its ln p_speed spreads wider. No fold's meta fit was skipped. Per fold, the
normal points' ΔLL against the market-only meta runs from −0.0044 to +0.0012, and it is positive in 10 to 17 of the
27 folds.

## Reading

- **E39 does not reject the null at the final price.** No variant's ΔLL against the market-only meta has a lower
  bound above 0. Over D4's window the normal points' ΔLL is between −0.00018 and +0.00003 nats per race, and their
  Δ pseudo-R² between −0.00007 and +0.00001: 08 §4's "plausibly 0", below its +0.0005 ceiling.
- Next to ln q, p_speed gets almost no weight. Where α's interval excludes 0 (drift 0.004 under the normal error,
  both methods) it is negative: given the final price, a runner p_speed rates higher wins slightly less often than
  the price alone says. This is a measured sign, not a claim. It buys nothing out of fold (+0.00003 and −0.00018).
  One reading is that the crowd already prices the horses' time form and leans on it a little too much.
- Alone, p_speed is far weaker than the market: 0.44 to 0.47 nats per race behind D4, with pseudo-R² 0.07 to 0.09
  against D4's 0.26. Drift 0.004 beats a static ability by about 0.03 nats per race, and PL beats Henery by about
  0.006. The race-level gain #305 measured (0.233 s of time RMSE) does not carry into a useful win forecast without
  the market.
- The Student-t points (508 marginal races, 44 meetings) are too few to tell anything: their ΔLL bounds reach
  ±0.002. Their standalone figures match the normal points' within the bounds.

## Alternatives rejected and why

- **Coupling the race-level speed model's error to the grid's t_df.** The grid is over `HorseSpeed`'s two parameters;
  `SpeedModel` keeps #305's primary Student-t (5 df) at every point, so a grid point changes only the horse layer and
  one race-level fit per fold serves all four points.
- **T-2 (D2's primary) as the decision time.** The ticket fixes the off. With results proxied at post + 30 min, T-2
  and the off differ only by results published in the last two minutes before the off.
- **The offset form (u = ln q + alpha · ln p_speed) for the marginal.** With ln q's coefficient fixed at 1, a member
  that only echoes the price earns weight (the test-author's synthetic market shows it); the stage-2 form with beta
  free is #92's C4 marginal, so alpha measures what p_speed adds beyond a refitted power of q.
- **A ridge on alpha.** Two coefficients on tens of thousands of races need no shrinkage; ridge 0 keeps the plain
  meeting sandwich interval, and alpha = 0 is the null itself.
- **Henery's scale from PL's (s × pi / sqrt 6).** That is a Gumbel error's SD, not the normal's ML scale; it is used
  only as the starting bracket of Henery's own one-parameter fit.
- **Race-by-race observation of the test runs.** Quadratic in the fold's races (#97's owner-closed finding); one
  batch per fold gives the same predictions by `HorseSpeed`'s batch and incremental agreement.

## What this rules out

The ticket's metric is log loss, not a return per ¥100. No variant here places a ticket, so no return is measured
(D16: none is ranked on return).

- At the final Win price over D4's window under the normal error, a gain from p_speed larger than +0.00021 nats per
  race is ruled out at D7's one-sided 95% level. That is the largest upper bound of the four normal variants, about
  +0.00008 in pseudo-R². With a static ability (drift 0) the bounds are +0.00002 (PL) and +0.00004 (Henery).
- p_speed as a replacement for the market is ruled out by a wide margin: its standalone upper bound is −0.41 nats
  per race.
- Not ruled out:
  - a gain from the Student-t points over the whole window (508 races here; bounds up to +0.0021, #651);
  - a gain at drift values outside the grid;
  - a gain at earlier board prices (T-2 odds), where the market knows less;
  - a gain in segments such as debuts or thin histories, where the price may lean on less form;
  - a gain jointly with C2's members (#98).

## Leaderboard rows

The registry family `japan-c3-p-speed-final-win` enters the leaderboard through #451's `legacy_rows` (every family
and variant of the Japan registry), so no wiring was added. Its sixteen option ids are
`japan-c3-p-speed-final-win/<method>:drift=<0|0.004>:t_df=<none|5>:<standalone|marginal>`. None has a return per
¥100, so all are listed unranked (D16). None failed. Their kind is the leaderboard's default, `other`: a family missing
from `leaderboard_backfill.LEGACY_KINDS` gets that kind, and the leaderboard's tests pin it so. Naming it a `member`,
like E32's family, needs those tests changed first (a proposed follow-up).

## Follow-ups filed

- #651 (proposed): speed up `HorseSpeed`'s Student-t fit and rerun the Student-t points over D4's window.
- Proposed, not filed here: give `japan-c3-p-speed-final-win` the leaderboard kind `member` (with its pinning test);
  rerun E39 at the T-2 board price once logged odds cover enough development days.
- Existing: #98 melds C3 with C2's members at C4, the gain beyond C2.
