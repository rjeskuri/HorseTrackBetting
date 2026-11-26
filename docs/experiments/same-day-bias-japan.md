# Experiment report: E32, the same-day draw and position bias as of t (Japan, #348)

Ticket #348 (epic #73, the C0–C6 testing scaffold). Does the day's draw and running-style bias, as known at the
decision time, add anything to the final Win price? **Nothing is adopted (D16).** Every figure is an aggregate over
the JRA results store (netkeiba race results; no JRDB, no JRA-VAN). No runner or race row appears.

- Registry family `japan-same-day-bias-final-win`, spec sha256
  `eb830398172df36eb7f5e291f7c69ee1d511a8b8b3a198e78846f4ee3657e027`, frozen in the registry before any store row was
  read. Five records (all scored, none failed): `2ef70c7878a6e57e`, `4a5072d23449b547`, `5074477737eea13d`,
  `9160b58976de2d50`, `336b66704cae6a12`, git `631affe`. Summary:
  `docs/experiments/registry/japan/japan-same-day-bias-final-win.md`.
- Code: `countries/japan/studies/same_day_bias.py`. Reproduce with
  `uv run python -m horsetrack.countries.japan.studies.same_day_bias --run --to 2026-11-26`.
- Store: `processed/jra/runners.parquet@6f3cec95347e0132`, `processed/jra/races.parquet@9cce511831074b64`, and #129's
  running style (`features/jra/handicap/<day>.parquet`). Development days 2014-06-07 to 2026-11-23. Scored races are
  in the test windows of #70's folds 1 to 27 (2020-04-01 to 2026-11-23): 22,959 races in 1,914 meetings. Fold 0
  (2020-Q1) only trains.
- Holdout: development days only. #114's scoped reader drops holdout days, the range ends before 2026-11-28, and
  `LockedHoldout.open` is not called.
- Full report (per-fold tables): `$HORSETRACK_DATA_DIR/reports/same-day-bias-japan-eb830398172d.md`. The run took
  5 minutes with a peak resident set of 1.0 GB.

## Hypothesis

E32 holds that the same-day bias at t adds to the final Win price. The feature follows the owner's answer on #97
(2026-11-26, item 4), "one shrunk bias column from #138's within_day_posterior". For each runner it is its draw
third's posterior mean (`draw_adj`) plus the posterior mean of its own #129 running style (`style_adj_<style>`, 0
without one). Both come from `track_bias.track_bias_features` at t = post − 2 min (D2's primary). They read only the
same day's earlier races at the venue and surface whose results were published strictly before t. Each fold's prior is
fitted on the days up to its train_end, with day_sd the pooled predictive estimate (#258, the module default).

The model is a conditional logit u = ln q + coef · bias, with q the final Win price normalised per race. coef = 0 is
the market exactly. **Null:** coef = 0 and walk-forward ΔLL = 0, meaning the market already prices the same-day bias.
A ΔLL whose D7 one-sided 95% lower bound stays at or below 0 does not reject the null.

## Variants tried

All five variants are registry records, and none failed. The coefficient CI is two-sided 95%, from coef ± 1.96 times
the meeting-clustered sandwich se. The ΔLL figure is in nats per race of winner log loss (D6), market minus model, so
a positive value means the model is better. Its bounds are D7's one-sided 95% bounds from a meeting bootstrap with
10,000 replicates and seed 348.

| Variant | Rows | Estimate | 95% interval |
| --- | --- | --- | --- |
| posterior:pooled_coefficient (E32) | the scored test-fold rows, 22,959 races | coef **+0.410** (se cluster 0.114) | +0.187 to +0.634 |
| history_prior:pooled_coefficient | the same rows, nothing published | coef +0.386 (se cluster 0.115) | +0.160 to +0.612 |
| posterior:walk_forward_delta_ll (E32) | fold k scored with the coef from folds < k | ΔLL **+0.000197** | lower −0.000026, upper +0.000418 |
| history_prior:walk_forward_delta_ll | the same | ΔLL +0.000159 | lower −0.000049, upper +0.000367 |
| same_day_update:walk_forward_delta_ll | posterior against history prior, paired | ΔLL +0.000038 | lower +0.000012, upper +0.000064 |

The per-fold coefficients from the training rows run from +0.05 (fold 1, 876 training races, CI −0.87 to +0.97) to
between +0.33 and +0.43 from 2023 on. Every fold from 2023-Q1 has a CI that excludes 0. The per-fold ΔLL swings in
sign from quarter to quarter, between about −0.0016 and +0.0017.

## Reading

- The coefficient CI excludes 0. Within the scored rows, the bias column has a positive weight beyond ln q, and the
  sign matches #138's construction (runners favoured by the day's draw and style do a little better than their price
  says). The column itself is small, though. It is a sum of posterior means on #138's share-of-rivals-beaten scale,
  where #258's day sds are 0.01 to 0.04, so coef 0.41 moves a runner's log-odds by a few hundredths at most.
- Out of fold, this does not become a measurable gain over the final price. The ΔLL is +0.0002 nats per race, and its
  one-sided lower bound is below 0. **E32 does not reject the null at the final price.**
- Most of the coefficient is the static history prior, not the same day. The prior alone gets coef 0.386 and ΔLL
  +0.00016. The same-day update adds +0.000038 over it, with a lower bound above 0.

## Reconciliation with #138

#138, #224 and #258 measured the within-day update against the prior alone, in a ridge conditional logit on the
winner with no market in the model (#129's `fit_handicap_score`). #258's gain was +0.000443 per level and +0.000463
pooled (ledger rows for #224 and #258). Here the same posterior is measured at the final price, so ln q is the offset
and the comparison is the same pair of columns. The same-day update's gain over the prior is +0.000038, positive, with
a lower bound of +0.000012. That is about a tenth of the gain without the market. The same-day bias is real but small,
and the final price already carries most of what it holds. The scored races (22,959 in 1,914 meetings) are #224's
22,887 plus the days the store has gained since.

## Alternatives rejected and why

- **#97's `same_day_bias` column.** It is not on main, and the owner's answer names #138's `within_day_posterior`.
- **Separate draw and style coefficients, or a column per level.** The owner's answer is one shrunk column.
- **A ridge penalty on coef.** One coefficient on 22,959 races needs no shrinkage, and at ridge 0 the interval is the
  plain meeting sandwich.
- **The calibrated market (D4) as the baseline.** The ticket's outline fixes ln q as the offset and ΔLL against q.
  The D4 baseline is listed as a follow-up.
- **The pooled coefficient on all rows, including fold 0.** It is fitted on the scored test-fold rows instead, where
  every row's bias is built from a prior that predates its fold.

## What this rules out

- At the final Win price, a gain from the same-day bias column larger than about +0.00042 nats per race (the
  one-sided upper bound) is ruled out on these days and folds. So is any gain from the same-day update over the
  history prior larger than about +0.000064.
- It does not rule out a small positive gain (the point estimate is +0.0002), a gain at earlier board prices (T-2
  odds rather than the final price), or a gain once publication times are logged rather than proxied at post + 30 min.

## Follow-ups filed

None filed by this run. Proposed for the orchestrator (`status:proposed` under #73): re-run E32 against the D4
calibrated-market baseline and at the T-2 board price once logged odds cover enough development days, and re-run when
`result_published_utc` replaces the post + 30 min proxy (#260).
