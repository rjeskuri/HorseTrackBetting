# Experiment report: the handicap score with #96's speed figure, re-measured (Japan, #219)

Ticket #219 (epic #73, the C0–C6 testing scaffold), the follow-up to #129. #129 turned each runner's past
performances into handicap factors and a one-line out-of-fold `handicap_score`. Its speed family used an interim
figure: the start's final time against the median winning time of its (venue, surface, distance, class) on the
in-fold days. #96's expected top-3 time has landed (`models/speed.py`, scored by #305). The speed family now reads it,
and this report re-measures #129's two comparisons beside #129's numbers.

**Nothing is adopted (D16).** Every figure is an aggregate over the JRA results store (netkeiba race results; no JRDB,
no JRA-VAN) and the calibrated-market Win baseline v0 built on it. No runner or race row appears.

- Registry family `japan-handicap-speed-96-final-win`, spec sha256
  `16b385befd691c9e957de14b56f465ed6cfda04bcaffcec42b199b23ebabcad8`, frozen in the registry before the table was built
  or any outcome was read. Six records, all scored, none failed, git `dd7d41b`. Summary:
  `docs/experiments/registry/japan/japan-handicap-speed-96-final-win.md`.
  - `v96:V1:handicap_alone` `f4e9d9b9…`, `v96:V2:meld_over_market_v0` `9cddf365…`
  - `interim_par:V1:handicap_alone` `e114e682…`, `interim_par:V2:meld_over_market_v0` `2635aee6…`
  - `swap:handicap_alone` `5f173c65…`, `swap:meld_over_market_v0` `9120fcb7…`
- Code: `countries/japan/handicap/factors.py` (`_speed_figures`, `build_handicap_table(..., out_dir=)`) and
  `countries/japan/studies/handicap_speed.py`. Reproduce with
  `uv run python -m horsetrack.countries.japan.studies.handicap_speed --run [--rebuild]`.
- Store: `processed/jra/races.parquet@9cce511831074b64`, `processed/jra/runners.parquet@6f3cec95347e0132`. Baseline:
  v0's out-of-fold `win_calibration_oof-aac501f927b0` (#114, #123), the one #129 used.
- Full report: `$HORSETRACK_DATA_DIR/reports/handicap-speed-japan-16b385befd69.md`. The rebuilt table is
  `features/jra/handicap-96/` (1350 race days); #129's table stays in `features/jra/handicap/`, untouched.
- Cost: the whole run (table build, out-of-fold scoring of both tables, six variants) took 1,298 s at a peak resident
  set of 1.10 GB.

## Hypothesis

Written into the frozen spec before anything was scored.

- **Null:** swapping the speed par for #96's expected time does not improve the handicap score. The gain
  ll(interim-par score) − ll(#96 score) is at most 0 on the common races.
- **H1 (V1):** the #96 score alone still beats equal chances, with a pseudo-R² within 0.02 of #129's 0.165. A negative
  gain over equal chances would be a bug.
- **H2 (V2):** the meld over v0 stays small, at most about 0.002 nats per race (#129: 0.0005), and its one-sided lower
  bound may not exclude 0. A gain above 0.01 would point to leakage before an edge.
- **H3:** the swap is worth little: under 0.01 nats per race either way.
- **Limit I knew of in advance:** in the per-day table every start is in the fold (`train_end` is the cut), so its
  figure is the par less the finish time with the day variant at its prior (0). The day variant enters only for a
  start after `train_end`, which the table never has. This re-measure therefore tests #96's six-key, Student-t par
  and its race table, not the same-day variant.

## Setup

Reproduced from #129's Setup, unchanged except where noted.

- **Data:** development days only, 2014-06-07 to 2026-11-15, the JRA store (races, runners). Every day is before the
  holdout (2026-11-28, D14). Every read goes through #114's scoped reader and `LockedHoldout` is never opened.
- **Figure:** `SpeedModel` (#96) with `SpeedSpec(PAR_KEYS, VARIANT_KEYS, same_day_variant=True, t_df=5.0)`, #305's
  primary error model, fitted on `speed_run.build_race_table(estimand="mean_placed_1_3")` over the races to each race
  day's cut. The figure of a start is the expected top-3 time of its race at post − 2 min (D2) minus the runner's
  finish time. A start whose race is not in the table, whose cell has no par, or with no finish time has no figure.
- **Build:** `build_handicap_table(2014-06-07, 2026-11-15)` wrote 1350 race days, 600,231 runner rows (as #129's did).
  Factors are as of each race day.
- **Score:** the default ridge conditional logit (alpha 1, 46 `SCORE_FEATURES`), out of fold over Japan's #70 scheme:
  327,080 scored rows and 23,799 races, 2020-01-05 to 2026-11-15, as before.
- **Scoring:** D6 through `core.eval.scoring` (#69's module, which was not on main when #129 was measured): winner log loss per race, a dead
  heat's k co-winners at 1/k, pseudo-R² against equal chances (ln n), gain = ll_baseline − ll_model in nats per race
  over the common races.
- **CIs:** `core.eval.bootstrap`, a paired bootstrap resampling meetings (venue × day), 95% one-sided (D7), and the
  same draws' 2.5% and 97.5% quantiles as the two-sided interval. Difference from #129: `scoring.yaml` now signs
  10,000 replicates (#129 used 2,000 in a stand-alone script), seed 219.
- **Same code on #129's table:** the `interim_par` rows rescore #129's kept table with this study's code, so the
  comparison does not depend on a change of method. They match #129's reported numbers to the digits it printed.

## Variants tried

Races / meetings: 23,799 / 1,985 for V1; 23,760 for the comparisons with v0; 22,850 / 1,908 for V2 (fold 0 only trains
the meld). Bounds are the one-sided 95% lower bound and the two-sided 95% interval.

| Variant | #96 figure (this ticket) | Interim par, same code | #129 as reported |
| --- | --- | --- | --- |
| V1: score alone vs equal chances, pseudo-R² | 0.1648 | 0.1650 | 0.165 |
| V1: gain, nats/race | +0.4276; lower bound +0.4182; CI [+0.4163, +0.4390] | +0.4281; +0.4187; [+0.4167, +0.4396] | +0.428; +0.419; [+0.417, +0.439] |
| Market v0 alone vs equal chances, same races | 0.2576 | 0.2576 | 0.258 |
| Score alone vs market v0 (ll_v0 − ll_score) | −0.2405; lower bound −0.2471; CI [−0.2484, −0.2324] | −0.2400; −0.2467; [−0.2480, −0.2318] | −0.240; −0.247; [−0.248, −0.232] |
| V2: meld exp(a ln p_cal + b ln h) over v0, gain | +0.000519; lower bound +0.000144; CI [+0.000068, +0.000953] | +0.000529; +0.000153; [+0.000078, +0.000965] | +0.00051; +0.00015; [+0.00009, +0.00095] |

V2 coefficients (last fold, 2026-Q4): a = 1.0532, b = −0.0616 (interim par 1.0538, −0.0623; #129 1.053, −0.062).
Across the 27 folds b runs from −0.0787 to +0.0104 (interim par −0.0787 to +0.0089; #129 −0.078 to +0.015).

V2 gain by calendar year (nats per race, about 3,450 races a year, 2,537 in 2020):

| Year | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| #96 figure | −0.00004 | +0.00105 | +0.00102 | −0.00025 | +0.00024 | +0.00042 | +0.00110 |
| #129 | −0.00011 | +0.00111 | +0.00094 | −0.00026 | +0.00023 | +0.00047 | +0.00112 |

The swap itself, paired on the same races (gain = ll(interim) − ll(#96), so positive means #96 is better):

| Variant | Races / meetings | Gain, nats/race | One-sided 95% lower bound | Two-sided 95% CI |
| --- | --- | --- | --- | --- |
| Score alone | 23,799 / 1,985 | −0.000493 | −0.00165 | [−0.00187, +0.00091] |
| Melded over v0 | 22,850 / 1,908 | −0.000011 | −0.000072 | [−0.000084, +0.000062] |

The figures themselves differ. On the 327,080 scored runner rows the new `speed_best`, `speed_mean3` and `speed_last`
correlate 0.98, 0.94 and 0.91 with the interim ones; the share with no figure is 10.6% against 10.2%, and the count of
figures (`speed_n`) is the same on 97.0% of rows. So the swap changes the inputs and not the score.

## Reading

- **H1 held, with a null swap.** The #96 score's pseudo-R² is 0.1648, 0.0002 under the interim score's, against the
  market's 0.2576 on the same races. It beats equal chances by at least 0.418 nats per race (one-sided 95%) and trails
  the market by at least 0.240.
- **H2 held.** The meld gains 0.0005 nats per race over v0 (lower bound +0.00014), far below the 0.01 that would point
  to leakage. The weight on the handicap score stays **negative** in the latest folds (b = −0.062), as in #129, and
  two of seven years are negative, as in #129. The pattern is unchanged: the gain over the market comes from fading
  the form score.
- **H3 held; the null is not rejected.** The swap moves the score alone by −0.0005 nats per race (two-sided CI
  [−0.0019, +0.0009]) and the meld by −0.00001 (CI [−0.00008, +0.00006]). Neither interval excludes 0, so #96's figure
  neither improves nor worsens the handicap score in a way these 23,799 races can detect. The meld's interval bounds
  the effect on V2 to about 0.00008 nats per race either way.
- **Why so little moves (a reading, not measured).** The speed figure is one family among several in a 46-feature
  score, and the two figures correlate 0.91 to 0.98. In the per-day table #96's day variant is
  at its prior for every start (see Hypothesis), so what was swapped is the par and its race table: six keys (layout
  and rail included; #305 found rail `unknown` on every race in its store copy) and a Student-t fit in place of the class
  median.

## Alternatives rejected and why

- **Fold-bounded or fitted-variant figures for in-fold days.** `SpeedModel` exposes no fitted day variant for the
  days it was fitted on, and `observe` refuses them, so the figure for a past in-fold start cannot use its own day's
  going without changing #96's model. Not run here.
- **Re-tuning the score (alpha, features) after the swap.** Not run: tuning on these folds would turn the CIs into
  post-selection numbers, and feature inclusion is the owner's decision (D16).
- **Scoring only the new table.** Rejected: without the same code on #129's table, a difference could come from the
  scoring change (`core.eval.scoring`, 10,000 replicates) rather than the figure.
- **Overwriting #129's table.** Not done: the new table is `features/jra/handicap-96/`, and `track_bias_report` and
  `same_day_bias` keep reading #129's.

## What this rules out

- At the one-sided 95% level the swap does not make the handicap score worse than the interim one by more than 0.0019
  nats per race, and it does not improve it by more than 0.0009 (two-sided CI of the paired gain on the score alone).
  Melded over v0 the bounds are about ±0.00008.
- The #129 result stands: the handicap score alone beats equal chances by at least 0.418 nats per race and trails the
  market by at least 0.240; the meld over v0 gains between 0.00007 and 0.00095 nats per race (two-sided 95%) with a
  negative weight on form. A large public-information edge from simple handicapping stays ruled out on 2020-2026
  development data.
- It does not rule out a gain from #96's **same-day variant** inside the score. That variant was at its prior for every
  start here (see Hypothesis), so this run does not test it.
- It is a single run on development data, with one alpha, one feature set and one meld form. It is not a holdout
  claim.

## Leaderboard rows

None. This study measures winner log loss, not a return per ¥100 at the final price, so it adds no ranked row
(D16): the six variants are registry records, listed unranked.

## Follow-ups filed

None filed by this run. Proposed for the orchestrator (`status:proposed` under #73):

- Give #96's `SpeedModel` a read of the fitted day variants of its in-fold days (or fit the day variant leave-one-day-out
  for the handicap table), then re-measure V1 and V2 with the same-day variant in the figure of past starts. This
  is the part of #219's request ("with its day variant") that the per-day table cannot test today.
- #218 (the negative weight on form within the market, by odds band and favourite rank) still stands.
