# Experiment report: the handicap score with #96's speed figure, re-measured (Japan, #219)

Ticket #219 (epic #73, the C0–C6 testing scaffold), the follow-up to #129. #129 turned each runner's past
performances into handicap factors and a one-line out-of-fold `handicap_score`. Its speed family used an interim
figure: the start's final time against the median winning time of its (venue, surface, distance, class) on the
in-fold days. #96's expected top-3 time has landed (`models/speed.py`, scored by #305). The speed family now reads it,
with its day variant, and this report re-measures #129's two comparisons beside #129's numbers.

**Nothing is adopted (D16).** Every figure is an aggregate over the JRA results store (netkeiba race results; no JRDB,
no JRA-VAN) and the calibrated-market Win baseline v0 built on it. No runner or race row appears.

Two runs are reported. **v1** (first pass, superseded) put every past start inside #96's fit, so the day variant was at
its prior for all of them and it measured the new par alone. **v2** (this report's result) gives each past start the
figure as of its own race day, with that day's earlier results observed, which is the "with its day variant" of the
ticket; it also fixes the V2 meld's embargo. v1's records stay in the registry.

- Registry family **`japan-handicap-speed-96-final-win-v2`**, spec sha256
  `553414bc28927d42c44289f135d3c7488b32081ad259c19124b69db6276be037`, frozen before the table was built or any outcome
  was read (it names v1's spec as superseded). Six records, all scored, none failed, git `5fcfc02`. Summary:
  `docs/experiments/registry/japan/japan-handicap-speed-96-final-win-v2.md`.
  - `v96:V1:handicap_alone` `e3ca71d0…`, `v96:V2:meld_over_market_v0` `42193606…`
  - `interim_par:V1:handicap_alone` `c620f1c9…`, `interim_par:V2:meld_over_market_v0` `6f757165…`
  - `swap:handicap_alone` `61ca4698…`, `swap:meld_over_market_v0` `b4113498…`
- v1 family `japan-handicap-speed-96-final-win`, spec sha256
  `16b385befd691c9e957de14b56f465ed6cfda04bcaffcec42b199b23ebabcad8`, six scored records (`f4e9d9b9…`, `9cddf365…`,
  `e114e682…`, `2635aee6…`, `5f173c65…`, `9120fcb7…`), summary `japan-handicap-speed-96-final-win.md`. Kept.
- A first attempt at the v2 run (same frozen spec) was killed by the coordinator: its score pass hung in the fit of fold
  26 (see Setup, "The fit"). It left no registry record. The fix (below) is in the code behind the six records above.
- Code: `countries/japan/handicap/factors.py` (`_own_day_expected`, `build_handicap_table(..., speed_at_start_day=True)`,
  `score_table`, `table_complete`) and `countries/japan/studies/handicap_speed.py`. Reproduce with
  `uv run python -m horsetrack.countries.japan.studies.handicap_speed --run [--rebuild] [--max-minutes N]`.
- Store: `processed/jra/races.parquet@9cce511831074b64`, `processed/jra/runners.parquet@6f3cec95347e0132`. Baseline:
  v0's out-of-fold `win_calibration_oof-aac501f927b0` (#114, #123) and its manifest, the one #129 used.
- Full report: `$HORSETRACK_DATA_DIR/reports/handicap-speed-japan-553414bc2892.md`. The v2 table is
  `features/jra/handicap-96-v2/` (1350 race days, with its completion marker). v1's `features/jra/handicap-96/` and
  #129's `features/jra/handicap/` stay as they were.

## Hypothesis

Written into the frozen spec before anything was scored. H1 to H3 are carried unchanged from v1, whose results were
known when v2's spec was frozen; they were not tuned to them.

- **Null:** swapping the speed par for #96's expected time does not improve the handicap score. The gain
  ll(interim-par score) − ll(#96 score) is at most 0 on the common races.
- **H1 (V1):** the #96 score alone still beats equal chances, with a pseudo-R² within 0.02 of #129's 0.165. A negative
  gain over equal chances would be a bug.
- **H2 (V2):** the meld over v0 stays small, at most about 0.002 nats per race (#129: 0.0005), and its one-sided lower
  bound may not exclude 0. A gain above 0.01 would point to leakage.
- **H3:** the swap is worth little: under 0.01 nats per race either way.
- **What v2 changes from v1:** the figure of each past start is #96's as of its own race day, so a start's figure
  carries that meeting's earlier results (the day variant); and the V2 meld trains only on v0 rows no later than the
  scored fold's training cutoff (D9's embargo).

## Setup

Reproduced from #129's Setup, unchanged except where noted.

- **Data:** development days only, 2014-06-07 to 2026-11-15, the JRA store (races, runners). Every day is before the
  holdout (2026-11-28, D14). Every read goes through #114's scoped reader and `LockedHoldout` is never opened.
- **Figure (v2):** `SpeedModel` (#96) with `SpeedSpec(PAR_KEYS, VARIANT_KEYS, same_day_variant=True, t_df=5.0)`, #305's
  primary error model, on `speed_run.build_race_table(estimand="mean_placed_1_3")`. For a start on race day E the model
  is fitted on the races of the days before E, has observed E's results, and gives its race's expected top-3 time at
  post − 2 min (D2); the figure is that time minus the runner's finish time. A race resulted after that moment (its own
  race included) never counts. The figure as of a day is fixed, so there is one fit per race day (after the first
  365 days of the store), and the table's past starts take their figure from that map. A start with no finish time, or in a cell with no earlier race,
  or in the first 365 days of the store (2010), has no figure. v1 instead fitted once per table day on every race to
  its cut, so no past start had a day variant.
- **Build:** `build_handicap_table(2014-06-07, 2026-11-15, speed_at_start_day=True)` wrote 1350 race days and 600,231
  runner rows (as #129's did) under `features/jra/handicap-96-v2/` and then a completion marker, which the study
  requires before it reuses a table. The builder's default stays #129's contract (`handicap_factors` per day), which
  `test_handicap_score.py` pins; the default destination is now `handicap-96-v2`, not #129's directory.
- **Score:** the default ridge conditional logit (alpha 1, 46 `SCORE_FEATURES`), out of fold over Japan's #70 scheme:
  327,080 scored rows and 23,799 races, 2020-01-05 to 2026-11-15, as before.
- **The fit:** the score pass's fold 26 hung inside `scipy`'s `trust-exact` (a NaN trust radius at the float limit,
  with the old `gtol` of 1e-9, below what the arithmetic gives; every other fold stopped by the "bad approximation"
  exit and was accepted by the gradient bound). The fit now uses the acceptance bound itself as `gtol`
  (`GRAD_TOL × races`), caps iterations at 100 and refuses a non-finite objective or Hessian; the study runs under a
  deadline (90 minutes) and appends a failed record if a run raises or times out. The #129 table's scores, which the
  `interim_par` rows read, were produced with the earlier tolerance and were not recomputed; a difference between the
  two tolerances is at the 1e-6 level of the coefficients.
- **Scoring:** D6 through `core.eval.scoring`: winner log loss per race, a dead heat's k co-winners at 1/k,
  pseudo-R² against equal chances (ln n), gain = ll_baseline − ll_model in nats per race over the common races.
- **V2 meld:** exp(a ln p_cal + b ln h) normalised within the race; (a, b) is the conditional logit fitted on the v0 rows
  of the folds before the scored one **and dated no later than that fold's training cutoff** (the `train_end` of v0's
  manifest, D9's one-week embargo); the first scored fold only trains. v1 admitted every row of the earlier folds, so
  for the 2024-Q2 fold it trained on March 31 races although the cutoff was March 30.
- **CIs:** `core.eval.bootstrap`, a paired bootstrap resampling meetings (venue × day), 95% one-sided (D7), and the
  same draws' 2.5% and 97.5% quantiles as the two-sided interval. Difference from #129: `scoring.yaml` now signs 10,000
  replicates (#129 used 2,000 in a stand-alone script), seed 219.
- **Same code on #129's table:** the `interim_par` rows rescore #129's kept table (pinned to `features/jra/handicap`
  and checked to differ from the #96 table) with this study's code. They agree with what #129 printed on V1 to the
  digits it printed. On V2 they agree on the point estimate and the last fold's coefficients but not on everything:
  the interim V2 CI is [+0.000070, +0.000955] against #129's [+0.00009, +0.00095] (different bootstrap settings, above),
  and b's largest fold value is +0.0220 against #129's +0.015 (smallest −0.0798 against −0.078). The meld code of the
  stand-alone script is not the same code, and I could not reproduce that range. (v1's interim run, without the embargo,
  gave +0.0089.) It does not move the point estimates.

## Variants tried

Races / meetings: 23,799 / 1,985 for V1; 23,760 for the comparisons with v0; 22,850 / 1,908 for V2 (fold 0 only trains
the meld). Bounds are the one-sided 95% lower bound and the two-sided 95% interval. "v1" is the superseded first run.

| Variant | v2: #96 figure as of the start's day | v1: #96, no day variant | Interim par, same code (v2 run) | #129 as reported |
| --- | --- | --- | --- | --- |
| V1: score alone vs equal chances, pseudo-R² | 0.1659 | 0.1648 | 0.1650 | 0.165 |
| V1: gain, nats/race | +0.4306; lower bound +0.4212; CI [+0.4193, +0.4419] | +0.4276; +0.4182; [+0.4163, +0.4390] | +0.4281; +0.4187; [+0.4167, +0.4396] | +0.428; +0.419; [+0.417, +0.439] |
| Market v0 alone vs equal chances, same races | 0.2576 | 0.2576 | 0.2576 | 0.258 |
| Score alone vs market v0 (ll_v0 − ll_score) | −0.2375; lower bound −0.2442; CI [−0.2455, −0.2292] | −0.2405; −0.2471; [−0.2484, −0.2324] | −0.2400; −0.2467; [−0.2480, −0.2318] | −0.240; −0.247; [−0.248, −0.232] |
| V2: meld over v0, gain | +0.000364; lower bound +0.000022; CI [−0.000040, +0.000762] | +0.000519; +0.000144; [+0.000068, +0.000953] (no embargo) | +0.000520; +0.000143; [+0.000070, +0.000955] | +0.00051; +0.00015; [+0.00009, +0.00095] |

V2 coefficients, last fold (2026-Q4), and the range of b over the 27 folds:

| | a (last) | b (last) | b over the folds |
| --- | --- | --- | --- |
| v2 figure | 1.0486 | −0.0546 | −0.0717 to +0.0341 |
| v1 figure (no embargo) | 1.0532 | −0.0616 | −0.0787 to +0.0104 |
| interim par, v2 run | 1.0542 | −0.0626 | −0.0798 to +0.0220 |
| #129 | 1.053 | −0.062 | −0.078 to +0.015 |

V2 gain by calendar year (nats per race, about 3,450 races a year, 2,537 in 2020):

| Year | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2 figure | −0.00023 | +0.00073 | +0.00091 | −0.00030 | +0.00027 | +0.00028 | +0.00078 |
| interim par, v2 run | −0.00010 | +0.00114 | +0.00096 | −0.00027 | +0.00023 | +0.00048 | +0.00112 |
| #129 | −0.00011 | +0.00111 | +0.00094 | −0.00026 | +0.00023 | +0.00047 | +0.00112 |

The swap itself, paired on the same races (gain = ll(interim) − ll(#96), so positive means #96 is better):

| Variant | Races / meetings | Gain, nats/race | One-sided 95% lower bound | Two-sided 95% CI |
| --- | --- | --- | --- | --- |
| v2, score alone | 23,799 / 1,985 | +0.002528 | +0.000783 | [+0.000414, +0.004600] |
| v2, melded over v0 | 22,850 / 1,908 | −0.000156 | −0.000258 | [−0.000277, −0.000034] |
| v1, score alone | 23,799 / 1,985 | −0.000493 | −0.00165 | [−0.00187, +0.00091] |
| v1, melded over v0 | 22,850 / 1,908 | −0.000011 | −0.000072 | [−0.000084, +0.000062] |

The figures themselves differ more than in v1. On the 327,080 scored runner rows the v2 `speed_best`, `speed_mean3` and
`speed_last` correlate 0.90, 0.89 and 0.84 with the interim ones (v1: 0.98, 0.94, 0.91); the share with no figure is
10.6% against 10.2%, and the count of figures (`speed_n`) is the same on 95.7% of rows.

## Reading

- **H1 held.** The v2 score's pseudo-R² is 0.1659, 0.0009 above the interim score's, against the market's 0.2576 on the
  same races. It beats equal chances by 0.4306 nats per race (one-sided 95% lower bound +0.4212) and trails the market
  by 0.2375 (two-sided CI [−0.2455, −0.2292]).
- **H2 held.** The meld gains 0.00036 nats per race over v0, far below the 0.01 that would point to leakage. Its
  one-sided lower bound is +0.000022, barely above 0, and the two-sided CI [−0.000040, +0.000762] includes 0, so the gain
  is not distinguishable from 0 at the two-sided level. The meld changes the market exponent as well as adding the score:
  a is 1.0486 and b is −0.0546 in the last fold, and b ranges from −0.0717 to +0.0341 across the 27 folds. The gain is
  the joint effect of those two terms and is not attributed to either. Two of seven years are negative, as in #129.
- **H3 held in size; the null is rejected for the score alone and not for the meld.** The day variant moves the score
  alone by +0.0025 nats per race (one-sided lower bound +0.00078; two-sided CI [+0.00041, +0.00460]): #96's figure with
  its day variant gives a better handicap score than the class par, by a small amount, 0.6% of the score's own
  gain over equal chances. Melded over v0 the swap is −0.00016 (two-sided CI [−0.00028, −0.00003]): the interim meld
  is slightly the better one, so whatever the day variant adds to the score alone is not information the market lacks.
  Both are far under 0.01. v1, without the day variant, had both near 0 (CI included 0): the day variant is what moved
  the score alone. These are two comparisons from one run on one development span; a single tail would not carry
  weight by itself.
- **Why the figure moved (a reading, not measured).** A start's time against the par of an average day was, for the
  interim figure, a mix of the horse and the going of that day; the day variant takes the going part out. That makes
  the figure a better ability signal inside the score, but the market already prices going and the meld has little left
  to add, which fits the negative b.

## Alternatives rejected and why

- **Fold-bounded figures with the variant from the fit's own days.** `SpeedModel.observe` refuses days the fit saw, so
  a past start's figure cannot be read from a fit that contains its own day. The one-fit-per-day construction above is
  the alternative that keeps the variant causal.
- **Per-start fits.** The figure as of a day is the same for every start of that day, so one fit per race day is
  enough (one fit per race day, not one per start per table day).
- **A bare table rebuild over #129's directory.** Not done: the builder's default destination is
  `features/jra/handicap-96-v2`, and `track_bias_report` and `same_day_bias` keep reading #129's table for the running
  style, which this change does not touch.
- **Re-tuning the score (alpha, features) after the swap.** Not run: tuning on these folds would turn the CIs into
  post-selection numbers, and feature inclusion is the owner's decision (D16).
- **Scoring only the new table.** Rejected: without the same code on #129's table, a difference could come from the
  scoring change (`core.eval.scoring`, 10,000 replicates) rather than the figure.

## What this rules out

- At the one-sided 95% level, #96's figure with its day variant does not make the score alone worse than the interim
  one, and improves it by at least 0.0008 nats per race (about 0.0025 estimated). Melded over v0, it is no better than
  the interim figure: the estimated difference is −0.00016 and the one-sided lower bound −0.00026.
- The #129 result stands: the score alone beats equal chances by at least 0.42 nats per race (one-sided 95%) and trails
  the market by 0.24; the meld over v0 gains at most 0.0008 nats per race (two-sided 95% upper end) with a negative
  weight on form. A large public-information edge from simple handicapping stays ruled out on 2020-2026 development
  data.
- It does not rule out a gain from #96's day variant in a form other than a speed figure inside a 46-feature ridge
  score, or in a market-aware model.
- It is a single run on development data, with one alpha, one feature set and one meld form. It is not a holdout claim.

## Leaderboard rows

None. This study measures winner log loss, not a return per ¥100 at the final price, so it adds no ranked row (D16):
the six variants are registry records, listed unranked.

## Follow-ups

Proposed for the orchestrator (`status:proposed` under #73); none is filed by this run:

- Tests for the new table path: `_own_day_expected` (a figure reads its day's earlier results and never its own race or
  a later one, one fit per day), `score_table`, `table_complete` and the marker, and the study's deadline and failed
  record. `test_handicap_score.py` pins the builder's default as `handicap_factors` per day, so the
  `speed_at_start_day` path has no committed test; a hook denied adding one here.
- #218 (the negative weight on form within the market, by odds band and favourite rank) still stands; this run's
  negative b joins the evidence.
