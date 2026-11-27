# Experiment report: the meld's weight on form within T-2 odds bands and favourite ranks (Japan, #218)

Ticket #218 (epic #73), a follow-up from #129's experiment report. **Nothing is adopted** (adoption is by rank,
owner 2026-11-27), and the holdout is not opened. **Every figure is withheld (D21).** The odds bands and the
favourite ranks are read from the JRA-VAN time-series Win board at T-2, a licensed source, so every gain, bound,
coefficient, count, rate and sign of this study, and the outcome of its family test, is derived from JRA-VAN data.
This report says what was pre-registered, what ran, where the full report lives and which registry records hold it.

- Registry family `japan-form-bands-t2-win`, spec sha256
  `3585a45d05f6ecfcf1de92d605d08982ec2b8c4def4d9932823c63c031655bad`. The spec is the owner's approved
  pre-registration (the draft of 2026-11-24 on #218, corrected on 2026-11-25 to read bands and ranks from the T-2
  board, with JRA-VAN named as the T-2 source the same day, and reframed on 2026-11-27). It was frozen in the
  registry before the run read any row.
- **The run of record** is the second run, at git `f4c0acd`, after the round-1 review of PR #473. Its twelve
  records, one per variant: run ids `4c7dea4e42c2f518` (P0), `20a9069c61117f4a` (B1), `4537f408d1a34c26` (B2),
  `570898b9f44b6f20` (B3), `aed79d91002c40bb` (B4), `9e195ed56d5874ec` (B5), `737cbfa04736738b` (R1),
  `81573ebb829227d0` (R2), `8d041ce3effb2fc9` (R3), `72de994ece80707b` (R4), `a4fa71a6043226c2` (BA) and
  `0294e37a29f7de22` (RA). Variants searched: 12 (the family size counts each variant once across runs).
- **The first run** (git `3e34967`, run ids `3fe170189fe62eba`, `04ad26d0cb345057`, `d02f95cd43c45b52`,
  `dabbc4558a368c0f`, `1a341e83bc72144b`, `859f8c530c79f2cb`, `d53b8b253b2647d1`, `579d34cb0e08de76`,
  `3758c6cecf22c7ec`, `b07c5d28368ec6f6`, `983ac7cfac9fcb80`, `c1476a950951ff48`) stays in the append-only
  registry and is superseded. The review found that its reader treated a starter missing from the v0 file as a
  scratch and did not count the races v0 leaves out, and that its private report gave no answer to the primary
  question and did not state the design's confounds (below). The frozen spec did not change, so the second run is
  in the same family.
- Every record carries the sources `jra` and `jravan`, so `docs/experiments/registry/japan/japan-form-bands-t2-win.md`
  shows its values as withheld. Its Status column (scored or failed) is shown, by the registry's design; the
  values behind each status are not.
- Code at `f4c0acd`: `countries/japan/studies/form_bands.py` (tested in `tests/countries/japan/test_form_bands.py`
  and `test_form_bands_reader.py`), the `reference_variants` argument of `core/eval/snooping.py` (tested in
  `tests/core/eval/test_snooping_reference.py`) and `Registry.frozen_spec`. To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.form_bands`. It reads one day of the time series at a
  time, WIN rows only, and prints only identifiers (family, spec hash, run ids, report path), never a figure. Each
  run writes its own private report, named by its UTC start time, so no earlier report is overwritten, and the
  report lists the ids of the records holding its results (appended, or reproduced from an identical earlier
  run).
- Data builds, the same for both runs: `processed/jra/runners.parquet@6f3cec95347e0132`,
  `processed/jra/races.parquet@9cce511831074b64`,
  `models/japan/v0/win_calibration_oof-aac501f927b0.parquet@e61c7078ac0a6e46`,
  `features/jra/handicap@51777b77c4c5c478` and `processed/jravan/odds_timeseries@6f5ac71d1926c803`.
- The full report of the run of record is under the data directory at
  `reports/form-bands-japan-3585a45d05f6-20261127T051515786538Z.md`, with its figures as JSON beside it (same
  stem, `.json`). The first run's report is `reports/form-bands-japan-3585a45d05f6.md`. All were written through
  `write_report` with the source `jravan`. The report holds the ranked leaderboard (the gain of the registered
  meld over v0, its one-sided bound and two-sided interval per variant), the drop counts by reason, the group
  coefficients and sizes, the primary question's section, SPA and StepM, and every fold's fit.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| Pre-registration | The meld form, the five odds bands, the four rank groups, the 12 variants, the primary question, the scoring, the meeting bootstrap (2000 resamples, seed 218) and SPA / StepM at 5% family-wise, before any run. | #218, owner comments of 2026-11-25 |
| T-2 board | Bands and favourite ranks come from the Win board at T-2 (D2's primary), read from the JRA-VAN time series. JRA-VAN is handled as JRDB is (D21). | #218, owner comments of 2026-11-25 |
| Reframe | The pre-registration is frozen in the registry before any run; all 12 variants are member rows, ranked, with CI and the variants-searched count; results are private. Adoption is by rank. | #218, owner comment of 2026-11-27 |
| D2 | T-2 is the primary decision time (`config/decision.yaml`). | `docs/problem-statement.md` §1, #14 |
| D6 | Winner log loss per race, a dead heat's winners at 1/k, void races dropped and counted, scratched runners removed and the probabilities renormalised over the starters. | §3, #14; the #69 answer |
| D7 | Meetings (venue x day) resampled; 95%, one-sided. | §3, #14 |
| D9 | Quarter folds, expanding, a 1-week embargo, 2020-Q1 the first quarter scored out of fold (`config/folds.yaml`). | §4, #14; the #70 answer |
| D14 | Development days only. The A2 holdout from 2026-11-28 is never read; every row is checked with `is_holdout_day`, and the data end on 2026-11-15. | §5, #14; the #71 answer |
| D21 | No figure derived from JRA-VAN (or JRDB) appears in the repository. No JRDB input is read. | §8, #14; owner, 2026-11-26 |

## Data and method

- **Rows.** The population is the store's races on the development race days 2020-01-01 to 2026-11-15. Each
  scored race holds its v0 starters from the v0 out-of-fold file (`win_calibration_oof-aac501f927b0`, #114's
  calibrated market, the input #129's V2 used): each starter's `p_cal`, #129's out-of-fold `handicap_score` as `h`
  (`features/jra/handicap/<day>.parquet`, built from the netkeiba store only), the store's winner and meeting, and
  its T-2 Win odds.
- **What v0 leaves out.** v0's generator writes no prediction for a race with a first-place dead heat, a race
  without a priced winner, or a starter without a final price. Such races cannot be scored against v0 here. Each
  one is dropped and counted by its reason (void, dead heat, no priced winner, or a confirmed starter without a v0
  row), with the counts in the private report. A missing v0 row is never read as a scratch: a runner the store
  does not flag as scratched but v0 lacks takes its race out. v0 is not rebuilt for these races, since the
  pre-registration names the v0 file. So the D6 rule is applied to the races v0 scores, and the races v0 leaves
  out (dead heats among them) are outside this study's population.
- **Scratches after T-2.** A runner the store flags as scratched but the T-2 board still prices is kept as a
  scratched row. It takes part in the ranking and is never scored. A race with a starter without `h` or without a
  T-2 price, a race whose T-2 board prices a runner the store does not hold, and a race without a winner, is
  dropped and counted.
- **The T-2 board.** For each race, each runner's latest WIN row at 2 minutes or more before the scheduled post,
  read as `studies.cross_pool.boards_at` reads it. A runner whose latest row is scratched or unpriced is off the
  board. The time series is a sequence of snapshots, so the board read at T-2 is the latest snapshot at or
  before T-2, not a capture taken at T-2.
- **Groups.** Odds bands [1, 3), [3, 7), [7, 15), [15, 50), [50, ∞), left-closed. Favourite ranks 1, 2-3, 4-6,
  7+, where a runner's rank is 1 plus the number of runners in its race priced strictly shorter on the T-2 board
  (a tie shares the better rank).
- **The meld.** Within each race, P(i wins) ∝ exp(a ln p_cal + Σ_g b_g 1[i in g] ln h), fitted by unpenalised
  conditional-logit maximum likelihood (`models.condlogit`).
- **Walk-forward.** Fold k is fitted on the starters of the folds before k whose race day is on or before fold k's
  `train_end` (#70's embargo) and scores fold k's races. 2020-Q1 only trains.
- **Scores.** Per race, ll_v0 - ll_variant in nats per race (positive: the meld beats v0), over the races every
  variant scores. Per variant: the mean, D7's one-sided 95% lower bound and the two-sided 95% interval from the
  paired meeting bootstrap (2000 resamples, seed 218). Per group: the fitted b by fold (median and range) and the
  group's runners and races.
- **Family test.** Hansen's SPA ("no variant beats v0") and Romano-Wolf StepM at 5% family-wise, over the 11
  non-reference variants, on the per-meeting sums of the per-race losses in date order, with 2000 replicates,
  seed 218 and block size 1 (meetings drawn i.i.d., as D7's bootstrap draws them). P0 is the reference and is not
  in the test. `core.eval.snooping` accepts P0 as a reference only because the frozen spec lists it among its
  variants and not among the family-test rules. The test reads the current run's records: a variant that scored
  in an earlier run and fails in this one is tested as failed, and the registry keeps both records.

## How to read the results

These caveats are part of the method, not findings. They hold whatever the figures show.

- **The gain of the registered meld over v0.** Every gain in the private report is the gain of the registered
  meld over v0. It is not a measure of form alone, for two reasons built into the frozen design.
- **Group intercept.** b_g multiplies ln h only inside group g. Within a race, ln h shares the race's normaliser
  and has a typical level, so b_g also acts as an intercept for the whole group. A b_g of either sign can shift
  the probability of a whole band or rank group, for example a group the market under- or over-prices, without
  using the form differences between its runners. So the sign of b_g does not by itself give the sign of a form
  effect.
- **Market refit.** Every variant refits the market weight a, while v0 is unchanged. Each gain, and the family
  test against v0, therefore credits the refit of a as well as the form terms.
- **No conclusion about the sign of the form effect.** That needs a comparison that holds the group intercepts
  and the market refit fixed. Such a comparison is a new family for the owner to approve (#474). It is not run,
  and nothing here stands in for it.

## Hypothesis

Stated in the pre-registration before any run. #129's single run found that, given v0, the meld's weight on the
form score was negative in most folds: runners the form score likes win slightly less often than the market says.
The question is whether that weight holds up within odds bands and favourite ranks. **Primary question:** is b < 0
and the gain over v0 > 0 in the favourite groups (rank 1 and ranks 2-3) more than in the longshot bands? A
negative b that appears only at [50, ∞) would read as the favourite-longshot bias already in v0, not as form being
overweighted. The null, the starting point, is b_g = 0 and no gain over v0 in every group.

The frozen spec defines no favourite-vs-longshot comparison rule: no measure or threshold says when the effect
in the favourite groups counts as stronger than in the longshot bands. No such rule is chosen after the run. So
the comparison itself is inconclusive under the frozen spec, whatever the figures are. This follows from the
spec, not from the data. The private report reads each group on its own (its b by fold and its gain of the
registered meld over v0) and gives no verdict. A deciding rule needs a new, owner-approved family. The group
readings are figure-derived and are not stated here.

## Variants tried

All 12 were registered before the run, and every run records all 12, whatever they show. They were scored on
the same development races and folds (D9 walk-forward, 2020-Q2 to 2026-Q4, 2020-Q1 training only). Each
variant's gain of the registered meld over v0, with its one-sided bound and two-sided 95% interval, is withheld
(D21) and is in the private report. Each record's status (scored or failed) is in the committed registry summary.

| Variant | Design | Run of record | First run (superseded) |
| --- | --- | --- | --- |
| P0 | one b over every runner (#129's V2, the reference) | `4c7dea4e42c2f518` | `3fe170189fe62eba` |
| B1 | one b for odds band [1, 3) only | `20a9069c61117f4a` | `04ad26d0cb345057` |
| B2 | one b for odds band [3, 7) only | `4537f408d1a34c26` | `d02f95cd43c45b52` |
| B3 | one b for odds band [7, 15) only | `570898b9f44b6f20` | `dabbc4558a368c0f` |
| B4 | one b for odds band [15, 50) only | `aed79d91002c40bb` | `1a341e83bc72144b` |
| B5 | one b for odds band [50, ∞) only | `9e195ed56d5874ec` | `859f8c530c79f2cb` |
| R1 | one b for the T-2 favourite only | `737cbfa04736738b` | `d53b8b253b2647d1` |
| R2 | one b for ranks 2-3 only | `81573ebb829227d0` | `579d34cb0e08de76` |
| R3 | one b for ranks 4-6 only | `8d041ce3effb2fc9` | `3758c6cecf22c7ec` |
| R4 | one b for ranks 7 and longer only | `72de994ece80707b` | `b07c5d28368ec6f6` |
| BA | all five band coefficients together | `a4fa71a6043226c2` | `983ac7cfac9fcb80` |
| RA | all four rank coefficients together | `0294e37a29f7de22` | `c1476a950951ff48` |

## Alternatives rejected and why

- **Bands and ranks from the final board** (the first draft): rejected by the owner on 2026-11-25 in favour of the
  decision-time board at T-2.
- **Run now on our own logged T-2 boards:** they cover two Kyoto days, none of them a development day of this
  span, so no race could be grouped (#218, 2026-11-25).
- **Park until the logger has enough days:** rejected for the JRA-VAN time series (owner, 2026-11-25).
- **JRDB pre-race odds or netkeiba time-series odds:** JRDB is excluded as an input (D21), and netkeiba's series
  would be a new scrape.
- **Other band edges, rank cuts, a ridge penalty, more features, or ln p_cal split by group:** excluded by the
  pre-registration. Any of them is a new family under a new ticket.
- **A block bootstrap over weeks for SPA / StepM:** the pre-registration names D7's meeting resampling. A block of
  one meeting is that resampling; a longer block would be a new choice.

## What this rules out

The figures are withheld (D21). What the result excludes depends on its intervals and on the family test, all
derived from the JRA-VAN board, and the private report gives it variant by variant. Two limits hold whatever the
figures are:

- Every result is a gain of the registered meld over v0. It includes the group-intercept and market-refit terms,
  so it rules nothing in or out about form alone (see "How to read the results").
- It covers only the races v0 scores. The races v0 leaves out (first-place dead heats, races without a priced
  winner, races with a starter without a final price) are outside it.

In every case it is a development-day, single-family result with one run of record. It is no holdout claim and
supports no adoption by itself.

## Follow-ups filed

- #474: a new family that holds the group effects and the market recalibration constant (group-intercept
  controls, ln p_cal split by group), so that a group's coefficient on form can be told apart from a group
  intercept and from a group-specific reshaping of v0. The owner approves it before any run.
- #475: the same family on our own logged T-2 boards (#130) once a training fold holds enough logged development
  days, as a second source for the bands.
- #476: `write_summary` refuses the registry while another family's id holds a `/`. This family's summary file
  was written from a view of the well-named families.
- #87 and #102 take this family's records as a candidate input, by rank.
