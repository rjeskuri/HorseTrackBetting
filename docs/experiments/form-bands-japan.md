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
- Twelve records, one per variant, all appended by the one run: run ids `3fe170189fe62eba` (P0),
  `04ad26d0cb345057` (B1), `d02f95cd43c45b52` (B2), `dabbc4558a368c0f` (B3), `1a341e83bc72144b` (B4),
  `859f8c530c79f2cb` (B5), `d53b8b253b2647d1` (R1), `579d34cb0e08de76` (R2), `3758c6cecf22c7ec` (R3),
  `b07c5d28368ec6f6` (R4), `983ac7cfac9fcb80` (BA) and `c1476a950951ff48` (RA). Variants searched: 12. Each
  record carries the sources `jra` and `jravan`, so `docs/experiments/registry/japan/japan-form-bands-t2-win.md`
  shows each one as withheld.
- Code at `3e34967`: `countries/japan/studies/form_bands.py` (tested in `tests/countries/japan/test_form_bands.py`)
  and the `reference_variants` argument of `core/eval/snooping.py`. To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.form_bands`. It reads one day of the time series at a
  time, WIN rows only, and prints only identifiers (family, spec hash, run ids, report path), never a figure.
- Data builds: `processed/jra/runners.parquet@6f3cec95347e0132`, `processed/jra/races.parquet@9cce511831074b64`,
  `models/japan/v0/win_calibration_oof-aac501f927b0.parquet@e61c7078ac0a6e46`,
  `features/jra/handicap@51777b77c4c5c478` and `processed/jravan/odds_timeseries@6f5ac71d1926c803`.
- The full report is under the data directory at `reports/form-bands-japan-3585a45d05f6.md`, with its figures as
  JSON beside it (`form-bands-japan-3585a45d05f6.json`). Both were written through `write_report` with the source
  `jravan`. It holds the ranked leaderboard (gain, one-sided bound and two-sided interval per variant), the group
  coefficients and sizes, the primary question's table, SPA and StepM, and every fold's fit.

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

- **Rows.** One row per v0 starter of the v0 out-of-fold file (`win_calibration_oof-aac501f927b0`, #114's
  calibrated market, the input #129's V2 used): its `p_cal`, #129's out-of-fold `handicap_score` as `h`
  (`features/jra/handicap/<day>.parquet`, built from the netkeiba store only), the store's winner and meeting, and
  its T-2 Win odds. Development race days 2020-01-01 to 2026-11-15. A runner priced on the T-2 board but not a v0
  starter (a scratch after T-2) is kept as a scratched row: it takes part in the ranking and is never scored. A
  race with a starter without `h` or without a T-2 price, and a race without a winner, is dropped and counted.
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
  in the test.

## Hypothesis

Stated in the pre-registration before any run. #129's single run found that, given v0, the meld's weight on the
form score was negative in most folds: runners the form score likes win slightly less often than the market says.
The question is whether that weight holds up within odds bands and favourite ranks. **Primary question:** is b < 0
and the gain over v0 > 0 in the favourite groups (rank 1 and ranks 2-3) more than in the longshot bands? A
negative b that appears only at [50, ∞) would read as the favourite-longshot bias already in v0, not as form being
overweighted. The null, the starting point, is b_g = 0 and no gain over v0 in every group.

What would show the hypothesis wrong: no favourite-group variant (R1, R2, B1) with b < 0 and a lower bound above
zero, or such an effect at least as strong in the longshot bands as in the favourite groups.

The answer to the primary question is in the private report. It is a figure-derived result and is not stated here.

## Variants tried

All 12 were registered before the run and all 12 are recorded, whatever they show. They were scored on the
same development races and folds (D9 walk-forward, 2020-Q2 to 2026-Q4, 2020-Q1 training only). Each variant's
gain, one-sided bound and two-sided 95% interval are withheld (D21) and are in the private report.

| Variant | Design | Registry run id |
| --- | --- | --- |
| P0 | one b over every runner (#129's V2, the reference) | `3fe170189fe62eba` |
| B1 | one b for odds band [1, 3) only | `04ad26d0cb345057` |
| B2 | one b for odds band [3, 7) only | `d02f95cd43c45b52` |
| B3 | one b for odds band [7, 15) only | `dabbc4558a368c0f` |
| B4 | one b for odds band [15, 50) only | `1a341e83bc72144b` |
| B5 | one b for odds band [50, ∞) only | `859f8c530c79f2cb` |
| R1 | one b for the T-2 favourite only | `d53b8b253b2647d1` |
| R2 | one b for ranks 2-3 only | `579d34cb0e08de76` |
| R3 | one b for ranks 4-6 only | `3758c6cecf22c7ec` |
| R4 | one b for ranks 7 and longer only | `b07c5d28368ec6f6` |
| BA | all five band coefficients together | `983ac7cfac9fcb80` |
| RA | all four rank coefficients together | `c1476a950951ff48` |

Whether any variant failed, and which, is part of the result and is in the private report.

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

Withheld (D21). What the result excludes depends on its intervals and on the family test, all derived from the
JRA-VAN board. The private report states it variant by variant. In every case it is a development-day,
single-family, single-run result: it is no holdout claim and supports no adoption by itself.

## Follow-ups filed

None filed by this pull request. Proposed (each a new pre-registered family, `status:proposed` under epic #73):

- The same meld with ln p_cal also split by group, so that a group's b on form can be told apart from a
  group-specific reshaping of v0's favourite-longshot curve, which the primary question's own caveat names.
- The same family on our own logged T-2 boards (#130) once a training fold holds enough logged development days,
  as a second source for the bands.
- #87 and #102 take this family's records as a candidate input, by rank.
