# Experiment report: the meld's weight on form within T-2 odds bands and favourite ranks (Japan, #218)

Ticket #218 (epic #73), a follow-up from #129's experiment report. **Nothing is adopted** (adoption is by rank,
owner 2026-11-27), and the holdout is not opened. **Every figure is withheld (D21).** The odds bands and the
favourite ranks are read from the JRA-VAN time-series Win board at T-2, a licensed source, so every gain, bound,
coefficient, count, rate and sign of this study, and the outcome of its family test, is derived from JRA-VAN data.
This report says what was pre-registered, what ran, where the full report lives and which registry records hold it.

There are three families, each holding the same 12 variants. **Variants searched: 36, over three families.**
The registry counts each family's 12 once, whatever the number of runs.

- **v1, the pre-registered family:** `japan-form-bands-t2-win`, spec sha256
  `3585a45d05f6ecfcf1de92d605d08982ec2b8c4def4d9932823c63c031655bad`.
  - Its spec is the owner's approved pre-registration: the draft of 2026-11-24 on #218, corrected on 2026-11-25
    to read bands and ranks from the T-2 board, with JRA-VAN named as the T-2 source the same day, and reframed on
    2026-11-27. It was frozen in the registry before the first run read any row.
  - **First run, under the frozen rule:** git `3e34967`, run ids `3fe170189fe62eba` (P0), `04ad26d0cb345057`
    (B1), `d02f95cd43c45b52` (B2), `dabbc4558a368c0f` (B3), `1a341e83bc72144b` (B4), `859f8c530c79f2cb` (B5),
    `d53b8b253b2647d1` (R1), `579d34cb0e08de76` (R2), `3758c6cecf22c7ec` (R3), `b07c5d28368ec6f6` (R4),
    `983ac7cfac9fcb80` (BA) and `c1476a950951ff48` (RA). Private report: `reports/form-bands-japan-3585a45d05f6.md`.
  - **Second run, a post-freeze correction:** git `f4c0acd`, run ids `4c7dea4e42c2f518` (P0), `20a9069c61117f4a`
    (B1), `4537f408d1a34c26` (B2), `570898b9f44b6f20` (B3), `aed79d91002c40bb` (B4), `9e195ed56d5874ec` (B5),
    `737cbfa04736738b` (R1), `81573ebb829227d0` (R2), `8d041ce3effb2fc9` (R3), `72de994ece80707b` (R4),
    `a4fa71a6043226c2` (BA) and `0294e37a29f7de22` (RA). Private report:
    `reports/form-bands-japan-3585a45d05f6-20261127T051515786538Z.md`.
  - The second run changed the eligibility after the freeze, following the round-1 review of PR #473: a
    confirmed starter missing from v0 now dropped its race, where the frozen `data.rows` rule read it as a
    scratch. So it is not a run under v1's frozen rule. It is superseded, and it stays in the append-only registry
    so that the count is honest.
- **v2, the first corrected family:** `japan-form-bands-t2-win-v2`, spec sha256
  `4d1f93ba902c368184c395c58a64bdf2c0a8e79b94d8b8893c82d406e8421256`.
  - The design is v1's exactly: the variants, groups, meld, walk-forward, scoring and inference are unchanged.
    Only the rows' eligibility (`data`, with `inputs.p_cal`, `rulings`, `approved` and `supersedes`) differs, as
    the round-1 and round-2 reviews of PR #473 corrected it (below).
  - Claude froze it under the owner's 2026-11-27 reframe, which lets Claude add variants and count them without
    a new sign-off. It was frozen before its run read any row.
  - Its run, **superseded by v3:** git `36e9f3b`, run ids `5468464cbff10aaa` (P0), `2399ba7c801ad791` (B1),
    `9621c016d2fd0277` (B2), `9676bf8f2bace0db` (B3), `583c30098ca50cfa` (B4), `a14b8280fae07b59` (B5),
    `f3d653bfe1da3910` (R1), `5c1ec1b381d2ca63` (R2), `5c14264b57c41e2c` (R3), `ce470397f4025dbb` (R4),
    `23ebf013690949f9` (BA) and `482c4f89b52df882` (RA). Private report:
    `reports/form-bands-japan-4d1f93ba902c-20261127T054829124881Z.md`, with its figures as JSON beside it (same
    stem, `.json`).
  - The round-4 review of PR #473 found that the round-3 exclusions (a race with no runner rows is void, the
    dead-heat embargo, a dead heat without a v0 reference) changed which races are eligible under v2's frozen
    hash. On the owner's decision of 2026-11-27 on #218, v2 stays frozen exactly as it is and the corrected rules
    go into v3. v2's run stays in the append-only registry so that the count is honest.
- **v3, the corrected family and the run of record:** `japan-form-bands-t2-win-v3`, spec sha256
  `dd7c9a1ee1020b536bb4aacaabd3c1e28b839572aa4e33f010754583688dc942`.
  - The design is v2's (and v1's) exactly. It differs from v2 only in `family_id`, `data`, `inputs.p_cal`,
    `rulings`, `approved` and `supersedes`, which names v2. The eligibility is the round-3 and round-4 rules
    (below).
  - Claude froze it under the owner's 2026-11-27 reframe and the owner's 2026-11-27 decision on #218 that the
    corrected rules go into v3. It was frozen before its run read any row.
  - Run of record: git `5f07f3a`, run ids `a525c2b93b170961` (P0), `4937871dbf56109d` (B1), `c5e62b1abe9d33eb`
    (B2), `c0a1cfe376d9daf5` (B3), `c96f0eb006e48de8` (B4), `afffb16a625a1e47` (B5), `d25b9ff88786bef1` (R1),
    `4b5ac1e45f50accc` (R2), `50e71d7d6cd2d971` (R3), `9dc55e7f6b4a083b` (R4), `6d1032d5e53fd841` (BA) and
    `0ce664ef462d745f` (RA). Private report: `reports/form-bands-japan-dd7c9a1ee102-20261127T105844050073Z.md`,
    with its figures as JSON beside it (same stem, `.json`).
- **Registry summaries.** Every record carries the sources `jra` and `jravan`, so
  `docs/experiments/registry/japan/japan-form-bands-t2-win.md`, `japan-form-bands-t2-win-v2.md` and
  `japan-form-bands-t2-win-v3.md` show every value as withheld. The Status column (scored or failed) is shown,
  by the registry's design; the values behind each status are not.
- **Code at `5f07f3a`:**
  - `countries/japan/studies/form_bands.py`, tested in `tests/countries/japan/test_form_bands.py` and
    `test_form_bands_reader.py`.
  - The `reference_variants` argument of `core/eval/snooping.py`, tested in
    `tests/core/eval/test_snooping_reference.py`, and `Registry.frozen_spec`.
  - To reproduce, run `uv run python -m horsetrack.countries.japan.studies.form_bands`. It freezes and records
    v3. It reads one day of the time series at a time (WIN rows only) and the store column-scoped, and it prints
    only identifiers (family, spec hash, run ids, report path), never a figure.
  - Each run writes its own private report, named by its UTC start time, so no earlier report is overwritten.
    The report lists the ids of the records that hold its results, whether appended or reproduced from an
    identical earlier run.
- **Data builds,** the same for all four runs: `processed/jra/runners.parquet@6f3cec95347e0132`,
  `processed/jra/races.parquet@9cce511831074b64`,
  `models/japan/v0/win_calibration_oof-aac501f927b0.parquet@e61c7078ac0a6e46`,
  `features/jra/handicap@51777b77c4c5c478` and `processed/jravan/odds_timeseries@6f5ac71d1926c803`.
- **Private reports.** All are under the data directory, written through `write_report` with the source
  `jravan`. Each holds:
  - the ranked leaderboard: the gain of the registered meld over v0, with its one-sided bound and two-sided
    interval per variant;
  - the drop counts by reason and the group coefficients and sizes;
  - the primary question's section;
  - SPA and StepM, and every fold's fit.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| Pre-registration | The meld form, the five odds bands, the four rank groups, the 12 variants, the primary question, the scoring, the meeting bootstrap (2000 resamples, seed 218) and SPA / StepM at 5% family-wise, before any run. | #218, owner comments of 2026-11-25 |
| T-2 board | Bands and favourite ranks come from the Win board at T-2 (D2's primary), read from the JRA-VAN time series. JRA-VAN is handled as JRDB is (D21). | #218, owner comments of 2026-11-25 |
| Reframe | The pre-registration is frozen in the registry before any run; all 12 variants are member rows, ranked, with CI and the variants-searched count; results are private. Later variants are added by Claude and counted, without a new sign-off (v2). Adoption is by rank. | #218, owner comment of 2026-11-27 |
| v3 | The round-3 eligibility rules (runnerless races void, the dead-heat embargo against the fold's `train_end`, dead heats without a v0 reference) and the fold-wide refit check go into a new family v3, frozen before its run, whose run is the run of record; v2 stays frozen exactly as it is. | #218, owner comment of 2026-11-27 |
| D2 | T-2 is the primary decision time (`config/decision.yaml`). | `docs/problem-statement.md` §1, #14 |
| D6 | Winner log loss per race, a dead heat's winners at 1/k, void races dropped and counted, scratched runners removed and the probabilities renormalised over the starters. | §3, #14; the #69 answer |
| D7 | Meetings (venue x day) resampled; 95%, one-sided. | §3, #14 |
| D9 | Quarter folds, expanding, a 1-week embargo, 2020-Q1 the first quarter scored out of fold (`config/folds.yaml`). | §4, #14; the #70 answer |
| D14 | Development days only. The A2 holdout from 2026-11-28 is never read; every row is checked with `is_holdout_day`, and the data end on 2026-11-15. | §5, #14; the #71 answer |
| D21 | No figure derived from JRA-VAN (or JRDB) appears in the repository. No JRDB input is read. | §8, #14; owner, 2026-11-26 |

## Data and method

The rows are v3's. The design is the same in all three families.

- **Rows.** The population is the store's races on the development race days 2020-01-01 to 2026-11-15, a race
  of the races table without runner rows included (it is void). A race's
  starters are the store's confirmed starters, those whose `is_scratched` is not True. Each starter carries:
  - its `p_cal` from the v0 out-of-fold file (`win_calibration_oof-aac501f927b0`: #114's calibrated market, the
    input #129's V2 used);
  - #129's out-of-fold `handicap_score` as `h` (`features/jra/handicap/<day>.parquet`, built from the netkeiba
    store only);
  - the store's winner and meeting;
  - its T-2 Win odds.
  A missing v0 row is never read as a scratch.
- **Dead heats: v0's fold fits refitted.** v0's file holds no prediction for a race with a first-place dead heat.
  - The study refits each of v0's fold fits from the store, as `models.v0.calibration.run_fit` makes them:
    `fit_win_calibration` on the fold's training rows of v0's development frame from 2014-06-07, over the same
    folds.
  - The refit must reproduce the file's `p_cal`, from the file's own normalised 1/odds, on every development row
    the file holds for that fold, inside the run's span or not (the rows outside it are checked, never scored).
    If it does not (the store's training rows have changed), the run stops. v2 checked only the rows the file
    holds and the run keeps.
  - A first-place dead heat whose starters are all priced gets its fold's refit prediction on its normalised
    1/odds. Each co-winner counts as a winner at 1/k (D6).
- **v0 rows checked against the store.** Every v0 row is checked against the store's current starter fields: the
  runner is held, not scratched and priced, with the same normalised 1/odds. A disagreement takes the race out.
  So does a store starter that is priced but has no v0 row.
- **Unnumbered runners.** The store's horse number may be missing. An unnumbered scratched runner leaves its
  race intact. An unnumbered confirmed starter takes its race out.
- **Races left out.** Each is counted once, under the first of these reasons that applies:
  1. void: no winner, or no runner rows (`dropped_void`);
  2. a winner, but none with a final price;
  3. a dead heat with an unpriced starter;
  4. not in v0 for another reason;
  5. an unnumbered confirmed starter;
  6. a v0 row the store contradicts;
  7. a confirmed starter without a v0 row;
  8. a dead heat outside every fold;
  9. a dead heat dated less than the fold scheme's `embargo_days` after its fold's `train_end`
     (`dropped_dead_heat_embargo`);
  10. a dead heat in a fold the v0 file holds no row of, so its refit has nothing to be checked against
      (`dropped_dead_heat_no_v0_reference`);
  11. a T-2 runner the store does not hold;
  12. a starter without `h`.

  Reasons 9 and 10, and the runnerless void, are v3's; v2's run did not apply them.

  Scoring then drops and counts any race with a starter off the T-2 board, and any race without a winner. The
  counts are in the private report.
- **Scratches after T-2.** A runner the store flags as scratched but the T-2 board still prices is kept as a
  scratched row. It takes part in the ranking and is never scored.
- **v1's rows** (both runs) took the v0 file's starters as given and left out the races that file omits. In the
  first run, any T-2 runner missing from v0 was read as a scratch, which is the frozen `data.rows` rule. The
  second run was the partial correction (above); v2 completed it, and v3 adds the round-3 and round-4 rules.
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
  in an earlier run and fails in this one is tested as failed, and the registry keeps both records. When fewer
  than three meetings are scored, the bootstrap behind SPA and StepM is undefined. The tests are then reported
  as not run, with that reason, and are never forced.

## How to read the results

These caveats are part of the method, not findings. They hold whatever the figures show.

- **The gain of the registered meld over v0.** Every gain in the private report is the gain of the registered
  meld over v0. It is not a measure of form alone, for two reasons built into the frozen design.
- **Group intercept.** b_g multiplies ln h only inside group g. Within a race, ln h shares the race's normaliser
  and has a typical level, so b_g also acts as an intercept for the whole group. Since ln h ≤ 0, a b_g < 0 adds
  b_g ln h ≥ 0 to each runner of the group. Against b_g = 0, it raises (or keeps) the group's share of the race,
  as an under-priced group would need, whether or not it uses the form differences between the group's runners.
  So the sign of b_g does not by itself give the sign of a form effect.
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

All 12 were registered before the first run, and every run records all 12, whatever they show. Over the three
families that makes 36 variants searched. Within a run, every variant is scored on the same development races
and folds (D9 walk-forward, 2020-Q2 to 2026-Q4, 2020-Q1 training only). v2's races differ from v1's, and v3's
from v2's, by the eligibility above. Each variant's gain of the registered meld over v0,
with its one-sided bound and two-sided 95% interval, is withheld (D21) and is in the private reports. Each
record's status (scored or failed) is in the committed registry summaries.

| Variant | Design | v3, run of record | v2 (superseded) | v1, first run | v1, second run (superseded) |
| --- | --- | --- | --- | --- | --- |
| P0 | one b over every runner (#129's V2, the reference) | `a525c2b93b170961` | `5468464cbff10aaa` | `3fe170189fe62eba` | `4c7dea4e42c2f518` |
| B1 | one b for odds band [1, 3) only | `4937871dbf56109d` | `2399ba7c801ad791` | `04ad26d0cb345057` | `20a9069c61117f4a` |
| B2 | one b for odds band [3, 7) only | `c5e62b1abe9d33eb` | `9621c016d2fd0277` | `d02f95cd43c45b52` | `4537f408d1a34c26` |
| B3 | one b for odds band [7, 15) only | `c0a1cfe376d9daf5` | `9676bf8f2bace0db` | `dabbc4558a368c0f` | `570898b9f44b6f20` |
| B4 | one b for odds band [15, 50) only | `c96f0eb006e48de8` | `583c30098ca50cfa` | `1a341e83bc72144b` | `aed79d91002c40bb` |
| B5 | one b for odds band [50, ∞) only | `afffb16a625a1e47` | `a14b8280fae07b59` | `859f8c530c79f2cb` | `9e195ed56d5874ec` |
| R1 | one b for the T-2 favourite only | `d25b9ff88786bef1` | `f3d653bfe1da3910` | `d53b8b253b2647d1` | `737cbfa04736738b` |
| R2 | one b for ranks 2-3 only | `4b5ac1e45f50accc` | `5c1ec1b381d2ca63` | `579d34cb0e08de76` | `81573ebb829227d0` |
| R3 | one b for ranks 4-6 only | `50e71d7d6cd2d971` | `5c14264b57c41e2c` | `3758c6cecf22c7ec` | `8d041ce3effb2fc9` |
| R4 | one b for ranks 7 and longer only | `9dc55e7f6b4a083b` | `ce470397f4025dbb` | `b07c5d28368ec6f6` | `72de994ece80707b` |
| BA | all five band coefficients together | `6d1032d5e53fd841` | `23ebf013690949f9` | `983ac7cfac9fcb80` | `a4fa71a6043226c2` |
| RA | all four rank coefficients together | `0ce664ef462d745f` | `482c4f89b52df882` | `c1476a950951ff48` | `0294e37a29f7de22` |

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
- It covers the races left in once the counted drops are made. First-place dead heats are in v3, scored with
  v0's refitted fold fits, unless they fall in the embargo after their fold's `train_end` or in a fold the v0
  file holds no row of. Races without a priced winner, and races with a starter that v0 cannot price, are
  outside it.

In every case it is a development-day result with one run of record (v3), searched over 36 variants in three
families. It is no holdout claim and supports no adoption by itself.

## Follow-ups filed

- #474: a new family that holds the group effects and the market recalibration constant (group-intercept
  controls, ln p_cal split by group), so that a group's coefficient on form can be told apart from a group
  intercept and from a group-specific reshaping of v0. The owner approves it before any run.
- #475: the same family on our own logged T-2 boards (#130) once a training fold holds enough logged development
  days, as a second source for the bands.
- #476: `write_summary` refuses the registry while another family's id holds a `/`. The three families' summary
  files were written from a view of the well-named families.
- #87 and #102 take this family's records as a candidate input, by rank.
