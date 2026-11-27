# Experiment report: the C1 final-price nowcast against the random walk and #116's drift (Japan, #85)

Ticket #85 (epic #73; 08 §8 B2, §3.2 C1, §4 E19). **Nothing is adopted** (D16), and the holdout is not opened.
**Every figure is withheld (D21).** The run trains and scores on the JRA-VAN time-series odds paths in the store,
which are a licensed source. This report therefore says what ran, where its report lives and which registry records
hold it. It gives no count, span, score, rate, sign or bound derived from that data. The figures for our own logged
days (Wide coverage, below) are not JRA-VAN data and are given.

- Registry family `japan-c1-nowcast-final-shares-v2`, spec sha256
  `6cc31a7fc685deebc6b05a31c515007da3d3000c3d0839621df936a48a5997f9`. The spec was frozen when the full run
  started, before that run read any row. It holds the hypothesis, the null, the data rules (the #69 late-scratch
  rule among them), the variants, the comparisons, the scores, the D2 grid, D9's fold scheme, D7's bootstrap and the
  model constants.
- **v2 supersedes the v1 family** `japan-c1-nowcast-final-shares` (spec sha256
  `c652ebf9d5da87478f3281ab66e8db9f960b5c0ce326af05019f0a345d41a0dd`, git `819b99b`, run ids `e8c261a0882090ba`,
  `f90461967e915071`, `d7f6cc3bb7e0d2a2`, `28b378db8473dd2f`, `13ab926148e96f76`; private report
  `nowcast-japan-c652ebf9d5da.md`). #85's review found that v1's spec and code dropped every late-scratch case
  whole, which breaks the owner's #69 rule in `core/eval/scoring.py`: scratched runners are removed and the
  probabilities renormalised over the starters, void races are dropped and counted, and model and baseline are
  scored on their common races, with the drops counted. The frozen v1 spec could not take the fix, so the fixed
  study is a new family. The registry is append-only, so v1's freeze and records stay as they are. They are
  superseded and are not evidence. v2 changes only the scratch clause (and the `dropped` list it implies) and adds
  a `supersedes` field. No model, constant, score, fold or grid changed. The fix came from the review against an
  existing owner rule, and v1's scores were not consulted for it.
- **Trial runs.** Two `--trial` runs came before the v1 freeze. They read days up to 2020-03-31 and 2020-06-30, so
  they scored the 2020-Q1 and 2020-Q2 folds. At the time `--trial` wrote no registry record. They sized memory and
  run time, and they showed the Place rule's check failing (below). **The spec's `place_shares` clause was added
  after those trial runs had scored folds.** That clause makes Place a diagnostic when the check fails, and v2
  keeps it. The code then changed only to save memory: fits take a day list instead of a copy of the cases, and
  Arrow's pool is released after each day.
  - The run up to 2020-03-31 is now in the registry, recorded after the fact. Its private report JSON
    (`$HORSETRACK_DATA_DIR/reports/nowcast-japan-cf21829add59-trial.json`) holds what a record needs: the gains,
    the git sha, the data builds (its odds build is the day files up to 2020-03-31), the spec hash, the sampling
    and the draws. Its five comparisons were appended through the registry API to family
    `japan-c1-nowcast-final-shares-trial`, with `params.trial = true` and a `params.recorded_after_the_fact` note.
    The run ids are `047d1f1a0272043f`, `a00893ac0cd5785d`, `22568a2dce18cf08`, `3572ab0903c12455` and
    `10904eca4c2ea311`. Their git sha is `ea5d0b4`, the HEAD the run named, but `studies/nowcast.py` was not yet
    committed then (it was first committed in `819b99b`). The JSON does not state the seed, so the records carry
    none.
  - The run up to 2020-06-30 is not recorded. The other trial wrote its report to the same file name, so nothing
    of this run survives to rebuild a record from. It is disclosed here instead.
  - From this review on, a `--trial` run appends its records to the family's `-trial` family
    (`japan-c1-nowcast-final-shares-v2-trial`), with `params.trial = true`. It never freezes or alters the family's
    spec. No v2 trial has been run.
- Code at `be905f8`: `core/eval/distributional.py` (CRPS, PIT, Gaussian log score), `countries/japan/models/nowcast.py`
  (the model, tested in `tests/countries/japan/test_nowcast.py` and `tests/core/eval/test_distributional.py`) and
  `countries/japan/studies/nowcast.py` (the run). To reproduce, run `uv run horsetrack japan nowcast-study`. It
  scores every test day with 200 draws per case, one day file in memory at a time. `--test-days-per-fold N` scores a seeded sample of test days. `--trial` records its runs as trials (above).
- Data builds: `processed/jravan/odds_timeseries@0fd1381d81daf570`, `processed/jravan/races.parquet@8c0dfc18f1894406`,
  the same as v1's.
- The full report is under the data directory at
  `$HORSETRACK_DATA_DIR/reports/nowcast-japan-6cc31a7fc685.md`, with its tables as JSON beside it
  (`nowcast-japan-6cc31a7fc685.json`). Both were written through `write_report` with the source `jravan`. The
  records are in `docs/experiments/registry/japan/japan-c1-nowcast-final-shares-v2.md`, and the after-the-fact
  trial records are in `japan-c1-nowcast-final-shares-trial.md`. Every one shows as withheld.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| Training data | Train on the JRA-VAN time-series paths in the store (`processed/jravan/odds_timeseries`, #369), add our own logged days as they accrue, and count a snapshot at its stamped minute, with no lag. | #85, owner comment of 2026-11-26; the as-of answer on #359 |
| D2 | The grid is T-60, T-30, T-16, T-5, T-2 and T-1, with T-2 primary (`config/decision.yaml`). | `docs/problem-statement.md` §1, #14 |
| D3 | Win, Place and Wide come first. The model is pool-generic and fits Wide wherever Wide paths exist. | §1, #14; #127 |
| D7 | The ticket's race-day clusters are the primary interval. D7's meeting clusters (venue x day) are reported beside them. Both are 95% one-sided with 10,000 replicates (`config/scoring.yaml`, seed 85). | §3, #14 |
| D8 | The primary score is the log predictive density of the final log share, with PIT and 50%/90% coverage per horizon. This ticket adds CRPS (and the KL of the final shares). The price baseline is #116's fitted drift by pool, odds band and horizon, scored beside the random walk. | §3, #14 |
| D9 | Quarter folds, quarterly refit, an expanding window, 22 quarters of minimum training and a 1-week embargo (`config/folds.yaml`). The first scored quarter is 2020-Q1. | §4, #14; the #70 answer |
| D10 | No race day before 2014-06-07 is read. The nowcast is the price half, and D10 keeps the earlier takeout regime out wherever the market enters, so the series' earlier history is not used. | §4, #14 |
| D11-D15 | Holdout A2 runs from 2026-11-28 and is fully locked. Each day is checked with `is_holdout_day`, which drops none of the days read, since all of them precede the holdout. The holdout is not opened. | §5, #14; the #71 answer |
| D16 | Adoption needs a holdout gain bound. Nothing is adopted here, and `current.json` still names #116's drift. | §6, #14 |
| D21 | No figure derived from JRA-VAN appears in the repository. Registry records carry the source `jravan`, so the committed summary withholds them. | §8, #14; owner, 2026-11-26 |
| p_fund | It is not an input in v0 (circularity risk). Whether it may enter is the owner's call. | #85 ticket |

## Data

- **Boards.** These are `processed/jravan/odds_timeseries/<day>.parquet`: 0B41 gives WIN, PLACE and the bracket
  quinella, 0B42 gives QUINELLA, and every file is exactly `logger.compact.SCHEMA`. Each day file is read on its
  own, by column and with the model's pools only. The board at t = post − τ takes each combination's latest row at
  or before t (by `snapshot_utc`). The lag boards are read the same way at t − 5 and t − 15. V_t is the latest
  `__pool__` total at or before t. Pools other than WIN, PLACE and QUINELLA_PLACE are ignored.
- **The final board** is each combination's latest row (the store's `is_final_capture`). V_final is the latest
  pool total.
- **Cases.** A case is one (race, pool, τ). A late scratch follows the owner's #69 rule (`core/eval/scoring.py`:
  scratched runners are removed and the probabilities renormalised over the starters). The store writes a scratched
  runner's rows with no price, so a combination priced on the board at t whose latest row at the close has no price
  is off the final board. Such a case is kept: every forecaster gets the board at t, the scratched runner included,
  its drawn shares are restricted to the combinations priced at the close and renormalised, and the case is scored.
  The private report counts these cases (`renormalised`). The fits (pool growth aside, which reads every race's
  totals) train only on the cases whose board at t is the final board, and #116's drift does the same. A case is
  dropped when its final board prices a combination the board at t does not (`added_by_close`), or when there is
  no board at t or no final board. The private report counts each kind of drop. A race without a post time gives
  no case.
- **Grades** are the jravan races table's `race_class`, and they enter only the pool-growth profile.
- **Place shares.** The series carries no per-runner Place money, so Place shares invert the low figures in closed
  form by the stake-back-share rule of `config/jra.yaml`. Where the rule holds, the inverted money totals 1 in units
  of the pool. The run reports that total on every development board, and **the check does not hold on JRA-VAN
  boards** (the private report has the figures). Place shares from the low figure are therefore not money shares
  there, and every Place row of the run is a diagnostic, not evidence. In the ticket's words, Win is shipped. Before
  the v1 frozen run, the high figures were also inverted by the same rule and did not total 1 either, so neither end of
  the Place range follows the simple rule. The final per-runner votes (H1, #368) were checked against the final WIN
  board on one day and did not reconcile with it exactly, so they were not used as Place money. That check was an
  exploration and is not part of either frozen run. **Superseded by #410 (below):** the inversion used a rule that
  took the takeout off the pool, not JRA's rule, which applies the rate to the whole dividend; v3, v4 and v5 read Place by
  JRA's rule, and v5 measures it against the final Place votes on the scored boards.
- **Wide.** The JRA-VAN series has no Wide pool. Our own logged development days (`processed/jra/odds_timeseries`)
  are 2026-11-21 and 2026-11-22, each with 12 Wide races, and neither file has a post time (`post_utc` is null, as
  the logger writes no race-card sidecar for them). So no Wide board at t can be placed, Wide is neither fitted nor
  scored, and the model's Wide path is exercised only by its code. Both days also fall in the last scored quarter,
  which no fold trains on. Our own WIN and PLACE days duplicate JRA-VAN days and are not read. A logged holdout
  day's file is skipped by its name and never read (D14). v1's code read every logged file and tagged holdout days
  only afterwards, which the review found. No logged day was a holdout day when either run read them, since the
  holdout starts 2026-11-28 and the logged days are the two above, so no holdout file was opened.
- **Board age at t.** The JRA-VAN paths are 5-6 minutes apart, so the board at a near-post τ can be minutes old.
  The run measures the age at t per pool and τ (from the board's oldest row to t) and reports it. Pool growth uses
  the board's own minutes to post, not τ. #116's drift is fitted on the same boards without its `max_board_age_s`
  limit. That limit (45 s at T-2, 30 s at T-1) was set for a logger capturing every ~30 s, and on these paths it
  would drop most of the drift's examples at T-2 and T-1. Without it, every forecaster is fitted and scored on the
  same boards.

## Hypothesis

The frozen spec reads:

- **E19.** "The structural nowcast (phi_t x s_late, logistic-normal late composition with velocity and
  favourite-rank features) forecasts the final ln-shares of the Win and Place pools better than the random walk
  q_final = q_t and than #116's fitted drift, per decision horizon: log score (D8, primary) and CRPS gains with
  one-sided 95% day-clustered bounds; PIT and 50%/90% coverage by horizon."
- **Null.** "No gain over the random walk at any horizon (the starting point)." Under D7 a gain claim for a (pool, τ)
  rests on its one-sided lower bound lying above 0. Against #116's drift, the same reading tells whether the
  structural form adds anything over D8's fitted baseline.

## Variants tried

Four forecasters were fitted on each fold's training days, and every scored case drew from each of them
with the same view, n and seed:

- `structural`: φ_t from the pool-growth profile. The late composition is logistic-normal around the board with
  three features (the share velocities over [t−5, t] and [t−15, t], and ln rank). β comes from least squares within
  race. σ maximises the training log score over at most 400 seeded cases per (pool, τ), with 64 common draws.
- `no_features`: the same with β = 0.
- `random_walk`: softmax(ln s_t + σ_rw ε). σ_rw is fitted per (pool, τ) by the same criterion as the structural
  model's σ.
- `drift_v0`: #116's `NowcastV0.fit` on the same boards, with `nowcast_v0.yaml`'s bands and prior_n. Its odds draws
  are read back into shares by the pool's share rule.

There is one registry record per comparison. Each holds, per pool and τ, the log-score, CRPS and KL gains with
race-day and meeting bounds, and the per-fold gains at T-2. All figures are withheld. The v1 records are listed for
the trail only: they are superseded (above).

| Comparison | v2 record | v1 record (superseded) |
| --- | --- | --- |
| `structural_vs_random_walk` (E19, primary) | `03c2a06563491ea5` | `e8c261a0882090ba` |
| `structural_vs_drift_v0` (D8's price baseline) | `bd887ac5e5959224` | `f90461967e915071` |
| `no_features_vs_random_walk` | `25a351d57949f866` | `d7f6cc3bb7e0d2a2` |
| `structural_vs_no_features` (what the features add) | `2337943e84865262` | `28b378db8473dd2f` |
| `drift_v0_vs_random_walk` | `e750e055f87e0d06` | `13ab926148e96f76` |

The private report also holds, per forecaster, pool and τ, the PIT deciles and the 50% and 90% coverage. It holds
σ_t, σ_rw and β per pool and τ in every fold, the pool-growth profile against the observed V_t / V_final, the board
age at t and the Place rule's check.

## Failures

- Place: the stake-back inversion of the low figure fails its money-total check on JRA-VAN boards (see Data), so
  Place is a diagnostic in v1 and v2. Only Win stands as a result there. #410 (below) replaces the rule.
- Wide: no case can be built (see Data). This is a coverage failure, not a modelling result.
- v1's scratch handling: v1 dropped late-scratch cases against the owner's #69 rule, so its five records are
  superseded by v2's (above). This is a failure of the v1 study design, found in review, not a modelling result.
- The command exited 1 at its last step, in both the v1 and the v2 full runs. Each time the report and all five
  records were written first. `write_summary` then raised on records of another ticket in the shared registry:
  #131's family ids (`japan/131-exacta-category-v1` and others) contain `/`, which cannot be a summary file name
  (tickets #344 and #391). The registry verifies and was not altered. The committed summary was regenerated by
  `write_summary` itself, over the registry restricted to the families already committed plus this ticket's. For v2
  that added `japan-c1-nowcast-final-shares-v2.md`, `japan-c1-nowcast-final-shares-trial.md` and their two lines of
  the index, and changed no other file. No record failed, and every variant scored on every fold.

## Alternatives rejected and why

- **Dropping late-scratch cases (the v1 family):** rejected in #85's review. It breaks the owner's #69 rule, under
  which scratched runners are removed and the probabilities renormalised over the starters.
- **Training on the renormalised cases too:** not done. A scratched runner has no final share, so its late-money
  ratio, which β's least squares reads, is undefined. The #69 rule is a scoring rule, and the fits keep the cases
  whose board at t is the final board, as in v1.
- **σ from the residual SD of ln r** (r = the late-money ratio): rejected. Where 1 − φ_t is small, rounded odds make
  r noisy and sometimes non-positive. A floor then gives outliers that would set σ. σ is instead fitted by the
  scoring rule the run reports (minimum-score estimation, consistent for a proper score). The random walk gets the
  same criterion, so the comparison is like for like. The floor R_FLOOR enters β's least squares only.
- **Place money from the H1 per-runner votes (#368):** not used. They are final votes only, with no time series, so
  they give no shares at t. On the day checked they did not reconcile exactly with the final WIN board either.
- **Place shares from the high figures:** rejected. Their inversion fails the same check in the other direction.
- **#116's `max_board_age_s` on its training examples:** not applied, for the reason given under Data.
- **The v0 drift reimplemented on shares:** not done. `NowcastV0.fit` runs unchanged on (low figure at t, final low
  figure) pairs built from the same boards. Only its draws are read back into shares, by the same share rule that
  gives the target.
- **The pre-2014 JRA-VAN history:** not used (D10).
- **A random φ (uncertain V_final):** not in v0. φ_t is a point forecast. V_final uncertainty does not move the
  shares, which are the scored quantity, but it would matter to a decision layer that sizes against the pool.
- **Wide fitted on our two logged days:** impossible. There are no post times, and no training fold contains them.

## What this rules out

The figures are withheld, so the repository states no size or direction of any gain, and nothing here supports
adoption: D16 needs a holdout bound, and the holdout is not opened. The run does rule out the stake-back inversion of
the board's low figures as a source of Place money shares on JRA-VAN boards. Until that is resolved, any Place
nowcast built on those shares forecasts a quantity that is not money over the pool. #410 (below) found the cause:
the inversion's rule was not JRA's.

## Follow-ups filed

None by this PR. Proposed, for the orchestrator to file under epic #73 as `status:proposed`:

- Place shares that are money shares: JRA's exact Place odds rule, or per-runner Place money where a source carries
  it at t. Until then, Place stays a diagnostic.
- A Wide nowcast on our own logged days, once they are compacted with post times (a race-card sidecar per venue)
  and enough development days have accrued to train a fold.
- Near-post boards: score T-2 and T-1 on our own logs, which capture every ~30 s, once they cover enough days. On
  5-6 minute paths those horizons mostly see an older board.
- Pre-registered model variants: σ by odds band (where #116's drift has its cells), and a random V_final.

## #410: Place shares by JRA's rule (families v3, v4 and v5)

Ticket #410 (epic #73). **Every figure is withheld (D21)**: the measurement and the re-run read JRA-VAN data. This
section states no count, rate, share, bound or quantifier derived from them; they are in the private reports named
below. Nothing is adopted, and the holdout is not opened.

- **Hypothesis.** v1 and v2 inverted Place's low figures by m_i (P lo_i − P + 1) = (1 − takeout) − (the P − 1
  largest other m_j), which takes the takeout off the pool and returns stakes at 1. JRA's rule
  (`core.pricing.stake_back_share_dividend`, pinned by `tests/core/test_pricing.py`) pays rate × (1 + D / (P s_i))
  per unit, rate = 1 − takeout applied to the whole dividend, truncated to 10 yen and floored at the minimum
  dividend. The hypothesis: under JRA's rule the printed figures follow from the money, and shares read from both
  figures are money shares within a stated tolerance. Null: they are not, and Place stays a diagnostic.
- **Measurement, v5** (`studies/nowcast.py`, `place_rule_measurement`; run inside the study after the folds). It
  validates the boards the study scores: the closing PLACE board of every scored Place case, that is the folds'
  test cases after any test-day sampling (v4 measured every extracted Place race, training-only days included; no
  broader development measurement is kept) (`models.nowcast.closing_boards`, every priced runner kept, a missing high figure kept as
  missing), read exactly as the cases read it (`models.nowcast.place_shares`: P from the priced runners, and the
  read used, `joint`, `undetermined` or `low_only`). Each is set against its race's PLACE votes (H1) in
  `processed/jravan/pool_votes.parquet`, read one year at a time through the store's scoped reader
  (`models.v0.store.read_store_table`, a price read whose pool, slot and day predicates run inside the scan, so
  holdout-day rows never reach the study; D14, #415). A board is validated only when its H1 record is final
  (データ区分 4 or 5) with positive votes on every priced runner; otherwise it is unavailable and counted by reason
  (no votes, votes not final, a priced runner without votes). Per read used in scoring and per P it reports
  coverage (validated over scored boards), whether the rule gives every printed figure from the votes, and how far
  the shares as scored are from the vote shares. Tolerance: 7.5% relative and 0.01 of the pool absolute on every
  runner. The tests derive it from the 10-yen truncation, which moves the share a figure of 2.0 implies by up to
  6.8%. Coverage and agreement are separate (#410's re-check): a determined read (`joint` or `low_only`) stands as
  a result (`place_result_readers`) only when it validated boards and its shares as scored are within the tolerance
  on at least `PLACE_MIN_WITHIN_TOLERANCE` = 0.95 of them. That value is a stated, reversible choice and a
  data-validity check on the share read, not an adoption gate: by the tolerance's derivation a read that recovers
  the money is within it on every board whose figures the rule prints from the votes, and the slack leaves room for
  boards whose votes and figures disagree for reasons outside the read (`share_reproduced` counts those), while a
  read that fails structurally misses wherever its failure applies (the low figures on the tests' synthetic
  floored-favourite board miss it). Each score row carries its closing board's read (`reader`). The gains of the private report (day, meeting
  and fold clusters) hold the Place cases of the reads that stand only; the `undetermined` boards' cases are reported
  apart (`gains_undetermined`), and, when some read stands, so are every other read's (`gains_unvalidated`, by
  read). When no read stands, the determined reads stay in the gains and the report calls their Place rows a
  diagnostic. One registry record: `scored` when a read stands; `failed` with `params.unvalidated` when none does or
  the measurement cannot run.
- **Variants tried**, measured on the same validated boards:
  - `joint`, the board path wherever a board prints a high figure on every runner: `models.nowcast.place_money_fit`
    reads both figures by least squares at each figure's truncation midpoint, re-weighted by the order of the
    money. Figures at the floor are left out, since they only bound the payment.
  - `undetermined`, the same joint fit where the figures above the floor hold some runners' money only as a sum
    (for instance two favourites at the floor at both ends, whose swap prints the same board). An SVD rank test
    of the used equations' design, with relative tolerance `PLACE_RANK_RTOL` = 1e-9, flags it (a ruling on #410's
    review). The board keeps its minimum-norm shares and is still scored, not dropped. The measurement counts it
    under its own read, and it is a diagnostic whatever its measurement: it never stands as a result
    (`place_result_readers` holds `joint` and `low_only` only).
  - `low_only`, the owner's fallback for a board without high figures (and for a joint fit that leaves a runner
    without money): the low figures alone, read in closed form by the same rule (`stake_back_money`), with floored
    figures read at the floor. It is reported as the read used on its own boards, and as a counterfactual on every
    validated board.
  - Number of places P as scored (from the priced runners), and again from the registered slots (H1), the cheap
    check of 2-place mismatches (P from the starters at sale, or after scratches).
- **v3's measurement, superseded.** v3 (below) measured the rule on the `final_odds` boards
  (`processed/jravan/final_odds.parquet@daaea94abd5d223c` against `processed/jravan/pool_votes.parquet@cd743283ba0f094a`),
  record `fbadab006dbc59e4`; its figures are in the private report `nowcast-japan-4cb84e2f74e2.md`. #410's review
  found that it did not validate what the study scores: it dropped runners without a high figure, so P and the read
  could differ from the scored board's; it took every positive H1 count as final, including a day-before close; and
  an empty measurement would have registered as `scored`. Its reads also filtered the days after reading each
  batch, not inside the scan; the export predates the holdout start (2026-11-28), so no holdout row existed to
  reach it. v3's records stay in the registry and support no Place claim here.
- **Result, v4, superseded.** v4's record `place_rule_vs_votes` `8810ca538e68f8e4` is `scored` because boards read
  `joint` or `low_only` had final votes. #410's re-check found that this took coverage for agreement (a read could
  stand with none of its boards within the tolerance), that v4 measured every extracted Place race rather than the
  scored ones, and that its Place gains pooled the `undetermined` boards' cases with the rest. v4's records stay in
  the registry; its Place verdict is superseded by v5's (below) and supports no Place claim here. Its figures are in
  the private report `$HORSETRACK_DATA_DIR/reports/nowcast-japan-3e07f46516e3.md`.
- **Result, v5.** The measurement ran on the closing boards of the scored Place cases. The `joint` read stands as
  a result: it meets the criterion stated above (`place_result_readers`), so the record `place_rule_vs_votes` `86795d4a880bbd20` is `scored`. The
  private report reads the Place rows of the `joint` boards as a result, which meets DoD (b) for them, and its
  Place gains hold only those boards' cases. Every other read is a diagnostic, not a result: the `undetermined`
  boards' Place cases are reported apart (`gains_undetermined`), as would be those of any `low_only` board
  (`gains_unvalidated`). Per read and per P, the coverage, the reproduction, the agreement and the errors are in
  the private report `$HORSETRACK_DATA_DIR/reports/nowcast-japan-a07a5be8273d.md` and are withheld here (D21); this
  report does not assert the tolerance on every board. v3's and v4's Place verdicts stay superseded.
- **The v3 re-run.** Family `japan-c1-nowcast-final-shares-v3`, spec sha256
  `4cb84e2f74e25d5cc1de9ec471dd84912e0ec09b1c16cfba3d1dc07d387fad83`, frozen at the start of the run before any
  development row was read (the owner's 2026-11-27 reframe: Claude pre-registers and logs every variant). v3
  supersedes v2 and changes only the Place share rule (`place_shares`), the measurement (`place_rule_measurement`)
  and #116's drift draws on Place. The drift's draws are printed as JRA prints them (truncated, floored) and read by
  the same rule as the boards. Win, the folds, the grid, the scores, the bootstrap and the model constants are v2's.
  v2's freeze and records are untouched and are superseded for Place. Git `8eda1d0`. Data builds: v2's, plus the two
  above. Private report `$HORSETRACK_DATA_DIR/reports/nowcast-japan-4cb84e2f74e2.md`, with JSON beside it.
  Records, all `scored`: `place_rule_vs_votes` `fbadab006dbc59e4` (#410's measurement),
  `structural_vs_random_walk` `22024f6ca46f2c7a`, `structural_vs_drift_v0` `65711e9b1951b6c7`,
  `no_features_vs_random_walk` `b8b06b6f158ae3e0`, `structural_vs_no_features` `eb239abaf9ab6446` and
  `drift_v0_vs_random_walk` `4963fec9404d273d`. They are summarised in
  `docs/experiments/registry/japan/japan-c1-nowcast-final-shares-v3.md`, where every one shows as withheld. The
  drift's Place draws are read through the least-squares fit, which accounts for most of the extra time over v2. As
  in v1 and v2, the command exited 1 at
  `write_summary` on #131's family ids containing `/` (#344, #391), after the report and all six records were
  written. The summary was regenerated the same way as before: committed families plus v3, which added the v3 file
  and its index line only. No v3 trial was run. v3's Place rows are superseded by v4's (its measurement, above).
- **The v4 run.** Family `japan-c1-nowcast-final-shares-v4`, spec sha256
  `3e07f46516e38793a7f8f1f9fd544002aa308cb7338c05405e53724a4a3a85b7`, frozen at the start of the run before any
  development row was read. It is not a trial run. v4 supersedes v3 and changes the Place measurement (above). It
  also adds the `undetermined` flag to the Place read, which labels the read and leaves the shares unchanged. The
  shares are scored as before: they are the minimum-norm least squares the fit already returned. Win, the folds,
  the grid, the scores, the bootstrap, the model constants and the Place share values are v3's. The v3 and v2
  freezes and records are untouched. Git `6fea17e`. Data builds: `processed/jravan/odds_timeseries@0fd1381d81daf570`,
  `processed/jravan/races.parquet@8c0dfc18f1894406` and, for the measurement,
  `processed/jravan/pool_votes.parquet@cd743283ba0f094a`; `final_odds` is no longer read. Private report
  `$HORSETRACK_DATA_DIR/reports/nowcast-japan-3e07f46516e3.md`, with JSON beside it. Records, all `scored`:
  `place_rule_vs_votes` `8810ca538e68f8e4`, `structural_vs_random_walk` `64c20ebb0818ac84`,
  `structural_vs_drift_v0` `5207a155965fe7fe`, `no_features_vs_random_walk` `07716dc7d44cb386`,
  `structural_vs_no_features` `b9a8f7775b946344` and `drift_v0_vs_random_walk` `a3fa2ccf6119a778`. They are
  summarised in `docs/experiments/registry/japan/japan-c1-nowcast-final-shares-v4.md`, where every one shows as
  withheld. As before, the command exited 1 at `write_summary` on #131's family ids containing `/`, after the
  report and all six records were written. The summary was regenerated the same way as for v3 (committed families
  plus v4), which added the v4 file and its index line only. No v4 trial was run. Superseded by v5 for Place.
- **The v5 run.** Family `japan-c1-nowcast-final-shares-v5`, spec sha256
  `a07a5be8273d481d96742d73c06ec381e0ab00daa6dad95655c1dfd97a39cc7c`, frozen at the start of the run before any
  development row was read. It is not a trial run. v5 supersedes v4 and changes the Place verdict and its
  measurement (above): coverage and agreement are separate, the boards measured are the scored ones, each score row
  carries its closing board's read, and the result gains leave out the Place cases of reads that do not stand. The
  fit's rank is computed only where the read is named (`place_shares` and each case's closing board), never in
  the draws; the Place share values are unchanged. Win, the folds, the grid, the scores, the bootstrap, the model
  constants and the Place share values are v4's. The v4, v3 and v2 freezes and records are untouched. Git
  `0f0df42`. Data builds: `processed/jravan/odds_timeseries@0fd1381d81daf570`,
  `processed/jravan/races.parquet@8c0dfc18f1894406` and, for the measurement,
  `processed/jravan/pool_votes.parquet@cd743283ba0f094a`. Private report
  `$HORSETRACK_DATA_DIR/reports/nowcast-japan-a07a5be8273d.md`, with JSON beside it. Records, all `scored`:
  `place_rule_vs_votes` `86795d4a880bbd20`, `structural_vs_random_walk` `ea2511d8a40f5a3f`,
  `structural_vs_drift_v0` `c22f1d7b8bd8dc27`, `no_features_vs_random_walk` `65cdf0a13048b04b`,
  `structural_vs_no_features` `f5dcf23571130cd4` and `drift_v0_vs_random_walk` `b38178e1d3825eeb`. They are
  summarised in `docs/experiments/registry/japan/japan-c1-nowcast-final-shares-v5.md`, where every one shows as
  withheld. As before, the command exited 1 at `write_summary` on #131's family ids containing `/`, after the
  report and all six records were written. The summary was regenerated the same way (committed families plus v5),
  which added the v5 file and its index line only. No v5 trial was run.
- **The shadow adapter.** `ShadowNowcast` (#116's draws interface) now reads a Place board's high figures as
  `race_view` does, so its draws start from the shares the study scores (#410's review). It is not part of the
  study's scoring.
- **Wide.** `stake_back_money` and `stake_back_odds` now follow JRA's rule for every stake-back pool, so Wide
  (QUINELLA_PLACE) and `ShadowNowcast`'s Place and Wide draws use it too. Wide is not measured against money here (a
  follow-up), and no Wide case exists (above).
- **Alternatives rejected.** The v2 rule, which the measurement replaces. The low figures alone on boards that
  print high figures, since they fail on floored favourites on the synthetic boards of the tests. Reading the drift's draws unprinted, since a
  midpoint read of an untruncated figure is biased by half a step.
- **What this rules out.** v2's check failed under a rule that is not JRA's (the takeout off the pool), so that
  failure says nothing about JRA's boards; whether JRA's rule reads them as money is v5's measurement. Nothing is
  adopted (D16).
- **Open question.** Any boards the rule does not reproduce, the 2-place boards among them: final figures
  published from a different count of starters or votes than H1's are one candidate. The measurement reports P
  from the registered slots for that; it is not fixed here.
- **Follow-ups**, proposed for the orchestrator: Wide's inversion measured against Wide money; the boards the Place
  rule does not reproduce; the boards read `undetermined`.
