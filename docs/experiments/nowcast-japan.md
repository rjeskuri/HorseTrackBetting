# Experiment report: the C1 final-price nowcast against the random walk and #116's drift (Japan, #85)

Ticket #85 (epic #73; 08 §8 B2, §3.2 C1, §4 E19). **Nothing is adopted** (D16), and the holdout is not opened.
**Every figure is withheld (D21).** The run trains and scores on the JRA-VAN time-series odds paths in the store,
which are a licensed source. This report therefore says what ran, where its report lives and which registry records
hold it. It gives no count, span, score, rate, sign or bound derived from that data. The figures for our own logged
days (Wide coverage, below) are not JRA-VAN data and are given.

- Registry family `japan-c1-nowcast-final-shares`, spec sha256
  `c652ebf9d5da87478f3281ab66e8db9f960b5c0ce326af05019f0a345d41a0dd`. The spec was frozen when the full run
  started, before that run read any row. It holds the hypothesis, the null, the data rules, the variants, the
  comparisons, the scores, the D2 grid, D9's fold scheme, D7's bootstrap and the model constants. Two runs with
  `--trial` came first. They wrote no records and read days up to 2020-03-31 and 2020-06-30, so they scored the
  2020-Q1 and 2020-Q2 folds. They sized memory and run time, and they showed the Place rule's check failing
  (below). After them the spec gained its `place_shares` clause, which makes Place a diagnostic when that check
  fails. The code then changed only to save memory: fits take a day list instead of a copy of the cases, and
  Arrow's pool is released after each day. No model, constant or score changed after the trial scores were read.
- Code at `819b99b`: `core/eval/distributional.py` (CRPS, PIT, Gaussian log score), `countries/japan/models/nowcast.py`
  (the model, tested in `tests/countries/japan/test_nowcast.py` and `tests/core/eval/test_distributional.py`) and
  `countries/japan/studies/nowcast.py` (the run). To reproduce, run `uv run horsetrack japan nowcast-study`. It
  scores every test day with 200 draws per case. On this 16 GB Mac it took about 20 minutes, with a peak
  resident memory of about 2.3 GB, one day file in memory at a time. `--test-days-per-fold N` scores a seeded sample
  of test days. `--trial` writes no records.
- Data builds: `processed/jravan/odds_timeseries@0fd1381d81daf570`, `processed/jravan/races.parquet@8c0dfc18f1894406`.
- The full report is under the data directory at
  `$HORSETRACK_DATA_DIR/reports/nowcast-japan-c652ebf9d5da.md`, with its tables as JSON beside it
  (`nowcast-japan-c652ebf9d5da.json`). Both were written through `write_report` with the source `jravan`. The
  records are in `docs/experiments/registry/japan/japan-c1-nowcast-final-shares.md`, where every one shows as
  withheld.

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
- **Cases.** A case is one (race, pool, τ). It is dropped when its combinations change between t and the close (a
  late scratch). It is also dropped when there is no board at t, no final board or no post time. The private report
  counts each kind of drop.
- **Grades** are the jravan races table's `race_class`, and they enter only the pool-growth profile.
- **Place shares.** The series carries no per-runner Place money, so Place shares invert the low figures in closed
  form by the stake-back-share rule of `config/jra.yaml`. Where the rule holds, the inverted money totals 1 in units
  of the pool. The run reports that total on every development board, and **the check does not hold on JRA-VAN
  boards** (the private report has the figures). Place shares from the low figure are therefore not money shares
  there, and every Place row of the run is a diagnostic, not evidence. In the ticket's words, Win is shipped. Before
  the frozen run, the high figures were also inverted by the same rule and did not total 1 either, so neither end of
  the Place range follows the simple rule. The final per-runner votes (H1, #368) were checked against the final WIN
  board on one day and did not reconcile with it exactly, so they were not used as Place money. That check was an
  exploration and is not part of the frozen run.
- **Wide.** The JRA-VAN series has no Wide pool. Our own logged days (`processed/jra/odds_timeseries`) are
  2026-11-21 and 2026-11-22, each with 12 Wide races, and neither file has a post time (`post_utc` is null, as the
  logger writes no race-card sidecar for them). So no Wide board at t can be placed, Wide is neither fitted nor
  scored, and the model's Wide path is exercised only by its code. Both days also fall in the last scored quarter,
  which no fold trains on. Our own WIN and PLACE days duplicate JRA-VAN days and are not read.
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
race-day and meeting bounds, and the per-fold gains at T-2. All figures are withheld.

| Comparison | Record |
| --- | --- |
| `structural_vs_random_walk` (E19, primary) | `e8c261a0882090ba` |
| `structural_vs_drift_v0` (D8's price baseline) | `f90461967e915071` |
| `no_features_vs_random_walk` | `d7f6cc3bb7e0d2a2` |
| `structural_vs_no_features` (what the features add) | `28b378db8473dd2f` |
| `drift_v0_vs_random_walk` | `13ab926148e96f76` |

The private report also holds, per forecaster, pool and τ, the PIT deciles and the 50% and 90% coverage. It holds
σ_t, σ_rw and β per pool and τ in every fold, the pool-growth profile against the observed V_t / V_final, the board
age at t and the Place rule's check.

## Failures

- Place: the stake-back inversion of the low figure fails its money-total check on JRA-VAN boards (see Data), so
  Place is a diagnostic. Only Win stands as a result.
- Wide: no case can be built (see Data). This is a coverage failure, not a modelling result.
- The command exited 1 at its last step. The report and all five records were written first. `write_summary` then
  raised on records of another ticket in the shared registry: #131's family ids (`japan/131-exacta-category-v1`
  and others) contain `/`, which cannot be a summary file name. The registry verifies and was not altered. The
  committed summary was regenerated by `write_summary` itself, over the registry restricted to the families already
  committed plus this one. That changed only `japan-c1-nowcast-final-shares.md` and one line of the index. No
  record failed, and every variant scored on every fold.

## Alternatives rejected and why

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
nowcast built on those shares forecasts a quantity that is not money over the pool.

## Follow-ups filed

None by this PR. Proposed, for the orchestrator to file under epic #73 as `status:proposed`:

- Place shares that are money shares: JRA's exact Place odds rule, or per-runner Place money where a source carries
  it at t. Until then, Place stays a diagnostic.
- A Wide nowcast on our own logged days, once they are compacted with post times (a race-card sidecar per venue)
  and enough development days have accrued to train a fold.
- Near-post boards: score T-2 and T-1 on our own logs, which capture every ~30 s, once they cover enough days. On
  5-6 minute paths those horizons mostly see an older board.
- Pre-registered model variants: σ by odds band (where #116's drift has its cells), and a random V_final.
