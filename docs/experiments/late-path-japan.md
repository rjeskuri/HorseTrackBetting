# Experiment report: the late-money path, E23 (ex post), E24 (ex ante) and E22 (Japan, #102)

Ticket #102 (epic #73; 08 §8 B9, §4 E22 to E24). **Nothing is adopted** (D16), and the holdout is not opened.
**Every figure is withheld (D21).** The run reads the JRA-VAN time-series odds paths in the store, a licensed
source. This report therefore says what ran, where its private report lives and which registry records hold it. It
gives no coefficient, standard error, interval, gain, p-value, count, span, coverage or sign derived from that data.
The paper's own figures, quoted below, are public.

- Registry family `japan-late-path-jravan`, spec sha256
  `89e7bb6d35c7b67749890f2c8b65ba3813632993058bd6150eadf6a981a2475c`. The spec was frozen when the full run started,
  before that run read any row. It holds the whole of `countries/japan/config/late_path.yaml` (the paper's versions,
  the baseline, E23's endings and windows, E24's features, E22's p_model and buckets, the bootstrap), the data
  rules, D2's grid, D9's fold scheme, D7's bootstrap and the name of every comparison.
- The full run: git `8c55bef`, `uv run horsetrack japan late-path-study` (source `jravan`, every development day).
  It exited 0, and every comparison scored; its 14 run ids are listed under "Variants tried".
- Data builds: `processed/jravan/odds_timeseries@0fd1381d81daf570`, `processed/jravan/races.parquet@8c0dfc18f1894406`,
  `processed/jravan/runners.parquet@714d13d1b6b6e5ed` and, for E22's p_model,
  `features/japan/rating-7ec8931213c6.parquet@7544cd84b8021f55` (#91's export).
- The full report is under the data directory at `$HORSETRACK_DATA_DIR/reports/late-path-japan-jravan-89e7bb6d35c7.md`,
  with its tables as JSON beside it (`late-path-japan-jravan-89e7bb6d35c7.json`). Both were written through
  `write_report` with the source `jravan`. The records are in
  `docs/experiments/registry/japan/japan-late-path-jravan.md`, and the trials' in `japan-late-path-jravan-trial.md`.
  Every one shows as withheld.
- Code: `countries/japan/studies/late_path.py` (the pure pieces, tested in `tests/countries/japan/test_late_path.py`)
  and `countries/japan/studies/late_path_study.py` (the run). To reproduce, run
  `uv run horsetrack japan late-path-study`; `--trial --end <day>` records its runs in the `-trial` family and
  freezes nothing.

### Trial runs, disclosed

Three `--trial` runs came before the freeze. Each read the development days through 2020-06-30, so each scored the
2020-Q1 and 2020-Q2 folds, which the full run scores again. Their records are in `japan-late-path-jravan-trial` with
`params.trial = true`.

1. Git `0323eae` (2026-11-26), spec `6cfc94f6dd50…`, the ticket's change to the last interim board only, with the
   two owner fields still pending. Records `2de0406d6e418dc8`, `9e76d8aee1cd76f0`, `56513542b12d9b6a` (scored) and
   `9c60edde7f384b3c` (E24, **failed**: a race with no WIN board at t raised instead of being counted). The fix,
   `707e99c`, counts such a case (`no_board_at_t` in the private report). Its private report was overwritten by the
   next trial, which wrote to the same file name.
2. Git `707e99c` (2026-11-26), the same spec. Records `22df3127ff2e0d62`, `3472dca2189a7f53`, `8ec10338c7a02ffd` and
   `5dc3a1f836a8eab0`, all scored; private report `late-path-japan-jravan-6cfc94f6dd50-trial.md`. Its summary step
   failed on #131's `/` family ids (then #344), after the records were written; #528 on main has since fixed it.
3. Git `7c2240c` (2026-11-28), spec `89e7bb6d35c7…`, the spec the full run froze, with the owner's answer in it.
   Fourteen records, all scored (`d4a65995cc33671d`, `f78178272420b962`, `aadbf0cce6844480`, `120197bb1cbda1ee`,
   `679fe9c362c6d98c`, `16d859a89bef7471`, `1d291e08be4be26b`, `9158e47479e359b2`, `072f1dec1f745761`,
   `f1736e0a4da2e0a1`, `d9caf85dab069095`, `298603ef1b9fde23`, `6580286551649460`, `65b1fe9e4a05dd03`); private
   report `late-path-japan-jravan-89e7bb6d35c7-trial.md`.

**The spec changed after trials had scored folds.** The two owner fields (E24's features, E22's p_model) were
approved as proposed, unchanged, and E23's `final` ending was added, all by the owner's answer on #102
(2026-11-28), after trials 1 and 2 had scored 2020-Q1 and 2020-Q2. The question that led to the `final` ending
gave the board spacing of the JRA-VAN series as its reason, and cited no result. After trial 3 the coder read its
private report to check that every E23 cell was populated, then changed only the order of the report's sections
(`8c55bef`); no spec field, record name or computation changed between trial 3 and the full run.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| E24's features | Exactly two: `path_change` = O_t / O_(t−5 min) − 1 and `c1_drift` = C1's out-of-fold predicted drift E[s_t / s_final] − 1, at the D2 grid with T-2 primary, nothing else. | #102, owner answer of 2026-11-28 |
| E22's p_model | C2's dynamic-rating out-of-fold ln p (#91), blended as softmax(α ln p_model + β ln q_τ) per D2 bucket. | #102, owner answer of 2026-11-28 |
| E23's endings | The late change to the last interim board before the close (primary), and to the final odds, as the paper does; at 5 minutes (primary), 10 and 20. | #102, owner answer of 2026-11-28 |
| Training data | The JRA-VAN time-series paths in the store; a snapshot counts at its stamped minute, with no lag. | #79; #85 owner comment of 2026-11-26 |
| D2 | The grid T-60, T-30, T-16, T-5, T-2, T-1, with T-2 primary (`config/decision.yaml`). | `docs/problem-statement.md`, #14 |
| D4 | The final-price baseline is the calibrated market (spline in ln q, 5 knots), fitted in-fold only, here on the path source's own final WIN board. | §2, #14 |
| D7 | One-sided 95% bounds of the gains, resampling race days (the ticket) and meetings (D7), 10,000 replicates (`config/scoring.yaml`, seed 102). | §3, #14 |
| D9 | Quarter folds, quarterly refit, an expanding window, 22 quarters of minimum training and a 1-week embargo (`config/folds.yaml`); the first scored quarter is 2020-Q1. | §4, #14; the #70 answer |
| D10 | No race day before 2014-06-07 is read. | §4, #14 |
| D11-D15 | Holdout A2 from 2026-11-28, fully locked: a holdout day's file is left out by its name (`is_holdout_day`) before any read. The holdout is not opened. | §5, #14; the #71 answer |
| D16 | Nothing is adopted; every comparison is a row on the leaderboard. | §6, #14; epic #447 |
| D21 | No figure derived from JRA-VAN enters the repository; every record's `sources` is `jravan`, so the summary withholds it. | §8, #14; owner 2026-11-26; #438 |

## Data

- **Paths.** `processed/jravan/odds_timeseries/<day>.parquet` (`late_path.path_files`), WIN rows only and by column,
  one day file in memory at a time. Holdout days are skipped by file name and never read; days before 2014-06-07
  are not listed.
- **Starters and winners.** A starter is a WIN runner priced on its final capture. A winner is a starter whose final
  row carries a WIN dividend that is not a refund; a dead heat has two, each counted at 1/k (#69). A race without a
  winner is void and dropped, counted.
- **The close** is the race's scheduled post time (a Ruling: every interim board of the series is printed before it,
  and the closing board, equal to the final odds, at or after it). The private report counts, by ending, window and
  year, the starters with a late change and those whose change is not 0.
- **q_final**, the baseline of every offset test, is D4's calibrated market fitted on each D9 fold's training days
  and predicting its test days, so only test days are scored. E23's regression, which has no baseline, reads every
  development day.
- **E24's inputs at t.** Each fold's C1 (`models.nowcast`, the #85 model with features) is fitted on that fold's
  training days and never on a test day (`ex_ante_features` refuses a nowcast fitted on the race's day or later).
  The board at t = post − τ takes each runner's latest row at or before t.
- **E22's p_model** is #91's out-of-fold rating rows, joined to the path source's runners by horse id and race; a race
  with any runner lacking one is left out, counted.

## Hypothesis

The frozen spec reads:

- **E23 (ex post).** "Given the final odds, a runner whose odds shortened late returns more at those odds
  (δ < 0), and ln q_final + late_change predicts the winner better than ln q_final." The regression is the paper's
  base spec: OLS of won × odds_final on 1, odds_final and the late change, with race-, day- and meeting-clustered
  sandwich SEs and a 95% two-sided Wald interval. The definition of done compares δ with the paper's range of −0.34
  to −0.42. The paper is arXiv 2509.14645 (Hanyu, Ishii, Otani & Teramoto, JRA 2004-2023): **v1** δ = −0.3559 (SE
  0.0381); **v3 of 2026-08-20** δ = −0.3386 (SE 0.0392), and −0.4201 with full controls. The paper's change runs to
  the final odds; the ticket's primary change runs to the last interim board, so the `final` ending is the paper's
  quantity and the `interim` ending the one known before the close.
- **E24 (ex ante).** "The observed path up to t and C1's predicted late drift predict the winner beyond the final
  price (expected about 0, 'very limited' in the paper)." The null, a gain of 0 at every τ, is the starting point.
- **E22.** "The weight α on the model in softmax(α ln p_model + β ln q_τ) falls toward the post, and a τ-indexed
  blend predicts better than one fixed blend" (Giacomini-White against one blend fitted on every bucket pooled).
- **What would show each wrong.** E23: a δ interval that excludes the paper's range, or an offset gain whose
  one-sided lower bound does not lie above 0. E24: a one-sided lower bound above 0 would contradict the expected
  null. E22: no Giacomini-White rejection, and no fall of α toward the post.

## Variants tried

One registry record per comparison, 14 in all, every one scored. E23's records hold one (ending, window) each; E24's
record holds a metric per τ; E22's a metric per bucket. All figures are withheld.

| Comparison | E | Ending, window | Split | Record |
| --- | --- | --- | --- | --- |
| `e23:paper_regression:late_change_to_interim_w5` (**headline**) | E23 | interim, 5 min | every development day, in sample | `eb010ea15a52adc7` |
| `e23:paper_regression:late_change_to_interim_w10` | E23 | interim, 10 min | every development day | `17c246c57fa78602` |
| `e23:paper_regression:late_change_to_interim_w20` | E23 | interim, 20 min | every development day | `9e771b0f2d56893b` |
| `e23:paper_regression:late_change_to_final_w5` | E23 | final, 5 min | every development day | `ea7c5cbee2618958` |
| `e23:paper_regression:late_change_to_final_w10` | E23 | final, 10 min | every development day | `615525d3121263a6` |
| `e23:paper_regression:late_change_to_final_w20` | E23 | final, 20 min | every development day | `3116f0153a661780` |
| `e23:offset:late_change_to_interim_w5_vs_final` | E23 | interim, 5 min | D9 test quarters, walk-forward | `719b7c7f89e5a1aa` |
| `e23:offset:late_change_to_interim_w10_vs_final` | E23 | interim, 10 min | D9 test quarters, walk-forward | `e26145dc8bdd18fb` |
| `e23:offset:late_change_to_interim_w20_vs_final` | E23 | interim, 20 min | D9 test quarters, walk-forward | `b75f02b21028a05b` |
| `e23:offset:late_change_to_final_w5_vs_final` | E23 | final, 5 min | D9 test quarters, walk-forward | `5c3ed9d8c795c446` |
| `e23:offset:late_change_to_final_w10_vs_final` | E23 | final, 10 min | D9 test quarters, walk-forward | `b90cc37b7fccb409` |
| `e23:offset:late_change_to_final_w20_vs_final` | E23 | final, 20 min | D9 test quarters, walk-forward | `35e0bc790b7b6bde` |
| `e24:offset:ex_ante_vs_final` (T-60 to T-1, T-2 primary) | E24 | features at t | D9 test quarters, walk-forward | `b1a41241487f1ebe` |
| `e22:alpha_by_tau_vs_fixed_blend` (per D2 bucket) | E22 | — | D9 test quarters' races, cross-fitted over 5 race groups | `1c7b17ae58801ec9` |

The private report holds, for every regression, δ with its race-, day- and meeting-clustered SE and 95% interval
and whether that interval meets the paper's range (`overlaps_paper_range`); for every offset test, the per-race
gain in nats with one-sided day and meeting bounds and the in-sample path coefficients; for E22, α and β per
bucket with their SEs, the fixed blend and the Giacomini-White statistic and p-value; and the counts behind each.
The offset tests fit their path coefficients walk-forward over the scored quarters (each quarter scored with
coefficients fitted on the earlier scored quarters only; the first scored quarter only trains). The trial records
(above) are variants of the same comparisons on a subset of days and are not evidence.

## Failures

- No comparison of the full run failed.
- Trial 1's E24 failed (above); the fix counts a race with no WIN board at t instead of failing the comparison.
- Trial 2's summary step failed on another ticket's family ids (#344, fixed by #528); its records were written first.

## Alternatives rejected and why

- **The paper's full-controls regression** (v3's −0.4201): not run. Only the base spec is pre-registered, and its
  controls are not in the store as the paper defines them. Proposed as a follow-up.
- **More path features for E24** (other windows, velocities, ranks): not run. The owner approved exactly two
  features; searching over path features is the risk 08 §4 names.
- **The final ending read from `odds_final` directly:** not used. `late_change(to="final")` takes each runner's last
  capture at or after the close, so a final board stamped before the scheduled post gives no change rather than
  folding an interim board into the "final" ending; the coverage table counts what each ending has.
- **The close at the actual off:** the series carries no off time, so the scheduled post time is the close (above).
- **Our own logged days (`--source jra-logs`)** as the path source: supported by the code but not run. Too few
  development days are logged to train a fold, and they would repeat JRA-VAN days.
- **Another p_model for E22** (C4's meta-model, #92's ability): not used; the owner approved #91's rating.
- **One E24 record per τ:** not done. E24 is one pre-registered comparison on D2's grid with T-2 primary, as the
  spec named it before the owner's answer; its τs are metrics of that record.

## What this rules out

The figures are withheld, so the repository states no size or direction of δ, of any offset gain, of E24's result or
of E22's weights, and nothing here supports adoption: D16 needs a holdout bound, and the holdout is not opened. The
definition of done's comparison (δ with its interval against −0.34 to −0.42, citing v1 and v3 above) and the ex-ante
result, null or not, are in the private report's "Headline" and E23, E24 and E22 tables. As a measurement, each
comparison now has a recorded result on every development day (E23's regression) or every scored quarter (the offset
tests and E22), so a later claim about the late-money path on JRA-VAN history must cite these records.

## Leaderboard rows

No comparison measures a return per ¥100 staked, so every row is unmeasured and unranked (D16), and every row is
built on JRA-VAN, so it is withheld (D21) and listed by option id only. This PR does not rebuild
`docs/leaderboard/japan.md`; the next `horsetrack japan leaderboard --public` lists these ids among the withheld rows.

- `japan-late-path-jravan/e23:paper_regression:late_change_to_interim_w5`
- `japan-late-path-jravan/e23:paper_regression:late_change_to_interim_w10`
- `japan-late-path-jravan/e23:paper_regression:late_change_to_interim_w20`
- `japan-late-path-jravan/e23:paper_regression:late_change_to_final_w5`
- `japan-late-path-jravan/e23:paper_regression:late_change_to_final_w10`
- `japan-late-path-jravan/e23:paper_regression:late_change_to_final_w20`
- `japan-late-path-jravan/e23:offset:late_change_to_interim_w5_vs_final`
- `japan-late-path-jravan/e23:offset:late_change_to_interim_w10_vs_final`
- `japan-late-path-jravan/e23:offset:late_change_to_interim_w20_vs_final`
- `japan-late-path-jravan/e23:offset:late_change_to_final_w5_vs_final`
- `japan-late-path-jravan/e23:offset:late_change_to_final_w10_vs_final`
- `japan-late-path-jravan/e23:offset:late_change_to_final_w20_vs_final`
- `japan-late-path-jravan/e24:offset:ex_ante_vs_final`
- `japan-late-path-jravan/e22:alpha_by_tau_vs_fixed_blend`
- The trial family `japan-late-path-jravan-trial` gives the same 14 option ids under its own family id, plus the
  first two trials' `e23:paper_regression`, `e23:offset:late_change_vs_final`, `e24:offset:ex_ante_vs_final` and
  `e22:alpha_by_tau_vs_fixed_blend`.

## Follow-ups filed

None by this PR. Proposed, for the orchestrator to file under epic #73 as `status:proposed`:

- E23 with the paper's full controls, pre-registered as a new family, to compare with v3's −0.4201.
- E23 and E24 on our own logged boards (`--source jra-logs`, captures every ~30 s), where the last interim board sits
  close to the post, once enough development days are logged to train a fold.
- Whether E24's two features enter the C4 meta-model (#87) as a member at the decision time: an owner decision on
  feature inclusion; the holdout checks the ranking (D15 as amended, epic #447).
