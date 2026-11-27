# Experiment report: expected top-3 time against the par-only baseline (Japan, #305, C3)

Ticket #305 (epic #73; 08 §3.2 C3, §8 B6, the standalone time test of E39). #96 built C3's race-level speed model
(`countries/japan/models/speed.py`). A race's top-3 time is the par of its cell, plus a day variant for its (day,
track, surface), plus noise. This study runs that model on #70's development folds. It asks whether the same-day
variant, read only from races resulted before the decision time, predicts a race's top-3 time better than the par
alone.

**Nothing is adopted (D16), and no bet is placed.** This is a time test. It says nothing yet about win probabilities
or the final price: that is the marginal test of E39 (#97, then C4).

Every figure here is an aggregate over the JRA results store (netkeiba race results: finishing positions and finish
times). No JRDB, no JRA-VAN, and no race row appears.

- Registry family `japan-speed-top3-c3`, spec sha256
  `72f5c802e96e40850b46fc9b85a0544f4f53a3b12255ea2cf3fae767f02c115f`. The spec was frozen before any outcome was
  read. It holds:
  - the hypotheses, the estimand and the race exclusions;
  - the availability rule, the model's keys and error models, and the decision time;
  - the fold scheme, the metric, the cells and margins, the bootstrap, and the owner decisions.
- Code at `2ab5bf4` (`countries/japan/studies/speed_run.py`, with #96's `models/speed.py` unchanged). To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.speed_run --run`.
- Store: `processed/jra/races.parquet@9cce511831074b64` and `processed/jra/runners.parquet@6f3cec95347e0132`, read
  column-scoped.
- The full report, with every cell of every variant, is under the data directory at
  `reports/speed-japan-72f5c802e96e.md`.
- This family's records are in `docs/experiments/registry/japan/japan-speed-top3-c3.md`.
  - The file is the one `write_summary` renders for this family, which since #334 writes one file per family.
- Run ids:

  | Variant | Run id |
  | --- | --- |
  | `t5:variant_on_vs_par_only` | `200c21323add835f4f1d28cb18feac561542a6ecab1d40fc1f708a1249c8decc` |
  | `t5:variant_off_vs_par_only` | `5ad32528cb783cb3c3d6b6083f4c17a575c335b0342d449b081757d39e421f6d` |
  | `normal:variant_on_vs_par_only` | `6076c40fd2fd4bdb1243956a2d6779116d07636fddc447ee21b03bd1f361a327` |
  | `normal:variant_off_vs_par_only` | `fa03b5a34bdcc6640d57663b4fcc0790f3acde92678bd0902caa20edb56665c8` |

- The whole run (both error models, 180 fits, every bootstrap) took 100 s at a peak RSS of 468 MB.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| Estimand | `top3_time_s` is the mean finish time of the runners placed 1–3, dead heats averaged. A race with no 3rd-placed runner uses the placed runners present, and is counted. Read per runner: each runner placed 1 or 2 counts once, and the runners placed 3rd share one slot at their mean. | #96 (2026-11-25); the dead-heat reading on #305 (2026-11-25) |
| Jump races | Excluded from the speed model, and counted. | #96 (2026-11-25) |
| Availability | A same-day result is available at the race's post_utc + 30 min, the same-day dividends rule. The variant at t reads only results available strictly before t. The as-of layer carries the same rule for the finish (races `finish_time_s`; runners `finish_pos`, `dead_heat`, `finish_time_s`), added by this ticket; the study applies it itself as `result_available_utc`, and the two agree on the instant (the view shows a finish from post + 30 min on; the study, reading strictly before t, is the stricter at that one instant). | #96 and #80 (2026-11-24/25); `config/availability.yaml` and `docs/specs/as-of.md`: the dividends #227, the finish #305 |
| D2 | The decision time is T-2, the primary point of the grid (`config/decision.yaml`). | `docs/problem-statement.md` §1, #14 |
| D7 | Bounds are meeting-clustered (venue x day), 95% one-sided (`config/scoring.yaml`; 10,000 replicates, seed 305). | §3, #14 |
| D9 | #70's quarterly expanding walk-forward folds, `min_train` 22 quarters, 1-week embargo (`config/folds.yaml`). | §4, #14; #70 |
| D10 | No price is read, so rows before 2014-06-07 may train. The folds start from the store's first day. | §4, #14 |
| D11–D15 | Holdout A2 from 2026-11-28 is fully locked. Every holdout day is dropped before the folds, and the holdout was not opened. The store ends 2026-11-23, so none was present. | §5, #14; #71 |
| D16 | Nothing is adopted. | #14 |
| D21 | No figure derived from a licensed source appears in the repository. None is read here. | #14 |

## Hypothesis

The null is the starting point: **the same-day variant does not lower the time RMSE of E[top-3 time] below the par
of the race's cell** (gain = RMSE(par only) − RMSE(variant on) ≤ 0).

- **E39, standalone time test.** With the same-day variant read from the races of the same day, track and surface
  that were resulted before t, E[top-3 time] has a lower time RMSE than the par alone on development test folds. The
  claim rests on the one-sided lower bound of the overall gain being above 0.
- **Check.** Without an injected prior, the variant off is the prior N(0, tau²), whose mean is 0, so E[top-3 time] is
  the par and the gain is exactly 0. The `variant_off` records confirm the pipeline, not a hypothesis.
- **The prior from 07 §5.3** concerns the market: speed figures add nothing over the final price. A time gain here
  does not test it.

## The race table

`build_race_table` reads only these columns:
- races: race id, meeting, race day, venue, surface, distance, course layout, rail, class and post time;
- runners: race id, finishing position and finish time.

The exclusions apply in order, and each is counted.

| Count | n |
| --- | --- |
| Races in the store (2010-01-05 to 2026-11-23) | 58,406 |
| Jump races, excluded | 2,139 |
| No post time, excluded | 0 |
| No runner placed 1–3 with a finish time, excluded | 0 |
| Kept | 56,267 |
| Of those, partly placed (no 3rd-placed runner) | 110 |
| Of those, missing timing (a runner placed 1–3 has no finish time; #416) | 0 |
| Of those, short orders (fewer than three runners placed 1–3; #416) | 0 |
| Of those, ties (a place in 1–3 shared; #416) | 335 |

Every race kept has a post time and a timed top 3. In the copied store `rail_setting` is empty and `course_config`
is null on most races, so the rail key is `unknown` on every race and the layout key on 88% (see Follow-ups).
The run predates #253, which now fills `rail_setting` in the store build from the cached JRA going pages; historical
rail still needs #90, and the rerun by rail and layout is #329.

## Variants tried

The folds are the 45 quarterly test windows from 2015-Q4 to 2026-Q4 (2026-Q4 runs to the store's last day,
2026-11-23).
- The first window opens on the first quarter boundary at least 22 quarters after the store's first day. Each window
  trains on every development race up to one week before it.
- Of the 37,147 test races, 37,047 were scored over 3,215 meetings.
- 100 test races were not scored: the fold's fit had never seen their cell, so they have no par.

`variant_on` and `variant_off` use the same fit and the same races as `par_only`, so every comparison is paired.

Two error models were run. `t5` is #96's Student-t error at 5 degrees of freedom, the df of #96's tests; the df is not
signed (see Open questions). `normal` is the normal error, run beside it. No variant failed.

| Variant | Races | Meetings | RMSE model (s) | RMSE par only (s) | Gain (s) | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `t5:variant_on_vs_par_only` | 37,047 | 3,215 | 1.009 | 1.242 | **+0.233** | +0.224 | +0.242 |
| `t5:variant_off_vs_par_only` | 37,047 | 3,215 | 1.242 | 1.242 | 0 | 0 | 0 |
| `normal:variant_on_vs_par_only` | 37,047 | 3,215 | 1.012 | 1.256 | **+0.245** | +0.234 | +0.255 |
| `normal:variant_off_vs_par_only` | 37,047 | 3,215 | 1.256 | 1.256 | 0 | 0 | 0 |

How to read the table:
- Gain = RMSE(par only) − RMSE(variant on). It is positive when the variant helps.
- lo and hi are the (1 − level) and level quantiles of 10,000 meeting-clustered paired replicates (D7, one-sided). Each
  replicate recomputes both RMSEs over the races of the drawn meetings.
- The overall gain is positive in each of the 45 folds under both error models. Per-fold gains run from +0.089 to
  +0.386 s (t5) and +0.095 to +0.462 s (normal).
- The fitted race noise is sigma 0.62–0.63 s across folds (t5). The day variant's SD is tau 0.77–0.85 s, larger than
  the race noise.

### By margin (t5, variant on against par only)

The margins below pool the cells. The normal error gives the same picture. On every margin row with 100 or more races,
its gain is within 0.09 s of the t5 gain. The larger differences, up to 0.13 s, are on rows of 40 races or fewer.
Every margin row of both models is in the full report.

| Surface | Races | Meetings | RMSE model | RMSE par | Gain | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dirt | 18,506 | 3,215 | 0.841 | 1.013 | +0.172 | +0.163 | +0.181 |
| turf | 18,541 | 3,214 | 1.154 | 1.435 | +0.282 | +0.267 | +0.297 |

| Track | Races | Meetings | RMSE model | RMSE par | Gain | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CKY | 3,845 | 335 | 1.054 | 1.372 | +0.318 | +0.294 | +0.342 |
| FKS | 2,397 | 218 | 0.944 | 1.180 | +0.236 | +0.202 | +0.270 |
| HAK | 1,580 | 132 | 0.955 | 1.268 | +0.313 | +0.271 | +0.352 |
| HSN | 5,411 | 466 | 0.977 | 1.148 | +0.171 | +0.156 | +0.187 |
| KOK | 2,747 | 246 | 0.965 | 1.252 | +0.288 | +0.250 | +0.326 |
| KYO | 4,726 | 404 | 1.047 | 1.311 | +0.263 | +0.236 | +0.291 |
| NAK | 5,333 | 461 | 1.022 | 1.223 | +0.201 | +0.179 | +0.224 |
| NII | 3,291 | 298 | 0.985 | 1.123 | +0.138 | +0.112 | +0.162 |
| SAP | 1,744 | 146 | 0.959 | 1.228 | +0.269 | +0.232 | +0.306 |
| TOK | 5,973 | 509 | 1.054 | 1.277 | +0.224 | +0.194 | +0.255 |

| Class | Races | Meetings | RMSE model | RMSE par | Gain | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MAIDEN | 12,810 | 3,215 | 1.111 | 1.216 | +0.105 | +0.098 | +0.111 |
| DEBUT | 3,336 | 2,220 | 1.176 | 1.377 | +0.201 | +0.184 | +0.219 |
| C1 | 10,814 | 3,214 | 0.886 | 1.201 | +0.315 | +0.300 | +0.330 |
| C2 | 5,076 | 3,158 | 0.906 | 1.230 | +0.324 | +0.302 | +0.344 |
| C3 | 2,225 | 2,090 | 0.851 | 1.213 | +0.362 | +0.327 | +0.399 |
| OP | 915 | 861 | 0.973 | 1.194 | +0.221 | +0.172 | +0.269 |
| L | 448 | 414 | 1.025 | 1.316 | +0.291 | +0.229 | +0.354 |
| G3 | 753 | 753 | 0.986 | 1.374 | +0.388 | +0.332 | +0.443 |
| G2 | 406 | 406 | 1.162 | 1.602 | +0.440 | +0.358 | +0.523 |
| G1 | 264 | 264 | 1.281 | 1.771 | +0.490 | +0.360 | +0.608 |

| Results seen before t | Races | Meetings | RMSE model | RMSE par | Gain | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9,517 | 3,215 | 1.158 | 1.158 | 0 | 0 | 0 |
| 1 | 5,364 | 2,857 | 1.084 | 1.352 | +0.268 | +0.253 | +0.284 |
| 2 | 5,941 | 2,977 | 1.017 | 1.311 | +0.295 | +0.278 | +0.311 |
| 3 | 5,709 | 3,105 | 0.938 | 1.268 | +0.330 | +0.310 | +0.349 |
| 4 | 5,279 | 3,055 | 0.858 | 1.215 | +0.357 | +0.335 | +0.378 |
| 5 | 3,343 | 2,687 | 0.858 | 1.211 | +0.353 | +0.327 | +0.379 |
| 6 or more | 1,894 | 1,661 | 0.791 | 1.158 | +0.367 | +0.328 | +0.408 |

"Results seen" is a diagnostic. It counts the races of the target's (day, track, surface) with a result available
strictly before its decision time.
- Where none is available, the variant is its prior and the gain is 0 by construction. That holds for 26% of scored
  races: the first one or two races of each surface each day, since a result arrives 30 minutes after the post and
  JRA posts are about 30–35 minutes apart.
- Maiden races tend to run early in the card, when fewer results are in. The by-class gains are not adjusted for
  that, so MAIDEN's small gain need not mean the variant fits maidens worse.

| Layout | Races | Meetings | RMSE model | RMSE par | Gain | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outer | 4,478 | 1,624 | 1.154 | 1.407 | +0.253 | +0.226 | +0.281 |
| inner | 11 | 11 | 2.044 | 2.098 | +0.054 | −0.283 | +0.258 |
| unknown | 32,558 | 3,215 | 0.987 | 1.218 | +0.230 | +0.221 | +0.240 |

By distance, every distance with 40 or more scored races has a lower bound above 0.
- The gain grows with distance, from +0.053 s at 1,000 m (lo +0.026) to +0.50 s at 2,600 m (lo +0.43).
- The four distances with 20 or fewer races (2,300, 3,200, 3,400 and 3,600 m) have bounds spanning 0.
- The full distance table is in the full report.

### By cell (track × surface × distance × layout × rail × class)

There are 618 scored cells. The full table, with each cell's races, meetings, both RMSEs, gain and bounds for all
four variants, is in the full report (`reports/speed-japan-72f5c802e96e.md`, under the data directory). A cell has a
median of 30 scored races (mean 60, at most 661).

| t5, variant on | Cells | lo > 0 | hi < 0 | Bounds span 0 |
| --- | --- | --- | --- | --- |
| All cells | 618 | 418 | 14 | 186 |
| Cells with 100+ races (20,770 races) | 103 | 87 | 0 | 16 |

What the cell counts mean:
- The 14 cells with hi < 0 each have 1 to 91 races. There are 618 one-sided bounds and no multiplicity control. If
  no cell had a true gain, chance alone would put about 31 cells below 0, so 14 is no evidence of a real loss in any
  cell.
- The normal error gives 417, 15 and 186.
- The 20 largest cells, t5, variant on:

| Track | Surface | Distance | Layout | Class | Races | Meetings | RMSE model | RMSE par | Gain | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NAK | dirt | 1800 | unknown | MAIDEN | 661 | 424 | 1.164 | 1.321 | +0.157 | +0.127 | +0.188 |
| NAK | dirt | 1200 | unknown | MAIDEN | 568 | 432 | 0.631 | 0.697 | +0.066 | +0.046 | +0.087 |
| HSN | dirt | 1800 | unknown | MAIDEN | 552 | 437 | 1.196 | 1.270 | +0.074 | +0.051 | +0.098 |
| TOK | dirt | 1600 | unknown | MAIDEN | 457 | 411 | 1.088 | 1.194 | +0.106 | +0.075 | +0.138 |
| NAK | dirt | 1800 | unknown | C1 | 399 | 348 | 0.807 | 1.090 | +0.282 | +0.225 | +0.340 |
| NAK | dirt | 1200 | unknown | C1 | 382 | 323 | 0.577 | 0.617 | +0.041 | −0.007 | +0.088 |
| KYO | dirt | 1800 | unknown | MAIDEN | 375 | 335 | 1.141 | 1.173 | +0.031 | +0.015 | +0.050 |
| TOK | dirt | 1600 | unknown | C1 | 372 | 342 | 0.637 | 0.986 | +0.349 | +0.284 | +0.413 |
| TOK | dirt | 1400 | unknown | MAIDEN | 371 | 366 | 0.932 | 1.020 | +0.088 | +0.053 | +0.127 |
| HSN | dirt | 1800 | unknown | C1 | 359 | 321 | 0.816 | 1.172 | +0.357 | +0.303 | +0.412 |
| NII | dirt | 1800 | unknown | MAIDEN | 342 | 265 | 1.027 | 1.054 | +0.026 | +0.002 | +0.051 |
| HSN | dirt | 1400 | unknown | MAIDEN | 326 | 317 | 0.735 | 0.791 | +0.056 | +0.029 | +0.084 |
| CKY | dirt | 1800 | unknown | C1 | 313 | 263 | 0.811 | 1.157 | +0.346 | +0.279 | +0.415 |
| TOK | dirt | 1400 | unknown | C1 | 308 | 294 | 0.573 | 0.917 | +0.344 | +0.273 | +0.416 |
| KYO | dirt | 1800 | unknown | C1 | 307 | 274 | 0.874 | 1.218 | +0.344 | +0.283 | +0.405 |
| NII | dirt | 1800 | unknown | C1 | 306 | 246 | 0.870 | 0.998 | +0.128 | +0.081 | +0.176 |
| CKY | dirt | 1800 | unknown | MAIDEN | 305 | 268 | 1.212 | 1.296 | +0.084 | +0.052 | +0.116 |
| NII | dirt | 1200 | unknown | C1 | 278 | 223 | 0.553 | 0.595 | +0.041 | −0.016 | +0.095 |
| KOK | turf | 1200 | unknown | MAIDEN | 276 | 223 | 0.861 | 0.881 | +0.021 | −0.020 | +0.062 |
| NII | dirt | 1200 | unknown | MAIDEN | 273 | 234 | 0.608 | 0.627 | +0.019 | −0.007 | +0.048 |

Rail is `unknown` on every cell, so the rail column is left out of this table.

## Alternatives rejected and why

- **Other estimands:** the winner's time (already in the store) and the 3rd-placed runner's time (the Place line).
  The owner chose the mean of the placed 1–3 on #96.
- **The per-placing dead-heat reading.** Under it, 1,1,3 would give (t1 + t3)/2. #305's answer chose the per-runner
  reading. It affects about 190 races.
- **Logged publication times for availability.** `result_published_utc` is null on every stored race, so the variant
  would never see data on development folds. The owner chose post + 30 min.
- **`core.eval.bootstrap.paired_bootstrap` on per-race squared-error differences.** It bounds the MSE difference, not
  the RMSE difference. `rmse_diff_bootstrap` keeps its cluster draw, its one-sided quantiles and the gain's sign
  (`core.eval.scoring.gain`), and recomputes both RMSEs on each replicate.
- **Training only from 2014-06-07 (D10's price era).** C3 reads no price, so D10 allows earlier rows. Starting at the
  store's first day puts the first test quarter at 2015-Q4, not 2020-Q1, and scores 45 folds instead of 28.
- **Fitting once and reusing that fit for the variant-off model.** #96's fit ignores `same_day_variant` today. Each
  fold fits twice (0.5 s a fit) anyway, so the comparison does not depend on that detail.
- **A multiplicity correction across cells.** The cells are descriptive. The claim rests on the overall row, which
  was the only pre-registered test.

## What this rules out

- **The null, overall and by margin.** The same-day variant lowers the time RMSE of E[top-3 time] by 0.233 s under
  the Student-t error, 19% of the par's 1.242 s (one-sided 95% lower bound +0.224 s). Under the normal error it is
  0.245 s (lower bound +0.234 s).
  - The gain is positive in every fold.
  - It is positive on both surfaces, at every track and in every class, each with a lower bound above 0.
  - It holds in every cell with 100 or more races, apart from 16 cells whose bounds span 0.
- **Where the gain comes from.** It comes from the day's earlier results: it is 0 when none is available by t, and it
  grows from +0.27 s with one result to +0.37 s with six or more. The day effect is larger than the race noise: tau
  is about 0.8 s against sigma of about 0.62 s.
- **What it does not show.**
  - Nothing about the final price. A better time forecast need not move win probabilities beyond what the board
    already prices. That is E39's marginal test (#97, then C4), and 07 §5.3's prior is that it collapses into the
    market.
  - Nothing about course or rail. Rail is unknown on every race and layout on 88% (#329).
- **What the result depends on.** The 30-minute availability rule is an owner decision, but its truth is an
  assumption in `docs/ledger.md`: the row "A JRA race's official result is published within 30 minutes of its post
  time".
  - If results are published later, the variant sees fewer races at T-2 than here. The gain for races 30–35 minutes
    after a previous post would then be overstated.
  - #225's logger sightings will test that assumption on logged days.
- **The error model.** Student-t at 5 df and the normal error agree within 0.012 s on the gain. The Student-t par has
  the lower out-of-sample RMSE (1.242 against 1.256 s). This comparison has no bounds.

## Open questions for the owner

- **The Student-t degrees of freedom of C3's error.** #96's outline signs a Student-t error but no df. This run used
  5, the df of #96's tests, and ran the normal error beside it. It is the owner's choice which one C3 carries forward.

## Follow-ups filed

- #328 (proposed): test a carry-over variant prior for a day's first races. Those are the 26% of races with no
  same-day result by T-2, where the gain is 0. The prior's source is a variant-method decision.
- #329 (proposed, blocked by #90): rerun this test by rail and layout once #90 fills them.
- #330 (proposed): keep the registry summary under the public-tree size cap, so a regenerated summary lists every run
  again.
- Already filed:
  - #97: horse relative speed and p_speed, the marginal E39 test against the final price.
  - #219: swap #129's interim speed par for #96's figure. It was waiting on this ticket.
  - #90 and #230: the cushion and moisture prior, E29.

## Complete-timing sensitivity (#416)

Ticket #416 (follow-up to #305, epic #73) checks the race-level target before later evaluations use it.
- **The gap it closes.** #305's builder counted a race as partly placed only when no timed runner was placed 3rd.
  So a race whose 1st and 2nd were placed but untimed, with only 3rd timed, passed as complete, and its label was
  t3 alone.
- **What changed.** Who placed is now read from `finish_pos` before any time. Each kept race gets three flags,
  counted separately:
  - missing timing: a runner placed 1–3 has no finish time;
  - short order: fewer than three runners placed 1–3 (a dead heat for 2nd, 1,2,2, is three runners);
  - tie: two or more runners share a place in 1–3.
- **The label is unchanged.** It is still the signed estimand over the placed runners that have a time. Changing it
  is the owner's decision. `complete_timing` marks the races whose runners placed 1–3 are all timed.

- Registry family `japan-speed-top3-c3-timing`, spec sha256
  `6e61cb2147ffce77a077db5957957cc83147299c0609cfe95b33d517b3f8f7bd`.
  - The spec is #305's unchanged, plus the coverage definitions and the sensitivity. It was frozen before any
    outcome was read.
  - #305's family keeps its frozen spec (`72f5c802e96e...`). Its four records were reproduced exactly, so none was
    appended again.
- Code at `997e0ef`. The store is the same as #305's (`races.parquet@9cce511831074b64`,
  `runners.parquet@6f3cec95347e0132`).
- The full report is under the data directory at `reports/speed-japan-72f5c802e96e-6e61cb2147ff.md`. The records are
  in `docs/experiments/registry/japan/japan-speed-top3-c3-timing.md`.
- Run ids:

  | Variant | Run id |
  | --- | --- |
  | `t5:variant_on_vs_par_only:complete_timing` | `190f75a5a966f8ff3e9196f7b57a2850bcbb1d658b3d4f23622a4569d6f4445d` |
  | `t5:variant_off_vs_par_only:complete_timing` | `ee82f1516de1ca32157b7cf77dc3570e429eb787548fa4b0a35555f28df305f6` |
  | `normal:variant_on_vs_par_only:complete_timing` | `992e2bd6303b454f00e4e625cdfac8a652ad442367758bb598340367449dfbc6` |
  | `normal:variant_off_vs_par_only:complete_timing` | `ebad83db21915ab940b2431c017554029b80605f844225910943da774237d87b` |

- The run (both error models, both arms) took 181 s at a peak RSS of 423 MB.
- It exited 1 at its last step, `write_summary`. The cause is another ticket's records in the shared registry: #131's
  family ids (`japan/131-exacta-category-v1` and others) contain `/` (#344, #391).
  - The report and the records had been written first.
  - The committed summary was regenerated by `write_summary` itself, as #85 did, over the registry restricted to the
    families already committed plus this one. That added this family's file and its index line, and changed no
    other file.

### Coverage of the kept races

| Count | n |
| --- | --- |
| Kept races | 56,267 |
| Missing timing | 0 |
| Short orders | 0 |
| Ties | 335 |
| Complete timing | 56,267 |

- No runner placed 1–3 lacks a finish time. All 168,951 such runners are timed.
- No race placed fewer than three runners. #305's 110 partly placed races are exactly the 110 dead heats for 2nd: three
  runners in the top 3, with nobody placed 3rd.
- The 335 ties are 77 dead heats for 1st, 110 for 2nd and 148 for 3rd. No race has two. They keep #305's convention:
  each runner placed 1 or 2 counts once, and the runners placed 3rd share one slot.

### The sensitivity

The complete arm reruns the fold loop on the races with complete timing only. It uses the published arm's fold
windows (walk-forward over every development race), so dropping races never moves a window. The dropped races are in
no fit. The same-day variant does not read their results, and they are not scored. The change is measured on the races
both arms score:
- delta = gain(complete) − gain(existing);
- its bounds come from a meeting-clustered paired bootstrap (D7, 95% one-sided, 10,000 replicates, seed 305), which
  recomputes all four RMSEs on each draw.

| Error model, prediction | Gain, existing (s) | Gain, complete timing (s) | Races in both | Change (s) | lo | hi |
| --- | --- | --- | --- | --- | --- | --- |
| t5, variant on | +0.233 (+0.224, +0.242) | +0.233 (+0.224, +0.242) | 37,047 | 0 | 0 | 0 |
| t5, variant off | 0 | 0 | 37,047 | 0 | 0 | 0 |
| normal, variant on | +0.245 (+0.234, +0.255) | +0.245 (+0.234, +0.255) | 37,047 | 0 | 0 | 0 |
| normal, variant off | 0 | 0 | 37,047 | 0 | 0 | 0 |

The two arms are the same races, folds and fits, so every gain and every bound agrees to the last digit. The change
is exactly 0.

### What this rules out

- **Untimed official finishers do not touch #305's comparison on this store.** None exists, so #305's published gains
  (+0.233 s under t5 and +0.245 s under normal) stand as they are.
- **It is not a test of the label.** The sensitivity can only move when some placed runner is untimed. A store with
  untimed placings, such as a rebuilt or extended one, would make it bite. That is why the flag and the counts are
  kept in the builder and in every run.
- **The primary label is unchanged,** pending the owner's decision on whether an incomplete race should keep its
  timed-runner mean, be dropped, or be labelled another way. On this store that choice changes no number.
- Nothing is adopted (D16). The holdout was not opened, and no holdout day was read.
