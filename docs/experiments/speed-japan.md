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
  - #305's family keeps its frozen spec (`72f5c802e96e...`). Its four records were reproduced at `997e0ef` in every
    field but the commit (theirs is `2ab5bf4`). The run's dedupe then ignored the commit, so none was appended again.
    Since #416's review the dedupe matches the commit too, so a rerun at a new commit appends its records and the
    registry names the code behind each result.
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
| Untimed official orders (excluded, not kept: placed 1–3, none timed) | 0 |

- No runner placed 1–3 lacks a finish time. All 168,951 such runners are timed.
- `n_untimed_order` counts the part of the "no runner placed 1–3 with a finish time" exclusion that has an official
  order (a runner placed 1–3) but no time in it, apart from races nobody placed in, such as an abandoned race. It
  was added after the run; its 0 follows from `n_no_placed_time` = 0 at the same store builds.
- No race placed fewer than three runners. #305's 110 partly placed races are exactly the 110 dead heats for 2nd: three
  runners in the top 3, with nobody placed 3rd.
- The 335 ties are 77 dead heats for 1st, 110 for 2nd and 148 for 3rd. No race has two. They keep #305's convention:
  each runner placed 1 or 2 counts once, and the runners placed 3rd share one slot.

### The sensitivity

The complete arm reruns the fold loop on the races with complete timing only. It uses the published arm's fold
windows (walk-forward over every development race), so dropping races never moves a window. The dropped races are in
no fit. The same-day variant does not read their results, and they are not scored. A fold with no complete-timing race
to train or to test on is skipped and recorded with its reason, and the arm fails only when no race is scored in both
arms; on this store every race has complete timing, so no fold is skipped. The change is measured on the races
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

## On the leaderboard (2026-11-27)

Under the reframe (epic #447; D16 to D18 as amended 2026-11-27 in #448), the results above are measured results, not
adoption verdicts. Nothing is adopted or rejected: every option, model and member will be a row on the planned
`horsetrack japan leaderboard` (#452; this revision does not have the command yet), ranked by its shrunk return per
¥100 at the final price, and a row with no measured return will be listed after the ranking, never dropped. Where
this report says "not adopted", "rules out" or "rejected", read a measured position. This section reopens and
re-runs nothing.

- Registry families: `japan-speed-top3-c3` (#305) and `japan-speed-top3-c3-timing` (#416, the complete-timing
  sensitivity).
- Both measure a time forecast, not a return, so, as task 4 (#451, planned) specifies, both will be listed with no
  return yet. The forecast's marginal against the final price is E39's test, open as #97.
- "Nothing is adopted" and what this report rules out now read as measured positions on the time forecast. The open
  questions for the owner above are unchanged.

## Carry-over variant prior for a day's first races (#328)

Ticket #328 (epic #73, follow-up of #305). At T-2, 26% of #305's scored races (9,517 of 37,047) had no result of their
own day, track and surface before t. There #96's same-day variant is its N(0, tau²) prior and #305's gain is 0 by
construction. This study gives the variant a carry-over prior on those races, in the owner's three kinds (#328,
2026-11-28), each its own pre-registered variant:
- **(a) previous_day**: the same track and surface's posterior variant on its latest earlier race day with a result
  available before t, decayed toward N(0, tau²) with a half-life of 7 days.
- **(b) other_surface**: the same day's posterior variant on the other surface (turf for dirt, dirt for turf) at the
  same track, from its results available before t, coupled with weight 0.5.
- **(c) both**: (a) and (b) combined by their information over the base prior.

The prior is `countries/japan/models/speed_prior.py` (`CarryOverPrior`); the study is
`countries/japan/studies/speed_prior_run.py`. #96's `SpeedModel` is unchanged, and so is its fit, which never reads an
injected prior.

- Registry family `japan-speed-top3-c3-prior`, spec sha256
  `2f451f09a852781e10ddacde9a86f511afceb8d811a60753ce96a580c8fbf241`, frozen before the store was read. It is #305's
  spec (estimand, race exclusions, availability, keys, error models, T-2, folds, bootstrap, decisions) with the three
  priors and their parameters, the prior definitions, the two baselines and results seen as the primary split.
- Code at `b39ffb9`. To reproduce, run `uv run python -m horsetrack.countries.japan.studies.speed_prior_run --run`.
- Store: the same builds as #305 and #416 (`processed/jra/races.parquet@9cce511831074b64`,
  `processed/jra/runners.parquet@6f3cec95347e0132`), read column-scoped. JRA results only; no JRDB, no JRA-VAN (D21).
- The full report, with every cell, margin and results-seen row of every variant, is under the data directory at
  `reports/speed-prior-japan-2f451f09a852.md`. The records are in
  `docs/experiments/registry/japan/japan-speed-top3-c3-prior.md`.
- 24 records, none failed (2 error models × 3 priors × 2 predictions × 2 baselines). The run took 397 s at a peak RSS
  of 527 MB.
- #305's baselines are recomputed in the same fold loop, from the same fits: par only and #305's `variant_on`
  reproduce #305's RMSEs (t5 1.242 s and 1.009 s; normal 1.256 s and 1.012 s). Every comparison is paired, on the
  same 37,047 races in 3,215 meetings over the same 45 folds.

### Hypothesis

The null is the starting point: **no carry-over prior lowers the time RMSE below #305's `variant_on` on the races with
no same-day result before t** (gain = RMSE(#305 variant on) − RMSE(prior, variant on) ≤ 0 on results seen 0).
- A kind's claim rests on the one-sided 95% lower bound of that gain being above 0 (D7, meeting clusters). The overall
  gain is reported beside it.
- Expected before the run: the gain shrinks toward 0 as the day's own results come in, because they dominate the
  prior.
- The parameters were fixed before any scored read; no measured persistence of the variant existed in the ledger:
  - half-life 7 days, one race week: a Sunday reads its Saturday at ρ = 0.91, and a Saturday reads the previous
    Sunday at ρ = 0.55;
  - surface weight 0.5, the midpoint of the allowed [0, 1]. The turf and dirt variants share the day's weather, wind
    and temperature, but moisture moves them in opposite directions (wet dirt rides faster, soft turf slower), so the
    sign of the coupling was uncertain before the run.

### Variants tried

The races with no same-day result first. Gain over #305's `variant_on` (s), with the one-sided 95% bounds lo and hi
(10,000 meeting-clustered paired replicates, seed 328). On these races #305's `variant_on` is the par (RMSE 1.158 s
under t5, 1.166 s under normal), so the gain over par only is the same figure.

| Prior | Error model | Races (results seen 0) | Gain (s) | lo | hi |
| --- | --- | --- | --- | --- | --- |
| previous_day | t5 | 9,517 | **+0.149** | +0.138 | +0.161 |
| previous_day | normal | 9,517 | **+0.150** | +0.138 | +0.163 |
| both | t5 | 9,517 | **+0.137** | +0.126 | +0.149 |
| both | normal | 9,517 | **+0.139** | +0.127 | +0.152 |
| other_surface | t5 | 9,517 | −0.027 | −0.031 | −0.022 |
| other_surface | normal | 9,517 | −0.027 | −0.032 | −0.022 |

The variant-on and variant-off predictions agree on these races (nothing of the day's own is in), so each row covers
both.

Every race, all 24 variants. Gain over #305's `variant_on` and over par only, each with its bounds:

| Prior | Error model | Prediction | Gain over #305 variant on | lo | hi | Gain over par only | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| previous_day | t5 | variant on | +0.046 | +0.040 | +0.052 | +0.279 | +0.269 | +0.289 |
| previous_day | t5 | variant off | −0.080 | −0.090 | −0.069 | +0.154 | +0.143 | +0.164 |
| other_surface | t5 | variant on | −0.021 | −0.024 | −0.019 | +0.212 | +0.204 | +0.220 |
| other_surface | t5 | variant off | −0.293 | −0.307 | −0.281 | −0.060 | −0.067 | −0.054 |
| both | t5 | variant on | +0.033 | +0.026 | +0.040 | +0.266 | +0.256 | +0.276 |
| both | t5 | variant off | −0.109 | −0.122 | −0.097 | +0.124 | +0.113 | +0.135 |
| previous_day | normal | variant on | +0.053 | +0.049 | +0.058 | +0.298 | +0.286 | +0.310 |
| previous_day | normal | variant off | −0.087 | −0.100 | −0.075 | +0.157 | +0.146 | +0.169 |
| other_surface | normal | variant on | −0.016 | −0.018 | −0.014 | +0.229 | +0.219 | +0.239 |
| other_surface | normal | variant off | −0.307 | −0.323 | −0.293 | −0.063 | −0.070 | −0.056 |
| both | normal | variant on | +0.044 | +0.039 | +0.049 | +0.289 | +0.278 | +0.301 |
| both | normal | variant off | −0.117 | −0.131 | −0.104 | +0.127 | +0.116 | +0.139 |

- "Variant off" is the prior alone, with no same-day result read. Against #305's `variant_on`, which reads them, it
  loses overall, as it must once the day's own results are in; against par only, (a) and (c) alone gain +0.12 to
  +0.16 s.
- #305's best, `t5:variant_on`, has RMSE 1.009 s; with the previous-day prior it is 0.964 s (normal: 1.012 s to
  0.958 s).

By results seen before t (t5, variant on, gain over #305's `variant_on`, bounds in brackets):

| Results seen | Races | previous_day | other_surface | both |
| --- | --- | --- | --- | --- |
| 0 | 9,517 | +0.149 (+0.138, +0.161) | −0.027 (−0.031, −0.022) | +0.137 (+0.126, +0.149) |
| 1 | 5,364 | +0.048 (+0.037, +0.059) | −0.028 (−0.035, −0.021) | +0.033 (+0.021, +0.045) |
| 2 | 5,941 | +0.009 (+0.000, +0.017) | −0.020 (−0.024, −0.015) | −0.004 (−0.013, +0.006) |
| 3 | 5,709 | −0.009 (−0.017, −0.002) | −0.018 (−0.023, −0.014) | −0.023 (−0.032, −0.014) |
| 4 | 5,279 | −0.019 (−0.030, −0.010) | −0.017 (−0.022, −0.013) | −0.032 (−0.044, −0.022) |
| 5 | 3,343 | −0.007 (−0.016, +0.002) | −0.012 (−0.016, −0.009) | −0.018 (−0.029, −0.008) |
| 6 or more | 1,894 | −0.021 (−0.044, −0.002) | −0.010 (−0.014, −0.006) | −0.032 (−0.058, −0.009) |

- Under the normal error the previous-day prior's losses from 3 results on are smaller (+0.003 at 3, −0.008 at 4,
  −0.001 at 5, −0.005 at 6 or more; only the bounds at 4 exclude 0). A plausible reason, not tested here: the t5
  posterior's reweighting starts from the prior's mean, so a confidently wrong prior is slower to give way.
- By surface (t5, variant on, every race): previous_day +0.050 s on dirt and +0.044 s on turf; other_surface −0.013 s
  and −0.028 s; both +0.041 s and +0.028 s. Every margin and cell is in the full report.

### Alternatives rejected and why

- **Fitting the half-life and the surface weight on the folds.** The owner's kinds are pre-registered variants with
  fixed parameters, frozen before any scored read. A fitted value is a new family, not a rerun of this one.
- **A negative surface weight.** The prior's contract keeps w in [0, 1], the owner's (b) as specified. The result
  suggests the coupling is negative or going-dependent; that is a follow-up, not a change to this family.
- **The fit's own in-sample variants for a previous day that is a training day.** The prior takes every previous
  day's residuals against the fold's par, the same rule on training, embargo and test days, all published before the
  test window. The fit's in-sample variant (#97's `in_sample_variant`) would treat training days differently from the
  embargo week.
- **A prior in the fit.** #96's ruling holds: an injected prior with a non-zero mean would shift the pars of every
  cell linked to that day.

### What this rules out

- **The null for (a) and (c) on the first races.** With no same-day result before t, the previous-day prior lowers the
  time RMSE by +0.149 s (t5; lower bound +0.138) against #305's `variant_on`, which there is the par: 1.158 s to
  1.008 s. Carrying both priors gives +0.137 s (+0.126). Under the normal error the figures are +0.150 s and +0.139 s.
- **Overall, the previous-day prior improves #305's variant**, by +0.046 s (t5, +0.040 to +0.052) and +0.053 s
  (normal), the gain coming from the races with zero or one same-day result.
- **(b) as specified loses everywhere.** At weight 0.5 the other surface's variant moves the first races the wrong
  way: −0.027 s on the first races and −0.021 s overall (t5), bounds below 0 in every results-seen row. Together with
  the a-priori reasoning above, it points to a coupling below 0 on part of the days. Adding (b) to (a) costs 0.012 s on
  the first races.
- **The prior holds the variant back once the day's results are in.** From three own results on, every kind's t5
  point estimate is slightly below #305's (−0.007 to −0.032 s; the bounds exclude 0 in all but previous_day at 5).
  A likely reason, not tested here: over one night (ρ = 0.91) the prior SD stays close to the previous day's
  posterior SD, so the prior is too tight when the going has changed overnight.
- It is a time forecast. It says nothing about win probabilities or the final price; that is E39's marginal test
  (#97, then C4). Nothing is adopted (D16). The holdout was not opened, and no holdout day was read.

### Leaderboard rows

The 24 variants are rows `japan-speed-top3-c3-prior/<error model>:<prior>:<prediction>_vs_<baseline>`, for example
`japan-speed-top3-c3-prior/t5:previous_day:variant_on_vs_variant_on`. Like #305's and #416's they measure a time
forecast, not a return per ¥100, so they are listed with no return and are not ranked. `leaderboard_backfill` gives an
unlisted family the kind `other`; #305's family is a `member` through `LEGACY_KINDS`, and listing this one there too
waits on the backfill tests' kinds table (#639, a follow-up below).

### Follow-ups

Filed as `status:proposed` tickets naming epic #73, not blocking #328:
- #637: a surface coupling that may be negative or depends on the going (for instance estimated on each fold's
  training days, or split by the moisture prior of #90 and #230), in a new family.
- #638: the previous-day prior with its decay estimated on the training folds, or a prior SD inflated as the day's own
  results come in, so that it gives way from three results on, in a new family.
- #639: list `japan-speed-top3-c3-prior` as a `member` in `leaderboard_backfill.LEGACY_KINDS`, with the backfill tests'
  kinds table.
- #640: E39's marginal test (#97) can carry the previous-day prior as a variant of the speed member.
