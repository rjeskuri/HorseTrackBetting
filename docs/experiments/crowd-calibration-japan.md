# Experiment report: crowd calibration at the final Win price (Japan, #77)

Ticket #77 (epic #72; 08 §8 A3, §4 E1, E3, E4). The study fits three de-vigs of the final Win price (normalised,
power-β, Shin) and measures the power β by year, by era, by a pre-registered segment family (E4) and by
market-entropy bins (E3). It measures the tail as observed over expected winners above decimal 51.0 under each de-vig,
the owner's choice (#77 (6)). It reports each de-vig's out-of-fold winner log loss against normalised. **Nothing is
adopted.** The owner picks the baseline de-vig (08 §9.4), and D4 already signs the calibrated market (a spline in
ln q, fitted in-fold) as the Win baseline. Every figure here is an aggregate over the JRA results store (netkeiba
race results; no JRDB, no JRA-VAN). No runner or race row appears.

- Registry family `japan-crowd-calibration-final-win`, spec sha256
  `4816f067ab5a50cb7dc07cd3dff85292532aa76a7b6b07d938f4c30dd5563c36`. The spec was frozen before any outcome was
  read. It holds the measure each hypothesis names (not its direction), the de-vigs, the β fit, the E4 definitions,
  the E3 cut points and their rule, the tail, the floor, the fold scheme, the bootstrap and the owner decisions.
- Code at `25594e8` (`countries/japan/studies/calibration.py`). Owner inputs are in
  `countries/japan/config/calibration.yaml`. To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.calibration --run`.
- Store: `processed/jra/runners.parquet@b3226f2a22bfce26`, `processed/jra/races.parquet@4e6a394a57e3101a`.
- The full report is under the data directory, at `reports/crowd-calibration-japan-4816f067ab5a.md`. Every record
  is in `docs/experiments/registry-japan.md`.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| D4 | The Win baseline is the calibrated market, fitted in-fold. This study measures the de-vigs beneath it and adopts none. | `docs/problem-statement.md` §2, #14 |
| D6 | Winner log loss per race. A dead heat scores each of its k co-winners at 1/k. | §3, #14; the dead-heat rule in the #69 answer (2026-11-24) |
| D7 | Bootstrap clusters are meetings (venue x day); bounds are 95% one-sided (`config/scoring.yaml`). | §3, #14 |
| D9 | Walk-forward quarters, refit quarterly, expanding window, 1-week embargo. The first scored quarter is 2020-Q1 (`config/folds.yaml`). | §4, #14; the #70 answer |
| D10 | Races before 2014-06-07 are excluded wherever the price enters, so no pre-reform price was read. | §4, #14 |
| D11, D12, D14, D15 | Holdout A2 from 2026-11-28, fully locked for outcomes and prices, opened only by the owner. Not opened here. | §5, #14; the #71 answer (`config/holdout.yaml`) |
| D21 | No figure derived from a licensed source appears in the repository. None is used here. | #14 |
| #77 (1) | E4 is the full list: R12, R11pm1 (races 10-12), g1_day_undercard, field_16_plus, debut (新馬), venue, heavy_going (不良), handicap (ハンデ), with the definitions in the spec. | #77, owner comment of 2026-11-25 |
| #77 (2) | E3 cut points are the quintiles of per-race normalised entropy over the development races of the first training window (2014-06-07 to 2019-12-31), frozen before any scored fold. | #77, 2026-11-25 |
| #77 (3) | The era table covers the post-reform era only, stated as such. β by year carries the drift. D10 is unchanged. | #77, 2026-11-25 |
| #77 (4) | In the no-floor-bound subset, races with a runner at odds 1.1 or under are dropped from fitting and from scoring. | #77, 2026-11-25 |
| #77 (5) | The tail is a runner priced above decimal 51.0 (above 50/1). | #77, 2026-11-25 |
| #77 (6) | The tail is measured as observed over expected winners above 51.0 under each de-vig, with meeting-clustered bounds, not as a β. The frozen family stands. A tail β is #285, pre-registered as its own family. | #77, owner answer of 2026-11-25 (c) |

The E3 cut points are 0.69521, 0.74673, 0.78515 and 0.82154. They come from the final Win odds alone: the columns
read were race_id, race_date, win_odds and is_scratched, and no outcome column. They cover 19,259 races on 608
development race days (2014-06-07 to 2019-12-28), computed by `--entropy-cut-points`. Each run recomputes them and
refuses a config that disagrees.

## Hypothesis

The frozen spec pre-registered the measures, not their directions. Its `hypotheses` read, in full: E1, "the power
beta of the final Win price by year and era; each de-vig's ΔLL against normalised and its tail calibration (observed
vs expected winners)"; E3, "the power beta by bin of the race's normalised market entropy"; E4, "the power beta in vs
out of each pre-registered segment". Its `adoption` adds "judged by the beta CI, not by ΔLL (08 §4 E1)". Each
direction below is quoted from 08 §4 or 06, both written before the freeze (2026-11-25 03:24 UTC). The failure rules
were written after the run, with this report, and are marked so. The null is the starting point: normalised 1/odds is
calibrated (β = 1), and no segment or entropy bin differs.

- **E1.** Direction (08 §4 E1): "The final Win price is FLB-miscalibrated, p ∝ q^β with β > 1, concentrated above
  50/1, drifting by era." 08 §4 E1 and 06 §1 expect power-β's gain over normalised to be small, measured in sample.
  No out-of-fold size was written before the run. Failure rule (written after the run): it fails if the β CI covers
  1.
- **E3.** Direction (08 §4 E3, from 06 H1): "the favourite is under-stated in concentrated races and over-stated in
  flat ones". Its restatement as β above 1 in concentrated races and below 1 in flat ones was written after the run.
  Failure rule (written after the run): it fails if the bins' CIs overlap 1 and one another.
- **E4.** Direction (08 §4 E4): "β differs across pre-registered crowd-composition segments", two-sided, with no
  sign per segment. 06 H2 (handicaps reverse the FLB) and H8 (venue and race-number differences) name signs for
  handicap, R12 and venue; the spec froze none. Failure rule (written after the run): it fails for a segment if its in
  and out CIs overlap.

## Variants tried

The family size is 17 variants (`Registry.family_size`), with 34 records.

**Run 1 (git `af0f234`): all 17 variants failed, and each failure is a registry record.** `runners.finish_pos` is
nullable, so `finish_pos == 1` was missing for the 2,732 starters with no placing, and every fit and score refused
`won`. The synthetic store had no unplaced starter. Commit `25594e8` counts an unplaced starter as a loser. The spec
did not change, so the rerun is the same family.

**Run 2 (git `25594e8`): all 17 variants scored.**

Data and split:
- The rows are development days from 2014-06-07 to 2026-11-15: 43,058 races and 600,231 priced starters. 2,150
  scratched runners are out. There are 66 dead heats and no void or unpriced race.
- Out-of-fold E1 covers 28 quarterly folds from 2020-Q1 to 2026-Q4 (the last is partial, to 2026-11-15). The
  expanding window has a 1-week embargo, and the first training window runs from 2014-06-07 to 2019-12-29. That
  makes 23,799 scored races in 1,985 meetings.
- The in-sample tables fit β once per group over every development race.
- Floor-bound means a runner at odds 1.1 or under. There are 195 such runners, in 195 races (0.45%).

CIs:
- The β CIs are 95% two-sided Wald (β ± 1.96 se, observed information).
- The ΔLL and tail bounds are the one-sided 5% and 95% quantiles of the paired bootstrap over meetings (D7; 10,000
  replicates, seed 77).

### E1: out-of-fold winner log loss, ΔLL against normalised

ΔLL is normalised's log loss minus the de-vig's, in nats per race. Positive means better than normalised.

| Subset | De-vig | Races | Mean log loss | ΔLL vs normalised | Lower (5%) | Upper (95%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all races | normalised | 23,799 | 1.92829 | | | |
| all races | power-β (refit per fold) | 23,799 | 1.92802 | +0.000269 | +0.000110 | +0.000428 |
| all races | Shin | 23,799 | 1.93954 | -0.01125 | -0.01344 | -0.00906 |
| no floor-bound races | normalised | 23,680 | 1.93398 | | | |
| no floor-bound races | power-β (refit per fold) | 23,680 | 1.93373 | +0.000243 | +0.000092 | +0.000394 |
| no floor-bound races | Shin | 23,680 | 1.94542 | -0.01144 | -0.01366 | -0.00923 |

The per-fold training β rises from 1.0128 (trained to 2019-12-29) to 1.0206 (trained to 2026-09-26). It ranges from
1.0096 to 1.0213 (se 0.0077 to 0.0052). The no-floor-bound folds run from 1.0089 to 1.0202.

### E1: β by era (post-reform era only)

D10 excludes every price before 2014-06-07, so no pre-reform price was read. The table covers the post-reform era
only (owner's answer 3 on #77).

| Subset | Era | Races | β | 95% CI |
| --- | --- | ---: | ---: | --- |
| all races | 2014-06-07 onward | 43,058 | 1.0206 | [1.0104, 1.0308] |
| no floor-bound races | 2014-06-07 onward | 42,863 | 1.0196 | [1.0093, 1.0298] |

### E1: β by year (in-sample)

2014 runs from 2014-06-07 and 2026 runs to 2026-11-15.

| Year | Races | β (all races) | 95% CI | β (no floor-bound) | 95% CI |
| --- | ---: | ---: | --- | ---: | --- |
| 2014 | 1,990 | 1.0151 | [0.9686, 1.0616] | 1.0145 | [0.9679, 1.0612] |
| 2015 | 3,454 | 1.0079 | [0.9718, 1.0440] | 1.0065 | [0.9704, 1.0427] |
| 2016 | 3,454 | 1.0111 | [0.9761, 1.0460] | 1.0102 | [0.9751, 1.0452] |
| 2017 | 3,455 | 1.0196 | [0.9843, 1.0549] | 1.0203 | [0.9848, 1.0558] |
| 2018 | 3,454 | 1.0168 | [0.9812, 1.0524] | 1.0150 | [0.9793, 1.0508] |
| 2019 | 3,452 | 1.0071 | [0.9712, 1.0429] | 1.0064 | [0.9704, 1.0423] |
| 2020 | 3,456 | 0.9934 | [0.9578, 1.0290] | 0.9930 | [0.9574, 1.0287] |
| 2021 | 3,456 | 1.0325 | [0.9959, 1.0692] | 1.0328 | [0.9961, 1.0696] |
| 2022 | 3,456 | 1.0558 | [1.0187, 1.0929] | 1.0539 | [1.0167, 1.0910] |
| 2023 | 3,456 | 1.0255 | [0.9890, 1.0620] | 1.0242 | [0.9876, 1.0607] |
| 2024 | 3,454 | 1.0446 | [1.0079, 1.0813] | 1.0393 | [1.0025, 1.0762] |
| 2025 | 3,455 | 1.0190 | [0.9830, 1.0550] | 1.0200 | [0.9838, 1.0563] |
| 2026 | 3,066 | 1.0200 | [0.9816, 1.0584] | 1.0190 | [0.9803, 1.0578] |

The no-floor-bound race counts are lower by 5 to 27 per year.

### E1: the tail (runners above decimal 51.0, out of fold)

Observed against expected winners under each de-vig. (O - E) per race is the tail's observed minus expected winners
per scored race, with meeting-clustered one-sided bounds. O / E is a point estimate only. The owner chose this, not a
β, as the ticket's tail measure (#77 (6)); the tail β is #285, pre-registered as its own family.

| Subset | De-vig | Tail runners | Observed | Expected | O / E | (O - E) per race | Lower (5%) | Upper (95%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all races | normalised | 111,564 | 654 | 788.4 | 0.830 | -0.00565 | -0.00732 | -0.00390 |
| all races | power-β | 111,564 | 654 | 752.5 | 0.869 | -0.00414 | -0.00580 | -0.00239 |
| all races | Shin | 111,564 | 654 | 299.0 | 2.188 | +0.01492 | +0.01324 | +0.01667 |
| no floor-bound races | normalised | 110,885 | 653 | 784.0 | 0.833 | -0.00553 | -0.00721 | -0.00377 |
| no floor-bound races | power-β | 110,885 | 653 | 750.1 | 0.871 | -0.00410 | -0.00577 | -0.00234 |
| no floor-bound races | Shin | 110,885 | 653 | 297.8 | 2.193 | +0.01500 | +0.01332 | +0.01676 |

### E4: β in vs out of each segment (in-sample, every development race)

Δβ = β_in - β_out is derived from the two rows. Its Wald CI treats the two fits as independent (they cover disjoint
races). It is a descriptive screen with no family-wise correction: D17's SPA and StepM belong to A6, not here.

| Segment | β in [95% CI] (races) | β out [95% CI] (races) | Δβ [95% CI] |
| --- | --- | --- | --- |
| R12 | 1.0804 [1.0423, 1.1185] (3,578) | 1.0158 [1.0052, 1.0264] (39,480) | +0.065 [+0.025, +0.104] |
| R11pm1 (races 10-12) | 1.0238 [1.0024, 1.0451] (10,754) | 1.0197 [1.0081, 1.0313] (32,304) | +0.004 [-0.020, +0.028] |
| g1_day_undercard | 1.0521 [1.0142, 1.0900] (3,190) | 1.0172 [1.0066, 1.0278] (39,575) | +0.035 [-0.004, +0.074] |
| field_16_plus | 1.0167 [1.0015, 1.0319] (17,710) | 1.0237 [1.0100, 1.0375] (25,348) | -0.007 [-0.028, +0.014] |
| debut (新馬) | 1.0060 [0.9728, 1.0392] (3,731) | 1.0221 [1.0114, 1.0328] (39,327) | -0.016 [-0.051, +0.019] |
| heavy_going (不良) | 0.9763 [0.9246, 1.0281] (1,572) | 1.0223 [1.0119, 1.0327] (41,486) | -0.046 [-0.099, +0.007] |
| handicap (ハンデ) | 0.9598 [0.9153, 1.0043] (2,450) | 1.0239 [1.0134, 1.0343] (40,608) | -0.064 [-0.110, -0.018] |

For g1_day_undercard, the 293 flat G1 races themselves are in neither side. Every other segment covers every race.

| Venue | Races | β | 95% CI |
| --- | ---: | ---: | --- |
| SAP | 2,064 | 1.0089 | [0.9609, 1.0569] |
| HAK | 1,872 | 1.0227 | [0.9709, 1.0744] |
| FKS | 2,976 | 0.9862 | [0.9474, 1.0251] |
| NII | 4,055 | 1.0275 | [0.9939, 1.0611] |
| TOK | 6,716 | 1.0581 | [1.0324, 1.0837] |
| NAK | 5,997 | 1.0332 | [1.0063, 1.0600] |
| CKY | 4,404 | 1.0153 | [0.9834, 1.0473] |
| KYO | 5,387 | 0.9896 | [0.9612, 1.0180] |
| HSN | 6,252 | 1.0188 | [0.9921, 1.0454] |
| KOK | 3,335 | 1.0069 | [0.9698, 1.0439] |

### E3: β by market-entropy bin (in-sample, every development race)

Each bin is the race's normalised entropy of the final Win odds, cut at the frozen quintiles of the first training
window. Bin 1 is the most concentrated market.

| Bin (entropy) | Races | β | 95% CI |
| --- | ---: | ---: | --- |
| 1: [0, 0.69521) | 9,182 | 1.0578 | [1.0390, 1.0767] |
| 2: [0.69521, 0.74673) | 8,642 | 1.0193 | [0.9978, 1.0408] |
| 3: [0.74673, 0.78515) | 8,493 | 1.0166 | [0.9932, 1.0401] |
| 4: [0.78515, 0.82154) | 8,143 | 0.9975 | [0.9718, 1.0232] |
| 5: [0.82154, 1] | 8,598 | 0.9690 | [0.9409, 0.9972] |

The bin-1 minus bin-5 difference is Δβ = +0.089 [+0.055, +0.123], derived as for E4. No race had fewer than two
priced runners.

## Alternatives rejected and why

- **Before and after 2014-06-07 (the ticket's era comparison).** It is not run. D10 excludes pre-reform prices, and
  the owner confirmed the post-reform era only (#77, answer 3).
- **A β fitted within the tail** (a separate slope above 51.0, or a spline in ln q). The owner chose observed over
  expected winners under each de-vig, with meeting-clustered bounds, as this ticket's tail measure, and the frozen
  family stands (#77 (6)). The tail β is #285, pre-registered as its own family. D4's spline is the signed way to
  model the curve; #275 scores it on the same folds and tail table.
- **Meeting-clustered β CIs.** The pre-registered β CI is Wald from the observed information, so races count as
  independent. The ΔLL and tail bounds do resample meetings (D7). A clustered β interval would be wider. It is
  #276.
- **A family-wise test over E4 and E3** (SPA, StepM, D17). That belongs to A6's pre-registered Family P study with
  the holdout. Here the segment and entropy β are in-sample screens.
- **Walk-forward segment and entropy β.** Not pre-registered. These β select nothing, so in-sample was the frozen
  choice.
- **Low turnover (E4).** It needs per-race sales, which are not ingested; #278 ingests them.
- **E2 (the board at t).** It needs a JRDB board; it is #103.
- **De-truncating floored odds.** The odds are used as displayed. Floor-bound races are handled by the
  no-floor-bound subset, and that subset changes β only in the third decimal.

## What this rules out

- **β = 1 over the post-reform development era.** The overall β is 1.0206 [1.0104, 1.0308], so normalised 1/odds is
  not exactly calibrated. The bias is mild: the CI rules out β above 1.031. The per-fold training β drifts up, from
  1.013 to 1.021 between the 2019 and 2026 windows. Single years have CIs that exclude 1 only in 2022 and 2024.
  Floor-bound races do not drive any of this: without them β is 1.0196 [1.0093, 1.0298].
- **A sizeable gain from the power correction.** Out of fold, power-β gains +0.00027 nats per race over normalised.
  The one-sided 95% lower bound is +0.00011, so the gain is above zero. The upper bound is +0.00043, which rules out a
  gain above about 0.0004 nats per race. That is far below any MDE: judge by the β CI, not by ΔLL (08 §4 E1).
- **Shin as the tote's de-vig.** Shin loses 0.0113 nats per race against normalised [-0.0134, -0.0091]. It
  attributes the 20% takeout to insider money and strips probability from longshots: tail O/E is 2.19. This matches
  D4's warning about Shin on tote pools.
- **A single β fixing the tail.** Above 51.0, winners come in at 0.83 of normalised's expectation and 0.87 of
  power-β's. The (O - E) bounds are below zero for both. The tail bias is not absorbed by one exponent.
- **E4 and E3 are not confirmed here.** The in-sample screens flag R12 (Δβ +0.065 [+0.025, +0.104]), handicap
  (-0.064 [-0.110, -0.018]) and a monotone entropy gradient (bin 1 minus bin 5, +0.089 [+0.055, +0.123]). They are
  in-sample, with no family-wise correction. They span years that overlap the research that proposed them. The
  entropy bins may also be confounded with how many tail runners a race has, which this study does not separate.
- **Large E4 effects, where the screen is null.** For R11pm1, field_16_plus and debut, the screen rules out
  |Δβ| above about 0.03, 0.03 and 0.05.

## Follow-ups filed

Each is filed under epic #72, `status:proposed` unless marked.

- #275: score D4's calibrated market (the spline in ln q) on the same folds, races and tail table as normalised and
  power-β.
- #276: meeting-clustered CIs for the power β (a cluster bootstrap or sandwich se), to check the Wald width.
- #277: separate the entropy gradient from the tail bias: β by entropy bin on runners at or under 51.0, or with the
  tail share controlled.
- #278: ingest JRA per-race Win sales, so E4's low-turnover segment can be tested.
- #285 (`status:ready`, by the owner's answer #77 (6)): fit a tail β, the conditional-logit slope with a separate
  slope above 51.0, pre-registered as its own family.
- Carry R12, handicap and the entropy gradient into A6's pre-registered family (max-statistic, holdout) as
  candidates, not findings. No new ticket: this is #88's scope, and A6 already takes segment β from #77's
  `segment_beta`.
