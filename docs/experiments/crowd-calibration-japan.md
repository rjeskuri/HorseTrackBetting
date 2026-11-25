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
  is in `docs/experiments/registry/japan/`.

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

The tail β fitted beside this table is #285's: see "Experiment report: the tail β (#285)" at the end of this file.
D4's spline on the same folds and tail table is #275's: see "Experiment report: D4's calibrated market on the
crowd-calibration folds (#275)" after it.

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

---

# Experiment report: the tail β (Japan, #285)

Ticket #285 (epic #72), the owner's answer (c) on #77 of 2026-11-25. #77 measured the tail above decimal 51.0 as
observed over expected winners per de-vig. This study fits the slope a tail runner gets: a conditional logit on
x = ln(1/odds) with a body slope β for every runner and an increment added for runners priced strictly above 51.0.
`tail β` = β + increment. **Nothing is adopted.** Every figure is an aggregate over the JRA results store (netkeiba
race results; no JRDB, no JRA-VAN). No runner or race row appears.

- Registry family `japan-tail-beta-final-win`, spec sha256
  `ee2e0cc5602402ed118302c9f1645e732191784c8c406d3f6ef0708fc34c232d`. It is its own family, not #77's. The spec was
  frozen in the registry before any development row was read. It holds the model, the comparators, the ΔLL rule, the
  fit and its CI, the tail threshold, the floor, the fold scheme, the bootstrap and the owner decisions.
- Code: `countries/japan/studies/tail_beta.py`; the records carry git `3250795`, and the next commit changes only the
  hypothesis id written to the records (see Variants tried). To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.tail_beta`.
- Store: `processed/jra/runners.parquet@b3226f2a22bfce26`, `processed/jra/races.parquet@4e6a394a57e3101a`, the
  builds #77 read. The full report is under the data directory, at `reports/tail-beta-japan-ee2e0cc56024.md`. The
  four records are in `docs/experiments/registry/japan/`.
- Holdout: development days only. `calibration.read_development` drops holdout days, and `LockedHoldout.open` is not
  called. No holdout outcome or price was read.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| #77 answer (c) (2026-11-25) | The tail measure is a tail β: the conditional-logit slope with a separate slope above 51.0, pre-registered as its own family, beside the observed over expected winners. | #77, owner answer of 2026-11-25 (listed as (6) above) |
| #77 (4) | The no-floor-bound subset drops races with a runner at odds 1.1 or under from fitting and from scoring. | #77, 2026-11-25 |
| #77 (5) | The tail is a runner priced above decimal 51.0 (`tail_min_odds` in `config/calibration.yaml`). | #77, 2026-11-25 |
| Floor 1.1 | A priced runner at or under odds 1.1 + 1e-9 is floor-bound (the Win pool's minimum dividend over the unit stake). | `config/jra.yaml`; #77 (4) |
| D6 | Winner log loss per race; a dead heat scores each of its k co-winners at 1/k. | §3, #14; the #69 answer (2026-11-24) |
| D7 | Bootstrap clusters are meetings (venue x day); bounds are 95% one-sided; 10,000 replicates (`config/scoring.yaml`). | §3, #14 |
| D9 | Walk-forward quarters, refit quarterly, expanding window, 1-week embargo; the first scored quarter is 2020-Q1 (`config/folds.yaml`). | §4, #14; the #70 answer |
| D10 | Races before 2014-06-07 are excluded wherever the price enters. | §4, #14 |
| D11, D12, D14, D15 | Holdout A2 from 2026-11-28, fully locked for outcomes and prices, opened only by the owner. Not opened here. | §5, #14; the #71 answer (`config/holdout.yaml`) |
| D21 | No figure derived from a licensed source appears in the repository. None is used here. | #14 |

The comparators (normalised 1/odds and power-β refit per fold) and ΔLL against both are the ticket's own outline. The
bootstrap seed (285) is an implementation choice for reproducibility: `scoring.yaml` signs no seed.

## Hypothesis

The null is the starting point: the price is one-slope, so the increment is 0 and the tail β equals the body slope.
The frozen spec pre-registered the measures (the tail β and its increment, overall, by year, with and without
floor-bound races, and ΔLL against normalised and power-β), not their directions. The direction below was written
after #77's tail table, which looked at the same development races, so this study cannot confirm it independently.

- **Direction.** #77 found tail runners winning less often than either de-vig says (O/E 0.83 under normalised, 0.87
  under power-β). A single slope that is too flat in the tail would show as an increment above 0, a tail β above the
  body slope.
- **Failure rule** (written with this report): it fails if the increment's 95% Wald CI covers 0. It also fails to add
  to power-β if the lower bound of ΔLL against power-β is at or below 0.

## Variants tried

The family has 4 variants (`Registry.family_size`) and 4 records, all scored: a walk-forward variant and an in-sample
by-year variant, each on all races and on no floor-bound races.

**One aborted attempt, no record lost.** The first run froze the family, read the development rows and scored all four
variants. The registry then refused the first record, because a hypothesis id must be E, L or P followed by digits,
and the code had used `T1`. Nothing was appended. The family stays frozen with its spec, which still names its
hypothesis `T1`. The spec cannot change, so the fix writes the records under `E1` (08 §4 E1, whose tail measure this
is) and leaves the spec as frozen. The rerun is the same family and wrote the four records above.

Data and split:
- The rows are #77's: development days from 2014-06-07 to 2026-11-15, 43,058 races and 600,231 priced starters. 2,150
  scratched runners are out; 66 dead heats. 195 floor-bound runners sit in 195 races (0.45%).
- Out of fold covers 28 quarterly folds from 2020-Q1 to 2026-Q4 (the last is partial). That makes 23,799 scored races
  in 1,985 meetings (23,680 without floor-bound races).
- In-sample slopes are one fit per group on every development race of the subset.
- CIs on slopes are 95% two-sided Wald (estimate ± 1.96 se, inverse observed information; tail β's se uses the full
  covariance). They treat races as independent, as #77's β CIs do. The ΔLL bounds are the one-sided 5% and 95%
  quantiles of the paired bootstrap over meetings (D7, seed 285).

### The slopes, overall (in-sample, every development race)

| Subset | Races | Tail runners | Tail winners | Body β | Increment | Tail β |
| --- | ---: | ---: | ---: | --- | --- | --- |
| all races | 43,058 | 204,012 | 1,218 | 0.9999 [0.9877, 1.0121] | +0.0450 [+0.0302, +0.0599] | 1.0449 [1.0316, 1.0582] |
| no floor-bound races | 42,863 | 202,917 | 1,215 | 0.9987 [0.9864, 1.0109] | +0.0453 [+0.0304, +0.0601] | 1.0439 [1.0306, 1.0573] |

### The slopes by year (in-sample)

2014 runs from 2014-06-07 and 2026 to 2026-11-15. Races are of the all-races subset; the no-floor-bound counts are
lower by 5 to 27 per year.

| Year | Races | Tail winners | Increment | Tail β | Increment (no floor-bound) | Tail β (no floor-bound) |
| --- | ---: | ---: | --- | --- | --- | --- |
| 2014 | 1,990 | 65 | +0.0151 [-0.0504, +0.0807] | 1.0233 [0.9644, 1.0821] | +0.0148 [-0.0508, +0.0803] | 1.0224 [0.9635, 1.0813] |
| 2015 | 3,454 | 99 | +0.0624 [+0.0103, +0.1145] | 1.0411 [0.9938, 1.0884] | +0.0631 [+0.0110, +0.1152] | 1.0400 [0.9927, 1.0873] |
| 2016 | 3,454 | 106 | +0.0440 [-0.0065, +0.0945] | 1.0345 [0.9892, 1.0798] | +0.0460 [-0.0047, +0.0968] | 1.0347 [0.9891, 1.0802] |
| 2017 | 3,455 | 111 | +0.0248 [-0.0253, +0.0749] | 1.0327 [0.9879, 1.0775] | +0.0264 [-0.0240, +0.0767] | 1.0342 [0.9892, 1.0793] |
| 2018 | 3,454 | 98 | +0.0530 [+0.0008, +0.1052] | 1.0449 [0.9983, 1.0915] | +0.0531 [+0.0008, +0.1054] | 1.0431 [0.9964, 1.0897] |
| 2019 | 3,452 | 85 | +0.0866 [+0.0318, +0.1414] | 1.0562 [1.0062, 1.1061] | +0.0859 [+0.0311, +0.1408] | 1.0549 [1.0050, 1.1049] |
| 2020 | 3,456 | 109 | +0.0288 [-0.0214, +0.0791] | 1.0084 [0.9635, 1.0534] | +0.0282 [-0.0221, +0.0785] | 1.0077 [0.9627, 1.0526] |
| 2021 | 3,456 | 88 | +0.0629 [+0.0082, +0.1177] | 1.0679 [1.0182, 1.1176] | +0.0617 [+0.0070, +0.1165] | 1.0675 [1.0178, 1.1172] |
| 2022 | 3,456 | 93 | +0.0192 [-0.0346, +0.0730] | 1.0663 [1.0185, 1.1141] | +0.0197 [-0.0341, +0.0735] | 1.0646 [1.0167, 1.1124] |
| 2023 | 3,456 | 81 | +0.0912 [+0.0350, +0.1473] | 1.0769 [1.0259, 1.1279] | +0.0912 [+0.0350, +0.1474] | 1.0754 [1.0244, 1.1265] |
| 2024 | 3,454 | 98 | +0.0146 [-0.0384, +0.0676] | 1.0524 [1.0057, 1.0990] | +0.0170 [-0.0360, +0.0700] | 1.0483 [1.0016, 1.0950] |
| 2025 | 3,455 | 99 | +0.0419 [-0.0106, +0.0944] | 1.0414 [0.9946, 1.0881] | +0.0415 [-0.0113, +0.0942] | 1.0421 [0.9951, 1.0892] |
| 2026 | 3,066 | 86 | +0.0342 [-0.0216, +0.0901] | 1.0384 [0.9886, 1.0881] | +0.0322 [-0.0238, +0.0882] | 1.0362 [0.9864, 1.0860] |

The increment is above 0 in every year, with a CI that excludes 0 in 5 of 13 (2015, 2018, 2019, 2021, 2023). The tail
β's CI excludes 1 in 5 of 13 (2019, 2021 to 2024). Floor-bound races change no figure by more than the third decimal.
These are 13 CIs with no family-wise correction. The year-by-year spread (tail β from 1.008 to 1.077) is within what
the CIs allow.

### Walk-forward: the slopes of each fold's training rows

The first fold trains to 2019-12-29 (19,259 races): body β 0.9900 [0.9720, 1.0080], increment +0.0493 [+0.0274,
+0.0711], tail β 1.0393 [1.0196, 1.0590]. Across the 28 folds the increment runs from +0.0445 to +0.0493 and the tail
β from 1.0346 to 1.0462, with the last fold (trained to 2026-09-26) at +0.0452 and 1.0449. The no-floor-bound folds
run from +0.0452 to +0.0500 and 1.0340 to 1.0454. Every fold's increment CI excludes 0. The per-fold figures are
registry metrics, one per fold, named `*[test_year=Y]`.

### Out-of-fold winner log loss, ΔLL of the tail β

ΔLL is the comparator's mean log loss minus the tail β's, in nats per race. Positive means the tail β is better.

| Subset | Races | Mean log loss (tail β) | ΔLL vs normalised | Lower (5%) | Upper (95%) | ΔLL vs power-β | Lower (5%) | Upper (95%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all races | 23,799 | 1.92770 | +0.000590 | +0.000245 | +0.000933 | +0.000321 | +0.000005 | +0.000629 |
| no floor-bound races | 23,680 | 1.93342 | +0.000560 | +0.000215 | +0.000903 | +0.000318 | -0.000001 | +0.000630 |

The comparators' means are 1.92829 (normalised) and 1.92802 (power-β) on all races, as in #77.

### Out-of-fold tail, runners above decimal 51.0

Observed against expected winners under the tail β, beside #77's table. This is a point estimate, with no bounds.

| Subset | Model | Tail runners | Observed | Expected | O / E |
| --- | --- | ---: | ---: | ---: | ---: |
| all races | tail β | 111,564 | 654 | 651.9 | 1.003 |
| no floor-bound races | tail β | 110,885 | 653 | 649.1 | 1.006 |

## Alternatives rejected and why

- **A hinge slope, continuous at 51.0.** The model is the ticket's and the tests': the extra slope multiplies x
  itself, so at 51.0 the model also steps the log-weight down by increment × ln 51 (about 0.18 at the increment of
  0.045). The increment therefore mixes a slope change and a level step in the tail. A hinge (x - ln(1/51) above 51.0)
  separates them. It was not pre-registered, so it is not run here: it would be its own family. It is the main
  follow-up below.
- **A spline in ln q** (D4's calibrated market). That is #275, on the same folds and tail table.
- **Meeting-clustered slope CIs.** The pre-registered CI is Wald from the observed information, as in #77. The ΔLL
  bounds do resample meetings. Clustered slope intervals would be wider: #276.
- **A threshold other than 51.0, or several thresholds.** The owner signed 51.0 (#77 (5)). The threshold is in the
  frozen spec, so another value is another family.
- **ΔLL by year.** The ticket asks for the slopes by year and ΔLL overall and by subset. Per-year ΔLL was not
  pre-registered.
- **Dropping a year with no tail winner.** On real data every year has 65 to 111 tail winners, so each year is
  identified. The code records a year without any tail winner as a table row, not a failed variant, because its
  increment has no maximum.

## What this rules out

- **A single slope for the whole price.** The increment is +0.0450 [+0.0302, +0.0599], so a zero increment is out and
  the tail's slope is at least 1.032 at the lower CI. The sign is the one #77's O/E pointed to: tail runners are
  over-priced, so the model gives them less than a slope-1 price does.
- **The body being off.** The body slope is 0.9999 [0.9877, 1.0121]: the CI rules out a body β above 1.012. #77's
  power β of 1.0206 [1.0104, 1.0308] is mostly the tail pulling one exponent up: with a separate tail slope the body
  is not distinguishable from 1. This is a statement about this two-slope form, with its step at 51.0.
- **A sizeable out-of-fold gain.** The tail β beats power-β by +0.000321 nats per race [+0.000005, +0.000629], and
  without floor-bound races the lower bound is -0.000001. The upper bound rules out a gain above about 0.0006 nats per
  race. Against normalised the gain is +0.000590 [+0.000245, +0.000933]. These are far below any MDE and do not argue
  for adopting the slope on log loss.
- **A tail bias left in the price after the tail β.** Observed over expected tail winners is 1.003 (654 against 651.9),
  against 0.83 and 0.87 under #77's de-vigs. That point estimate carries no bounds, and it is on the data that set the
  slope's scale.
- **Floor-bound races driving the tail.** Without them the increment is +0.0453 [+0.0304, +0.0601].
- **What is not ruled out.** The tail β is not stable by year to better than about ±0.05, and its CIs treat races as
  independent. These are the same races as #77's tail table, so the two tables are one observation, not two. The
  result does not show that the tail β adds anything a calibrated market (D4) lacks.

## Follow-ups filed

Each is proposed under epic #72, `status:proposed`; the orchestrator files them.

- A tail slope that is continuous at 51.0 (a hinge on x - ln(1/51)), its own family, to separate the slope change from
  the level step.
- #275: D4's spline on the same folds and tail table (already filed).
- #276: meeting-clustered CIs for the slopes (already filed).

---

# Experiment report: D4's calibrated market on the crowd-calibration folds (Japan, #275)

Ticket #275 (epic #72), from #77's review. #77 compared the normalised, power-β and Shin de-vigs of the final Win
price out of fold. This study scores D4's calibrated market beside them on the same folds, with the same ΔLL and the
same tail table. The calibrated market is the natural cubic spline in ln q of `models.v0.calibration`
(`fit_win_calibration`, reused, 5 knots), q the normalised share of the race's priced runners, refitted on each fold's
training rows. **Nothing is adopted.** The owner picks the baseline de-vig (08 §9.4). Every figure is an aggregate
over the JRA results store (netkeiba race results; no JRDB, no JRA-VAN). No runner or race row appears.

- Registry family `japan-calibrated-market-final-win`, spec sha256
  `46f53f0b0fc33aa562a6efa9062b8d00ddd71cfb79ead9f28e30ed81022aa9d9`. It is its own family, not #77's. The spec was
  frozen in the registry before any development row was read. It holds the spline (model, knots, knot quantile range,
  dead-heat rows), the de-vigs, the comparator, the ΔLL rule, the subsets, the tail, the floor, the fold scheme, the
  bootstrap and the owner decisions.
- Code: `countries/japan/studies/calibrated_market.py`; the records carry git `19deb19`. To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.calibrated_market`. A rerun on the same store finds its
  records already present and appends none.
- Store: `processed/jra/runners.parquet@6f3cec95347e0132`, `processed/jra/races.parquet@9cce511831074b64`. These are
  later builds than #77's (`b3226f2a22bfce26`, `4e6a394a57e3101a`): the store was rebuilt after #77, and it adds
  three development days (see Data and split). The full report is under the data directory, at
  `reports/calibrated-market-japan-46f53f0b0fc3.md`. The eight records are in `docs/experiments/registry/japan/`.
- Holdout: development days only. `calibration.read_development` drops holdout days, and `LockedHoldout.open` is not
  called. No holdout outcome or price was read.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| D4 | The Win baseline is the calibrated market (a spline in ln q), fitted in-fold. This study scores it beside the de-vigs and adopts nothing. | `docs/problem-statement.md` §2, #14 |
| #77 (4) | The no-floor-bound subset drops races with a runner at odds 1.1 or under from fitting and from scoring. | #77, 2026-11-25 |
| #77 (5) | The tail is a runner priced above decimal 51.0 (`tail_min_odds` in `config/calibration.yaml`). | #77, 2026-11-25 |
| Floor 1.1 | A priced runner at or under odds 1.1 + 1e-9 is floor-bound (the Win pool's minimum dividend over the unit stake). | `config/jra.yaml`; #77 (4) |
| D6 | Winner log loss per race; a dead heat scores each of its k co-winners at 1/k. | §3, #14; the #69 answer (2026-11-24) |
| D7 | Bootstrap clusters are meetings (venue x day); bounds are 95% one-sided; 10,000 replicates (`config/scoring.yaml`). | §3, #14 |
| D9 | Walk-forward quarters, refit quarterly, expanding window, 1-week embargo; the first scored quarter is 2020-Q1 (`config/folds.yaml`). | §4, #14; the #70 answer |
| D10 | Races before 2014-06-07 are excluded wherever the price enters. | §4, #14 |
| D11, D12, D14, D15 | Holdout A2 from 2026-11-28, fully locked for outcomes and prices, opened only by the owner. Not opened here. | §5, #14; the #71 answer (`config/holdout.yaml`) |
| D21 | No figure derived from a licensed source appears in the repository. None is used here. | #14 |

The comparison itself (the spline beside #77's three de-vigs, ΔLL against normalised, #77's tail table) is the
ticket's outline. The bootstrap seed (275) is an implementation choice for reproducibility: `scoring.yaml` signs no
seed. The spline's knot count (5) and knot quantile range (0.05 to 0.95) are `fit_win_calibration`'s defaults, D4's
live model.

## Hypothesis

The null is the starting point: the spline adds nothing to normalised 1/odds, so its ΔLL is 0 and its tail O/E is
normalised's. The frozen spec pre-registered the measures (each de-vig's and the spline's out-of-fold ΔLL against
normalised, and the tail's observed vs expected winners, with and without floor-bound races), not their directions.
The direction and failure rules below were written with this report, after the run. #77 and #285 had already looked
at the same development races and found the tail over-priced, so this study cannot confirm the direction
independently.

- **Direction.** A spline in ln q can bend where one exponent cannot, so it should take up the tail bias #77 found
  (O/E 0.83 under normalised, 0.87 under power-β) and gain at least power-β's ΔLL.
- **Failure rules** (written after the run): the spline fails to beat normalised if its ΔLL lower bound is at or
  below 0. It fails to absorb the tail if the (O - E) per race bounds exclude 0.

## Variants tried

The family has 8 variants (`Registry.family_size`) and 8 records, all scored, on one run: normalised, the D4 spline,
power-β and Shin, each on all races and on no floor-bound races. No variant failed and no run was repeated.

Data and split:
- Development days from 2014-06-07 to 2026-11-23: 43,130 races and 601,265 priced starters. 2,156 scratched runners
  are out; 66 dead heats. 195 floor-bound runners sit in 195 races (0.45%). That is #77's rows plus 72 races on three
  development days after #77's run (2026-11-21, 22 and 23, 24 races each), all before the holdout start.
- Out of fold covers 28 quarterly folds from 2020-Q1 to 2026-Q4 (the last is partial, to 2026-11-23). That makes
  23,871 scored races in 1,991 meetings (23,752 without floor-bound races). The first fold trains on 19,259 races to
  2019-12-29, the last on 42,578 races to 2026-09-26.
- Dead heats: the spline is fitted on #77's development rows, so a race with a dead heat for first is kept and each
  co-winner counts as a winner in the fit, as in power-β's fit. D4's live `run_fit`
  (`models/v0/calibration.py`, `_development_frame`) leaves those races out of its fit. The spec states this, and all
  four de-vigs see the same races.
- The ΔLL and tail bounds are the one-sided 5% and 95% quantiles of the paired bootstrap over meetings (D7, seed
  275).

### Out-of-fold winner log loss, ΔLL against normalised

ΔLL is normalised's log loss minus the de-vig's, in nats per race. Positive means better than normalised.

| Subset | De-vig | Races | Mean log loss | ΔLL vs normalised | Lower (5%) | Upper (95%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all races | normalised | 23,871 | 1.92850 | | | |
| all races | D4 spline (refit per fold) | 23,871 | 1.92752 | +0.000977 | +0.000499 | +0.001446 |
| all races | power-β (refit per fold) | 23,871 | 1.92823 | +0.000267 | +0.000105 | +0.000426 |
| all races | Shin | 23,871 | 1.93974 | -0.01125 | -0.01350 | -0.00901 |
| no floor-bound races | normalised | 23,752 | 1.93417 | | | |
| no floor-bound races | D4 spline (refit per fold) | 23,752 | 1.93324 | +0.000927 | +0.000445 | +0.001399 |
| no floor-bound races | power-β (refit per fold) | 23,752 | 1.93393 | +0.000240 | +0.000086 | +0.000392 |
| no floor-bound races | Shin | 23,752 | 1.94561 | -0.01144 | -0.01369 | -0.00920 |

Power-β and Shin match #77's figures (+0.000269 and -0.01125 on all races) on the rebuilt store with the 72 added
races. That is the same backtest recomputed, not an independent sample: per-fold mean log loss for normalised,
power-β and Shin equals #77's to 1e-9 in 27 of 28 folds, and only the last fold differs, by the 72 added races. #77's
ledger rows stay single-run. The per-fold training β runs from 1.0096 to 1.0213 as in #77. The spline's per-fold mean log loss is a
registry metric, one per fold.

### Out-of-fold tail, runners above decimal 51.0

#77's layout. Observed against expected winners under each de-vig. (O - E) per race is the tail's observed minus
expected winners per scored race, with meeting-clustered one-sided bounds. O / E is a point estimate only.

| Subset | De-vig | Tail runners | Observed | Expected | O / E | (O - E) per race | Lower (5%) | Upper (95%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all races | normalised | 111,905 | 655 | 790.7 | 0.828 | -0.00568 | -0.00742 | -0.00396 |
| all races | D4 spline | 111,905 | 655 | 643.7 | 1.018 | +0.00048 | -0.00125 | +0.00220 |
| all races | power-β | 111,905 | 655 | 754.6 | 0.868 | -0.00417 | -0.00590 | -0.00245 |
| all races | Shin | 111,905 | 655 | 299.8 | 2.185 | +0.01488 | +0.01315 | +0.01661 |
| no floor-bound races | normalised | 111,226 | 654 | 786.3 | 0.832 | -0.00557 | -0.00731 | -0.00383 |
| no floor-bound races | D4 spline | 111,226 | 654 | 641.1 | 1.020 | +0.00054 | -0.00119 | +0.00228 |
| no floor-bound races | power-β | 111,226 | 654 | 752.2 | 0.869 | -0.00414 | -0.00587 | -0.00240 |
| no floor-bound races | Shin | 111,226 | 654 | 298.6 | 2.190 | +0.01496 | +0.01322 | +0.01670 |

## Alternatives rejected and why

- **ΔLL of the spline against power-β, with its own bounds.** The spec's comparator is normalised, as in #77. The
  point difference of the means is 0.00071 nats per race on all races (1.92823 - 1.92752), with no bounds: it is
  derived, not a pre-registered measure. A paired comparison is a follow-up.
- **The spline against #285's tail β.** #285 scored on #77's store builds, not these, so the races differ by 72 and
  the two cannot be paired here. The paired comparison is a follow-up.
- **Dropping dead-heat races, as D4's live `run_fit` does.** Not run. The spec keeps #77's rows so the four de-vigs
  score the same races. 66 of 43,130 races have a dead heat, so the effect on the fit is likely small, but it is not
  measured.
- **Another knot count or knot range.** The spec freezes D4's live defaults (5 knots, quantiles 0.05 to 0.95).
  Another setting is another family.
- **Pinning #77's store builds.** The store is rebuilt in place, so #77's builds are not available to read. The
  study reads the current development rows through `read_development`, and power-β and Shin recompute #77's figures on them.

## What this rules out

- **The spline adding nothing over normalised.** Its ΔLL is +0.000977 nats per race [+0.000499, +0.001446]; without
  floor-bound races +0.000927 [+0.000445, +0.001399]. The upper bound rules out a gain above about 0.0015 nats per
  race. That is far below any MDE: the gain is real in sign and small in size.
- **A tail bias left after the spline.** Its tail O/E is 1.018 (655 against 643.7), and the (O - E) per race bounds
  cover 0 [-0.00125, +0.00220]. Scaled to the 23,871 scored races, the bounds are about -30 to +53 tail winners
  against 655 observed. Normalised and power-β leave the tail over-priced, with bounds below 0. Shin over-corrects.
- **Floor-bound races driving any of this.** Without them the spline's ΔLL moves by 0.00005 and its tail O/E by
  0.002.
- **What is not ruled out.** Whether the spline beats power-β or #285's tail β by more than chance is not tested
  here (no paired bounds). The spline's fit and #77's and #285's tables use the same development races, so they
  are one look at the data, not independent confirmations. The direction written above came after those looks. The
  holdout is untouched, so nothing here is an out-of-sample confirmation in 08 §5's sense.

## Follow-ups filed

Each is filed under epic #72, `status:proposed`.

- #320: a paired comparison of D4's spline against power-β and #285's tail β, on one store build, as its own family
  (ΔLL with meeting-clustered bounds).
- #321: the spline fitted on `run_fit`'s rows (dead-heat races left out), to check the study's spline matches D4's
  live fit, or align the two.
