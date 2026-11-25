# Experiment report: the power β with meeting-clustered intervals (Japan, #276)

Ticket #276 (epic #72). #77's β intervals are 95% Wald intervals, β ± 1.96 se, that treat races as independent, while
its ΔLL bounds are clustered by meeting (D7). This study re-reports #77's β tables (by year and era, by E4 segment, by
E3 entropy bin) with two intervals per row: #77's Wald one, unchanged, and one from a meeting-clustered sandwich se.
**Nothing is adopted.** Whether β's CIs adopt the meeting clusters is the owner's call, so #77's family
`japan-crowd-calibration-final-win`, its spec and its Wald tables are unchanged. Every figure is an aggregate over the
JRA results store (netkeiba race results; no JRDB, no JRA-VAN). No runner or race row appears.

- Registry family `japan-beta-clustered-final-win`, spec sha256
  `d7bf2d8a9ee9c0a289b080f95e453af4616906c067091c3681143bf0437c251c`. It is its own family, not #77's. The spec was
  frozen in the registry before any development row was read. It holds the β fit, both intervals' formulas, the
  segments, the E3 cut points and their rule, the floor, the fold scheme and bootstrap carried from #77 for
  provenance (unused: the fits are in sample and the clustered se is a sandwich), and the owner decisions.
- Code: `countries/japan/studies/beta_clustered.py` and `fit_power_beta(..., cluster=)` in `calibration.py`; the 11
  records carry git `7eb1531`. To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.beta_clustered --run`.
- Store: `processed/jra/runners.parquet@6f3cec95347e0132`, `processed/jra/races.parquet@9cce511831074b64`. This is a
  later build than #77's (`@b3226f2a...`, `@4e6a394a...`): development days now run to 2026-11-23, so the Wald rows
  here can differ from #77's in the last digits (era β 1.0205 on 43,130 races, against 1.0206 on 43,058).
- Holdout: development days only. `calibration.read_development` drops holdout days, and `LockedHoldout.open` is not
  called. No holdout outcome or price was read.
- The full report is under the data directory, at `reports/beta-clustered-japan-d7bf2d8a9ee9.md`. The 11 records are
  in `docs/experiments/registry-japan.md`.

## The clustered interval

β is the conditional-logit maximum on x = ln(1/odds), as in #77. Its Wald se is the observed information to the power
-1/2. The clustered se is the sandwich I⁻¹ (Σ_c S_c²) I⁻¹, square-rooted, where S_c is the sum over meeting c's fitted
races of each race's score at β̂ (x of the winner minus the winners' count times the race's mean x). A meeting is
venue × day (`races.meeting_id`), as in D7. It takes the small-sample factor G/(G - 1) on the variance, G being the
meetings of the fitted races. Both intervals are 95% two-sided normal, β ± 1.96 se. `se ratio` is the clustered se
over the Wald se: above 1 the Wald interval is the narrower.

The fit is tested on synthetic data with planted within-meeting correlation (each meeting draws its winners at its own
β around 1, sd 0.4): there the clustered se is about 1.5 times the Wald se and matches an independent first-principles
sandwich within 2%; with no correlation the ratio is about 1.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| D7 | Bootstrap clusters are meetings (venue x day). This study uses the same clusters for a sandwich se, not a bootstrap. Whether β's CIs adopt them is not signed: the owner's call. | §3, #14; `config/scoring.yaml` |
| D10 | Races before 2014-06-07 are excluded wherever the price enters. The era table covers the post-reform era only. | §4, #14 |
| D11, D12, D14, D15 | Holdout A2 from 2026-11-28, fully locked for outcomes and prices, opened only by the owner. Not opened here. | §5, #14; the #71 answer |
| D21 | No figure derived from a licensed source appears in the repository. None is used here. | #14 |
| #77 (1), (2), (4) | The E4 segments, the E3 cut points (0.69521, 0.74673, 0.78515, 0.82154) and the floor-bound subset (odds 1.1 or under) are #77's, from `config/calibration.yaml`. | #77, 2026-11-25 |

## Hypothesis

The null is the starting point: races within a meeting are independent given the price, so the clustered se equals the
Wald se (ratio 1) and #77's Wald intervals are right. The frozen spec pre-registered the comparison (se cluster over
the Wald se, per row), not its direction. The direction below is the ticket's concern, written before the run: the
races of a meeting share going, weather and the day's crowd, so their scores may move together, and then the Wald
interval would be too narrow (ratio above 1).

- **Failure rule** (written with this report): the hypothesis fails if the ratio is not clearly above 1 across the
  year, era and segment rows, and the clustered intervals do not change which of #77's β intervals exclude 1.

## Variants tried

The family has 11 variants and 11 records, all scored, none failed: β by year and era on all races and on no
floor-bound races (2), one per E4 segment (8), and E3's entropy bins (1). One run, no failed or aborted attempt.

Data and split:
- Development days from 2014-06-07 to 2026-11-23: 43,130 races in 3,597 meetings and 601,265 priced starters. 2,156
  scratched runners are out; 66 dead heats; no void or unpriced race. 195 floor-bound runners sit in 195 races.
- Every fit is in sample, one per group over every development race of the group. No fold, no resampling.

### β by year and era (all races)

| Year | Races | Meetings | β | Wald 95% CI | Clustered 95% CI | se ratio |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| 2014 | 1,990 | 166 | 1.0151 | [0.9686, 1.0616] | [0.9731, 1.0572] | 0.905 |
| 2015 | 3,454 | 288 | 1.0079 | [0.9718, 1.0440] | [0.9733, 1.0425] | 0.958 |
| 2016 | 3,454 | 288 | 1.0111 | [0.9761, 1.0460] | [0.9772, 1.0449] | 0.968 |
| 2017 | 3,455 | 288 | 1.0196 | [0.9843, 1.0549] | [0.9845, 1.0548] | 0.995 |
| 2018 | 3,454 | 288 | 1.0168 | [0.9812, 1.0524] | [0.9857, 1.0478] | 0.872 |
| 2019 | 3,452 | 288 | 1.0071 | [0.9712, 1.0429] | [0.9688, 1.0453] | 1.067 |
| 2020 | 3,456 | 289 | 0.9934 | [0.9578, 1.0290] | [0.9606, 1.0263] | 0.922 |
| 2021 | 3,456 | 288 | 1.0325 | [0.9959, 1.0692] | [0.9978, 1.0672] | 0.946 |
| 2022 | 3,456 | 288 | 1.0558 | [1.0187, 1.0929] | [1.0189, 1.0928] | 0.996 |
| 2023 | 3,456 | 288 | 1.0255 | [0.9890, 1.0620] | [0.9904, 1.0605] | 0.961 |
| 2024 | 3,454 | 288 | 1.0446 | [1.0079, 1.0813] | [1.0121, 1.0772] | 0.887 |
| 2025 | 3,455 | 288 | 1.0190 | [0.9830, 1.0550] | [0.9829, 1.0552] | 1.005 |
| 2026 | 3,138 | 262 | 1.0191 | [0.9811, 1.0570] | [0.9812, 1.0569] | 0.999 |
| Era (2014-06-07 onward) | 43,130 | 3,597 | 1.0205 | [1.0103, 1.0307] | [1.0107, 1.0303] | 0.963 |

Without floor-bound races the years' ratios run from 0.878 to 1.070 (median 0.964), and the era row is β 1.0195,
Wald [1.0093, 1.0297], clustered [1.0097, 1.0294], ratio 0.964. In both subsets the years whose CIs exclude 1 are 2022
and 2024, under the Wald and under the clustered interval alike.

### β in vs out of each E4 segment

Ratios are the clustered se over the Wald se. Δβ is not re-derived here.

| Segment | In: races (meetings) | In: β, clustered 95% CI | In: ratio | Out: β, clustered 95% CI | Out: ratio |
| --- | ---: | --- | ---: | --- | ---: |
| R12 | 3,584 (3,584) | 1.0814 [1.0435, 1.1193] | 0.997 | 1.0157 [1.0054, 1.0259] | 0.968 |
| R11pm1 | 10,772 (3,594) | 1.0237 [1.0027, 1.0446] | 0.982 | 1.0196 [1.0085, 1.0308] | 0.963 |
| g1_day_undercard | 3,201 (292) | 1.0527 [1.0161, 1.0893] | 0.967 | 1.0171 [1.0069, 1.0273] | 0.962 |
| field_16_plus | 17,742 (3,535) | 1.0163 [1.0016, 1.0310] | 0.970 | 1.0240 [1.0106, 1.0373] | 0.972 |
| debut | 3,741 (2,518) | 1.0066 [0.9743, 1.0389] | 0.973 | 1.0220 [1.0116, 1.0323] | 0.970 |
| heavy_going | 1,577 (329) | 0.9791 [0.9264, 1.0318] | 1.019 | 1.0222 [1.0122, 1.0321] | 0.961 |
| handicap | 2,454 (2,143) | 0.9594 [0.9150, 1.0038] | 1.000 | 1.0238 [1.0138, 1.0338] | 0.957 |

R12 has one race a meeting, so its clustered se is the race-level sandwich, and the ratio near 1 is that sandwich
agreeing with the information. Per venue (clustered 95% CI, ratio): TOK 1.0566 [1.0316, 1.0815] 0.974; NAK 1.0332
[1.0082, 1.0581] 0.930; NII 1.0275 [0.9941, 1.0608] 0.993; HAK 1.0227 [0.9733, 1.0720] 0.953; HSN 1.0188 [0.9925,
1.0450] 0.986; CKY 1.0153 [0.9847, 1.0460] 0.961; SAP 1.0089 [0.9650, 1.0529] 0.916; KOK 1.0069 [0.9704, 1.0433] 0.985;
KYO 0.9907 [0.9635, 1.0178] 0.957; FKS 0.9870 [0.9513, 1.0227] 0.923.

### β by E3 entropy bin

| Bin (normalised entropy) | Races | Meetings | β | Wald 95% CI | Clustered 95% CI | se ratio |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| 1: [0, 0.6952) | 9,199 | 3,283 | 1.0575 | [1.0387, 1.0763] | [1.0389, 1.0761] | 0.989 |
| 2: [0.6952, 0.7467) | 8,656 | 3,347 | 1.0199 | [0.9984, 1.0414] | [0.9989, 1.0410] | 0.981 |
| 3: [0.7467, 0.7851) | 8,506 | 3,330 | 1.0159 | [0.9925, 1.0394] | [0.9936, 1.0382] | 0.952 |
| 4: [0.7851, 0.8215) | 8,156 | 3,270 | 0.9972 | [0.9715, 1.0229] | [0.9720, 1.0224] | 0.983 |
| 5: [0.8215, 1] | 8,613 | 3,199 | 0.9696 | [0.9415, 0.9978] | [0.9421, 0.9972] | 0.982 |

## Alternatives rejected and why

- **A meeting bootstrap for β.** It would resample 3,597 meetings and refit β each time, and D7's bootstrap is
  one-sided for ΔLL. The sandwich is one pass over the same score contributions, two-sided like #77's Wald interval,
  and gives the comparison per row without a seed. Not run.
- **A race-level (HC) sandwich only.** It checks the information against the scores but cannot show a meeting effect.
  It is the R12 row's case and is covered by that row.
- **Clustering by race day or by venue.** D7 signs meetings (venue x day) and no other cluster was pre-registered. Not
  run.
- **Re-reporting by changing #77's tables.** Excluded by the ticket: #77's family and spec are frozen, and adoption
  is the owner's.

## What this rules out

- A meeting-level dependence in β's score large enough to widen the Wald interval. Across the 26 year rows the ratio
  runs from 0.87 to 1.07, with a median of 0.96; no row's clustered interval is more than 7% wider than its Wald
  interval, against about 1.5 times in the planted synthetic case. The era row, with 3,597 meetings, has a ratio of
  0.963.
- That the Wald intervals overstate how well β is known, in the other direction: the clustered intervals are the
  same width or a few percent narrower, and no conclusion of #77's β tables changes. The years whose CI excludes 1
  (2022, 2024), R12's in-segment β, the top and bottom entropy bins' β against 1 and the era β's CI excluding 1 are
  the same under both intervals; handicap's in-segment CI still covers 1.
- It does not rule out dependence the sandwich cannot see in a year's 288 meetings: a sandwich se from G meetings has
  a relative error of roughly 1/√(2G), about 4% here, so a ratio of 0.95 to 1.05 in one row is consistent with 1. Nor
  does it say which interval to publish: that is the owner's call. This is a single run on #77's development races;
  β by segment is in sample, with no family-wise correction, as #77's was.

## Follow-ups filed

None. Adopting the clustered interval for β (in `calibration.py`'s tables, or in the A6 family's segment β) waits on
the owner's call; if made, it needs a new spec, because #77's family stays frozen.
