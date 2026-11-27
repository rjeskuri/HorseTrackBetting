# Experiment report: floor, tail and favourite avoidance rules L2, L3 and L7 on the same races (Japan, #82)

Ticket #82 (epic #72; 08 §8 A4, price part; §5.1 L2, L3, L7; §4 E3, E7, E12, E14, R3; 04 §1.3). Family L, #81's
paired design: each rule is compared with its stated comparator on the same races, by each side's mean return per
unit staked (payout / stake) at final dividends and the paired per-race difference between them, each with
meeting-clustered one-sided bounds. Under the owner's reframe of 2026-11-27, every option is a ranked row: the rule's
tickets and its comparator's are both rows, an avoid rule drops nothing from the data, and there is no pass/fail.
**Nothing here is adopted.** Every figure below is an aggregate over the JRA results store (netkeiba race results,
final Win odds and dividends) and the shipped boost calendar. The 100-vs-110 prediction reads the JRA-VAN H1 votes;
its figures are withheld (D21) and are in a private report only. There is no race or runner row here.

- Registry family `japan-family-l-floor-tail-favourite`, spec sha256
  `a3bfd6a65fa8aeff537b6fe0eb5c82e7b8c5ae80e7ed490a3fef93bdab49aae7`. `run_study` froze the spec before it read any
  row; the freeze line is the record of the owner's inputs (there is no sign-off key). The spec holds the rules, the
  tail odds and ranks, the entropy cut points, the comparators, the selections, the decision time (T-2), the board
  fields, the bootstrap, the Place-votes reading and the 50 variant names (`family_size` 50).
- Code at `88c40aa` (`countries/japan/studies/avoidance_price.py`, built on #81's `studies/avoidance.py`). Owner
  inputs are in `countries/japan/config/avoidance_price.yaml`. To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.avoidance_price --run`. The family is frozen, so a rerun with
  this spec adds runs of the same variants. Another spec is refused (`FamilyAlreadyFrozen`).
- Store: `processed/jra/races.parquet@9cce511831074b64`, `processed/jra/runners.parquet@6f3cec95347e0132`,
  `processed/jra/dividends.parquet@b3d635352bb007df`, `processed/jra/odds_timeseries (2 days)@7bd8e900bd0a1e6e`
  (the same builds as #81's run). The JRA-VAN build read by the 100-vs-110 check is named in the private report.
- The full report is under the data directory, at `reports/floor-tail-favourite-japan-a3bfd6a65fa8.md`, with
  `-dropped.csv` beside it. The 100-vs-110 table is in `reports/floor-tail-favourite-japan-a3bfd6a65fa8-plus10-jravan.md`
  and its `.csv` (private, JRA-VAN-derived). Every record is in
  `docs/experiments/registry/japan/japan-family-l-floor-tail-favourite.md`; the two 100-vs-110 records show no values
  there (licensed source).
- One run, 2026-11-27: 128 s, 1.38 GB peak resident memory. A first attempt at `f0d90a1` was stopped before it
  scored or appended anything (it would have recorded the at-t 100-vs-110 check as scored with zero favourites on an
  empty board); its only trace is the family's freeze line, which this run's identical spec reused.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| D2 | The board at t is read at post - 2 min (`config/decision.yaml` `primary_min`). | `docs/problem-statement.md` §1, #14 |
| D3 | Every JRA pool is in research scope, exotics at tier 2. | §1, #14; amended 2026-11-24 (#127) |
| D7 | Bootstrap clusters are meetings (venue x day); bounds are 95% one-sided (`config/scoring.yaml`). | §3, #14 |
| D10 | Only races from 2014-06-07. | §4, #14 |
| D11, D14, D15 | Holdout A2 from 2026-11-28 is fully locked for outcomes and prices. It is not read or opened here. | §5, #14; `config/holdout.yaml` |
| D17 | The family size counts every variant, each cut point and pool included: 50. Every record carries it as `variants_searched`. | §6, #14 |
| D21 | No figure from a licensed source is in the repository. The 100-vs-110 prediction is JRA-VAN-derived and withheld. | §8, #14; owner 2026-11-26 (strict) |
| Floor | A board price at or below 1.1 is at the floor. | owner, 2026-11-24 |
| #82 answer | `l3_tail_odds: [51.0, 101.0]` (fractional 50/1 and 100/1, #77's convention); `l7_entropy_cut_points`: #77's quintiles (0.6952092433071905, 0.7467255002267639, 0.7851488760086348, 0.8215443360884407, from `config/calibration.yaml`), applied to final and board-at-t entropy; `sets: every_set`; `favourite_ties: lower_horse_no`; `l7_favourite_boxes: top3_box`; `l7_skipped_race: stake_back`; `plus10_inputs: jravan_pool_votes`. Confirmed: L3's ranks and L7's box read only the WIN board at t (BRACKET_QUINELLA excluded); L2 reads the PLACE and Wide boards at t only. | #82, owner comment of 2026-11-26 |
| Reframe | Every rule x variant is a ranked row against its comparator with n, CI and the variants searched; no D18 pass/fail; no sign-off key; an avoid rule drops nothing from the data. | #82, owner comment of 2026-11-27 |

Fixed from the request without a choice: the exotic tail ranks 9th or worse and 13th or worse by WIN odds (E12); a
handicap is `races.weight_type == "handicap"` (#77).

## Hypothesis

The frozen spec's `hypotheses`, in full:

- **L2 (E14, R3).** "The return of the base strategy less its tickets at the floor against the same tickets; R3: the
  100-vs-110 check of floor-bound placed favourites."
- **L3 (E12).** "The return of the strategy less its tail tickets against the strategy with the tail."
- **L7 (E3, E7).** "The return with no favourite Place in the flattest-entropy quintile, and no favourite Win, Place
  or box in handicaps, against the same tickets in all races."

The null is the starting point: no difference between a rule and its comparator. A positive paired difference is the
rule returning more per unit staked than its comparator on the same races.

## Variants tried

The family has 50 variants (`Registry.family_size`) and 50 records, all from one run at git `88c40aa`. 24 scored, all
at the final price. 26 are recorded as failed with data pending: every board-at-t variant.

Data and split:
- Development days from 2014-06-07 to 2026-11-23: 43,130 run races on 3,597 meetings, 601,265 starters, 2,454
  handicaps. The dividend rows read are WIN, PLACE, QUINELLA_PLACE, QUINELLA, FORECAST, TRIO and TIERCE: 474,675.
- No day in the store is a holdout day (the store ends 2026-11-23). The readers drop holdout days anyway; the votes
  reader refuses a holdout race; `LockedHoldout` was not opened.
- Nothing is fitted: every development race is used once (`paired_same_races:development`).

Selections:
- `final (look-ahead)`: the choice reads the final Win odds (`runners.win_odds`), which are not known at a decision
  time. Every scored row below is look-ahead.
- `board at t (T-2)`: odds_timeseries through `read_asof` at the race's `post_utc` - 2 min. 24 races on the two
  logged days (2026-11-21 and 2026-11-22) have WIN, PLACE and Wide boards, but `post_utc` is null on every row (no
  post-time sidecar, #352), so there is no t and the board at t is empty. All 26 board-at-t variants are data pending.

Each side's return is its mean over the paired races of payout / stake; the paired difference is the mean of rule
minus comparator. Lower and upper are the 5% and 95% quantiles of the bootstrap over meetings (D7; 10,000 replicates,
seed 82), each a one-sided 95% bound, per variant, with no family-wise correction. Under `stake_back`, an L7 race the
rule skips returns its stake (1.0).

### Paired comparisons at the final price (look-ahead)

The base strategy is one unit on every starter (WIN, PLACE) or every combination of the starters (exotics), flat.
L2's comparator is the same tickets with those at the floor; L3's, the same tickets with the tail; L7's, the same
favourite tickets in every race. "Races changed" counts the paired races where the rule drops a ticket or skips the
race. No race was dropped from any scored variant.

| Rule | Pool | Cut | Races | Rule return [lower, upper] | Comparator return [lower, upper] | Paired difference | Lower (5%) | Upper (95%) | Races changed |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| L2 | WIN | floor | 43,130 | 0.7245 [0.7138, 0.7353] | 0.7247 [0.7140, 0.7355] | -0.00018 | -0.00030 | -0.00004 | 195 |
| L3 | WIN | above 51 | 43,130 | 0.7941 [0.7877, 0.8005] | 0.7247 [0.7140, 0.7355] | +0.06939 | +0.05864 | +0.08002 | 42,208 |
| L3 | PLACE | above 51 | 43,130 | 0.7968 [0.7936, 0.8000] | 0.7316 [0.7269, 0.7364] | +0.06520 | +0.06040 | +0.06992 | 42,208 |
| L3 | WIN | above 101 | 43,130 | 0.7860 [0.7778, 0.7941] | 0.7247 [0.7140, 0.7355] | +0.06127 | +0.05240 | +0.06986 | 38,404 |
| L3 | PLACE | above 101 | 43,130 | 0.7862 [0.7825, 0.7901] | 0.7316 [0.7269, 0.7364] | +0.05464 | +0.05070 | +0.05852 | 38,404 |
| L3 | QUINELLA_PLACE | rank 9 or worse | 43,130 | 0.7613 [0.7561, 0.7666] | 0.6189 [0.6126, 0.6255] | +0.14238 | +0.13375 | +0.15059 | 41,120 |
| L3 | QUINELLA | rank 9 or worse | 43,130 | 0.7342 [0.7258, 0.7427] | 0.6018 [0.5910, 0.6129] | +0.13244 | +0.11902 | +0.14535 | 41,120 |
| L3 | FORECAST | rank 9 or worse | 43,130 | 0.7273 [0.7184, 0.7362] | 0.5989 [0.5870, 0.6113] | +0.12837 | +0.11382 | +0.14240 | 41,120 |
| L3 | TRIO | rank 9 or worse | 43,130 | 0.7031 [0.6930, 0.7135] | 0.5350 [0.5231, 0.5473] | +0.16813 | +0.15248 | +0.18334 | 41,120 |
| L3 | TIERCE | rank 9 or worse | 43,130 | 0.7060 [0.6943, 0.7179] | 0.5664 [0.5513, 0.5825] | +0.13958 | +0.12054 | +0.15776 | 41,120 |
| L3 | QUINELLA_PLACE | rank 13 or worse | 43,130 | 0.6934 [0.6875, 0.6994] | 0.6189 [0.6126, 0.6255] | +0.07447 | +0.06860 | +0.08016 | 30,701 |
| L3 | QUINELLA | rank 13 or worse | 43,130 | 0.6825 [0.6718, 0.6933] | 0.6018 [0.5910, 0.6129] | +0.08068 | +0.07147 | +0.08975 | 30,701 |
| L3 | FORECAST | rank 13 or worse | 43,130 | 0.6809 [0.6693, 0.6929] | 0.5989 [0.5870, 0.6113] | +0.08198 | +0.07216 | +0.09167 | 30,701 |
| L3 | TRIO | rank 13 or worse | 43,130 | 0.6231 [0.6123, 0.6345] | 0.5350 [0.5231, 0.5473] | +0.08810 | +0.07612 | +0.10017 | 30,701 |
| L3 | TIERCE | rank 13 or worse | 43,130 | 0.6512 [0.6377, 0.6654] | 0.5664 [0.5513, 0.5825] | +0.08481 | +0.06922 | +0.10007 | 30,701 |
| L7 | PLACE | flattest quintile | 43,130 | 0.8863 [0.8820, 0.8906] | 0.8410 [0.8360, 0.8460] | +0.04525 | +0.04240 | +0.04818 | 8,613 |
| L7 | WIN | handicap | 43,130 | 0.8009 [0.7916, 0.8101] | 0.7864 [0.7767, 0.7958] | +0.01452 | +0.01198 | +0.01707 | 2,454 |
| L7 | PLACE | handicap | 43,130 | 0.8533 [0.8485, 0.8582] | 0.8410 [0.8360, 0.8460] | +0.01229 | +0.01086 | +0.01372 | 2,454 |
| L7 | QUINELLA_PLACE | handicap, top-3 box | 43,130 | 0.8167 [0.8092, 0.8243] | 0.8032 [0.7953, 0.8110] | +0.01358 | +0.01140 | +0.01574 | 2,454 |
| L7 | QUINELLA | handicap, top-3 box | 43,130 | 0.8057 [0.7948, 0.8164] | 0.7887 [0.7774, 0.7997] | +0.01705 | +0.01413 | +0.01995 | 2,454 |
| L7 | FORECAST | handicap, top-3 box | 43,130 | 0.7797 [0.7687, 0.7903] | 0.7611 [0.7498, 0.7721] | +0.01859 | +0.01573 | +0.02147 | 2,454 |
| L7 | TRIO | handicap, top-3 box | 43,130 | 0.7845 [0.7624, 0.8073] | 0.7711 [0.7478, 0.7956] | +0.01339 | +0.00634 | +0.02017 | 2,454 |
| L7 | TIERCE | handicap, top-3 box | 43,130 | 0.6982 [0.6777, 0.7195] | 0.6767 [0.6552, 0.6988] | +0.02151 | +0.01553 | +0.02730 | 2,454 |

Board at t (T-2), every rule: L2 WIN, PLACE and QUINELLA_PLACE; L3 at both tail odds (WIN, PLACE) and both ranks (five
exotic pools); L7 flattest quintile, handicap WIN and PLACE, and the five boxes. 26 variants, each its own registry
record, all data pending on the 24 logged races (empty board: no `post_utc`).

### L7: what the skipped favourites returned (derived)

Under `stake_back` the rule's side counts each skipped race at 1.0, so an L7 rule row ranks above its comparator
whenever the skipped favourites returned under 1; the paired difference is (k / N) x (1 - r), with N the paired races,
k the skipped ones and r the skipped favourites' own mean return. The table solves for r from the recorded figures,
and for the rest of the races from the comparator's mean. This is arithmetic on the records, not a registry metric:
its bounds come from the difference's bounds with k held fixed and are approximate (the bootstrap redraws k).

| Variant | Races skipped (k) | Skipped favourites' return r [approx. lower, upper] | The other races | All races |
| --- | ---: | --- | ---: | ---: |
| Favourite PLACE, flattest quintile | 8,613 | 0.7734 [0.7588, 0.7877] | 0.8579 | 0.8410 |
| Favourite WIN, handicaps | 2,454 | 0.7448 [0.7000, 0.7894] | 0.7889 | 0.7864 |
| Favourite PLACE, handicaps | 2,454 | 0.7840 [0.7588, 0.8092] | 0.8445 | 0.8410 |
| Top-3 QUINELLA_PLACE box, handicaps | 2,454 | 0.7614 [0.7233, 0.7996] | 0.8057 | 0.8032 |
| Top-3 QUINELLA box, handicaps | 2,454 | 0.7003 [0.6494, 0.7517] | 0.7940 | 0.7887 |
| Top-3 FORECAST box, handicaps | 2,454 | 0.6733 [0.6227, 0.7236] | 0.7664 | 0.7611 |
| Top-3 TRIO box, handicaps | 2,454 | 0.7648 [0.6456, 0.8886] | 0.7715 | 0.7711 |
| Top-3 TIERCE box, handicaps | 2,454 | 0.6220 [0.5202, 0.7270] | 0.6800 | 0.6767 |

### Ranked options (mean return per ¥100 staked, final price, look-ahead)

Each scored comparison gives two options; a comparator shared by several variants (every WIN ticket serves L2 and both
L3 WIN cut-offs) is listed once. L7 rule rows count a skipped race as ¥100 back, so their place in this list reflects
stake-back as much as betting (see the derived table above). The rank orders the options and is not a test. No option
returns above its stake.

| Rank | Option | Return per ¥100 | Lower (5%) | Upper (95%) |
| ---: | --- | ---: | ---: | ---: |
| 1 | `L7:flattest_quintile:PLACE:final_look_ahead` | 88.63 | 88.20 | 89.06 |
| 2 | `L7:handicap:PLACE:final_look_ahead` | 85.33 | 84.85 | 85.82 |
| 3 | favourite PLACE, every race (comparator) | 84.10 | 83.60 | 84.60 |
| 4 | `L7:handicap:QUINELLA_PLACE_top3_box:final_look_ahead` | 81.67 | 80.92 | 82.43 |
| 5 | `L7:handicap:QUINELLA_top3_box:final_look_ahead` | 80.57 | 79.48 | 81.64 |
| 6 | favourite top-3 QUINELLA_PLACE box, every race (comparator) | 80.32 | 79.53 | 81.10 |
| 7 | `L7:handicap:WIN:final_look_ahead` | 80.09 | 79.16 | 81.01 |
| 8 | `L3:PLACE:above_51:final_look_ahead` | 79.68 | 79.36 | 80.00 |
| 9 | `L3:WIN:above_51:final_look_ahead` | 79.41 | 78.77 | 80.05 |
| 10 | favourite top-3 QUINELLA box, every race (comparator) | 78.87 | 77.74 | 79.97 |
| 11 | favourite WIN, every race (comparator) | 78.64 | 77.67 | 79.58 |
| 12 | `L3:PLACE:above_101:final_look_ahead` | 78.62 | 78.25 | 79.01 |
| 13 | `L3:WIN:above_101:final_look_ahead` | 78.60 | 77.78 | 79.41 |
| 14 | `L7:handicap:TRIO_top3_box:final_look_ahead` | 78.45 | 76.24 | 80.73 |
| 15 | `L7:handicap:FORECAST_top3_box:final_look_ahead` | 77.97 | 76.87 | 79.03 |
| 16 | favourite top-3 TRIO box, every race (comparator) | 77.11 | 74.78 | 79.56 |
| 17 | `L3:QUINELLA_PLACE:rank_9_or_worse:final_look_ahead` | 76.13 | 75.61 | 76.66 |
| 18 | favourite top-3 FORECAST box, every race (comparator) | 76.11 | 74.98 | 77.21 |
| 19 | `L3:QUINELLA:rank_9_or_worse:final_look_ahead` | 73.42 | 72.58 | 74.27 |
| 20 | every PLACE ticket (comparator) | 73.16 | 72.69 | 73.64 |
| 21 | `L3:FORECAST:rank_9_or_worse:final_look_ahead` | 72.73 | 71.84 | 73.62 |
| 22 | every WIN ticket (comparator) | 72.47 | 71.40 | 73.55 |
| 23 | `L2:WIN:final_look_ahead` | 72.45 | 71.38 | 73.53 |
| 24 | `L3:TIERCE:rank_9_or_worse:final_look_ahead` | 70.60 | 69.43 | 71.79 |
| 25 | `L3:TRIO:rank_9_or_worse:final_look_ahead` | 70.31 | 69.30 | 71.35 |
| 26 | `L7:handicap:TIERCE_top3_box:final_look_ahead` | 69.82 | 67.77 | 71.95 |
| 27 | `L3:QUINELLA_PLACE:rank_13_or_worse:final_look_ahead` | 69.34 | 68.75 | 69.94 |
| 28 | `L3:QUINELLA:rank_13_or_worse:final_look_ahead` | 68.25 | 67.18 | 69.33 |
| 29 | `L3:FORECAST:rank_13_or_worse:final_look_ahead` | 68.09 | 66.93 | 69.29 |
| 30 | favourite top-3 TIERCE box, every race (comparator) | 67.67 | 65.52 | 69.88 |
| 31 | `L3:TIERCE:rank_13_or_worse:final_look_ahead` | 65.12 | 63.77 | 66.54 |
| 32 | `L3:TRIO:rank_13_or_worse:final_look_ahead` | 62.31 | 61.23 | 63.45 |
| 33 | every QUINELLA_PLACE combination (comparator) | 61.89 | 61.26 | 62.55 |
| 34 | every QUINELLA combination (comparator) | 60.18 | 59.10 | 61.29 |
| 35 | every FORECAST combination (comparator) | 59.89 | 58.70 | 61.13 |
| 36 | every TIERCE combination (comparator) | 56.64 | 55.13 | 58.25 |
| 37 | every TRIO combination (comparator) | 53.50 | 52.31 | 54.73 |

### L2: the 100-vs-110 check (Plus-10, 04 §1.3)

Floor-bound placed favourites at the final price: the favourite on the final Win odds (ties to the lower horse
number) whose Place paid at most 110. From the jra store alone:

| Selection | Races | Floor-bound placed favourites | Paid 100 (Plus-10 withheld) | Paid 110 |
| --- | ---: | ---: | ---: | ---: |
| final (look-ahead) | 43,130 | 10,236 | 607 | 9,629 |
| board at t (T-2) | 24 logged | data pending (empty board) | | |

The prediction of each of these dividends (100 or 110) from its race's settled PLACE pool reads the JRA-VAN H1 votes
(the jravan `pool_votes` table, #368): sales from the pool row net of refunds, the race's Place payout rate from the
boost calendar, the placed horses from the jra dividends, and the frozen Place rule (the net pool first returns the
placed stakes, the profit is split equally over the placed horses and rounded down to 10 yen a ticket, then
`predicted_floor_dividend`). The predicted x realised table, its n and every count are withheld (D21): they are in
the private report `floor-tail-favourite-japan-a3bfd6a65fa8-plus10-jravan.md` under the data directory. A spot check
after the run (not a recorded run) found that the frozen equal-split rule does not reproduce JRA's realised Place
dividends from those votes, off the floor too, which agrees with #85's finding that JRA's Place figures do not follow
the stake-back-share rule. The table is therefore not yet a test of the Plus-10 prediction: the rule needs fixing
first (follow-up below).

## Alternatives rejected and why

- **A decision-time selection on history from the JRA-VAN time series.** The jravan `odds_timeseries` holds WIN,
  PLACE, BRACKET_QUINELLA and QUINELLA boards on history, which could give L2, L3 and L7 a board at t before the
  logger's days. It is another ticket's ground (#404 covers similar work) and would make those rows JRA-VAN-derived
  (withheld). Not built here; proposed as a follow-up.
- **Taking t from the card's post time when the capture has none.** The frozen spec defines t from the capture's
  `post_utc`, as #81 does. #352 backfills the post-time sidecar instead.
- **Reading the Place votes through `read_label`.** The view reads a label-only table whole; `pool_votes` is about
  1.2 GB. The reader prunes row groups by race_id statistics and reads five columns in 1M-row batches, PLACE rows of
  the wanted development races only.
- **Rebuilding the dividend of a race at a rate other than its own.** The Place payout rate comes from
  `takeout.race_takeout`, so the boosted races' 85% (ultra premium) is used where the calendar has it.
- **`drop_race` for tied Win prices and no L7 boxes.** The owner chose `lower_horse_no` and `top3_box`.
- **A family-wise correction over the 50 variants.** The reframe ranks options; the leaderboard (#449) shrinks within
  each pool. The bounds here are per variant, and the family size is on every record.

## What this rules out

All of this is look-ahead on history, development days 2014-06-07 to 2026-11-23. Nothing below adopts a rule, and no
option here returns above its stake.

- **L3, the tail, at the final price.** Dropping WIN or PLACE tickets on runners whose final WIN odds are above 51 or
  101 raises the per-race mean return by +0.055 to +0.069 per unit staked, every lower bound above +0.050. Dropping
  exotic combinations holding a runner ranked 9th or worse raises it by +0.128 to +0.168 (lower bounds +0.114 or
  more), and 13th or worse by +0.074 to +0.088. This is the favourite-longshot bias of the final price that the
  ledger's tail-hinge row (#294's run) measures; whether it survives a choice at t, when late money has not yet
  moved the tail, is not measured.
- **L2, the floor, in the WIN pool at the final price.** Dropping the WIN tickets priced at or below 1.1 (195 races)
  changes the race's mean return by -0.00018 (bounds -0.00030 and -0.00004). Floor-priced Win tickets returned
  slightly more than the race's other tickets, so the rule ranks just below its comparator; the effect is under 0.1%
  of stake. L2 on the PLACE and Wide boards has no history row (no final board of those pools on history) and no
  board at t yet.
- **L7 at the final price.** Every L7 rule row returns more than its comparator, as stake-back implies whenever the
  skipped favourites lose. The derived table is the informative read: the favourite's Place in the flattest-entropy
  quintile returned about 0.773 against about 0.858 in the other races, and in handicaps the favourite's Win about
  0.745 against 0.789, its Place about 0.784 against 0.845, and the top-3 QUINELLA and FORECAST boxes about 0.70 and
  0.67 against 0.79 and 0.77. The TRIO box's handicap return is not told apart from the rest (0.765, approximate
  bounds 0.646 to 0.889).
- **The 100-vs-110 check, realised part.** Of 10,236 floor-bound placed favourites, 607 (5.9%) were paid 100, so
  Plus-10 was withheld in about one floor-bound case in seventeen at the final price. The prediction of which ones is
  withheld (D21) and, as frozen, not yet a valid test (above).
- **Nothing at a decision time.** All 26 board-at-t variants are data pending.

## Follow-ups proposed

To be filed under epic #72 as `status:proposed`:

- Re-run this frozen family once the logged days have post times (#352) and as more days are logged, so the 26
  board-at-t variants score next to the look-ahead rows.
- Find the Place payout rule that reproduces JRA's realised Place dividends from the H1 final votes (on non-floor
  placed horses first), then re-run the 100-vs-110 check as a new family (the rule text is in this spec). The figures
  stay private (D21).
- A board-at-t version of L2, L3 and L7 on history from the jravan `odds_timeseries` (WIN and PLACE boards), labelled
  JRA-VAN-derived and withheld; check the overlap with #404 first.
- Leaderboard rows (#449): L7's rule rows need the skipped and kept favourites as their own rows per ¥100 actually
  staked, so stake-back does not lift them.
