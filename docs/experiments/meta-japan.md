# Experiment report: the C4 meta-model at the final Win price (Japan, #87)

Ticket #87 (epic #73; 08 §8 B3, §4 E40, E9, R13). The run fits the C4 meta-model on D9's development folds. It
fits stage 2 at the final Win price with the weights unbounded, and the residual (offset) form on the final price and
on the D4 calibrated market. It takes each member's encompassing coefficient beside the final price, scores each
variant's out-of-fold ΔLL per race against the D4 baseline, and tabulates the E9 information ceiling for Win.
**Nothing is adopted** (D16): no holdout is opened.

**Every member-based figure is withheld (D21).** The members are HKTrack's stored out-of-fold predictions. The
research-24 form model reads JRDB. The two "no JRDB" members are not confirmed JRDB-free (see Data). The owner allows
JRDB as an input to private models on this Mac only, with every JRDB-fed result kept out of the repository, pull
requests, tickets and comments (#87, owner comment of 2026-11-24). This report therefore says only that those runs
happened, where their report lives and which registry records hold them. It gives no number, sign or bound from them.
Two variants read no member: q against the baseline, and the baseline's own ceiling table. Both are scored on the
store alone, and their figures are below.

- Registry family `japan-c4-meta-final-win`, spec sha256
  `690d41225b38fa490f410c480bc979c652e5c77faf38471a97b44f818fff4259`. The spec was frozen (2026-11-25 05:27:07 UTC)
  before any outcome was read. It holds the hypotheses, the decision rules, the members and the member file's sha256,
  the variants, the folds, the bootstrap, the ceiling thresholds and the owner decisions.
- Code at `30e5bc9`: `countries/japan/models/condlogit.py` and `models/meta.py` (the model, tested in
  `tests/countries/japan/test_condlogit.py` and `test_meta.py`), and `countries/japan/studies/meta.py` (the run).
  The meeting-clustered bounds, the report's rule and rules-out sections and the level-2 window table were added at
  `647879c`, after the outcomes were read (review findings on PR #308; see Hypothesis). To reproduce, run
  `uv run python -m horsetrack.countries.japan.studies.meta --run --members ~/Projects/HKTrack/outputs/oof/m-v1-20261115-6102e309.parquet`.
- Store: `processed/jra/runners.parquet@6f3cec95347e0132`, `processed/jra/races.parquet@9cce511831074b64`. Members:
  `HKTrack:outputs/oof/m-v1-20261115-6102e309.parquet@b0c5a8298e0befe8`, read by column (keys and the three member
  columns only). HKTrack was read, never written or run.
- The full report, with every table, is under the data directory at `$HORSETRACK_DATA_DIR/reports/meta-japan-690d41225b38.md`
  (written through `write_report` with the sources jra and jrdb), re-rendered at `647879c`. The records are in
  `docs/experiments/registry-japan.md`, where the 36 member-based records show as withheld. The re-run reproduced
  the 26 first-run records exactly and did not append them again; it appended only the 12 `meeting_clustered:`
  records.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| D1 | A: condition on the final price. Every variant conditions on the final Win price. | `docs/problem-statement.md` §1, #14 |
| D4 | The Win baseline is the calibrated market (#114's spline in ln q), fitted in-fold. It is fitted on each fold's training rows of the store, with dead-heat races left out of its fit as #114's `run_fit` does. | §2, #14 |
| D6 | Winner log loss per race, a dead heat at 1/k per co-winner (the #69 answer). Pseudo-R² against equal chances is reported beside nats per race. | §3, #14 |
| D7 | Meeting (venue x day) clusters, 95% one-sided (`config/scoring.yaml`; 10,000 replicates, seed 87). The ΔLL and ceiling bootstraps resample meetings. The coefficient rules were frozen with 95% two-sided Wald intervals instead; the meeting-clustered one-sided bounds were added after the outcomes were read (see Hypothesis). | §3, #14 |
| D9 | "Quarter folds, quarterly refit, expanding, 5 seasons minimum, nested OOF, 1-week embargo" (`config/folds.yaml`). The fold calendar is computed over the store's development starters from 2014-06-07. The baseline and the level-2 meta follow the folds, refit, window, nesting and embargo. The 5-season minimum holds for the baseline but **not at level 2 on any scored fold** (see Data). Level 1 is HKTrack's season walk-forward. Whether this window is accepted for the replication is put to the owner on #87. | §4, #14; the #70 answer |
| D10 | No price before 2014-06-07. | §4, #14 |
| D11-D15 | Holdout A2 from 2026-11-28, fully locked. No holdout row is read, and the holdout is not opened. | §5, #14; the #71 answer |
| D16 | Adoption needs a holdout gain lower bound above 0 at the D7 level and a positive encompassing weight with its interval above 0. Nothing is adopted here. | §6, #14 |
| D21 | No figure derived from a licensed source appears in the repository. | §8, #14 |
| #87 | JRDB may be a model input in private models on this Mac only, and its results are never published. The meta-model's shrinkage target is α = 0, the calibrated market alone. | #87, owner comment of 2026-11-24 |

## Data

- **The store.** 43,130 development races (2014-06-07 to 2026-11-23) and 601,265 priced starters. 2,156 scratched
  runners are out. There are no void or unpriced races, and 66 races are dead heats. 28 walk-forward quarters start at
  2020-Q1. The baseline is predicted from 2023-Q1, the member file's span.
- **Scored folds.** 12 quarters, 2024-Q1 to 2026-Q4. 2023 is HKTrack's seed season (its stage 2 is fitted from it and
  it is not scored there), so here too it only trains the meta. The store-only variants cover 10,047 races in 838
  meetings, through 2026-11-23. The member variants stop at the member file's last day, 2026-11-15.
- **The members.** There are three, each a column of `m-v1-20261115-6102e309`'s out-of-fold file:
  - `form` is `p_fundamental_oof`, research-24's form model (HKTrack `docs/research/24-process-review-2026-11-22.md`
    §2: the ten-member stage-1 ensemble with no odds input). HKTrack's `primary_oof` is `blend` on every row, so this
    column equals `p_blend_oof`. It reads JRDB's `f_jrdb_*` indices.
  - `lgbm_no_jrdb` and `clogit_no_jrdb` are `p_lgbm_no_jrdb_oof` and `p_clogit_no_jrdb_oof`. Each drops only the
    `f_jrdb_*` columns (HKTrack `config/experiments/ten-model-ensemble.yaml`, `drop_prefixes: [f_jrdb_]`). They
    keep 21 pedigree features keyed on `horse_pedigree`, which HKTrack's JRDB build rewrites with JRDB's UKC horse
    master as the primary source (`src/hktrack/sources/jra/jrdb/pedigree.py`, `merge_pedigree`). In the v1 frame
    the members trained on, the two UKC-only columns (`f_sire_line`, `f_dam_sire_line`) are null on every row. That
    points to a netkeiba-only table at the time. But the UKC merge code landed in HKTrack about 15 hours before v1
    was built (commit `8385f71`, 2026-11-20), and the table then in place cannot be traced. Their aggregates are
    therefore treated as JRDB-derived and withheld.
- **The join.** It is on (race_id, horse_id) against the store's priced starters. A race with a starter the file
  lacks would be dropped. The counts each way are in the private report. Member probabilities enter as logs,
  normalised over the race's starters. The final price and outcomes come from this repository's store, never from
  HKTrack's `win_odds` or `is_winner`.
- **Level 1 is out of fold.** HKTrack's walk-forward is by season, expanding (`src/hktrack/eval/walkforward.py`:
  "Nothing fitted for test season S ever sees S or later"). The run checked that every row's `fold_season` is its
  race's calendar year. Level 2 is out of fold through `models.meta.oof_predictions` on D9's folds: each fold's meta is
  fitted on the member rows on or before its `train_end` and predicts its test quarter. Three leaks are stated, not
  hidden:
  - HKTrack chose its member set, features and hyperparameters with these seasons visible (ledger: research-24's
    test seasons were also used for model selection).
  - `lgbm_top12`, one of the form model's ten members, drops features ranked by a file whose tree count was chosen on
    2023. 2023 is not scored here, but it trains every fold's meta.
  - HKTrack embargoes nothing between seasons beyond the year-end break.
- **The level-2 window is below D9's 5-season minimum on every scored fold.** The member file starts on 2023-01-05,
  and each fold's meta trains on the member rows from then to its `train_end`. The frozen spec says only "the first
  folds" fall short; in fact all 12 do, from 0.98 to 3.73 years. The fold calendar is otherwise D9's, and the member
  file cannot give more. Whether this window is accepted for the replication is put to the owner on #87; until then
  every member result carries this caveat. The level-2 training span per scored fold (a coverage count of the store
  and the member file, no outcome):

  | Scored fold | Level-2 training rows | Years | Training races (meetings) |
  | --- | --- | --- | --- |
  | 2024-Q1 | 2023-01-05 to 2023-12-28 | 0.98 | 3,456 (288) |
  | 2024-Q2 | 2023-01-05 to 2024-03-30 | 1.23 | 4,344 (362) |
  | 2024-Q3 | 2023-01-05 to 2024-06-29 | 1.48 | 5,220 (435) |
  | 2024-Q4 | 2023-01-05 to 2024-09-28 | 1.73 | 6,047 (504) |
  | 2025-Q1 | 2023-01-05 to 2024-12-28 | 1.98 | 6,910 (576) |
  | 2025-Q2 | 2023-01-05 to 2025-03-29 | 2.23 | 7,738 (645) |
  | 2025-Q3 | 2023-01-05 to 2025-06-28 | 2.48 | 8,601 (717) |
  | 2025-Q4 | 2023-01-05 to 2025-09-27 | 2.73 | 9,501 (792) |
  | 2026-Q1 | 2023-01-05 to 2025-12-28 | 2.98 | 10,365 (864) |
  | 2026-Q2 | 2023-01-05 to 2026-03-28 | 3.23 | 11,187 (933) |
  | 2026-Q3 | 2023-01-05 to 2026-06-27 | 3.48 | 12,051 (1,005) |
  | 2026-Q4 | 2023-01-05 to 2026-09-26 | 3.73 | 12,951 (1,080) |

## Hypothesis

The frozen spec reads as follows:
- **E40.** "Stage 2 at the final Win price (u = alpha * m + beta * ln q, unbounded) replicates research-24's stage-2
  gain over the market (HKTrack docs/research/24 section 2; ledger, figures withheld) in this harness, as
  out-of-fold ΔLL per race against the D4 calibrated market; the residual (offset) form gives alpha_T > 0; each
  member's encompassing coefficient beside ln q."
- **E9.** "The out-of-fold ln(pi / q_final) of each meta variant in the Win pool: the q-mass above each threshold
  (overlays, one-sided) and the q-weighted RMS."

The decision rules were frozen with them:
- The replication holds when `stage2:q:form`'s ΔLL lower bound is above 0. Otherwise it is a documented failure.
- α_T > 0 for a member and market when the 95% Wald CI of its δ, in the pooled offset fit over the scored rows
  (in-sample), lies above 0.
- An encompassing coefficient is positive when its 95% Wald CI lies above 0.
- E9 is descriptive, with the break-even threshold at ln 1/(1 − 0.20) = ln 1.25 (`jra.yaml` Win takeout).

**The Wald rules are provisional.** The α_T and encompassing rules were frozen with 95% two-sided Wald intervals from
the inverse information, which treat races as independent. The owner's D7 fixes the inference unit (meetings, venue
x day) and level (95% one-sided), and D16 reads the encompassing interval as an adoption leg. The meeting-clustered
bound, coef − 1.645 × the meeting sandwich se, was added **after the outcomes were read** (review finding on PR
#308). It is the interval that matches D7. The frozen spec is unchanged, so the private report reads each rule both
ways: as frozen (Wald) and at D7 (clustered). The Wald-rule outcomes are provisional. The replication rule needs no
second reading: its ΔLL lower bound is already the meeting bootstrap's one-sided 95% bound.

The null is the starting point: the members add nothing beside the final price (α = 0, δ = 0, the owner's shrinkage
target). 08's E40 and E9 rows give expected sizes. They are not quoted here, since they come from research-24's
licensed-source fit.

E40's third clause, "estimand A beats B on shadow EV calibration", needs the C6 shadow layer and is not run here.

## Variants tried

There are 38 variants (`Registry.family_size`), one record each, and every one scored: the 26 of the frozen spec
and 12 `meeting_clustered:` variants added after the outcomes were read. ΔLL is the D4 baseline's winner log loss
minus the variant's, in nats per race (positive: better than the calibrated market). Bounds are the 5% and 95%
quantiles of the meeting bootstrap. Coefficients are fitted per fold on the fold's training rows, with no bounds (08
R13). The frozen spec's CIs on them are 95% two-sided Wald. Each `meeting_clustered:` record holds the same
coefficients with D7's bounds instead: coef ∓ 1.645 × the meeting-clustered sandwich se (`models.condlogit`, no
small-sample factor), each one-sided 95%. The market snapshot is the final Win price from the JRA store.

### Store only (publishable)

| Variant | Record | Races (meetings) | Result |
| --- | --- | --- | --- |
| `market:normalised` (E40): normalised q against the D4 baseline, the offset form at δ = 0 | `755c5f9153746daa` | 10,047 (838) | mean log loss 1.91131 against 1.91037; ΔLL −0.000939 [−0.001695, −0.000165]; Δ pseudo-R² −0.00036 |
| `ceiling:calibrated_market` (E9): the D4 baseline's own ln(p_cal / q) | `b5a0f5de1c7aa4b0` | 10,047 (838) | q-mass with p_cal above q 0.7747 [0.7698, 0.7795]; above ratio 1.1, 1.25, 1.5 and 2: 0 [0, 0]; q-weighted RMS 0.0582 [0.0577, 0.0587] |

By fold, `market:normalised`'s ΔLL runs from −0.00301 (2025-Q3) to +0.00153 (2024-Q2), above 0 in 3 of the 12
quarters (`docs/experiments/registry-japan.md`).

### Member-based (withheld, D21)

Each row below ran and scored. Its figures are in the private report and its registry record only.

| Member | `member_alone` | `stage2:q` | `offset:q` | `offset:cal` | `encompassing` |
| --- | --- | --- | --- | --- | --- |
| `form` | `3837d834ddfe73dd` | `3ff20ac058412349` | `e4b05ac245f69c8a` | `bc68f2979b251384` | `5ff72d4b5bc1cf0d` |
| `lgbm_no_jrdb` | `b6df4261df999e03` | `bbc999ed4a280da9` | `6db3eacd075e2387` | `4aea3cdec4fe6f07` | `a76c3e5cde01f94c` |
| `clogit_no_jrdb` | `779719932f4669ed` | `058b34d2a1c02ed7` | `d5cf515b84fb2ad3` | `9f8e3e2e06e7dcfb` | `a68cf1f8dacd3428` |

The E9 ceiling record of each meta variant:

| Member | `ceiling:stage2:q` | `ceiling:offset:q` | `ceiling:offset:cal` |
| --- | --- | --- | --- |
| `form` | `0e4934ca28c2832e` | `e1db285e6cd67f89` | `bd561c2353e8a33b` |
| `lgbm_no_jrdb` | `32ceaf6b323fb100` | `52119bca9fe64fee` | `47e3cd7416eb2e14` |
| `clogit_no_jrdb` | `73a73ff0b7d3d691` | `28d317bf261fcfa8` | `ae086c38d607fe52` |

The `meeting_clustered:` record of each meta and encompassing variant (added after the outcomes were read):

| Member | `stage2:q` | `offset:q` | `offset:cal` | `encompassing` |
| --- | --- | --- | --- | --- |
| `form` | `ce83d75662f8f8df` | `9aa931c8377b9399` | `953b4012786ec1de` | `f3d839ea734c92d5` |
| `lgbm_no_jrdb` | `6f2619105302ab55` | `0ae748a019de3c35` | `f5e691f5dc044c08` | `b8000c7f6bd4c164` |
| `clogit_no_jrdb` | `e5b88f730ca19d65` | `14335d32291f5624` | `eea4081f05420809` | `a996cf59ecbdcb70` |

What each form is:
- `member_alone`: the member's own probability.
- `stage2:q`: u = α·m + β·ln q, α and β unbounded.
- `offset:q`: u = ln q + δ·m.
- `offset:cal`: u = ln p_cal + δ·m. At δ = 0 this is the D4 baseline itself.
- `encompassing`: `models.meta.encompassing`, the member's weight beside ln q, unpenalised, in-sample over the scored
  rows.

## Alternatives rejected and why

- **Bounded weights** (HKTrack's `COEF_BOUNDS = (0, 5)`). 08 R13 requires α unbounded. Ledger row: HK's α of 0.00
  was that bound.
- **HKTrack's own stage-2 output** (`p_combined_oof`) or its market columns. The replication has to refit the meta
  in this harness, on D9's folds, against the D4 baseline, at this store's final price.
- **Scoring 2023.** It is HKTrack's seed season, and `lgbm_top12`'s feature ranking chose its tree count on it. It
  only trains.
- **Segment weights α(z).** The code supports them (interaction columns, tested). No segment set is pre-registered
  for member weights: the owner's #125 set is for the Win residual's slopes on ln q, and E42 names other segments.
  They are left to the selection study (#88).
- **A quadratic market term γ (ln q)².** The code supports it (`quadratic=True`). Research-24's final-price stage 2
  had none. The quadratic-vs-board question is E2 (#103).
- **A ridge toward α = 0.** The owner signed the target, not its strength. The DoD asks for α fitted unbounded, and
  nothing is adopted, so every fit is the unpenalised MLE (ridge 0). The code shrinks member terms only, never the
  market's.
- **Precision-weighted α (E41).** It needs the horse model's σ_i (#98).
- **One fit on all three members together.** The form model already blends the other two, which are two of its ten
  members.
- **Wald as the only coefficient interval.** The frozen spec's coefficient CI is Wald, so races count as
  independent, which D7 does not sign. The meeting-clustered bound is now reported beside it (added after the
  outcomes were read; see Hypothesis). A meeting bootstrap of the coefficients was not used: it would refit every
  meta 10,000 times, and the sandwich is the same unit and level.
- **A member without JRDB** (#91's dynamic rating). It does not exist yet. It is the publishable replication.

## What this rules out

- **Normalised 1/odds as the Win baseline.** Out of fold on 2024-Q1 to 2026-11-23, normalised q loses to D4's
  calibrated market by 0.00094 nats per race. Normalised's upper (95%) bound is −0.00017, so the calibration gain
  is above 0.
  This is the same direction as #77's power-β gain over normalised (+0.00027, 2020-2026), larger and on fewer
  races. Single run, one window.
- **Overlays from calibration alone.** The calibrated market never puts p_cal/q above 1.1 on any runner of the 10,047
  races. It lifts runners holding 77% of the q-mass above q (the favourites), with a q-weighted RMS of ln(p_cal/q) of
  0.058. Any E9 overlay at the 1.25 break-even must therefore come from a member, not from the market's own
  calibration.
- **The member variants.** What they rule out is not stated here (D21). The private report states it. Its "Decision
  rules" section applies each frozen rule mechanically: the replication verdict for `stage2:q:form` from its ΔLL
  lower bound (replicated, or a documented failure), α_T > 0 for each offset variant, and a positive encompassing
  weight for each member, each read both as frozen (Wald) and at D7 (meeting-clustered). Its "What this rules out"
  section reads each ΔLL, δ, encompassing and break-even q-mass bound at D7. Every statement there is conditional on
  the level-2 window above, which is below D9's minimum and awaits the owner on #87.

## Follow-ups filed

- #91 and #98 (both `status:ready`): a member without JRDB inputs, then the meta over the horse, speed and rating
  members. Both rerun this family's design with publishable figures.
- #88 (`status:ready`): the selection study reads α_s from segment interaction columns.
- #99 (`status:ready`): the ceiling table for the other pools.
- #103 (`status:ready`): the quadratic market term on the board (E2).
- Meeting-clustered (sandwich) bounds on the meta coefficients and the encompassing weight: done in this PR (the
  `meeting_clustered:` records), so no follow-up ticket is needed. #276 is the counterpart for #77's β.
- D9's 5-season minimum at level 2: put to the owner on #87 (accept this window for the replication, or not).
