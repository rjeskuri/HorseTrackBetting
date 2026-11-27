# Experiment report: the C2 top-3 Plackett-Luce ability at the final Win price (Japan, #92)

Ticket #92 (epic #73; 08 §8 B4, §3.2 C2, §4 E33, E34; 05 §2.2, §2.4, §2.5). The run fits the C2 ability member, a
discounted top-3 Plackett-Luce model of the finishing order with #91's dynamic rating and partial-pooled jockey,
trainer and sire, out of fold on D9's quarters, and measures it against the final Win price: its correlation with
ln q_final, its standalone results, its stage-2 weight beside ln q, its marginal in C4 against the market-only
stage-2 meta, and the two pre-registered comparisons E33 (top-3 against winner-only training) and E34 (pooling against
#129's as-of rates). **Nothing is adopted** (D16), and the holdout is not opened. **Every figure is withheld (D21).**
The sire comes from the JRA-VAN pedigree (the owner's answer on #92), a licensed source, so every result of the run
is built on JRA-VAN data. This report says what ran, where its report lives and which registry records hold it. It
gives no count, span, score, rate, sign, bound or selected value derived from the run.

**The result of record is family `japan-c2-ability-final-win-v2`.** It supersedes `japan-c2-ability-final-win` (v1),
whose 14 records stay in the registry and are listed under Superseded run. #92's review (round 1) found two faults in
v1 that change results: its member's C4 gain was measured only against D4's calibrated market, which differs from the
stage-2 meta in its market term too, so it could not isolate the member; and its rates' window ended on the last race
day before the race rather than #129's day − 1. v2 adds the market-only stage-2 meta and the member's paired marginal
against it, and uses #129's window. The review's two other findings, which do not change results, are also fixed: the
committed index keeps every family it lists, and each run's files are named by its data builds as well as its spec.

- Registry family `japan-c2-ability-final-win-v2`, spec sha256
  `12b1eed010954df8fc9d4ad976ed48d7e53efebaf3662b794b32057214b8109f`. The spec was frozen when the full run started,
  before the run read any row. It holds what v1's held (the hypotheses, the null, the decision rules, the level-1
  model, features, entities and grids, the rates comparator, the thin-history definition, the level-2 meta and
  baseline, the by-class table, the folds with the signed dates, D7's bootstrap and the export), plus the market-only
  meta, the C4 marginal's decision rule, #129's window for the rates, the artifact naming and what it supersedes.
- **No trial run.** Before each freeze the study ran only on synthetic races. For v1, it ran end to end through
  `compute_study` and `run_study` against a temporary registry, and its shared-fit discount selection was checked
  equal to `models.ability.fit_discounts` on synthetic folds. For v2, `tests/countries/japan/test_ability_study.py`
  pins the market-only meta, the marginal, the rate window against #129's own `_connection_rates`, the committed index
  and the artifact names, and `compute_study` was run end to end on synthetic races to check that it records every
  planned variant. The fit's speed was measured on synthetic races of the store's size. Before v1's freeze the real
  store was read once to check the inputs' structure (row and key coverage, missing values, the fold calendar). No
  model was fitted and nothing was scored in that check.
- Code: `countries/japan/models/ability.py` (the model, tested in `tests/countries/japan/test_ability.py`; written at
  `7b5185f`, made faster at `dad330f` without changing what a fit stops at) and `countries/japan/studies/ability.py`
  (the run), at `efb2276`. To reproduce, run `uv run horsetrack japan ability-study`. On this 16 GB Mac the run took
  about 2 h 25 min with a peak resident memory of about 1.8 GB, one fit at a time. A rerun appends only records the
  family does not hold.
- Data builds: `processed/jra/runners.parquet@6f3cec95347e0132`, `processed/jra/races.parquet@9cce511831074b64` (the
  same JRA store build as #91's run and v1's) and `processed/jravan/horse_pedigree.parquet@891fea934fadd013` (v1's).
- The full report is under the data directory at
  `$HORSETRACK_DATA_DIR/reports/ability-japan-12b1eed01095-9a3f58d49fb3.md`, with its tables as JSON beside it
  (`ability-japan-12b1eed01095-9a3f58d49fb3.json`); both were written through `write_report` with the sources `jra`
  and `jravan`. The per-runner export for the meld (#98) is at
  `$HORSETRACK_DATA_DIR/features/japan/ability-japan-12b1eed01095-9a3f58d49fb3.parquet`: every level-1 out-of-fold
  starter with `race_id`, `horse_id`, `race_date`, `oof_fold`, the fold's rating pair and discounts, `u`, `se_u` (σ_i)
  and `p_ability`. The stem is the spec hash and a digest of the data builds, and every v2 record names it in
  `data_builds.ability_artifacts`, so a rerun on refreshed data writes new files and leaves these as they are. The
  records are in `docs/experiments/registry/japan/japan-c2-ability-final-win-v2.md`, and every one shows as withheld.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| Features | #91's rating plus pooled jockey, trainer and sire; card features are left to #93-#95. | #92, owner answer 1, 2026-11-26 |
| Sire | From the JRA-VAN pedigree (`horse_pedigree.sire_id`, #370/#371), for full sire coverage from 2010 with no selection; results are withheld (D21). | #92, answer 2, 2026-11-26 |
| E34 | #129's connection rates (365-day strike rate shrunk 20 runs toward the field rate) for jockey, trainer and sire are the comparator, on races with a runner whose jockey, trainer or sire has fewer than 30 prior starts in the store since 2010; the paired race-level winner ΔLL of the stage-2 meta with each member; D7's meeting bootstrap. | #92, answer 3, 2026-11-26 |
| Splits | Level 1 out of fold from 2015-Q1, trained on results (no price) from 2010; D9's quarterly expanding folds with a 1-week embargo; inner folds every quarter inside a fold's training rows from 2011-Q1; the discounts and #91's rating pair re-selected each quarter. | #92, answer 4, 2026-11-26 (signed off) |
| By class | A by-class table, so the thin-history debut and maiden question is visible; the N5 disagreement-tail curve when N5 lands. | Owner, 2026-11-27 (#92 comment) |
| D1 | Condition on the final price: the member is measured beside the final Win q. | `docs/problem-statement.md` §1, #14 |
| D4 | The Win baseline is the calibrated market (spline in ln q), fitted in-fold. | §2, #14 |
| D6 | Winner log loss per race, a dead heat at 1/k per co-winner; pseudo-R² against equal chances. | §3, #14; #69 |
| D7 | Meeting (venue x day) clusters, 95% one-sided, 10,000 replicates (`config/scoring.yaml`, seed 92). | §3, #14 |
| D9 | Quarter folds, quarterly refit, expanding, 1-week embargo (`config/folds.yaml`); at level 2 the first scored quarter is 2020-Q1, and its training rows with a level-1 member start at 2015-Q1, five seasons. | §4, #14; #70 |
| D10 | Level 1 reads no price; level 2 reads prices from 2014-06-07 only. | §4, #14 |
| D11-D15 | Holdout A2 from 2026-11-28, fully locked; every read drops holdout days; not opened. | §5, #14; #71 |
| D16 | Nothing is adopted. | §6, #14 |
| D21 | No figure derived from JRA-VAN appears in the repository; the records carry the source `jravan`. | §8, #14 |

## Data

- **Level 1** (results only): every not-scratched starter of the JRA store's run races with a winner, from
  2010-01-05 (netkeiba results; the same races and starters as #91's rating input), with `jockey_key`, `trainer_key`
  and the JRA-VAN `horse_pedigree.sire_id` joined on `horse_id`. A starter without a placing has no `finish_pos`: it
  stays in every choice set and is never chosen. Positions 4th and later are not used.
- **Rating:** #91's `DynamicRating` over the same starters (a starter without a placing tied last), its member log
  probability at each (drift, scale) of #91's grids.
- **Rates** (E34's comparator): #129's window, vectorised over every race day: over the not-scratched runners of the
  store on the race days in [day − 365, day − 1] (#129's `_connection_rates` at cut = the race's day − 1), (wins +
  20 p0) / (runs + 20) with p0 the window's win rate, per jockey, trainer and sire; a level the window lacks gets p0 as
  the formula gives, and a missing key gets p0 too, where #129 gives null (the model needs a finite feature); the
  feature is ln(rate). A race day whose window holds no runner (the store's first) gets ln rate 0, a constant within
  race. v1 ended the window on the last race day before the race instead, so after a break between meetings its
  window started earlier than #129's (see Superseded run).
- **Thin history:** a runner whose jockey, trainer or sire has fewer than 30 not-scratched starts in the store on
  race days before the race's day since 2010 (a missing sire counts 0); a thin race has at least one among its priced
  starters.
- **Level 2:** `studies.meta.store_frame`, the priced starters of run races with a priced winner from 2014-06-07, and
  each member's ln p renormalised over the race's priced starters. Every priced starter has a level-1 row.
- Counts (starters, races, sire coverage, thin runners and races, by class) are in the private report.

## Hypothesis

The frozen spec reads:

- **E33.** "Discounted top-3 Plackett-Luce training beats winner-only training on the same features: Win ΔLL at the
  final price after stage 2 (08 §4 E33; expected +0.001 to +0.004 standalone)."
- **E34.** "Partial pooling of jockey, trainer and sire beats #129's as-of rates on races with thin entity history but
  adds nothing at the final price overall (08 §4 E34; expected 0.001 or less, concentrated in debut and maiden races;
  a null at the final price expected)."
- **C2.** "The pooled top-3 member earns stage-2 weight beside ln q_final (standalone and marginal in C4; 08 §3.2 C2)."
- **Null**, the starting point: the member carries nothing the final price lacks (α = 0); top-3 training is no better
  than winner-only; pooling is no better than the rates.

Decision rules (D7's bounds, one-sided 95%):

- E33: top-3 beats winner-only when the meeting-bootstrap lower bound of the paired per-race winner ΔLL of the stage-2
  meta with the pooled top-3 member against the stage-2 meta with the winner-only member, over the scored races, lies
  above 0.
- E34 thin: pooling beats the rates on thin history when the same bound for the pooled meta against the rates meta,
  over the scored races with a thin-history runner, lies above 0. E34 overall: descriptive, the same ΔLL over every
  scored race (the marginal ablation in C4).
- Stage-2 weight: α > 0 when the meeting-clustered lower bound of the pooled member's α, pooled over the scored rows
  in-sample, lies above 0. Stage-2 gain: the pooled meta's out-of-fold ΔLL against the calibrated market has its
  meeting-bootstrap lower bound above 0.
- C4 marginal (added in v2): the pooled member adds to the market term when the meeting-bootstrap lower bound of the
  paired per-race winner ΔLL of the stage-2 meta with the pooled member against the market-only stage-2 meta (the
  same market term, training rows and folds, no member), out of fold over the scored races, lies above 0. The
  stage-2 gain above compares two models that differ in their market term too (a linear term in ln q against D4's
  spline, fitted on other rows), so only this rule isolates the member.
- Descriptive: the pooled member's Pearson correlation with ln q_final (raw and within race) with meeting-bootstrap
  bounds; the member alone against the calibrated market and each member's pseudo-R²; the top-3 member alone against
  the winner-only member alone; the market-only meta against the calibrated market and its β per fold (added in v2).

## Method

- **Model** (`AbilityModel(k=3)`): u = β · rating + jockey + trainer + sire effects, each effect N(0, τ_e²) with τ_e²
  by empirical Bayes (MacKay's evidence fixed point), the penalised MAP by Newton's method. The likelihood is the
  discounted top-3 Plackett-Luce, discounts (1, w2, w3); a dead heat counts each co-placed runner at 1/m in the shared
  stage. σ_i is the Laplace standard error of u (an unseen level at effect 0 with its prior variance).
- **Members:** pooled (above, discounts selected); winner-only (the same at (1, 0, 0), E33's comparator); rates (the
  same likelihood on [rating, ln jockey rate, ln trainer rate, ln sire rate] with no entity effects, discounts selected
  the same way, E34's comparator).
- **Folds:** D9's quarters over the level-1 dates. Every quarter from 2011-Q1 is fitted on all earlier rows to its
  embargo and predicts its own rows; the quarters from 2015-Q1 are the outer folds, and an outer fold's inner folds are
  the quarters from 2011-Q1 inside its training rows. The two dates are set explicitly: `walk_forward`'s min_train
  does not count a partial first quarter (the store starts on 2010-01-05), so a 20-quarter minimum would start level 1
  at 2015-Q2 and #91's 4-quarter inner minimum would start the inner folds at 2011-Q2.
- **Selection,** per outer fold: the rating pair by #91's rule (the lowest mean winner log loss of the rating member
  alone over the fold's inner races), then the discounts in the 15-entry grid (w1 = 1, w2 and w3 in {0, 0.25, 0.5,
  0.75, 1}, w3 ≤ w2) with the lowest summed inner winner log loss at that pair, ties to the earlier entry, for the
  pooled and the rates member each. An inner fold's fit is the same for every outer fold whose training rows hold it
  (expanding windows from one origin), so each (pair, quarter, discounts) fit is made once and shared; the rule is
  `fit_discounts`' (checked equal on synthetic folds).
- **Stage 2** (#87's `fit_meta`): u = α · m + β · ln q, unbounded, ridge 0, per member, on each D9 fold's training
  rows that have a level-1 member (from 2015-Q1), predicting the fold's test quarter; α is also pooled over the scored
  rows in-sample. Scored: every D9 fold, 2020-Q1 to 2026-Q4 (to 2026-11-23).
- **Market-only stage 2** (v2, `market_only_oof`): u = β · ln q (q normalised within race), β unbounded, by
  conditional logit on the same rows and folds as the members' metas, predicting each fold's test quarter. The C4
  marginal (`c4_marginal`) is the paired per-race winner ΔLL of the pooled member's meta against it, so the two
  forecasts share the market term and the training rows and differ only by the member.

## Variants tried

One registry record per variant in `japan-c2-ability-final-win-v2` (git `efb2276`), all figures withheld.

| Variant | E | Record | What it holds |
| --- | --- | --- | --- |
| `selection:inner_fold` | E33 | `2e34550a575dba18` | Per outer fold, the selected rating pair and discounts (pooled and rates) and their inner log loss. |
| `correlation:ln_q` | E33 | `ca5cd31a6f3bf3c1` | The pooled member's Pearson correlation with ln q_final, raw and within race, with meeting bounds; per fold. |
| `member_alone:ability` | E33 | `4b982f80e2c0b32f` | The pooled member alone against the calibrated market (ΔLL, bounds, Δ pseudo-R²); each member's and q's pseudo-R². |
| `stage2:q:ability` | E33 | `3f10860c4ed00ac5` | The stage-2 meta with the pooled member against the calibrated market out of fold; α and β per fold (Wald); α pooled in-sample. |
| `meeting_clustered:stage2:q:ability` | E33 | `fb343ec0fb821e1d` | The same coefficients with D7's meeting-clustered bounds (the stage-2 weight rule's bound). |
| `stage2:q:ability_winner_only` | E33 | `a3e08eaf5729df71` | As `stage2:q:ability`, for the winner-only member. |
| `meeting_clustered:stage2:q:ability_winner_only` | E33 | `eb46051405195a0a` | Its meeting-clustered bounds. |
| `stage2:q:ability_rates` | E34 | `e7f94e010864bee8` | As `stage2:q:ability`, for the rates member. |
| `meeting_clustered:stage2:q:ability_rates` | E34 | `fb1906a7674e2cd9` | Its meeting-clustered bounds. |
| `stage2:q:market_only` | E33 | `ba98f5b596b37bf8` | New in v2: the market-only stage-2 meta (u = β · ln q) against the calibrated market out of fold; β per fold (Wald; meeting-clustered bounds in the private report). |
| `marginal:stage2:ability_vs_market_only` | E33 | `c3ecafa3859825bf` | New in v2, the C4 marginal rule: the paired ΔLL of the pooled member's stage-2 meta against the market-only meta, per fold and over the scored races, with meeting bounds. |
| `E33:stage2:top3_vs_winner_only` | E33 | `8a64e54ae991c2bd` | E33's rule: the paired ΔLL of the top-3 meta against the winner-only meta, per fold and over the scored races, with meeting bounds. |
| `E33:level1:top3_vs_winner_only` | E33 | `ae2f42ffe497af60` | Descriptive: the same for the members alone. |
| `E34:stage2:pooled_vs_rates:thin` | E34 | `e7ae075cc32a29ae` | E34's rule: the paired ΔLL of the pooled meta against the rates meta on the thin races, with meeting bounds. |
| `E34:stage2:pooled_vs_rates:all` | E34 | `4ec1c5d53313f3a5` | Descriptive: the same over every scored race (the marginal ablation). |
| `by_class` | E34 | `5fb369316930231f` | Per race class: the stage-2 and member-alone ΔLL against the calibrated market, E33's and E34's ΔLL, with bounds, and the thin-race share. |

Bounds are D7's: the meeting bootstrap (ΔLL, correlation) or the meeting-clustered sandwich (α, β), 95% one-sided
each.

The private report also holds the selection per outer fold (the rating pair, the pooled and rates discounts and the
inner log losses), each fold's α and β with Wald and meeting-clustered bounds, and the by-class table.

## Failures

None in v2: all 16 variants scored, the command exited 0, and every fit converged. The committed summary was written
by the fixed `write_committed_summary`: `write_summary` over this study's families only, then the committed index with
their lines added. It added `docs/experiments/registry/japan/japan-c2-ability-final-win-v2.md` and its line in the
index, and changed no other file. v1's file is unchanged. A full `write_summary` would still raise on #131's family
ids, which contain `/` (#344, #391).

## Superseded run (v1)

Family `japan-c2-ability-final-win`, spec sha256 `46e2b59865e8f9510645a3f5cc333ace5df33161307c243ce2f375fb5e79ae06`,
git `8cde358`, the same data builds. All 14 variants scored and none failed. Its private report and export stay at
`$HORSETRACK_DATA_DIR/reports/ability-japan-46e2b59865e8.md` (with its `.json`) and
`$HORSETRACK_DATA_DIR/features/japan/ability-japan-46e2b59865e8.parquet`; v2's names differ, so v2 left them as they
were. Its records stay in the registry (append-only), and its summary file stays in the index. It is superseded for
two reasons. First, it had no market-only meta, so its member's C4 gain was only against D4. Second, its rates' window
ended on the last race day before the race, so after a break between meetings the E34 comparator took in race days
that #129 leaves out. Every v1 variant is in v2 under the same name.

| Variant | E | Record |
| --- | --- | --- |
| `selection:inner_fold` | E33 | `b4a3ce05a5705f14` |
| `correlation:ln_q` | E33 | `56a03bc9e5c529f3` |
| `member_alone:ability` | E33 | `3fbeee8ed3751b57` |
| `stage2:q:ability` | E33 | `0fea89fcfa1944a4` |
| `meeting_clustered:stage2:q:ability` | E33 | `560bfa326598dd81` |
| `stage2:q:ability_winner_only` | E33 | `29e4685b84844bb9` |
| `meeting_clustered:stage2:q:ability_winner_only` | E33 | `3f9b103a5d58289d` |
| `stage2:q:ability_rates` | E34 | `766ca15849d88234` |
| `meeting_clustered:stage2:q:ability_rates` | E34 | `5edbb3d7b0cd9d8a` |
| `E33:stage2:top3_vs_winner_only` | E33 | `a3c1c548be2e650b` |
| `E33:level1:top3_vs_winner_only` | E33 | `d1c4708aa883085e` |
| `E34:stage2:pooled_vs_rates:thin` | E34 | `b3761f1f8e50f3cd` |
| `E34:stage2:pooled_vs_rates:all` | E34 | `4466c8aee39d70c9` |
| `by_class` | E34 | `6f988fc8c7189a39` |

## Alternatives rejected and why

- **D4 as the only comparator for the member's C4 gain (v1):** the stage-2 meta is linear in ln q, D4 is a spline in
  ln q fitted on other rows, so their ΔLL mixes the member with the change of market term. v2 keeps that comparison and
  adds the paired marginal against the market-only stage-2 meta.
- **v1's rate window** (ending on the last race day before the race): it differs from #129's (cut = day − 1) after a
  break between meetings; v2 uses #129's.
- **A full Bayesian fit (NumPyro, 08 B4's other option):** not used. The penalised MAP with empirical-Bayes variances
  and the Laplace σ_i is deterministic, needs no new dependency, and is fast enough for the run's several thousand
  fits; the tests pin its shrinkage, σ_i and the training-window rule.
- **The netkeiba pedigree cards for the sire:** rejected by the owner (answer 2): they come from races from 2021, so an
  earlier runner has a sire only if it raced again later, which selects on later racing.
- **Calling `fit_discounts` once per outer fold:** the same rule, but each outer fold would refit every inner quarter
  at every grid entry. The run shares each (pair, quarter, discounts) fit across the outer folds instead.
- **Choosing w2 and w3 on the test folds** (the risk 08's E33 row names): never; inner folds only.
- **#91's `select` and `walk_forward`'s min_train for the signed dates:** they would start the inner folds at 2011-Q2
  and level 1 at 2015-Q2 (see Method), so the signed dates are set explicitly.
- **A per-runner Bernoulli score for E34** (offered with the proposal): the owner chose the paired race-level winner
  ΔLL.
- **Card features (weight, draw, age, sex, layoff):** left to #93-#95 by the owner.
- **Place and Wide ΔLL against their own final prices** (08's E33 names them beside Win): outside this ticket, whose
  outline scores Win at the final price after stage 2; proposed below.
- **The N5 disagreement-tail curve:** N5 does not exist yet; the owner asked for it when N5 lands.

## What this rules out

The figures are withheld, so the repository states no size or direction of any result, and nothing here supports
adoption: D16 needs a holdout bound, and the holdout is not opened. The private report reads each frozen rule against
its bound. v2 is one run on one store build, recorded in the ledger as single-run. v1's figures are superseded: cite
v2's records, and read the member's marginal in C4 from `marginal:stage2:ability_vs_market_only`, not from the
stage-2 gain against D4.

## Follow-ups filed

None by this PR. Proposed, for the orchestrator to file under epic #73 as `status:proposed`:

- E33's Place and Wide half: the top-3 and winner-only members through C5's ordering against the Place and Wide final
  prices (08 §4 E33).
- The N5 disagreement-tail curve for this member, by class, when N5 lands (owner, 2026-11-27).
- Feed v2's export (u, σ_i, p_ability; `ability-japan-12b1eed01095-9a3f58d49fb3.parquet`) into #98's meld.
- Note in `core/eval/folds.py` (or its spec) that min_train does not count a partial first unit, so a study whose
  signed split names dates should set them explicitly, as this one does.

## On the leaderboard (2026-11-27)

Under the reframe (epic #447; D16 to D18 as amended 2026-11-27 in #448), the results above are measured results, not
adoption verdicts. Nothing is adopted or rejected: every option, model and member is a row on
`horsetrack japan leaderboard` (#452), ranked by its shrunk return per ¥100 at the final price, and a row with no
measured return is listed after the ranking, never dropped. Where this report says "not adopted", "rules out" or
"rejected", read a measured position. This section reopens and re-runs nothing.

- Registry families: `japan-c2-ability-final-win-v2` (the result of record) and `japan-c2-ability-final-win` (v1,
  superseded; its records stay on the record and on the board as superseded, not deleted).
- Task 4 (#451) maps no return for either, so both are listed with no return yet. Their records are withheld (D21):
  the public leaderboard shows their ids only, with no figure. A member's return in the leaderboard's format is a
  follow-up of epic #447, not a re-run started here.
- "Nothing is adopted" above now reads as a member measured against the final Win price and waiting for a return,
  not as a verdict on it.
