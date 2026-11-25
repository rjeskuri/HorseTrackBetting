# Experiment report: the C2 dynamic horse rating at the final Win price (Japan, #91)

Ticket #91 (epic #73; 08 §3.2 C2, §4 E38; 05 §3.2, §4.4-4.5). The run rates every JRA horse with a forward-filtered
dynamic rating, turns the rating into a within-race Win member, and measures that member against the final Win price
on D9's development folds: its correlation with ln q_final, its standalone pseudo-R², and its weight α in #87's
stage-2 meta-model beside ln q, unbounded. **Nothing is adopted** (D16): no holdout is opened. Every figure is an
aggregate over the JRA results store (netkeiba race results; no JRDB, no JRA-VAN). No runner or race row appears.

- Registry family `japan-c2-rating-final-win`, spec sha256
  `7ec8931213c607c1a76ded5e191b84b0d69b415c5c16ec30427375403fc4f21f`. It holds the hypothesis, the three decision
  rules, the rating's fixed parameters and its two grids, the selection rule, the meta, the folds and the bootstrap.
- **Disclosure: the frozen spec is not a clean pre-registration.** Before the family was frozen, the study's code
  was dry-run once on the real development data (no registry write, no file), and its outcomes were seen. The spec
  that was then frozen and run is byte-for-byte the one of that dry run: no grid, rule or parameter was changed after
  it, and the frozen run reproduces the dry run's figures exactly. The one change made after the first frozen run is
  the `meeting_clustered:stage2:q:rating` record (below), which puts the frozen stage-2 rule's own bound into the
  registry; the spec is unchanged.
- **Splits in the spec, awaiting the owner's sign-off.** Two choices in the frozen spec define splits, an owner
  decision, and are not settled by D9 or D10: the inner selection folds use D9's scheme over each outer fold's
  training rows with min_train cut from 22 to 4 quarters (some cut is forced: 22 quarters inside a 22-quarter window
  leaves no inner fold), and the rating reads results (placings only, no price) from the store's first day,
  2010-01-05, for its burn-in. Both were in the dry run above and are unchanged; both stand only with the owner's
  approval.
- Code: `countries/japan/models/rating.py` (the model, tested in `tests/countries/japan/test_rating.py`) and
  `countries/japan/studies/rating.py` (the run), at `bd4422c` for the first four records and at `e8a089a` for the
  fifth. To reproduce, run `uv run python -m horsetrack.countries.japan.studies.rating --run`.
  A rerun appends only records the family does not hold. The export was widened at `e233bf2` (below); the rerun
  there reproduced all five records and appended none.
- Store: `processed/jra/runners.parquet@6f3cec95347e0132`, `processed/jra/races.parquet@9cce511831074b64`.
- The full report is under the data directory, at `reports/rating-japan-7ec8931213c6.md`. Every record is in
  `docs/experiments/registry/japan/japan-c2-rating-final-win.md`.
- The per-runner export for the meld (#98) is at `features/japan/rating-7ec8931213c6.parquet`: every priced starter
  (601,265 rows, 2014-06-07 to 2026-11-23) at each (drift, scale) some fold selected (one pair here, drift 0 and
  scale 1.5), with `race_id`, `horse_id`, `race_date`, `drift_var_per_day`, `scale`, `selected_by` (the folds whose
  inner selection chose the pair), `oof_fold`, `mu_before`, `sigma_before` (σ_h(t)) and the member's `logp`. Every
  value is forward-filtered: it reads earlier races only, at a fixed pair. `oof_fold` names the one fold a row is out
  of fold for, the fold whose test quarter holds it (328,114 rows in 28 folds); it is null for the 273,151 rows
  before 2020-Q1. A row in a fold's training window is not out of fold for that fold, since the fold's pair was
  selected on those rows. For fold k the meld trains on k's training rows at k's pair and tests on the rows with
  `oof_fold` k, as the stage-2 fit here does.

## Owner decisions used

| Decision | What it fixes here |
| --- | --- |
| D1 | Condition on the final price: the member is measured beside the final Win q. |
| D4 | The Win baseline is the calibrated market (spline in ln q), fitted in-fold. |
| D6 | Winner log loss per race, a dead heat at 1/k per co-winner; pseudo-R² against equal chances. |
| D7 | Meeting (venue x day) clusters, 95% one-sided bounds (`config/scoring.yaml`). |
| D9 | Walk-forward quarters, quarterly refit, expanding, 1-week embargo; first scored quarter 2020-Q1 (`config/folds.yaml`). Every training window is at least 5.5 seasons (from 2014-06-07). |
| D10 | No price before 2014-06-07: the rating reads no price before it. Its results window from 2010-01-05 awaits sign-off (above). |
| D11-D15 | Holdout A2 from 2026-11-28, fully locked. No holdout day is read; not opened. |
| D16 | Nothing is adopted. |
| #87 (2026-11-24) | The meta-model shrinkage target is α = 0, the market alone; here α is unbounded and unshrunk (08 R13). |

## Hypothesis

E38 (08 §4): a forward-filtered dynamic rating member correlates below 0.94 with ln q_final and earns stage-2 weight
beside ln q. The prior that α moves is 10% or less; 08 expects members to correlate 0.94-0.98 with the market. The
frozen rules:

- distinct: the member's Pearson correlation with ln q over the scored runners is below 0.94 (descriptive).
- stage-2 weight: α > 0 when the meeting-clustered one-sided 95% lower bound of α, pooled over the scored rows
  in-sample, is above 0.
- stage-2 gain: the stage-2 meta's out-of-fold ΔLL against the calibrated market has its meeting-bootstrap lower
  bound above 0.

The null, the starting point: the rating carries nothing the final price lacks (α = 0).

## Method

- Rating (`DynamicRating`): a Gaussian random walk per horse, prior N(0, 1) at its debut, drift variance per day
  between its races, a spell variance of 0.25 added after more than 90 days off, and Weng and Lin's (2011)
  Bradley-Terry full-pair update with performance variance 1. It runs over every starter of the store's run races in
  race order (58,406 races, 821,458 starters, 2010-01-05 to 2026-11-23, holdout days excluded); a starter without a
  placing (中, 失) ranks tied last. Only the filtered values before each race (mu_before, sigma_before) feed the
  member.
- Member (`rating_member`): the softmax within race of (mu_before − the race's mean mu_before) / sqrt(scale² +
  sigma_before²), over the race's priced starters.
- Selection: drift in {0, 1e-4, 3e-4, 1e-3, 3e-3} per day and scale in {0.5, 0.75, 1, 1.5, 2, 3}. For each D9 fold
  the pair with the lowest mean winner log loss over the inner folds (D9's quarters over the fold's training rows,
  min_train cut to 4 quarters, a split awaiting sign-off) scores the fold's test quarter.
- Stage 2 (#87's `fit_meta`): u = α · m + β · ln q, unbounded, ridge 0, fitted on each fold's training rows with the
  member at the fold's pair and predicting its test quarter; α is also pooled over the scored rows in-sample.
- Scored: 28 folds, 2020-Q1 to 2026-Q4 (to 2026-11-23), 23,871 races in 1,991 meetings, from 43,130 development races
  with 601,265 priced starters (2014-06-07 to 2026-11-23).

## Variants tried

All five variants scored; none failed.

| Variant | Record | Result |
| --- | --- | --- |
| `selection:inner_fold` | `8e02f66cf5589c27` | Every fold picks drift 0 and scale 1.5 (inner log loss 2.270-2.279 per race; the worst grid pair 2.608-2.627). |
| `correlation:ln_q` | `78f9241b0f0be8f8` | Pearson of the member's ln p with ln q: 0.724 over the scored runners (0.726 within race); 0.645 to 0.773 by fold. |
| `member_alone:rating` | `37641dee8b216689` | Mean log loss 2.2781 against the calibrated market's 1.9275: ΔLL −0.3506 nats per race (bounds −0.3593 and −0.3419), Δ pseudo-R² −0.135. Pseudo-R² against equal chances: member 0.1223, q 0.2570, calibrated market 0.2574. |
| `stage2:q:rating` | `0e99ba0798a379d8` | α pooled −0.0447 (Wald 95% −0.0674 to −0.0219). Out of fold, ΔLL −0.000505 nats per race against the calibrated market (bounds −0.000966 and −0.0000447). Per fold α is −0.029 to −0.007, below 0 in all 28 folds; β on ln q 1.013 to 1.035. |
| `meeting_clustered:stage2:q:rating` | `4458a2a8040d2247` | The pooled α's meeting-clustered one-sided 95% bounds −0.0635 and −0.0258. Per fold, the upper bound is below 0 in 20 of 28 folds (every fold from 2022-Q1). |

Bounds are D7's: the meeting bootstrap (ΔLL) or the meeting-clustered sandwich (α), 95% one-sided each.

The frozen rules read: distinct, met (0.724 < 0.94). Stage-2 weight, not shown: the lower bound is −0.0635. Stage-2
gain, not shown: the lower bound is −0.000966.

## Alternatives rejected and why

- TrueSkill-Through-Time (smoothing): its estimate at t reads later races, which the leakage rule forbids for
  features (05 §3.2).
- TrueSkill's Thurstone factor graph and Glicko-2: Weng-Lin's Bradley-Terry rule gives closed-form multi-runner
  updates that need no message passing over 18-runner fields, and its guarantees (winner up, last down, no variance
  widens) are what the tests pin.
- A member that ignores sigma: the meld (05 §4.4-4.5) needs sigma, and the member pulls an uncertain rating toward the
  field's mean.
- Selecting prior_var, perf_var, spell_var or spell_days too: the ticket names only the drift; the scale is the
  member's own logit scale and is cheap to choose with it. The other four stay fixed in the spec.

## What this rules out

- At D7's level, on development folds: a positive stage-2 weight on this rating beside ln q. α ≥ 0 is ruled out
  (upper bound −0.0258), and so is α below −0.0635. Given the final price, a horse's higher past-results rating
  means a slightly lower win chance than its price says: the market over-weights what this rating measures.
- Any gain of the rating's stage-2 meta over the calibrated market out of fold: ΔLL ≥ 0 is ruled out (upper bound
  −0.0000447 nats per race). On the same races normalised q itself trails the calibrated market by about 0.00098
  nats per race (the q row above), so stage 2 on ln q with this member recovers part of the spline's calibration
  but not all of it.
- E38's expected 0.94-0.98 correlation does not hold for this member: it correlates 0.72 with the market, and alone
  it explains less than half of the market's pseudo-R² (0.122 against 0.257).
- The random walk adds nothing at inner-fold log loss: every fold picks drift 0, the grid's lower edge (a variance
  cannot go below it), so this rating is in effect a static ability with spell jumps.

This is one run on one store build; it ships as single-run in the ledger. Nothing is tested on the holdout, so
nothing is adopted.

## Follow-ups filed

None filed with this change. Proposed for the orchestrator to file under epic #73 as `status:proposed`:

- Feed the export into #98's meld with sigma_h(t), and test the negative α beside the other members.
- Test whether the negative α is #218's negative weight on form within the market (by odds band), using this
  member as the form measure.
- A pre-registered rerun that also selects prior_var and spell_var (and a class-based debut prior) by inner folds,
  since drift sits at its lower edge.
