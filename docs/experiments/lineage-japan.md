# Experiment report: the lineage prior in the ability member, E37 (Japan, #538, C2)

Ticket #538 (epic #73; 08 §4 E37, §3.3; 05 §4.1-4.3), split out of #95. #95 built the lineage prior
(`countries/japan/models/lineage.py`: a sire × surface × distance-band prior on the utility scale of #92's
Plackett-Luce model, the damsire at a weight) and E37's segment scoring (`studies/lineage.py`). This study feeds the
prior into #92's ability member for debut and thin-history runners and asks whether the member improves on debut,
first-time-surface and first-time-distance races, standalone and at the final Win price.

**Nothing is adopted (D16-D19), and no bet is placed.** Every variant × segment is a member row on the leaderboard.

Every figure here is an aggregate over the JRA results store (netkeiba race results), the netkeiba pedigree
(`processed/jra/horse_pedigree`) and the JRA final Win prices. No JRDB, no JRA-VAN, and no race or runner row appears.
#92's member as fitted reads the JRA-VAN sire, so this study's base member swaps that sire for the netkeiba sire name;
its records' sources are `jra` only.

- Registry family `japan-c2-lineage-e37-netkeiba-v3`, spec sha256
  `f5c6a8bf139f32ed95a53c08170ec2540383c48669670755f6d4de6128e3f2c1`, frozen before any outcome was read. It
  supersedes `japan-c2-lineage-e37-netkeiba-v2` (spec `c7a99ecd88ce…`), which superseded
  `japan-c2-lineage-e37-netkeiba` (spec `f49b85f4af0b…`); "Variants tried" gives both failed runs. Every record of all
  three stays.
- Code at `35fefd9` (`countries/japan/studies/lineage.py`; #92's `studies/ability.py` gained keyword-only options
  whose defaults keep #92's fits). To reproduce: `uv run horsetrack japan lineage-study`.
- Store: `processed/jra/runners.parquet@6f3cec95347e0132`, `processed/jra/races.parquet@9cce511831074b64`,
  `processed/jra/horse_pedigree.parquet@f0ce063e067fb596`, read column- and date-scoped.
- The full report and its JSON are under the data directory at
  `reports/lineage-japan-f5c6a8bf139f-130112341e22.{md,json}`; the committed registry summaries are
  `docs/experiments/registry/japan/japan-c2-lineage-e37-netkeiba-v3.md` (and the `-v2`, v1 and v1 `-trial` files).
- The run took 6,626 s at a peak RSS of 1.41 GB: 1,440 base fits (4,308 s), four lineage features (232-423 s each)
  and 160 lineage-member fits.

## Owner decisions used

| Decision | What it fixes here | Where it was signed |
| --- | --- | --- |
| #92's design | Level 1 from 2010 on results only, out of fold from 2015-Q1 on D9's quarters, inner folds from 2011-Q1, #91's rating pair and the discounts selected per fold; thin history is fewer than 30 prior starts. | #92 (2026-11-26) |
| D7 | Bounds are meeting-clustered (venue × day), 95% one-sided (`config/scoring.yaml`; seed 538). | §3, #14 |
| D9 | Quarterly expanding walk-forward folds, 1-week embargo (`config/folds.yaml`). | §4, #14; #70 |
| D10 | The level-1 model reads no price, so rows before 2014-06-07 may train; level 2 reads the price era only. | §4, #14 |
| D11-D15 | Holdout A2 from 2026-11-28 is fully locked. Holdout days are dropped by the store readers, and the holdout was not opened. The store ends 2026-11-23, so none was present. | §5, #14; #71 |
| D16-D19 | Nothing is adopted; every variant ranks (owner reframe, 2026-11-27). | #447 |
| D21 | No figure derived from a licensed source appears here. None is read: the pedigree is netkeiba's, and a UKC-merged table is refused. | #14 |

## Hypothesis

The null is the starting point: **the lineage feature adds nothing** (ΔLL ≤ 0 on every segment, standalone and at the
final price).

- **E37.** A sire × surface × band prior, the damsire at a weight, fed into #92's ability member as an unpenalised
  feature for runners with fewer than 30 prior starts (0 otherwise), lowers the member's winner log loss on debut,
  first-time-surface and first-time-distance races, standalone and after #87's stage 2 at the final price.
- **Expected size: small.** Pedigree-only predictive correlation is about 0.08 in JRA data, and the market is strong in
  debut and maiden races (08 §3.3; 07 §5.2).
- **The decision rules (frozen):** a variant improves on a segment when the meeting-bootstrap one-sided 95% lower bound
  of its paired per-race winner ΔLL lies above 0, standalone (lineage member against base member) and at the final
  price (the stage-2 meta with the lineage member against the stage-2 meta with the base member). Against the
  market-only stage-2 meta, and the base member's own marginal, are descriptive.

## Design

All of it is in the frozen spec (`studies/lineage.py`, `build_spec`).

- **The pedigree.** `processed/jra/horse_pedigree` (netkeiba's pedigree cards) keyed on its normalised names: the sire
  is `sire_name_norm`, the damsire `dam_sire_name_norm` (its id columns are empty). `netkeiba_pedigree` refuses a table
  carrying any column a UKC (JRDB) merge adds, so the file `jrdb.tables` writes back after a merge cannot be read. It
  covers mostly foals of 2018-2024, so the share of starters with a netkeiba sire rises from under 1% before 2015 to
  about 80% in 2021 and about 100% from 2024 (the coverage table below).
- **The base member.** #92's pooled member (`studies.ability`: the top-3 Plackett-Luce `AbilityModel` on #91's rating
  plus jockey, trainer and sire effects shrunk by empirical Bayes), with the netkeiba sire name in place of the JRA-VAN
  sire. The same level-1 frame (every starter of a run race with a winner from 2010, no price), D9's quarters with the
  1-week embargo, inner folds from 2011-Q1, out of fold from 2015-Q1, and #92's rules for the rating pair and the
  discounts (all 15 entries of `DISCOUNT_GRID`) per outer fold. A fit carries sire effects only when its training rows
  hold at least 50 distinct netkeiba sires (a count of ids), which holds from the 2016-10-01 quarter on; earlier fits
  have jockey and trainer effects only. v1 had no such rule, and its run stopped (below).
- **The feature.** One `LineagePrior` per D9 quarter, fitted on the level-1 runs before 00:00 UTC of the quarter's
  first day, from the first quarter whose earlier runs hold progeny runs of at least 50 stallions (2016-10-01; earlier
  rows have feature 0): u = g(sire, cell) + w · g(damsire, cell), g(s, c) = g_sire[s] + d[s, c], d ~ N(0, τ_cell²), τ_sire² by
  empirical Bayes, top-3 stages at discounts (1, 1, 1), no trainer effect. A run's feature is its quarter's prior mean
  at its horse, surface (turf or dirt) and #438's distance band (up to 1400, 1800, 2200 m, longer) when the horse has
  fewer than 30 prior starts (its level-1 runs on earlier race days; a debut has 0), and 0 otherwise. A jump race has no
  cell. Training and test rows alike read only runs before their quarter (`test_lineage_run.py` pins it).
- **The lineage members.** For each variant, the base member at the fold's selected rating pair and discounts, plus the
  feature as an unpenalised coefficient: an ablation of one feature, not reselected. An outer fold whose training rows
  have the feature 0 throughout (those ending before the first prior) takes the base member's fit, coefficient 0.
- **Level 2.** #92's: the priced starters of the price era (`studies.meta.store_frame`), D4's calibrated market out of
  fold, and #87's stage-2 meta (u = α · m + β · ln q, unbounded, ridge 0) of each member, beside the market-only stage-2
  meta (u = β · ln q), each on the same rows and folds. A race with a starter lacking any member is dropped.
- **The window.** The scored folds are the level-2 D9 folds from the first quarter from which every quarter has at least
  80% of its level-1 starters with a netkeiba sire (a count; no outcome read).
- **Segments.** `run_segments` over the level-1 frame: debut, first time on the surface, first time in the band; a race
  is in a segment when any of its priced starters is flagged, and `all` is every scored race. Each ΔLL is the mean
  paired per-race winner ΔLL with D7's meeting bootstrap, one-sided 95% (10,000 replicates, seed 538).

## Variants tried

The data. 821,458 level-1 starters in 58,406 races (2010-01-05 to 2026-11-23); 301,292 of them have a netkeiba sire.
The netkeiba sire share of a quarter's starters is 0% in 2014-Q1, 2.8% in 2016-Q4, 15.4% in 2019-Q1, 27.1% in 2020-Q1,
78.7% in 2021-Q2, 85.6% in 2021-Q3 and 100% from 2024. The coverage rule opens the window at **2021-07-01**: 22 scored
D9 folds (2021-Q3 to 2026-Q4), 18,663 races in 1,556 meetings, of which 4,237 have a debut runner, 6,947 a runner new
to the surface and 9,185 a runner new to the band. No race was dropped for a missing member. The sire block and the
prior both start at 2016-10-01 (23 of the 64 level-1 quarters fit without sire effects). Each lineage member fits the
feature on 40 of the 48 outer folds: the 8 from 2015-Q1 to 2016-Q4 train before the first prior and take the base
member's fit.

The base member's selection: discounts (1, 0.5, 0) on 46 of 48 outer folds and (1, 0.25, 0) on 2; rating pair
(drift 0, scale 1.5) on 32 folds and (drift 1e-4, scale 1.5) on 16. Each lineage member's coefficient on the feature
lies between 0.46 and 0.50 on every scored fold (0.015 to 0.04 on the first folds after the prior starts).

**Segment ΔLL** (mean per-race winner ΔLL, positive when the lineage member's side is better; in brackets the meeting
bootstrap's one-sided 95% lower and upper bounds, 10,000 replicates). Standalone is the lineage member alone against
the base member alone; "vs base" the stage-2 meta with the lineage member against the stage-2 meta with the base
member; "vs market only" the stage-2 meta with the lineage member against the market-only stage-2 meta.

| Variant (τ_cell, damsire weight) | Segment | Races | Standalone | Stage 2 vs base | Stage 2 vs market only |
| --- | --- | ---: | --- | --- | --- |
| `lineage_t0.1_d0` (0.1, 0) | debut | 4,237 | +0.0076 [+0.0047, +0.0104] | −0.00001 [−0.00004, +0.00002] | −0.00005 [−0.00014, +0.00005] |
| | first surface | 6,947 | +0.0066 [+0.0045, +0.0087] | −0.00003 [−0.00005, −0.00000] | −0.00014 [−0.00022, −0.00006] |
| | first distance | 9,185 | +0.0068 [+0.0050, +0.0085] | −0.00003 [−0.00005, −0.00001] | −0.00014 [−0.00021, −0.00008] |
| | all | 18,663 | +0.0079 [+0.0066, +0.0092] | −0.00001 [−0.00003, +0.00000] | −0.00010 [−0.00015, −0.00006] |
| `lineage_t0.1_d0.25` (0.1, ¼) | debut | 4,237 | +0.0089 [+0.0057, +0.0120] | −0.00003 [−0.00007, +0.00001] | −0.00006 [−0.00016, +0.00004] |
| | first surface | 6,947 | +0.0073 [+0.0050, +0.0096] | −0.00004 [−0.00007, −0.00001] | −0.00015 [−0.00024, −0.00007] |
| | first distance | 9,185 | +0.0075 [+0.0056, +0.0094] | −0.00004 [−0.00006, −0.00001] | −0.00015 [−0.00022, −0.00008] |
| | all | 18,663 | +0.0089 [+0.0075, +0.0103] | −0.00002 [−0.00004, −0.00000] | −0.00011 [−0.00016, −0.00006] |
| `lineage_t0.3_d0` (0.3, 0) | debut | 4,237 | +0.0105 [+0.0073, +0.0136] | −0.00002 [−0.00007, +0.00002] | −0.00006 [−0.00016, +0.00005] |
| | first surface | 6,947 | +0.0082 [+0.0058, +0.0106] | −0.00004 [−0.00007, −0.00000] | −0.00015 [−0.00024, −0.00006] |
| | first distance | 9,185 | +0.0073 [+0.0053, +0.0093] | −0.00004 [−0.00007, −0.00001] | −0.00015 [−0.00022, −0.00008] |
| | all | 18,663 | +0.0091 [+0.0077, +0.0106] | −0.00002 [−0.00004, +0.00000] | −0.00010 [−0.00015, −0.00006] |
| `lineage_t0.3_d0.25` (0.3, ¼) | debut | 4,237 | +0.0123 [+0.0089, +0.0157] | −0.00003 [−0.00009, +0.00002] | −0.00007 [−0.00018, +0.00004] |
| | first surface | 6,947 | +0.0093 [+0.0067, +0.0119] | −0.00005 [−0.00010, −0.00001] | −0.00016 [−0.00026, −0.00007] |
| | first distance | 9,185 | +0.0082 [+0.0060, +0.0104] | −0.00005 [−0.00009, −0.00001] | −0.00016 [−0.00024, −0.00009] |
| | all | 18,663 | +0.0103 [+0.0087, +0.0119] | −0.00003 [−0.00005, +0.00000] | −0.00011 [−0.00017, −0.00006] |

The base member's own marginal (its stage-2 meta against the market-only stage-2 meta): debut −0.00004 [−0.00011,
+0.00004], first surface −0.00011 [−0.00018, −0.00004], first distance −0.00011 [−0.00017, −0.00006], all −0.00009
[−0.00012, −0.00005].

**Stage 2 against D4's calibrated market** (every scored race; ΔLL with one-sided 95% bounds) and the member's weight
α pooled in-sample over the scored rows (meeting-clustered one-sided 95% bounds):

| Meta | ΔLL vs calibrated market | Pooled α |
| --- | --- | --- |
| `stage2:q:ability_netkeiba` (base) | −0.00094 [−0.00142, −0.00045] | −0.036 [−0.057, −0.014] |
| `stage2:q:lineage_t0.1_d0` | −0.00095 [−0.00143, −0.00046] | −0.037 [−0.059, −0.015] |
| `stage2:q:lineage_t0.1_d0.25` | −0.00096 [−0.00144, −0.00047] | −0.035 [−0.057, −0.013] |
| `stage2:q:lineage_t0.3_d0` | −0.00096 [−0.00144, −0.00046] | −0.035 [−0.057, −0.013] |
| `stage2:q:lineage_t0.3_d0.25` | −0.00096 [−0.00145, −0.00047] | −0.033 [−0.055, −0.011] |
| `stage2:q:market_only` | −0.00085 [−0.00133, −0.00036] | — |

**Failed runs, kept in the record.**

- `japan-c2-lineage-e37-netkeiba` (v1): its 31 variants failed together, `RuntimeError: empirical Bayes did not
  converge in 100 updates`, at the base member's 2014-Q1 fit, whose training rows held 3 netkeiba sires from 7 runs.
  #92's secant steps on the sire block's ln τ² cycled with period 4 there, although that block's fixed point exists
  (its residual crosses 0 near ln τ² = 1.5). v2 adds the frozen count rules of "Design".
- `japan-c2-lineage-e37-netkeiba-v2`: the selection, the base member's and the market-only stage-2 metas and the base
  marginal scored (11 records); the 20 lineage records failed, `ValueError: the design is not identified: singular
  information`, because the 8 outer folds before 2017-Q1 train on rows whose feature is 0 throughout. v3 adds the
  empty_feature rule.
- `japan-c2-lineage-e37-netkeiba-trial`: the `--trial` pipeline check (results from 2021, 3 discount entries, spec not
  frozen), 31 records, scored. Not read for this report.

## Alternatives rejected and why

- **The JRA-VAN pedigree over the full window.** It would cover every horse since 2010, but its figures are withheld
  (D21) and it doubles the run. Not run; filed as a follow-up, to be its own withheld family.
- **Reselecting the discounts for each lineage member.** Fifteen fits per quarter per variant instead of one per outer
  fold. The ablation keeps the base member's fold selection, so the feature is the only difference.
- **More tau_cell values, a second band scheme.** The grid is two tau_cell values (0.1, 0.3, the learnable range of 08
  §3.3) × two damsire weights (0, ¼), on #438's bands, to stay within a few hours on one machine.
- **Nicks and dosage.** A further variant, expected to rank low; not this ticket.
- **The relationship matrix (an animal model).** The prior needs only the sire and the damsire (#95).
- **A trainer effect in the prior mean.** The ability member already carries the trainer effect.

## What this rules out

These are measurements on 22 folds from 2021-Q3, not verdicts. Nothing is adopted, and no return per ¥100 is measured
here: a member is scored by its winner log loss.

- **Standalone, the null is ruled out for every variant and segment.** All 16 lower bounds lie above 0, from +0.0045
  (τ_cell 0.1, no damsire, first surface) to +0.0089 (τ_cell 0.3, damsire ¼, debut). On every segment the gain is
  largest at τ_cell 0.3 with the damsire at ¼, and largest of all on debuts, +0.0123 [+0.0089, +0.0157] nats per race.
  It is not confined to the segments: over every race it is +0.0079 to +0.0103, because about 289,000 level-1 runs
  carry a nonzero feature (fewer than 30 prior starts and a sire with a prior).
- **At the final price, a gain is ruled out.** The stage-2 meta with a lineage member never beats the stage-2 meta with
  the base member. Every upper bound is at most +0.00002 nats per race. On first-surface and first-distance races every
  upper bound is at or below 0, so there the lineage meta is measurably, if very slightly, worse. The prior's
  information is already in the final Win price.
- **No member adds to the market here.** Against the market-only stage-2 meta, every lineage meta and the base member
  lose on first-surface, first-distance and all races (upper bounds below 0), and gain at most +0.00005 on debuts. The
  members' pooled stage-2 weight α is negative, about −0.035 (bounds exclude 0). Beside ln q, the member's ability
  ranking counts slightly against a runner.
- **Every stage-2 meta here, market-only included, is worse than D4's calibrated market** on these folds: about −0.0009
  nats per race, bounds excluding 0. The linear ln q term of stage 2 is a weaker market model than D4's spline here,
  so the C4 comparisons above are the like-for-like ones.
- This matches the expectation of 08 §3.3: the pedigree prior is learnable and helps the member alone, but the market
  in debut and maiden races already prices it.

## Leaderboard rows

Each record is one member row (`research.leaderboard_backfill.LEGACY_KINDS`: kind `member`). None has a return per
¥100, so each row is listed unranked (D16). None is withheld: the sources are `jra` only.

- `japan-c2-lineage-e37-netkeiba-v3/<variant>`, 31 rows, all scored:
  - `selection:inner_fold`;
  - `level1:lineage_t{0.1,0.3}_d{0,0.25}`;
  - `stage2:q:{ability_netkeiba, lineage_t…, market_only}`;
  - `e37:{ability_netkeiba, lineage_t…}:{debut, first_surface, first_distance, all}`.
- `japan-c2-lineage-e37-netkeiba-v2/<the same 31 variants>`: 11 scored, 20 failed.
- `japan-c2-lineage-e37-netkeiba/<the same 31 variants>`: all failed.
- `japan-c2-lineage-e37-netkeiba-trial/<the same 31 variants>`: the trial, scored.

## Follow-ups filed

Proposed, to be filed under epic #73 as `status:proposed`:

- **E37 on the JRA-VAN pedigree over #92's full window (withheld).** The same members with JRA-VAN sire and damsire ids
  from 2020-Q1, beside #92's own member, as a separate withheld family (D21).
- **The sire-prior's information at the final price, by odds band.** Standalone gains of about 0.01 nats per race
  that vanish after stage 2 suggest that the market prices pedigree. N5's disagreement tail would show whether any odds
  band keeps a residual.
