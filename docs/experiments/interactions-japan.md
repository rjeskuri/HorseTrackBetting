# Experiment report: C2 interactions (jockey, trainer, track) at the final Win price (Japan, #93)

Ticket #93 (epic #73; 08 §3.3, §4 E35, E36, §8 B5; 05 §4.1-4.3). The run adds interaction blocks to #92's
Plackett-Luce ability model, measures the cell-size distribution of each interaction first, estimates each block's τ
with bounds, and measures each block member against the model without the block, out of fold, standalone and at the
final Win price through #87's stage-2 meta, with its tail return at the final price. **Nothing is adopted** (D16, D19
as amended on #448): each block is a ranked member row. The holdout is not opened. Every figure is an aggregate over
the JRA results store (netkeiba race results, final Win odds and Win dividends; no JRDB, no JRA-VAN), so the figures
are public. No runner or race row appears. A horse × jockey estimate is predictive, never causal (booking endogeneity, 05 §4.2), and
never a bet trigger.

- **Current:** registry family `japan-c2-interactions-final-win-v2`, spec sha256
  `2dde55836222c16552431d0037f0aabf4fbc55e814e79d139d5d7aa90e24d2e0`, frozen when its run started, before any row
  was read. It supersedes
  `japan-c2-interactions-final-win` (v1, spec `3060dfbf9626cf7d46186153de6ee4b6c0b65111acadbefc464d5aca300bded0`),
  found wrong in #93's review (round 1) on two counts: v1's tail settled a winner at its final odds over the race's
  number of winners instead of the dividend it paid, and its records named one git revision for level-1 fits made at
  two. v2's spec is v1's with the tail settled at the recorded Win dividend (below), a `supersedes` and a `provenance`
  entry; its level1, tau and folds sections are v1's, byte for byte. Both specs hold the hypotheses, the null, the
  rules, the base and every block member with its columns, the fit, the debut and transfer definitions, horse ×
  trainer's treatment, the cell table, τ's bounds, level 2, the tail, the folds, D7's bootstrap and the leaderboard
  option ids. Before v1's freeze the study ran only on synthetic races and the real store was read for key counts
  only (rows, horses, cells per interaction; no outcome) to size the fits.
- **Run history (four launches, two specs).** v1's first launch (git `022d4fc`) was stopped by hand after 5 of the
  48 level-1 folds were saved (2015-Q1 to 2016-Q1; this report said 4 before): trainer × track's dense fit
  alternated U and W and would have taken many hours. The second (`56d0669`, W in the joint Newton step) fitted
  afresh, matching the first's 5 folds (every other member's out-of-fold u bit for bit, trainer × track's within
  7e-6), and
  was stopped by hand after 40 folds, when horse_aptitude's empirical Bayes did not converge in 100 updates at
  2024-Q2, 2024-Q3 and 2024-Q4 (a weakly identified variance whose fixed-point residual is nearly flat). The third
  (`b5a726c`: the fixed point's step doubles while a residual keeps its sign, and a fit with local terms stops at a
  1e-4 residual in ln τ²; a 22-line change to `models/interactions.py` and the resume) resumed the second's checkpoint,
  fitted only the member-folds without a prediction (horse_aptitude 2024-Q2 to 2026-Q4, every other member from
  2025-Q1) and completed with v1's 53 records. v2's launch (git `109dce4`) fitted nothing: `adopt_v1_checkpoint`
  took v1's completed checkpoint, stamped each member-fold with the revision that fitted it (`56d0669` where its u
  equals the stopped second launch's saved u bit for bit, else `b5a726c`), carried the second launch's three failed
  fits forward, and took the last fold's τ intervals, the information per run and the rating pairs from v1's report;
  the run then re-scored level 2. Fits by revision: `56d0669` 237 member-folds (folds 2015-Q1 to 2024-Q4;
  horse_aptitude to 2024-Q1), `b5a726c` 51 (the rest). Each v2 record after level 1 names them per member and fold
  (`params.fit_revisions`). This is a mixed run: no fold was refitted at one revision for this report.
- **Which records are current.** v2's 56 records are current: the 53 variants and the second launch's three
  failed horse_aptitude fits (`fit:horse_aptitude:<fold>@56d0669`, status failed). v1's records stay in the registry
  (nothing is deleted): its 53 scored records are superseded (the tail's settlement, one revision named for two),
  and 109 records were appended to v1 by `record_v1_launches` (`params.backfilled`) for the launches that wrote none:
  each stopped launch's 53 planned variants as failed, at `022d4fc` and at `56d0669`, and the second launch's three
  failed fits (it kept the first fold's error only, so the other two say so). v1 now holds 162 records.
- Code: `countries/japan/models/interactions.py` (the blocks, tested in `tests/countries/japan/test_interactions.py`)
  and `countries/japan/studies/interactions.py` (the run). To reproduce, run
  `uv run horsetrack japan interaction-study`; it resumes the level-1 checkpoint
  `$HORSETRACK_DATA_DIR/features/japan/interactions-japan-level1-e2cb2f08123f.npz` (keyed by the spec's level1, tau
  and folds sections and the store build; one file, published by rename). On this 16 GB Mac v1's level-1 fits took
  about 4 h in total, peak resident memory 2.7 GB, one fit at a time; v2's re-score took 165 s (the rating grid,
  level 2 and the records; peak resident memory 1.5 GB). v1's three checkpoints are kept under
  `features/japan/interactions-japan-3060dfbf9626-faff16a8cb38-level1*`.
- Data builds: `processed/jra/runners.parquet@6f3cec95347e0132` and `processed/jra/races.parquet@9cce511831074b64`
  (the store build of #91's and #92's runs); v2 also reads `processed/jra/dividends.parquet@b3d635352bb007df`.
- The full report is at `$HORSETRACK_DATA_DIR/reports/interactions-japan-2dde55836222-28bea4f0bfa3.md`, with its
  tables as JSON beside it; the per-runner export (each member's out-of-fold u, for #98's meld) is at
  `$HORSETRACK_DATA_DIR/features/japan/interactions-japan-2dde55836222-28bea4f0bfa3.parquet`. v1's are beside them
  (`interactions-japan-3060dfbf9626-faff16a8cb38.*`). These are the paths v1's and v2's records cite. Since #93's
  review (round 5) a launch's artifacts are named by the spec, the builds and a digest of its results, and an existing
  one is never rewritten: a later launch that changes a result (a retry that recovers a failed fold) writes
  `interactions-japan-<spec>-<builds>-<results>.*` beside these, and only its new records cite it.
- **The families' registry summaries are not committed.** `write_summary` renders each over the public-tree guard's
  256 KB cap (`scripts/check_public_tree.py`, rule `size`), and a hand-cut copy is not allowed. The study writes them
  unchanged to `$HORSETRACK_DATA_DIR/reports/registry-summary-japan-c2-interactions-final-win-v2.md` and
  `...-final-win.md` and leaves their index lines out until #330 lets `write_summary` split them. Every figure below
  is in v2's.

## Owner decisions used

| Decision | What it fixes here |
| --- | --- |
| #93 reframe (owner, 2026-11-27) | Each block (horse × jockey, trainer × track, structured aptitude, trainer debut prior) is a ranked member row: ΔLL standalone and at the final price with CI, and its tail return at the final price. Horse × trainer is not fitted where it is aliased, and is recorded with its cell count. Claude pre-registers every block and pooling variant; no needs-input stop. |
| D1 | Condition on the final price: each member is measured beside the final Win q. |
| D4 | The Win baseline is the calibrated market (spline in ln q), fitted in-fold. |
| D6 | Winner log loss per race, a dead heat at 1/k per co-winner. |
| D7 | Meeting (venue × day) clusters, 95% one-sided bounds, 10,000 replicates (`config/scoring.yaml`, seed 93). |
| D9 | Quarter folds, quarterly refit, expanding, 1-week embargo (`config/folds.yaml`); level 2 scores every fold from 2020-Q1. |
| D10 | Level 1 reads no price; level 2 reads prices from 2014-06-07 only. |
| D11-D15 | Holdout A2 from 2026-11-28, fully locked; every read drops holdout days; not opened. |
| D16, D19 (as amended, #448) | Rank, not adopt; flat stakes. |
| D21 | Only the JRA results store is read, so the records carry the source `jra` and the figures are public. #92's sire (JRA-VAN) is left out of the base for this reason. |

## Data

- **Level 1** (results only): every not-scratched starter of the JRA store's run races with a winner, from 2010-01-05,
  with `jockey_key`, `trainer_key`, the race's venue, distance and surface, and race_utc (the post time, JST, in UTC;
  every race has one). A starter without a placing has no `finish_pos`: it stays in every choice set and is never
  chosen. `debut` marks a horse's first start in the frame (from 2010); `transfer` marks a run after a stable change
  of trainer (`transfer_step`, the new trainer kept for 2 consecutive runs, read as of each run).
- **Rating:** #91's `DynamicRating` over the same starters, its member at the (drift, scale) #91's rule picks per
  outer fold.
- **Level 2:** `studies.meta.store_frame`, the priced starters of run races with a priced winner from 2014-06-07, and
  each member's ln p renormalised over the race's priced starters.
- **Settlement (v2):** the store's `dividends` rows of pool WIN for those races, joined by race and the runners'
  horse number (`win_paid`): a runner pays its dividend / ¥100, a refund row its stake, no row 0. A race with no WIN
  row at all is unsettled (an interrupted store refresh), not a race every starter lost: its starters' return is
  missing, a scored one fails each tail variant and one that only trains is never read (no race here). The claim
  bands stay at the final odds (p × O_final).

- Counts: 821,458 level-1 starters in 58,406 races (2010-01-05 to 2026-11-23), 3,929 without a placing, 78
  dead-heat races; 86,502 horses (so 86,502 debut starts), 67,331 starts after a stable change of trainer; 572,772
  out-of-fold starters from 2015-Q1. Level 2: 601,265 priced starters in 43,130 races; scored, 23,871 races over 1,991
  meetings in 28 folds (2020-Q1 to 2026-Q4). Settlement: 43,196 WIN dividend rows, every one joined to a priced
  starter, every race with one; 2 paid starters not placed first and 2 first-placed starters without a dividend
  (dividends paid before a disqualification).
- #91's rule picked (drift 0, scale 1.5) in 32 outer folds and (drift 1e-4, scale 1.5) in 16.
- The base's mean information per run on the last fold's training rows is 0.106 nats at discounts (1, 0.5, 0.25),
  against 08 §3.3's 0.176 for top-3 training.

## Hypothesis

The frozen spec reads:

- **E35.** "Horse × jockey: τ_hj < 0.1 and its cells add nothing at the final price (08 §4 E35; the negative control;
  booking endogeneity makes any estimate predictive, never causal, and it is never a bet trigger)."
- **E36.** "Trainer × track (east/west travel): τ ≥ 0.05, learnable, with a small standalone gain that is probably
  priced (08 §4 E36)."
- **E93.** "Horse × track/course/distance in the structured form, the trainer debut prior and the transfer step
  (08 §3.3; §8 B5): each measured standalone and at the final price; horse × trainer is aliased with the horse where
  the trainer is constant and is not fitted."
- **Null**, the starting point: no block carries anything the base or the final price lacks: τ = 0 and a paired ΔLL
  of 0 standalone and at the final price; the tail returns the takeout.

The rules are descriptive (rank, not adopt): τ's empirical-Bayes value per fold and its profile-evidence bounds (each
one-sided 95%) on the last outer fold's fit, "learnable" read from τ's lower bound against 08 §3.3's 0.05; the paired
per-race winner ΔLL of each block member alone against the base alone (standalone) and of its stage-2 meta against the
base's stage-2 meta (at the final price), meeting bootstrap, one-sided 95% bounds; each member's stage-2 meta against
the market-only stage-2 meta (the C4 marginal); and the tail: realised Win return per ¥100 of every scored runner by
band of its stage-2 claim p × O_final (bands at the 50th, 90th and 99th percentiles), the top band the leaderboard row.
v2 settles each runner at the store's recorded Win dividend for its race and horse number, over the ¥100 unit stake
(0 without a dividend row); v1 settled a placed first at its final odds over the race's number of them.

## Method

- **Base** (`AbilityModel(k=3)`, #92's model without the sire): u = β · rating + jockey + trainer effects, each
  N(0, τ_e²) with τ_e² by empirical Bayes (MacKay's evidence fixed point), the penalised MAP by Newton's method; the
  discounted top-3 Plackett-Luce likelihood at fixed discounts (1, 0.5, 0.25) (not selected: #92's selection is E33's
  and is withheld with its sire).
- **Blocks** (`models.interactions.fit_with_blocks`), each with its own N(0, τ²) prior and τ² by the same fixed point:
  - horse_x_jockey (E35): a variance component per (horse, jockey) cell;
  - trainer_x_track (E36): a variance component per (trainer, venue) cell plus the one-factor east/west term, the
    rank-one U_trainer · W_venue with W centred at unit scale;
  - horse_aptitude (E93): the structured a_h + b_h·z(d) + c_h·z(d)² + e_h·1{dirt} (a prior per coefficient; z the
    distance standardised over the training rows) plus ⟨U_h, W_track⟩, a rank-one (horse, venue) block; 08 §3.3's
    pedigree prior on (b_h, e_h) is #95's;
  - trainer_debut (E93): a debut indicator and a variance component per (trainer, debut) cell, so a debut runner's
    level is shrunk toward its trainer's debut mean beyond the trainer's main effect, which the base already uses as
    every runner's prior mean (08 §3.3's "trainer effect as the debut prior mean of a_h");
  - transfer_step (E93): the transfer indicator as a feature.
  Blocks of more than 5,000 parameters (horse × jockey and the horse aptitude's per-horse terms on the store) take
  block-Jacobi steps: the MAP is exact and only their posterior variances are block-diagonal. Against the exact fit
  on synthetic races, τ of a planted per-horse slope came out 0.398 against 0.400, and the mixed test fixture's
  horse-level blocks within 10%. A block strongly coupled to a base entity (trainer × venue against the trainer's main
  effect) is biased toward 0 on that path, so trainer × track (3,799 cells) stays exact.
- **τ's bounds:** the profile of the Laplace evidence over the block's prior scale (other groups at their
  empirical-Bayes values, a low-rank W at its fitted direction), each bound one-sided at 95% (χ²₁), on the last outer
  fold's fit (2026-Q4, the most training rows). For the structured block τ is the prior sd of its term averaged over
  the training rows; for a low-rank block it is U's (W at unit scale), so all τ are on the log-odds scale of u.
- **Folds:** D9's quarters over the level-1 dates; level 1 out of fold from 2015-Q1, each quarter fitted on every
  earlier row to its embargo. Each fold's fit starts from the previous fold's (variances and coefficients): the path,
  not the fixed point.
- **Level 2** (#87's `fit_meta`): u = α · m + β · ln q, unbounded, ridge 0, per member, on each D9 fold's training rows
  that have every member, predicting its test quarter; the market-only meta u = β · ln q on the same rows and folds.
  Scored: every D9 fold, 2020-Q1 to 2026-Q4 (to 2026-11-23).

## Results

### Cell sizes (measured first: prior runs only)

Prior runs in the starter's cell at each out-of-fold start (572,772 starts from 2015-Q1; counts from 2010), by
`cell_sizes`: a cell's runs in races with a strictly earlier post time. Record `cells:<interaction>`.

| Interaction | Cells in the frame | Mean | Median | p90 | Share with 0 | Share with 1-4 | Share with 100+ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| horse × jockey | 428,297 | 1.5 | 0 | 4 | 51.2% | 39.2% | 0% |
| horse × trainer | 98,405 | 8.3 | 5 | 21 | 11.4% | 34.4% | 0% |
| trainer × track (venue) | 3,799 | 411 | 345 | 879 | 0.2% | 0.6% | 82.2% |
| horse × track (venue) | 327,391 | 2.0 | 1 | 5 | 38.7% | 48.0% | 0% |
| horse × course (venue, surface, distance) | 504,223 | 0.9 | 0 | 3 | 60.1% | 35.2% | 0% |
| trainer × debut (debut starts only) | 622 | 242 | 236 | 410 | 0.3% | 0.9% | 85.8% |
| horse | 86,502 | 9.4 | 6 | 23 | 9.9% | 31.3% | 0% |

08 §3.3's sizes hold: horse × jockey cells are 1-10 runs (p90 4), per-course horse cells mostly empty, trainer ×
track cells in the hundreds. Horse × trainer: 75,912 of 86,502 horses (87.8%) kept one trainer, and 78.2% of the
out-of-fold starts are in such a horse's cell; `drop_aliased` drops the block on those horses' rows and keeps it on
every row (some horses move). Record `horse_x_trainer:not_fitted` (status failed: not fitted, aliased).

### τ per block (the last outer fold, 2026-Q4; profile-evidence bounds, each one-sided 95%)

Records `tau:<member>` (each fold's empirical-Bayes τ, and these bounds). n½ = 1/(I τ²) at 08's I = 0.176 and at the
measured 0.106; "at n½" is the share of out-of-fold starts whose cell (structured and low-rank: the row entity) had
at least n½ prior runs.

| Block (member) | τ | Lower | Upper | Levels | τ over the 48 folds (min-max) | n½ at 0.176 | Share at n½ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| horse × jockey (E35) | 0.001 (floor) | 0 | 0.0169 | 424,329 | 0.001 every fold | n/a | 0 |
| trainer × track, variance component (E36) | 0.0267 | 0 | 0.0460 | 3,786 | 0.001-0.033 | 7,945 | 0 |
| trainer × track, one-factor term (E36) | 0.0499 | 0.0417 | 0.0593 | 636 trainers | 0.035-0.056 | 2,281 | 48.0% |
| distance aptitude, structured (E93) | 0.115 | 0.094 | 0.140 | 85,294 horses | 0.003-0.115 | 431 | 0 |
| horse × track, rank one (E93) | 0.001 (floor) | 0 | 0.0157 | 85,294 horses | 0.001 every fold | n/a | 0 |
| trainer debut, variance component (E93) | 0.426 | 0.392 | 0.480 | 620 | 0.408-0.445 | 31 | 94.9% |

The debut indicator's coefficient is −0.53 per fold (median; −0.61 to −0.41) and the transfer step's −0.061 (−0.097 to
−0.010): a debut runner, and a runner after a stable change of trainer, run below what the base gives them. The
structured aptitude's τ grows with the training window (0.003 in 2015 to 0.115 in 2026-Q4). Its rank-one horse × track
term stays at the floor in every fold.

### ΔLL, standalone and at the final price

Paired per-race winner ΔLL over the 23,871 scored races (positive: the block member beats the base), D7's meeting
bootstrap, one-sided 95% bounds. Records `level1:<member>_vs_base` and `marginal:stage2:<member>_vs_base`.

| Member | Standalone ΔLL (lower, upper) | At the final price, stage 2 vs base stage 2 (lower, upper) |
| --- | --- | --- |
| horse_x_jockey (E35) | +0.00000006 (+0.00000006, +0.00000007) | +0.0000000002 (−0.0000000001, +0.0000000005) |
| trainer_x_track (E36) | +0.00037 (+0.00005, +0.00068) | −0.0000102 (−0.0000207, +0.0000002) |
| horse_aptitude (E93) | +0.00103 (+0.00084, +0.00122) | −0.0000059 (−0.0000123, +0.0000005) |
| trainer_debut (E93) | +0.00561 (+0.00455, +0.00669) | −0.0000218 (−0.0000573, +0.0000135) |
| transfer_step (E93) | +0.00058 (+0.00041, +0.00076) | −0.0000034 (−0.0000093, +0.0000025) |

The C4 marginal (each member's stage-2 meta against the market-only stage-2 meta, records
`marginal:stage2:<member>_vs_market_only`): the base +0.000315 (+0.000140, +0.000493); the block members +0.000293 to
+0.000315, every lower bound above +0.00013. Against D4's calibrated market (records `stage2:q:<member>`) every
stage-2 meta is −0.0004 (bounds about −0.00087 and +0.00007) and the market-only meta −0.00071 (−0.00116, −0.00026).
Each member's stage-2 weight α pooled over the scored rows is negative, −0.052 to −0.053 (meeting-clustered
one-sided 95% bounds about −0.074 and −0.032).

### Tail return at the final price

Realised Win return per ¥100 (flat stakes, settled at the recorded Win dividend) of every scored runner, by band of
its stage-2 claim p × O_final, D7's meeting bounds. Records `tail:stage2:<member>` (v2). The stage-2 metas claim at
most 0.98: none ever claims value.

| Forecast | Top 1% (claims from the 99th percentile): return per ¥100 (lower, upper) | 90th-99th percentile | Mean claim, top 1% |
| --- | --- | --- | --- |
| trainer_debut | 83.1 (78.0, 88.4) | 85.6 (82.2, 89.4) | 0.853 |
| trainer_x_track | 80.8 (75.8, 86.0) | 85.0 (81.7, 88.7) | 0.854 |
| base | 80.5 (75.5, 85.6) | 84.9 (81.6, 88.5) | 0.854 |
| horse_x_jockey | 80.5 (75.5, 85.6) | 84.9 (81.6, 88.5) | 0.854 |
| horse_aptitude | 80.5 (75.5, 85.4) | 85.2 (81.9, 88.8) | 0.853 |
| transfer_step | 80.3 (75.3, 85.5) | 84.9 (81.5, 88.4) | 0.854 |
| market only (stage 2) | 76.8 (73.4, 80.3) | 84.2 (80.5, 88.0) | 0.843 |
| calibrated market (reference) | 95.5 (83.6, 107.6) | 81.0 (77.9, 84.1) | 0.868 |

Each top band holds 3,282 bets.

**The settlement's effect (v2 against v1).** Of the scored runners, 77 are settled otherwise than v1 did: 75
co-winners of the scored dead heats, each at its own dividend where that is not its odds over 2, and the two runners
of one race whose Win dividend was paid on a runner the store shows disqualified (no placing): that runner pays and
the store's first-placed runner does not (over the whole price era, 2 such races and 132 runners settled otherwise). None of the 77 is in a
top 1% band, so every top band, and with it every leaderboard row, is unchanged to the last digit; the lower bands
move by at most 0.05 per ¥100 (e.g. the base's bottom half 65.54 to 65.50). Every ΔLL, τ and coefficient is v1's
exactly (they read no dividend).

### Failures

- `horse_x_trainer:not_fitted` (status failed by design): aliased with the horse where the trainer is constant, its
  cell counts in the record.
- `fit:horse_aptitude:2024-04-01@56d0669`, `...2024-07-01@56d0669`, `...2024-10-01@56d0669` (status failed): the
  second launch's fits of horse_aptitude, where empirical Bayes did not converge in 100 updates (that launch kept the
  first fold's error only; the other two records say so). `b5a726c` refitted those member-folds, and the scored
  records use those fits.
- In v1 only: the two launches stopped by hand, each as its 53 planned variants with status failed (at `022d4fc`
  and `56d0669`), and the same three failed fits.
- No other variant failed.

### Variants tried

One registry record each, current in `japan-c2-interactions-final-win-v2` (git `109dce4`), beside the superseded
v1 record of the same variant in `japan-c2-interactions-final-win` (git `b5a726c`, kept):

| Variant | E | v2 record (current) | v1 record (superseded) |
| --- | --- | --- | --- |
| `cells:horse_x_jockey` | E35 | `a301662c85ad1e22` | `eb6336969a60f9d8` |
| `cells:horse_x_trainer` | E93 | `e4da628b1117a435` | `0fd26e9143219591` |
| `cells:trainer_x_track` | E36 | `5a6f36daf76cf45c` | `37b6c7d26203c91c` |
| `cells:horse_x_track` | E93 | `8d8b1d767d78ef67` | `c7900733d33e2d65` |
| `cells:horse_x_course` | E93 | `ff7cea9b60a33e11` | `573374d680df8889` |
| `cells:trainer_x_debut` | E93 | `6d638156fabd252e` | `919e6888146de55e` |
| `cells:horse`, `cells:jockey`, `cells:trainer` | E93 | `f2f2b7015c454442`, `ae59ae956ebe4475`, `eff05b22ba2ffccc` | `54b34a5747e22bdf`, `c6d9cbe569c63e27`, `1fc2888ffd8107a5` |
| `horse_x_trainer:not_fitted` (failed: aliased) | E93 | `f8411dd3e0b17e20` | `521410aff97003a7` |
| `selection:rating_pair` | E93 | `fd18aceda5a40513` | `657e198cd31047b4` |
| `fit:horse_aptitude:2024-04-01@56d0669`, `...2024-07-01@...`, `...2024-10-01@...` (failed) | E93 | `2d674d88af256a6c`, `1ae9c4b21758ef0d`, `909a42125db819cd` | backfilled (below) |
| `tau:horse_x_jockey` | E35 | `10cea3fb2518290c` | `a1321e8f74a2980d` |
| `tau:trainer_x_track` | E36 | `27ad552408174b26` | `46544e70616e81e7` |
| `tau:horse_aptitude` | E93 | `973af3f675e83cf5` | `4822bc2f4ad7244f` |
| `tau:trainer_debut` | E93 | `c99f5d34214c1399` | `0ede06c422de9be4` |
| `tau:transfer_step` (its coefficient) | E93 | `4f0b5aba3e874c15` | `873d50a9395d9a33` |
| `stage2:q:base` and `meeting_clustered:` | E93 | `93a7a41105de05a0`, `e8e03d5b80b656ee` | `ec88c1b345c82b80`, `6a3b84d838d52d56` |
| `stage2:q:horse_x_jockey` and `meeting_clustered:` | E35 | `a8eadd0c363fdc46`, `913cb164dff5ea54` | `55dd161a6e5ef7f2`, `4fcf82d97115df1a` |
| `stage2:q:trainer_x_track` and `meeting_clustered:` | E36 | `b9e603d178f4983a`, `2b22a1d74dab5662` | `fe10041dd9db7cf9`, `2cc2172a56dbf96d` |
| `stage2:q:horse_aptitude` and `meeting_clustered:` | E93 | `99e91e8838e5476d`, `5d8da9b9094a8a73` | `fe70eaad34da8d4e`, `1d5c73874b30998e` |
| `stage2:q:trainer_debut` and `meeting_clustered:` | E93 | `f4a8f2a4239ed991`, `63ce4bb015f7b81f` | `3ef4cb587c124863`, `40672fb2f88d01c7` |
| `stage2:q:transfer_step` and `meeting_clustered:` | E93 | `ce19c31ca709163d`, `5c28fb1df9e8d548` | `3e78f3a85f8071a3`, `57e1f4d0c2e2c24e` |
| `stage2:q:market_only` | E93 | `8bdee811267e1c7a` | `7224b54cab4cddd9` |
| `marginal:stage2:base_vs_market_only` | E93 | `77994772b78ffa17` | `5cec59468c7e9d4d` |
| `level1:horse_x_jockey_vs_base`, `marginal:stage2:horse_x_jockey_vs_base`, `..._vs_market_only` | E35 | `5f6dd4d375645d52`, `410a87ea914285c6`, `2b016a3f68d3f84b` | `5cb911707b70e922`, `aefa6c025361d702`, `1309c0460e5f0513` |
| `level1:trainer_x_track_vs_base`, `marginal:stage2:trainer_x_track_vs_base`, `..._vs_market_only` | E36 | `f3e34be4db9ce22a`, `975c09bcfe6470e9`, `e809c254a1601ed7` | `5cc2a0ba4fd61462`, `4b4fa15701124e14`, `55092857c92e1e70` |
| `level1:horse_aptitude_vs_base`, `marginal:stage2:horse_aptitude_vs_base`, `..._vs_market_only` | E93 | `c8046867f97e4f00`, `04641159927051e5`, `17f22d96c6123ea5` | `cbff36abbe426ee2`, `caaed8de68e07d1c`, `500e8fbbb0551784` |
| `level1:trainer_debut_vs_base`, `marginal:stage2:trainer_debut_vs_base`, `..._vs_market_only` | E93 | `7d03f809d136285d`, `511f69b341718cce`, `45fecea076de165a` | `a5771fada4e33491`, `8194a36e24d2e053`, `1e824a8b9b01450d` |
| `level1:transfer_step_vs_base`, `marginal:stage2:transfer_step_vs_base`, `..._vs_market_only` | E93 | `b26fdc2109ecef91`, `bbaf367e8aafee80`, `aedbfaa98a86d325` | `7ca7c130e7d1e6c7`, `848bb7f26da5e19f`, `bc18f5d74732df5f` |
| `tail:stage2:base`, `horse_x_jockey`, `trainer_x_track` | E93, E35, E36 | `9f33ce46d4b7805f`, `f710e419f75b522f`, `ebedf4829efe5be2` | `f2c0d0d04267dc23`, `67df43ee6976648b`, `7ec8e7ad5189326e` |
| `tail:stage2:horse_aptitude`, `trainer_debut`, `transfer_step` | E93 | `74d40e3e5b92ec25`, `914065add8ab4fdb`, `5e2115c418c36791` | `a73c1e6dbdaf9648`, `58ccea1005e84998`, `49711e35d2fd5fda` |
| `tail:stage2:market_only`, `tail:calibrated_market` | E93 | `195c66b2b5ccd503`, `8f23cf390afba95d` | `5d38c44fb47f1daf`, `a68064f9a1ddf84c` |

v1 also holds the 109 records `record_v1_launches` appended (`params.backfilled`): the 53 planned variants of each
stopped launch with status failed (git `022d4fc` and `56d0669`, the error naming the stop), and the three failed
horse_aptitude fits (git `56d0669`).

## Alternatives rejected and why

- **#92's base with the sire:** the sire is the JRA-VAN pedigree's, which would withhold every figure here (D21). The
  sire's terms (and 08 §3.3's sire × surface × distance) are #95's.
- **#92's per-fold discount selection:** its selected values are E33's and withheld with the sire; fixed discounts
  (1, 0.5, 0.25) keep the base and every block member on one likelihood, which is what the paired comparisons need.
- **Horse × trainer as a fitted block:** aliased with the horse where the horse keeps one trainer (08 §3.3), and the
  base has no static horse term to tell it from; the trainer debut prior and the transfer step stand in, and the
  block is recorded as not fitted with its cell counts.
- **Per-course horse cells (horse × venue × surface × distance as a variance component):** cells of a few runs
  (the cell table), which 08 §3.3 says fail; the structured form and the rank-one horse × track term replace them.
- **A full Bayesian fit (NumPyro):** not used, as in #92; the penalised MAP with empirical-Bayes variances is
  deterministic, needs no new dependency and fits the store in memory.
- **Dense exact fits for every block:** the per-horse blocks have 86,000 to 430,000 parameters on the store; their
  P × P information does not fit in memory, so they take block-Jacobi steps (exact MAP, block-diagonal variances).
- **Per-fold τ intervals:** the profile costs tens of MAP solves per block; it is computed once, on the last fold.
- **One joint member with every block:** not pre-registered here (each block is measured against the same base);
  proposed below for #98's meld.
- **Place and Wide:** outside this ticket, which scores Win at the final price; proposed below.
- **Refitting every fold at one revision for v2:** about 4 h of level-1 fits on this machine to replace fits from
  `56d0669` whose code differs from `b5a726c` only in the fixed point's step and stopping rule; v2 instead re-scores
  the saved fits and names each member-fold's revision in every record (a mixed run, disclosed above).
- **A checkpoint keyed by the fitting code (invalidated on any code change):** it would have discarded v1's fits for
  a change that touches only level 2 (the settlement); the key covers the spec's level1, tau and folds sections and
  the store build, and the revision of each fit is recorded instead.

## Verdict table: 08 §3.3 and 05 §4.3, claim by claim

Each in-scope claim as the ticket states it (08 §3.3, 05 §4.3 and 08 §8 B5, quoted from #93; the 05 and 08 reports
are local and partly JRDB-derived, so none of their figures is copied here), against this study's evidence. Every
figure is v2's, from the records named; the ΔLLs are nats per race over the 23,871 scored races, one-sided 95% meeting
bounds in brackets, and "at the final price" is the block member's stage-2 meta against the base's. A ΔLL is the
member's as scored, with every term it fits; no member fits one of its terms without the others, so a τ interval
describes that term's fit, never its share of the member's gain.

| Claim (as #93 states it) | Evidence here (records) | Conclusion |
| --- | --- | --- |
| Horse × jockey cells hold 1–10 runs, so only τ_hj is estimable | Prior runs per out-of-fold start: mean 1.5, median 0, p90 4; 51% of starts are the pair's first, none in a cell of 100+ (`cells:horse_x_jockey`). τ_hj at the empirical-Bayes floor in all 48 folds, upper bound 0.017 on the last (`tau:horse_x_jockey`) | Holds: most cells hold 0 to 4 prior runs, so only the spread τ_hj is estimated; it is at the floor, and a τ_hj above 0.017 is ruled out. |
| ... and τ_hj is never a bet trigger | Standalone +0.00000006 (+0.00000006, +0.00000007); at the final price +0.0000000002 (−0.0000000001, +0.0000000005) (`level1:horse_x_jockey_vs_base`, `marginal:stage2:horse_x_jockey_vs_base`) | Consistent: nothing to trigger on. The estimate is read as predictive only (booking endogeneity, 05 §4.2); no rule here bets on it. |
| Horse × trainer is aliased with horse | 75,912 of 86,502 horses (87.8%) keep one trainer, and 78.2% of out-of-fold starts are such a horse's; `drop_aliased` drops the block on those horses' rows (`horse_x_trainer:not_fitted`, `cells:horse_x_trainer`) | Holds where the trainer is constant: the block is not fitted, and its record (status failed, not fitted) carries the cell counts. |
| Horse × course cells fail, hence the structured form | Horse × course (venue, surface, distance) cells: mean 0.9 prior runs, 60.1% of starts a cell's first (`cells:horse_x_course`). The horse_aptitude member's terms: the structured distance and surface aptitude, τ 0.115 (0.094, 0.140), and the rank-one horse × track term, at the floor (upper 0.016). The member standalone +0.00103 (+0.00084, +0.00122); at the final price −0.0000059 (−0.0000123, +0.0000005) (`tau:horse_aptitude`, `level1:horse_aptitude_vs_base`, `marginal:stage2:horse_aptitude_vs_base`) | Holds for the cells: per-course cells are mostly empty. The horse_aptitude member, the structured form and the rank-one horse × track term fitted together, gains standalone with its lower bound above 0; of its two terms only the structured one is off τ's floor, but the gain is the member's. It adds nothing at the final price. |
| Trainer × track is learnable if τ ≥ 0.05 | One-factor east/west term τ 0.0499 (0.0417, 0.0593); cell-by-cell variance component 0.0267 (0, 0.0460) (`tau:trainer_x_track`). The member, both terms together, standalone +0.00037 (+0.00005, +0.00068); at the final price −0.0000102 (−0.0000207, +0.0000002) (`level1:trainer_x_track_vs_base`, `marginal:stage2:trainer_x_track_vs_base`) | τ ≥ 0.05 is not shown for either term: the frozen rule reads learnability from τ's lower bound, and the one-factor term's 0.042 sits below 0.05 (a τ below 0.042 is ruled out for it); the cell-by-cell component's interval reaches 0, and a τ above 0.046 is ruled out for it. The trainer_x_track member, both terms fitted together, gains standalone; which term produced the gain is not measured. Priced at the final price. |
| Drop horse × trainer in favour of a trainer debut prior (08 §8 B5) | τ 0.426 (0.392, 0.480) over 620 trainer × debut cells; the debut indicator −0.53 per fold (median); standalone +0.00561 (+0.00455, +0.00669); at the final price −0.0000218 (−0.0000573, +0.0000135) (`tau:trainer_debut`, `level1:trainer_debut_vs_base`, `marginal:stage2:trainer_debut_vs_base`) | The trainer_debut member, the debut indicator and the trainer × debut component fitted together, has the largest standalone gain of the members, its lower bound above 0; the two terms' shares are not separated. It adds nothing at the final price. |
| ... and a transfer step, applied only after a stable change (08 §8 B5) | 67,331 starts after a stable change of trainer (the new trainer kept for 2 consecutive runs); coefficient −0.061 per fold (median; −0.097 to −0.010); standalone +0.00058 (+0.00041, +0.00076); at the final price −0.0000034 (−0.0000093, +0.0000025) (`tau:transfer_step`, `level1:transfer_step_vs_base`, `marginal:stage2:transfer_step_vs_base`) | The transfer_step member, its one indicator, gains standalone with its lower bound above 0; priced at the final price. |
| Expected gain is about 0.42·τ²·w nats per race, before asking whether the market has it | Both sides are nats per race (each ΔLL here is a mean over the scored races), so w is a pure number: the w that reconciles the rule with a member's standalone ΔLL is ΔLL / (0.42 · Στ²) over the member's τ terms, one w for both terms of a two-term member; in brackets, the span of the printed bounds (ΔLL's lower over τ's upper, its upper over τ's lower; not a confidence interval). horse_x_jockey: τ at the floor, so any w from 0.0005 (at τ's upper bound, 0.017) reconciles +0.00000006. trainer_x_track: 0.27 (0.021, 0.93); 0.35 were the whole gain the one-factor term's, 1.2 were it the cell-by-cell component's. horse_aptitude: 0.19 (0.10, 0.33), its rank-one term at the floor adding almost nothing to Στ². trainer_debut: 0.074 (0.047, 0.10) were the whole gain the variance component's; less by the debut indicator's share, which is not separated. transfer_step: no τ (one coefficient), so the rule does not apply. Arithmetic on the rows above; no new read | Indeterminate. 05 §4.3 defines w per cell, w(n) = nIτ²/(1 + nIτ²), the posterior weight on a cell of n runs, states the gain for a race whose every runner sits in such a cell, and qualifies the rule as "second order, independent offsets, the share of runners covered". How w averages over cells of different sizes, and which I it takes (08's 0.176 for top-3 training, or the base's measured 0.106), is not stated in 05, 08 or #93, and this report publishes no runner-level w(n), so the rule's predicted gain is not computed here and the reconciling w is not checked against one. No measurement contradicts the rule: every member's span reaches below w(n)'s ceiling of 1. The market's side is determined: at the final price every member adds at most +0.000014 per race, whatever w is. |

## What this rules out

Measurements, not verdicts; nothing is adopted or rejected. One run on one store build (single-run in the ledger).

- **E35 (horse × jockey):** τ_hj sits at the floor in every fold, with an upper bound of 0.017 on the last: a horse ×
  jockey spread above 0.017 on the log-odds scale is ruled out (one-sided 95%) beside the base. Its cells add 6e-8
  nats per race standalone and nothing at the final price. E35's "τ_hj < 0.1, cells useless" holds. Read as
  predictive only (booking endogeneity); never a bet trigger.
- **E36 (trainer × track):** the member fits two terms together. Their fits: the one-factor east/west term's τ is
  0.050 (0.042, 0.059), steady over the folds, and the cell-by-cell variance component's 0.027 (0, 0.046). The rule
  reads "learnable" from the lower bound against 0.05, and 0.042 sits below it, so τ ≥ 0.05 is not shown; τ below
  0.042 is ruled out for the one-factor term, and τ above 0.046 for the cell-by-cell component. Standalone the member
  gains +0.00037 nats per race (lower bound +0.00005); no member fits one term without the other, so which term
  produced the gain is not measured. At the final price the member gains nothing (−0.00001, upper bound +0.0000002):
  "probably priced" holds.
- **E93 (aptitude, debut prior, transfer step):** each member gains standalone with its lower bound above 0:
  trainer_debut most (+0.0056 nats per race; the debut indicator, −0.53, and the trainer × debut component, τ 0.43:
  the trainer's debut runners differ from its later ones by a spread of 0.4 on the log-odds scale), then
  horse_aptitude (+0.0010; the structured distance and surface term, τ 0.115, and the rank-one horse × track term,
  at the floor with upper 0.016) and transfer_step (+0.0006; its one coefficient, −0.06 after a stable change). Each
  gain is the member's, all its terms fitted together; a τ describes its term's fit, not its share of the gain.
- **At the final price nothing adds:** every block member's paired ΔLL against the base's stage-2 meta has its upper
  bound at or below +0.000014 nats per race, and four of the five point estimates are negative. A gain above
  +0.000014 per race from any of these members, given the base and the final price, is ruled out (one-sided 95%). The
  final Win price already holds what the members measure.
- **The base itself** (#92's model without the sire) adds to the market-only stage-2 meta: +0.000315 nats per race
  (+0.000140, +0.000493), with a negative stage-2 weight (α ≈ −0.053), as #91's rating had: the market overweights what
  the ability member measures. Neither stage-2 meta beats D4's calibrated market (upper bounds near +0.00007).
- **Tail:** no stage-2 meta ever claims value (p × O below 1 for every runner), and the base's and every block
  member's top band returns 80 to 83 per ¥100 at the recorded dividends, below the Win break-even of 100 with every
  upper bound under 89. The block members' top bands are indistinguishable from the base's 80.5; the market-only
  meta's is 76.8 (upper 80.3).

## Leaderboard rows

Option ids (spec `leaderboard`), the top claim band of each stage-2 meta at the final Win price, raw (the
leaderboard's shrinkage within the pool is the planned `core.eval.leaderboard`'s; this family searched 6 members and
the market-only meta), from v2's `tail:stage2:*` records, which supersede v1's (equal here: no top-band runner's
settlement changed). Ordered by realised return per ¥100 at the recorded dividends:

| Option id | Return per ¥100, top 1% (lower, upper) | Bets |
| --- | --- | --- |
| `japan:WIN:final:c2-interactions:trainer_debut:stage2:claim-top-1pct` | 83.1 (78.0, 88.4) | 3,282 |
| `japan:WIN:final:c2-interactions:trainer_x_track:stage2:claim-top-1pct` | 80.8 (75.8, 86.0) | 3,282 |
| `japan:WIN:final:c2-interactions:base:stage2:claim-top-1pct` | 80.5 (75.5, 85.6) | 3,282 |
| `japan:WIN:final:c2-interactions:horse_x_jockey:stage2:claim-top-1pct` | 80.5 (75.5, 85.6) | 3,282 |
| `japan:WIN:final:c2-interactions:horse_aptitude:stage2:claim-top-1pct` | 80.5 (75.5, 85.4) | 3,282 |
| `japan:WIN:final:c2-interactions:transfer_step:stage2:claim-top-1pct` | 80.3 (75.3, 85.5) | 3,282 |
| `japan:WIN:final:c2-interactions:market_only:stage2:claim-top-1pct` | 76.8 (73.4, 80.3) | 3,282 |

Horse × trainer adds no row (not fitted). The calibrated market's top band (95.5, 83.6 to 107.6) is a reference, not
an option of this study.

## Follow-ups filed

Filed under epic #73 as `status:proposed`:

- #330: let `write_summary` split a family file over the public-tree cap, so v1's and v2's summaries can be
  committed (both are over 256 KB).
- #570: tests for `models.interactions`' local (block-Jacobi) path and its fixed-point step: a horse-level block
  fitted with DENSE_MAX forced low agrees with the dense fit, and a weakly identified variance converges within
  EB_MAX.
- #571: one C2 member with every interaction block together, exported for #98's meld.
- #572: 08 §3.3's pedigree prior on the aptitude's (b_h, e_h), with #95's lineage (after #95).
- #573: the trainer debut prior on debut and maiden races at the final price, by class.
- #574: Place and Wide: the block members through C5's ordering against their own final prices.

## #571: the all-blocks member (family v3; pre-registered, results pending the local run)

Ticket #571 (follow-up of #93, for #98's meld). One C2 member, `all_blocks`, carries every block and extra feature of
#93's members fitted together, so the meta-model can weigh C2 as a whole instead of six members that each repeat the
base. **No result is reported here yet:** the real run reads the local store and is run on the owner's machine
(`uv run horsetrack japan interaction-study`); this section is the pre-registration and what that run will produce.
Nothing is adopted (D16, D19): the member is a ranked member row.

- **Family and spec.** Registry family `japan-c2-interactions-final-win-v3` supersedes v2. Its spec
  (`studies.interactions.build_spec`) is v2's (`v2_spec`, which rebuilds v2's frozen spec byte for byte, sha256
  `2dde55836222c16552431d0037f0aabf4fbc55e814e79d139d5d7aa90e24d2e0`) with `family_id`, `ticket` (571), `supersedes`,
  the hypothesis E571, the member's blocks and extra features, its `level1.all_blocks` description and its leaderboard
  option added; every other section (the base, the six members, level 2, the tail, the folds, D7's bootstrap) is v2's.
  At this commit it hashes to `e91949f24e8d38e3cc6ec4374ec64978167084236807e616f58b2831a2d97ceb`; the launch freezes
  it with `freeze_family` before any outcome is read, and v2's frozen spec and its 56 records are not touched (no
  record is appended to v2).
- **Hypothesis (E571, frozen).** "Every #93 block fitted together in one member (horse × jockey; trainer × track with
  its one-factor east/west term; the structured aptitude with its rank-one horse × track term; the trainer debut prior
  with the debut indicator; the transfer step): measured standalone against the base and at the final price against
  the base's and the market-only stage-2 meta, with its tail." **Null**, the starting point: no gain over the base
  standalone or at the final price, and a tail at the takeout. v2's single-block gains are each member's alone; whether
  they add when fitted together is what the member measures, not something assumed.
- **Method.** The base (`AbilityModel(k=3)` on #91's rating, jockey and trainer, plus the debut indicator and the
  transfer step) with the six blocks `horse_x_jockey`, `trainer_x_track`, `trainer_x_track_factor`,
  `distance_aptitude`, `horse_x_track` and `trainer_debut`, one `fit_with_blocks` fit per outer fold with every τ² by
  empirical Bayes together, on the same D9 quarters, rating pairs and out-of-fold rows as #93's members. Each block
  keeps its own exact or block-Jacobi path (the choice is per term, by its size). The fixed point and MAP start from
  the member's own previous fold, the first fold from 0. τ's bounds are on the last outer fold's fit, keyed by member
  and block (the member shares its blocks' names with #93's members).
- **Level-1 reuse.** A v3 launch without a checkpoint of its own copies v2's
  (`interactions-japan-level1-e2cb2f08123f.npz`, never changed) to v3's key and resumes it: v2's six members keep
  their out-of-fold u, τ, fitting revisions (`56d0669`, `b5a726c`) and three failed horse_aptitude fits, and only
  the all-blocks member is fitted, on every fold, at the launch's revision. The run is mixed again; every record
  names its member-folds' revisions (`params.fit_revisions`), and the failed fits are v3 records too.
- **What the run will produce.** In v3: the member's τ per block and coefficient per feature per fold, with the last
  fold's bounds (`tau:all_blocks`); its stage-2 meta and its meeting-clustered bounds (`stage2:q:all_blocks`,
  `meeting_clustered:stage2:q:all_blocks`); its standalone paired ΔLL against the base (`level1:all_blocks_vs_base`); its
  ΔLL at the final price against the base's stage-2 meta (`marginal:stage2:all_blocks_vs_base`) and against the
  market-only meta (`marginal:stage2:all_blocks_vs_market_only`), each with D7's meeting bounds; and its tail at the
  final price by claim band (`tail:stage2:all_blocks`), whose top band is the leaderboard row
  `japan:WIN:final:c2-interactions:all_blocks:stage2:claim-top-1pct`. v2's six members and the market-only meta are
  re-scored beside it in v3 (their level 2 unchanged). The export #98's meld reads gains the column `u_all_blocks`,
  per level-1 out-of-fold runner from 2015-Q1, beside each member's u and its `oof_fold`.
- **Results:** pending the local run; the row and its ΔLLs with their bounds are added here by that run's PR, from
  v3's records.
