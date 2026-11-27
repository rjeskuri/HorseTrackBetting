# The T-2 cross-pool signal study, E25 (ticket #435, N4)

Status: **PRE-REGISTERED.** Claude pre-registers this spec, with no owner sign-off, on the owner's
decision in the comment on #435 headed "Reframe, owner 2026-11-27" (posted 2026-11-27). That
comment replaces Outline 2, "Stop with needs-input for the owner's sign-off, then freeze the
family.", with "Freeze the family in the registry. No sign-off stop."; replaces the definition of
done's "the owner's sign-off of the spec is on this ticket, and the freeze comes before every
scored record" with "the freeze comes before every scored record."; and replaces the rule "a spec
the owner approves before the run: a needs-input step" with "Claude pre-registers the spec and
freezes it before the run. No owner sign-off." The same comment makes both offsets registered
variants, each ranked, and puts every settled selection and N5 band, ranked, into the
candidate-cell file. `build_spec`'s `decisions` carries the same citation. The freeze is the
gate; there is no code gate on a sign-off.

Family id: **`japan-cross-pool-t2-v1`**. `uv run python -m
horsetrack.countries.japan.studies.cross_pool_run --freeze --from DAY --to DAY --seed N` freezes
the spec that `cross_pool_run.build_spec(date_from=..., date_to=..., seed=...)` builds, with
`Registry.freeze_family` in `$HORSETRACK_DATA_DIR/registry/japan.jsonl` (append-only: a freeze
cannot be undone). `--to` is required by every mode (`--freeze`, `--run`, `--trial`), so the run
takes the frozen span end and never a default of "today". The freeze comes before any scored
read: `run_cross_pool` raises `FamilyNotFrozen`, before any store file is opened, unless the
registry's frozen hash for the family equals the hash of the spec for the same `--from`, `--to`
and `--seed`; when another spec is frozen, the refusal names that spec's `--from`, `--to` and
`--seed`. A change to anything below is a new family, frozen again before it is scored. This page
holds no figure from the JRA-VAN data (D21): every count, rate, span, sign and bound of the run
lives in the private report under `$HORSETRACK_DATA_DIR/reports/`.

## Trial runs

A trial (`--trial`, `run_cross_pool(..., trial=True)`) runs the same code with no freeze, on
`cross_pool_run.TRIAL_SCHEME` (D9's shape, quarters and a 7-day embargo, with 4 training
quarters, so a few years of days score), and appends one record per variant, failures included,
to the trial family **`japan-cross-pool-t2-trial`** (params `trial`, the span and the trial fold
scheme in each record). A trial never freezes or writes `japan-cross-pool-t2-v1`; its private
report and cell file carry "trial" in their names, so they never overwrite the family's, and the
CLI writes no public page for it.

Real-store smoke and trial runs before the freeze checked only that the pipeline completes and
its memory; their records are in the trial family. No output figure of theirs was read to shape
this spec: the changes after them (the pair variant, per-pool settlement, the retrospective
label, the `top1` tie rule, the required `--to`, pool availability as a metric, the pair model's
complete-outcome rule, failed records from the store's discovery on) came from the review
findings on PR #493.

## Hypothesis and null

- **E25**: at the decision time t, the Quinella pool's support against the Win-implied
  (Harville) support adds to the final price: β ≠ 0 in u = offset + β · x, and the walk-forward
  winner log-loss gain over the offset, ΔLL, is positive. Per runner, `x_cross` (the top-2
  support) over the final Win price; per pair, `log_ratio` (the pair price at t against the
  Win-implied pair) over the final Quinella price.
- **Null**: β = 0 and ΔLL = 0. The final price already holds the cross-pool signal. The null is
  the starting point: a gain whose D7 one-sided 95% lower bound is at or below 0 does not reject
  it.
- Target: the race's winner (Win pool) at the final Win odds; for the pair variant, the pair(s)
  the final QUINELLA dividends pay, at the final Quinella odds.
- Adoption: none. This is a measurement; nothing is staked or adopted on it.

## Decision time and features

- t = post - 2 minutes: D2's primary (`config/decision.yaml` `primary_min`, read as
  `cross_pool_run.MINUTES_BEFORE`).
- The boards at t are #434's `studies.cross_pool.boards_at`: per (race, pool, combination) the
  latest row of the JRA-VAN time series (`processed/jravan/odds_timeseries/<day>.parquet`, 0B41
  Win and 0B42 Quinella) with `minutes_to_post` at or above 2. No row after t is read: the day
  file is read with that filter.
- `x_cross` = ln(q_top2_t / h_top2_t): the runner's Quinella top-2 support at t (the sum of the
  normalised Quinella shares of the pairs holding it) over its plain-Harville top-2 support from
  the Win shares at t (λ2 = 1), from #434's `cross_features`.
- `fav_x_cross` = is_fav · x_cross, is_fav the Win favourite at t (the lowest horse number on a
  tie): the favourite's support across pools.
- `log_ratio` = ln(q_pair_t / h_pair_t) per T-2 starter pair (a pair of the Win starters at t):
  its normalised Quinella share at t over its plain-Harville probability from the Win shares at t
  (λ2 = 1), from #434's `pair_features`. Summing pairs into runners (as `x_cross` does) can hide
  a move between pairs that leaves every runner's support unchanged; the pair variant reads the
  pairs themselves.
- A zero-vote Quinella quote reads as unavailable (#434's default, `zero_vote="unavailable"`).

## Variants and family size

Two feature sets times two offsets give four Win variants, and the pair model is the fifth; the
family size is **5**, one registry record each (failures included):

| Variant | Features | Offset | Choice set |
| --- | --- | --- | --- |
| `x_cross:normalised` | x_cross | normalised | the race's final Win starters |
| `x_cross:calibrated` | x_cross | calibrated | the race's final Win starters |
| `x_cross+fav:normalised` | x_cross, fav_x_cross | normalised | the race's final Win starters |
| `x_cross+fav:calibrated` | x_cross, fav_x_cross | calibrated | the race's final Win starters |
| `pair:quinella` | log_ratio | quinella | the race's T-2 starter pairs |

- `normalised`: ln q, q = (1/O) / Σ (1/O) over the race's priced starters, O the final Win odds
  (the jravan `final_odds`, pool WIN).
- `calibrated`: ln of D4's calibrated market, `models.v0.calibration.fit_win_calibration` (a
  natural cubic spline in ln q, 5 knots), fitted on each fold's training rows and applied to its
  training and test rows (out of fold on the test rows, as #114 and #87 used it). Both offsets are
  scored on the same folds, from the first scored quarter.
- `quinella`: ln q_pair, q_pair = (1/O) / Σ (1/O) over the race's T-2 starter pairs, O the final
  Quinella odds (`final_odds`, pool QUINELLA).
- The fit is the conditional logit u = offset + X β within the race, unpenalised maximum
  likelihood (`models.condlogit.fit_conditional_logit`, through `cross_pool_run.fit_fold`): a dead
  heat's winners at 1/k, a race with no winner dropped, the standard error the meeting-clustered
  sandwich, a 95% two-sided Wald interval.

## The pair model (`pair:quinella`)

- Choice set: the T-2 starter pairs of each race whose Quinella board at t prices every one
  (every `log_ratio` finite).
- Outcome: the pair(s) the final QUINELLA dividends (`dividends`) pay; a dead heat's paid pairs
  are co-winners at 1/k, averaged in the log loss as the Win variants' dead heats are. A race is
  kept only when every paid pair is in its choice set: a dead heat paying `1,2` and `1,3` whose
  horse 3 has no Win quote at t is dropped (`pair_excluded_incomplete_outcome`), never fitted with
  `1,2` as its sole winner; a race none of whose paid pairs is in the choice set is dropped under
  `pair_excluded_no_paid_starter_pair`.
- Offset `quinella`, feature `log_ratio` (λ2 = 1, zero votes unavailable), fitted per D9 fold on
  the earlier folds' rows only (`walk_forward` over the pair rows' race dates, the same scheme),
  scored by the pair log-loss gain per race (`cross_pool_run.score_pair_folds`), D7 bounds overall
  and by class group, the pooled coefficient with its meeting-clustered interval.
- Settled in QUINELLA only, claim = p_pair × the final Quinella odds, with its offset alone,
  `offset:quinella`.

## Folds, span and the holdout

- D9: `core.eval.folds.walk_forward` over the study rows' race dates with `config/folds.yaml`
  (calendar quarters, expanding, `min_train` 22 quarters, a 7-day embargo counted back from the
  fold's first scored day). Each fold's β and spline are fitted on that fold's training rows only,
  and its test races are scored out of fold. The fold scheme is frozen in the spec.
- Span: `--from` (default 2014-06-07) to `--to` (required), both frozen. D10: nothing before
  2014-06-07 enters; a span starting earlier is refused before any read.
- D14: development days only. A holdout day's file (`core.eval.holdout.is_holdout_day("japan",
  d)`) is dropped by its name before it is opened or stat'd; the jravan tables are read through
  `models.v0.store.read_store_table`, which drops holdout days, or filtered to the development
  race ids, with a kept holdout day refused. `LockedHoldout` is never opened.

## Exclusions (each counted in the private report)

- a race with no row in the jravan `races` table;
- a race with fewer than 2 priced final Win starters;
- a race with no T-2 Quinella board;
- a race in which a priced final starter has no T-2 row or a missing feature (a Quinella board at
  t that misses a starter pair gives no feature, #434);
- a race with no winner (void);
- a runner without a final Win price, or marked scratched in the results, is dropped and the race
  renormalised;
- an unmapped race class counts in `all` only;
- a race whose pool has no dividend row (the pool not formed, or a special payout) is left out of
  that pool's settlement;
- for the pair variant: a race with fewer than 2 T-2 starter pairs; a race whose Quinella board
  at t misses a starter pair (`pair_excluded_incomplete_t2_quinella_board`); a race with no row
  in `races`; a race with no QUINELLA dividend row (`pair_excluded_no_quinella_dividend`); a race
  in which a starter pair has no final Quinella price (`pair_excluded_unpriced_final_pair`); a
  race none of whose starter pairs is paid (`pair_excluded_no_paid_starter_pair`); a race with a
  paid pair outside its choice set while another is in it (`pair_excluded_incomplete_outcome`).

## Metric and bounds

- ΔLL per scored race = ll_offset - ll_model, winner log loss, a dead heat's winners averaged
  (D6); positive means the model helps.
- D7: the meeting (venue × day) bootstrap, `config/scoring.yaml` (`n_boot`, level 0.95,
  one-sided), seed `--seed` (default 435), lower and upper one-sided bounds; overall and by class
  group (newcomer, maiden, conditions, open, graded; `research.pool_scan.CLASS_GROUPS`).
- Coefficients per fold, and pooled over the scored rows with their out-of-fold offset, each with
  the meeting-clustered sandwich standard error and its 95% interval.

## Settlement at the final dividend

Settlement, the N5 curves and the candidate cells are a **retrospective diagnostic at the closing
(final) prices, not decision-time (T-2) rules**: each claim uses the final price twice, in the
offset and as the odds, so a late price move can put a combination in or out of a selection after
t, and no selection is a bet that could be placed at t. The conditional-logit test keeps the
final-price offset; the cells say so in their `study` value.

- Scorers: the four Win variants and their offsets alone, and the pair variant and its offset
  alone. The offset baselines are named **`offset:normalised`** and **`offset:calibrated`** (under
  the Win variants) and **`offset:quinella`** (under `pair:quinella`): each offset alone, β = 0.
  They are settled, in N5 and in the cells, and have no registry record of their own: their
  returns are recorded as `offset_return[<pool>:<selection>]` metrics on the records of the
  variants with the same offset (a variant's own returns are `return[<pool>:<selection>]`).
- Pools: the Win variants and their offsets in WIN, QUINELLA and FORECAST (the store's name for
  Exacta); the pair variant and `offset:quinella` in QUINELLA only.
- Probabilities from the Win scorer's Win probabilities by Harville (λ2 = 1): Win p_a; Quinella
  P({a, b}) = p_a p_b / (1 - p_a) + p_b p_a / (1 - p_b); Exacta P(a, b) = p_a p_b / (1 - p_a).
  The pair scorers price their own pairs.
- Claim = p × the pool's final odds (`final_odds`), over every priced combination both of whose
  legs are scored starters.
- Selections: `value`, every priced combination with claim > 1; `top1`, per race the combination
  with the largest claim. Claims within a relative 1e-9 of the race's largest tie, and a tie goes
  to the lowest final odds, then the lowest combination text. Under an offset alone every Win
  claim is mathematically 1 / Σ (1/O), so the `offset:normalised` Win `top1` is the final
  favourite (and the `offset:quinella` top1 the final Quinella favourite), never a pick by
  floating-point noise.
- Return per unit stake: dividend / 100 at the final dividend (`dividends`) when the combination
  paid, else 0; a dead heat's paid combinations all pay. Never the board's odds
  (`cross_pool_run.settle`).
- Bounds: the D7 meeting bootstrap, one-sided 95%, overall and by class group;
  se_mean_return = (upper - lower) / (2 z_level).
- A pool with nothing to settle (no priced combination in the scored races, as when a store
  lacks FORECAST final odds), or whose settlement raises, is unavailable for that scorer: counted,
  named in the result (`unavailable_pools`), the private report and the records' metrics
  (`settlement_unavailable[<scorer>/<pool>]`, value 1, n 0, on the variant's record, and for an
  offset alone on the records of the variants with that offset). It is never a param: params are
  part of the variant's identity (`Registry.family_size`), so a rerun of the frozen spec with the
  pool restored is another run of the same five variants. The fits, gains, coefficients and the
  other pools are kept, and the variant's record stays `scored`.

## N5

`core.eval.tail.tail_curve` per (scorer, pool) over every priced combination: claim the scorer's
claim, return the settled return, cluster the meeting, quantiles (0.5, 0.75, 0.9, 0.95, 0.99),
`by="class_group"`, D7's `n_boot` and level, seed `--seed`.

## Candidate cells

`$HORSETRACK_DATA_DIR/reports/candidate-cells/cross-pool.json` (through `write_report`), #430's
schema, one object per cell: `study, family_id, cell_id, pool, class_group, odds_band,
payout_rate, n_bets, n_race_days, n_meetings, mean_return, se_mean_return, shrunk_edge,
mean_odds`.

- `study` is `cross-pool/final-price-diagnostic`: the cell is a retrospective diagnostic at the
  closing prices, not a decision-time rule.
- The cells: every settled selection (scorer × pool × selection × segment, the segments `all`
  and the five class groups) and every N5 band (scorer × pool × segment × band); the cell id is
  `<scorer>/<pool>/<selection>/<segment>` or `<scorer>/<pool>/n5/<segment>/<band>`.
- `class_group` is null for `all`; `odds_band` is null for a selection and the claim-quantile band
  (`claim q0.9-0.95`) for an N5 band; `shrunk_edge` is null; `payout_rate` is the pool's base rate
  in `config/jra.yaml` (1 - takeout); `mean_return` is per unit stake at the final dividend.
- Ranking: `mean_return` descending, an empty cell last. Nothing is dropped for being null.

## Outputs

- The private report `$HORSETRACK_DATA_DIR/reports/cross-pool-japan-<spec hash[:12]>.md`
  (counts and exclusions, unavailable settlement pools, coefficients, gains, settlement, N5,
  ranked cells), written through `core.eval.reports.write_report` with `sources=("jravan",)`; a
  trial's is `cross-pool-japan-<hash[:12]>-trial.md` and its cells `cross-pool-trial.json`.
- One `RunRecord` per variant in the japan registry, `sources=("jravan",)`, status `scored` or
  `failed`; then `write_summary` regenerates `docs/experiments/registry/japan/`. A crash from the
  store's discovery and build fingerprinting on (a store missing a table, say) appends a failed
  record per variant to the family being run (the trial family for a trial) and is raised; the
  record names every data build a full run names, `@missing` or `@unavailable` after the path of
  a table it could not fingerprint, so its variant is a full run's. A refusal (an unfrozen
  family, a span before 2014-06-07, an out dir inside the repository) appends nothing and opens
  no store file.
- After `--run`, the public `docs/experiments/cross-pool-japan.md` from `render_public_report`:
  what ran, the family, the spec hash, the family size, the run ids, the private report's path
  and the planned settlement pools (which of them were available is only in the private report);
  every figure withheld (D21).
