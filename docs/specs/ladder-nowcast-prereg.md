# The live ladder nowcast against the random walk, pre-registration (ticket #629)

Status: **PROPOSED**, to be frozen before any scored read. Spec sha256 (`_spec_hash(family_spec())`):
`414b743b45de089a1699fc7a0ac5486bb1b7523ba801e1e9ba89afaf3032d75e`.

Family: `japan-ladder-nowcast`, one variant `nowcast-v0-vs-random-walk`. The spec is
`horsetrack.countries.japan.research.ladder_nowcast.family_spec()`, plain JSON;
`uv run horsetrack japan ladder-nowcast freeze` freezes it with `Registry.freeze_family` in
`$HORSETRACK_DATA_DIR/registry/japan.jsonl` (append-only: a freeze cannot be undone).
`ladder_nowcast.run` reads no ladder file unless the family is frozen with exactly this spec, so a
change to the spec after the freeze is a new family, frozen again before it is run. There is no
sign-off gate and no pass/fail verdict (#447): a nowcast that does not beat the random walk is a
result, recorded and ranked like any other.

This measures the projector the shadow page shows (#627), live, on our own captures. #85's nowcast
study scored the fit offline on JRA-VAN history; that is a different measurement.

## Hypothesis

The page's nowcast projection of a later checkpoint's odds is closer to the observed odds than a
random walk, the projection that the odds stay where they are: the MAE gain is above 0 on the
primary pairs. The null, the starting point: it is not.

## Input

- The sealed ladder logs `$HORSETRACK_DATA_DIR/shadow/japan/<YYYY-MM-DD>.ladder.jsonl` (#628,
  docs/specs/shadow-log.md, "The ladder log"), read through `LadderLog.verify()` and `read()`.
- **Development days only (D14).** A day with `is_holdout_day("japan", day)` is skipped before its
  path is looked at: its file is never opened, it never counts toward the minimum, and the holdout
  is not opened.
- **Sealed and verified only.** A file that is not sealed (the day is still being written, or it
  stopped early), whose chain does not verify (`ShadowLogTampered`) or whose records were logged
  as a holdout day (`HoldoutSealed`) is skipped and counted in the report by reason.
- Live and replay records (`mode`) are both scored: the projection is the same function of the
  odds; the report gives the count of replay records.

## Minimum

At least **4** development days holding a sealed ladder log (`MIN_DAYS`). With fewer the run is a
recorded failure: a `failed` registry record and a report, no rows. Four days give roughly ten
meetings, the bootstrap's clusters; fewer leave intervals that a single meeting dominates.

## Pairs

Every `ok` record at checkpoint k in (20, 10, 5, 2) and every later checkpoint j in its `projected`
(10, 5, 2 or 0, the final) is one pair, scored when the projection is not null and the key
(race_id, pool, combination) has an `ok` record at j. With O_k the record's odds, P its projection
for j and O_j the observed odds at j:

- **nowcast**: log error ln(P / O_j).
- **random_walk**, the null: the projection is O_k, log error ln(O_k / O_j). Both predictors are
  scored on exactly the same pairs.
- **direction hit**: sign(ln(P / O_k)) == sign(ln(O_j / O_k)), the sign three-valued (-1, 0, +1).
  The random walk never calls a move, so it hits exactly the keys whose odds did not move.

Exclusions, counted in the report and never scored, in this order: a `no_board` record (one per
missing checkpoint); a null projection (counted before its target is looked up, so the count does
not depend on the target); a target key that is not a later checkpoint; a target with no `ok`
record (a `no_board` checkpoint or a key missing from it); a projection that is not above 0.

## Cells and metrics

A cell is (pool, odds band, from k, to j, predictor). The band is `band_of(O_k, bands)`, the number
of cut points at or below O_k, with the nowcast's own cut points (`config/nowcast_v0.yaml`
`bands`: 2, 5, 10, 20, 50, 100), so a cell's band is the band the projector used. Each (pool, k, j,
predictor) also has an all-bands row (`band` = `"all"`).

Per cell: `n` (pairs), `n_meetings`, the **median log error** (the bias; the mean of the middle two
for an even n), its **two-sided 95% interval** (`ci_lo`, `ci_hi`), the **MAE** (mean absolute log
error), the **hit rate** and, on the nowcast's rows, the **MAE gain**: the random walk's MAE minus
the nowcast's on the same pairs, with its **one-sided lower bound** (`gain_lo`) at
`config/scoring.yaml`'s level (0.95, D7). A gain claim rests on that lower bound.

**Bootstrap.** Clusters are meetings (venue x day, D7; a race_id without its race number):
meetings are resampled with replacement, each draw taking all of the meeting's pairs in the cell;
both predictors and the gain use the same draws. Intervals are percentiles of the replicates.
Each cell's generator is seeded from the run seed and the cell, so a cell's interval does not
depend on which other cells exist. The command's defaults: seed 20261129, 2,000 replicates.

## Primary and secondary pairs

The primary pair is the page's rule, the previous rung to the next (#627,
`board.shadow_feed.PREVIOUS_RUNG[j] == k`): 20->10, 10->5, 5->2, and the final from T-2. Every
other (k, j) pair in the log (20->5, 20->2, 20->final, 10->2, 10->final, 5->final) is a secondary
row, kept and reported. The report's first figure is the primary pairs, all bands, per pool: the
nowcast's MAE beside the random walk's, the gain and its lower bound, the hit rates and the biases.

## Records and D21

Every run, failures included, appends one registry record of `japan-ladder-nowcast` (ticket 629,
hypothesis P629, sources `jra` and `jravan`: the odds are our own JRA captures, the projections come
from the nowcast fitted on JRA-VAN history, D21's withheld source). Its metrics are each cell's
median log error with its interval, MAE, hit rate and MAE gain with its lower bound; its notes the
run time, the days scored and the nowcast ids seen. The report, one file per run, is written
through `core.eval.reports.write_report` under `$HORSETRACK_DATA_DIR/reports/`, which refuses the
repository; the registry stays private. No figure of this study appears in a public file: the PR's
experiment report cites the registry record and the private report, never a figure.
