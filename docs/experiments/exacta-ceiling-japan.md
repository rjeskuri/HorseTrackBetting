# Experiment report: N2, the exacta look-ahead ceiling on final boards (Japan, #431)

Ticket #431 (epic #72). **Every figure is withheld (D21)**: the run reads the JRA-VAN store, a
licensed source, so this report says what ran and where its figures are, and gives no count, rate,
span, sign, bound or rank derived from that data. The selection is **look-ahead**: it reads the
final exacta and Win boards, which hold money placed after T-2. Nothing here is a betting rule or
a betting-edge claim, nothing is adopted, and the holdout is not opened.

- Registry families of the run (v2), each frozen with `Registry.freeze_family` before the run
  read any row (`uv run horsetrack japan exacta-ceiling freeze`; spec
  `docs/specs/exacta-ceiling-prereg.md`, code `exacta_ceiling.family_specs()`). Win
  probabilities: the normalised implied Win shares of the final Win board, as #131's E2
  rule-v5 on the T-2 board:

  | family | spec sha256 |
  | --- | --- |
  | `japan-431-exacta-ceiling-v2` | `ad9fd640840b856429af188a7c56b83820f4a89cc58b6cb86e6ddcdfb9122d4f` |
  | `japan-431-exacta-ceiling-plain-harville-v2` | `c5cfc892e5f71fcbbcf0579bb6d7985e0d35249180c815c010e3524ca081f90a` |
  | `japan-431-exacta-ceiling-descriptive-v2` | `33c5115be87fa9e64f5027d6e8a1b5347a920b4861bfdd8439280fea36e041f7` |
- Superseded families (v1, `exacta_ceiling.superseded_specs()`): Win probabilities from #114's
  out-of-fold calibrated `p_cal` on the races with a #114 row, as #131's superseded rule-v4.
  Frozen and recorded; their members stay ranked variant rows, labelled with that p source and
  span; never executed again:

  | family | spec sha256 |
  | --- | --- |
  | `japan-431-exacta-ceiling` | `acf0cd5499bbdf30a1bae4cb072e0a84c256a01a618f3a490c93ccd4175ccfc3` |
  | `japan-431-exacta-ceiling-plain-harville` | `5016c7045665463f4191a9f6a4c11e2c1408f7165eeb02d31c09a2dba4e44beb` |
  | `japan-431-exacta-ceiling-descriptive` | `5877f77e68d434d873fec240c685b9432437a6129576f8869a8affdff85348d8` |
- Command: `uv run horsetrack japan exacta-ceiling run --from 2014-06-07 --to <last day>`;
  development days only (`is_holdout_day`), none before 2014-06-07 (D10), every race with final
  Win and exacta boards; `LockedHoldout` is never opened. A race whose exacta pool has refund
  votes (a scratch after sales) is excluded and counted. A run that scores no race records its
  members as failed and writes no report.
- Private outputs, through `write_report` with the source `jravan`:
  `$HORSETRACK_DATA_DIR/reports/exacta-ceiling-japan-<commit>-<build>-seed<seed>.md` and `.json`
  (every ranked member with its p source, family size and the variants searched, the
  descriptives, the exclusions and the span), and the ranked member file
  `$HORSETRACK_DATA_DIR/reports/candidate-cells/exacta-ceiling.json` (written by the pre-registered seed
  431 only).
- Variants searched: 64, every member family of the study: 16
  members at #115's full-development lambda2 and the same 16 at plain Harville, for
  each p source (v2 and v1); family size 16 each.

## Hypothesis

E43 as #131, read on the final boards (N2): some exacta combination category, fav_second or
no_fav, is underbet against the Win-derived discounted Harville probability far enough that
buying it when q / p < k returns more than break-even (rho above 1 / rate) at the final dividend.
The null, the starting point: the exacta pool prices each category like the Win pool, so every
member's rho is about 1 (below break-even, takeout and breakage in). Even a member above
break-even here is a ceiling, not a strategy: a selection on the final board is not a proven
upper bound on the T-2 rule (informed late money), a ledger assumption.

## Variants tried

The 16 members of #131's E2 unchanged: `fav_second|k=0.6|all`, `fav_second|k=0.6|headline`, `fav_second|k=0.7|all`, `fav_second|k=0.7|headline`, `fav_second|k=0.8|all`, `fav_second|k=0.8|headline`, `fav_second|k=0.9|all`, `fav_second|k=0.9|headline`, `no_fav|k=0.6|all`, `no_fav|k=0.6|headline`, `no_fav|k=0.7|all`, `no_fav|k=0.7|headline`, `no_fav|k=0.8|all`, `no_fav|k=0.8|headline`, `no_fav|k=0.9|all`, `no_fav|k=0.9|headline`; each at #115's full-development lambda2
(`japan-431-exacta-ceiling-v2`) and at plain Harville, lambda2 = 1 (`japan-431-exacta-ceiling-plain-harville-v2`), with p from the final Win
board's normalised implied shares; and the same 32 with p from #114's `p_cal` on the races with a
#114 row (`japan-431-exacta-ceiling`, `japan-431-exacta-ceiling-plain-harville`: superseded, recorded, ranked as variant rows). Each
member's look-ahead rho, its one-sided 95% meeting-bootstrap upper bound (the CI column), lower
bound, return per 100 yen and its upper bound, n selected, n races and n meetings: withheld (D21).
Descriptives (`japan-431-exacta-ceiling-descriptive-v2`, and v1's `japan-431-exacta-ceiling-descriptive`): D = sum q / sum p per category
and cell, and the Quinella x order split, each with a two-sided 95% meeting-bootstrap interval:
withheld (D21). A member with no bet keeps its row, ranked last, with an unavailable marker in the
registry; a failed run is recorded too (the registry summary lists every run's status).

## Alternatives rejected and why

- A kill rule or an open/closed verdict: replaced by the owner's reframe (2026-11-27); every
  member is a ranked row and nothing closes or is parked on this result.
- The real lower tertile of the final exacta pool totals as the headline's "less popular": not
  added; #131's race-card proxy keeps the members comparable with E2 and the family small.
- #114's out-of-fold `p_cal` as the run's Win probabilities (v1): superseded by v2 because
  #131's frozen E2 is rule-v5, whose p_c(t) is the normalised implied Win shares of the Win
  board on every day with Win and exacta boards; v1's rows stay ranked (Sol review round 1 of
  24d080f).
- Own impact from the logged odds: the JRA-VAN final votes give each combination's money and the
  pool's net total directly, so the payout identity needs no odds inversion.

## What this rules out

withheld (D21): the private report gives each member's look-ahead rho with its one-sided 95% upper
bound, against break-even 1 / rate. Whatever it shows, the selection reads the final boards, so a
member's row is a look-ahead ceiling for the leaderboard, never a betting rule.

## Leaderboard rows

Option ids `<family>:<category>|k=<cut>|<cell>` for the 16 members above in each member family
(`japan-431-exacta-ceiling-v2`, `japan-431-exacta-ceiling-plain-harville-v2`, `japan-431-exacta-ceiling`, `japan-431-exacta-ceiling-plain-harville`), each a look-ahead row ranked by rho
and labelled with its p source and span; rank and figures withheld (D21).

## Follow-ups filed

- The same members on T-2 boards (#131's E2 v5) as #130-logged development days accrue, and in
  the holdout ranking check through #447's leaderboard.
