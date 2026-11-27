# Experiment report: N2, the exacta look-ahead ceiling on final boards (Japan, #431)

Ticket #431 (epic #72). **Every figure is withheld (D21)**: the run reads the JRA-VAN store, a
licensed source, so this report says what ran and where its figures are, and gives no count, rate,
span, sign, bound or rank derived from that data. The selection is **look-ahead**: it reads the
final exacta and Win boards, which hold money placed after T-2. Nothing here is a betting rule or
a betting-edge claim, nothing is adopted, and the holdout is not opened.

- Registry families, each frozen with `Registry.freeze_family` before the run read any row
  (`uv run horsetrack japan exacta-ceiling freeze`; spec `docs/specs/exacta-ceiling-prereg.md`,
  code `exacta_ceiling.family_specs()`):

  | family | spec sha256 |
  | --- | --- |
  | `japan-431-exacta-ceiling` | `acf0cd5499bbdf30a1bae4cb072e0a84c256a01a618f3a490c93ccd4175ccfc3` |
  | `japan-431-exacta-ceiling-plain-harville` | `5016c7045665463f4191a9f6a4c11e2c1408f7165eeb02d31c09a2dba4e44beb` |
  | `japan-431-exacta-ceiling-descriptive` | `5877f77e68d434d873fec240c685b9432437a6129576f8869a8affdff85348d8` |
- Command: `uv run horsetrack japan exacta-ceiling run --from 2014-06-07 --to <last day>`;
  development days only (`is_holdout_day`), none before 2014-06-07 (D10), races with #114's
  out-of-fold Win probabilities; `LockedHoldout` is never opened.
- Private outputs, through `write_report` with the source `jravan`:
  `$HORSETRACK_DATA_DIR/reports/exacta-ceiling-japan-<commit>-<build>.md` and `.json` (the
  ranked members, the descriptives, the exclusions and the span), and the ranked member file
  `$HORSETRACK_DATA_DIR/reports/candidate-cells/exacta-ceiling.json`.
- Variants searched: 32 (16 members at #115's
  full-development lambda2 and the same 16 at plain Harville); family size
  16 each.

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
(`japan-431-exacta-ceiling`) and at plain Harville, lambda2 = 1 (`japan-431-exacta-ceiling-plain-harville`). Each member's look-ahead rho,
its one-sided 95% meeting-bootstrap upper bound (the CI column), lower bound, return per 100 yen
and its upper bound, n selected, n races and n meetings: withheld (D21). Descriptives
(`japan-431-exacta-ceiling-descriptive`): D = sum q / sum p per category and cell, and the Quinella x order split,
each with a two-sided 95% meeting-bootstrap interval: withheld (D21). A member with no bet keeps its
row, ranked last, with an unavailable marker in the registry; a failed run is recorded too (the
registry summary lists every run's status).

## Alternatives rejected and why

- A kill rule or an open/closed verdict: replaced by the owner's reframe (2026-11-27); every
  member is a ranked row and nothing closes or is parked on this result.
- The real lower tertile of the final exacta pool totals as the headline's "less popular": not
  added; #131's race-card proxy keeps the members comparable with E2 and the family small.
- Win probabilities from the final Win price over a longer span: not added; #114's out-of-fold
  calibrated probabilities are #131's source, so the ceiling is comparable with E2.
- Own impact from the logged odds: the JRA-VAN final votes give each combination's money and the
  pool's net total directly, so the payout identity needs no odds inversion.

## What this rules out

withheld (D21): the private report gives each member's look-ahead rho with its one-sided 95% upper
bound, against break-even 1 / rate. Whatever it shows, the selection reads the final boards, so a
member's row is a look-ahead ceiling for the leaderboard, never a betting rule.

## Leaderboard rows

Option ids `japan-431-exacta-ceiling:<category>|k=<cut>|<cell>` and `japan-431-exacta-ceiling-plain-harville:<category>|k=<cut>|<cell>`
for the 16 members above, each a look-ahead row ranked by rho; rank and figures withheld (D21).

## Follow-ups filed

- The same members on T-2 boards (#131's E2 v5) as #130-logged development days accrue, and in
  the holdout ranking check through #447's leaderboard.
