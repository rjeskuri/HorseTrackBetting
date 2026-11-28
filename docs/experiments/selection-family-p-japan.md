# Experiment report: Family P selection rules on the boards at t (Japan, #462)

Ticket #462 (epic #72), the development-fold run of #88's pre-registered Family P. **Nothing is adopted.** **Every figure is withheld (D21).** The run reads the JRA-VAN time-series boards (#369) and the jravan races, runners, final odds and dividends, a licensed source, so this page says what ran and where its figures live. It gives no count, span, coefficient, gain, probability, return, sign or bound derived from that data.

- Registry family `japan-family-p-selection-v1`, spec sha256 `b44ee710203c414517aba6f2d7cc763ebd29c9854029383ad3a2719e999ff802`: `dataclasses.asdict(load_family("japan"))` of `countries/japan/config/selection_family.yaml`, the family the owner signed on #88 (2026-11-26), frozen in the Japan registry before any store file was opened. Any edit of the yaml is a new family id.
- Family size: 9 variants, the 3 rules and the 6 dropped E4 variants, each dropped variant recorded as a failed variant before anything was scored (08 §5.2 step 1).
- Private report: `$HORSETRACK_DATA_DIR/reports/selection/selection-family-p-japan.md`, written through `core.eval.reports.write_report`.
- Code: `countries/japan/studies/selection_run.py` (the frame and the run) over `countries/japan/studies/selection.py` (#88's study). Reproduce with `uv run horsetrack japan selection-run` (from 2014-06-07 to the last development day the jravan boards hold; seed 462, the ticket number, which chooses nothing).
- Holdout: development days only. A holdout day's file is dropped by its name and never opened (D14), the store tables are read with the holdout days dropped, the study refuses a holdout row, and the `LockedHoldout` it holds is never opened. Nothing before 2014-06-07 enters (D10: the calibrated-market benchmark refuses the pre-reform prices).

## Hypothesis

Each rule claims that, in its segment, the calibrated market (D4's spline in ln q, refitted on each of D9's walk-forward folds) misprices the Win pool in the direction the rule bets, and that its bets selected on the board at t = post - 2 min (D2) return more than the takeout gives back. **Null:** no segment departure (beta_s = 1) and no return above the takeout, the starting point.

- **P3 (E7), `P3-handicap-5-50`:** in handicap races the favourite-longshot bias reverses; beta_s < 1, so runners at 5.0 to 50.0 on the board at t are under-bet.
- **P4 (E4, E8, E42), `P4-r12-favourite`:** in race 12 the favourites win more often than priced; beta_s > 1, bets on the favourite at t (rank_t 1, joint favourites both).
- **P6 (E3), `P6-concentrated-favourite`:** in the most concentrated markets (entropy bin 1 of the Win board at t, E3's signed cut points) the favourite is under-priced; beta_s > 1, bets on the favourite at t.

What would show each wrong: a log-loss gain over the calibrated market whose one-sided lower bound is not above 0, or a fitted beta_s on the other side of 1; StepM not finding the rule superior over the family; a posterior P(omega_s > 1/(1 - t)) on the rule's selected bets that is not high.

## Variants tried

Every rule is evaluated on D9's walk-forward development folds (`config/folds.yaml`), its selection read from the board at T-2 and its bets settled at the final Win dividend.

| Variant | Claim | Segment | Selection at t | Status | Stage 1 (ΔLL, one-sided bound) | StepM | P(omega_s > 1/(1 - t)) | ROI at the final dividend | Rung |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `P3-handicap-5-50` | beta, below | weight_type handicap | odds_t in [5.0, 50.0] | withheld (D21) | withheld (D21) | withheld (D21) | withheld (D21) | withheld (D21) | S0 |
| `P4-r12-favourite` | beta, above | R12 in | rank_t 1 | withheld (D21) | withheld (D21) | withheld (D21) | withheld (D21) | withheld (D21) | S0 |
| `P6-concentrated-favourite` | beta, above | entropy_t_bin 1 | rank_t 1 | withheld (D21) | withheld (D21) | withheld (D21) | withheld (D21) | withheld (D21) | S0 |
| `P4-r11pm1` | dropped before the freeze | E4 segment | none | failed (dropped) | not run | not run | not run | not run | S0 |
| `P4-g1-day-undercard` | dropped before the freeze | E4 segment | none | failed (dropped) | not run | not run | not run | not run | S0 |
| `P4-field-16-plus` | dropped before the freeze | E4 segment | none | failed (dropped) | not run | not run | not run | not run | S0 |
| `P4-debut` | dropped before the freeze | E4 segment | none | failed (dropped) | not run | not run | not run | not run | S0 |
| `P4-venue` | dropped before the freeze | E4 segment | none | failed (dropped) | not run | not run | not run | not run | S0 |
| `P4-heavy-going` | dropped before the freeze | E4 segment | none | failed (dropped) | not run | not run | not run | not run | S0 |

The stages, each run for every rule whatever the others found (the owner's reframe, 2026-11-27):

1. Log loss: the winner log-loss gain of the segment departure over the calibrated market, fitted in fold and scored out of fold, with #69's paired bootstrap over meetings (one-sided, D7).
2. Max statistic: #75's SPA and Romano-Wolf StepM over the scored rules, level 0.05, stationary blocks of 6 meetings, 10000 replicates.
3. Omega: #78's model on each rule's selected bets at the final Win odds, odds-band fixed effects cut at 2, 5, 10, 20, 50 and 100: the expected return, P(omega_s > 1/(1 - t)) and the predictive ROI of the next 100 bets, reported only.
4. ROI: one unit to Win on each selected bet at the final dividend.

Survivors (stage 1 passed and StepM superior) are a diagnostic: withheld (D21). The frame's counts (races, starters, races with a board at t, the races left out with no post time, fewer than two starters or no winner): withheld (D21), in the private report. Every rule sits at S0, D19's default; none is above S1, and a rule moves only through the leaderboard and the owner.

## Alternatives rejected and why

- P1 and P2 (Place rules): the study pays Win dividends and scores the winner only; E15 keeps its own family (#83). Deferred by the owner on #88.
- P4-alpha: needs the owner to name a C4 member; P4-omega: needs a runner-level segment; P5: needs C4 at the decision time; P7: E24 has not run. Each deferred by the owner on #88, with what is missing in the yaml's `deferred`.
- The E4 variants: dropped by the owner on #88 after #77's in-sample screen; recorded as failed variants so the family size counts the search.
- Selecting on the final board: look-ahead (08 §5.2); the study refuses any rule input not declared from the board at t or the card as of t.

## What the result rules out

withheld (D21): the reading rests on each rule's stage-1 bound, StepM's finding and P(omega_s > 1/(1 - t)) in the private report. Under the null, none of the three segments departs from the calibrated market at T-2 in the direction it bets, and none of the rules' selections returns more than the takeout gives back.

## Leaderboard rows

Every rule and dropped variant is a row of the Japan leaderboard (`horsetrack japan leaderboard`; the Family P adapter in `countries/japan/research/leaderboard_backfill.py`: a Win rule selected at t, its return per ¥100 at the final dividend, n in bets). All are withheld (D21), listed by option id only:

- `japan-family-p-selection-v1/P3-handicap-5-50`
- `japan-family-p-selection-v1/P4-r12-favourite`
- `japan-family-p-selection-v1/P6-concentrated-favourite`
- `japan-family-p-selection-v1/P4-r11pm1` (unmeasured, dropped)
- `japan-family-p-selection-v1/P4-g1-day-undercard` (unmeasured, dropped)
- `japan-family-p-selection-v1/P4-field-16-plus` (unmeasured, dropped)
- `japan-family-p-selection-v1/P4-debut` (unmeasured, dropped)
- `japan-family-p-selection-v1/P4-venue` (unmeasured, dropped)
- `japan-family-p-selection-v1/P4-heavy-going` (unmeasured, dropped)

## Follow-ups filed

None yet.
