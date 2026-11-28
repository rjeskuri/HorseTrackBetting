# Experiment report: #94's competition terms on the C2 member at the final Win price (Japan, #531)

Ticket #531 (epic #73; #94's outline 4; 08 §3.3; 05 §3.4, §4.2, H12). The study adds #94's competition terms
(`countries/japan/models/competition.py`) to #92's pooled C2 member. It measures each variant standalone and at
the final Win price, and its tail return at the final price, so that each variant becomes a ranked member row on the
leaderboard. It also splits the member's stage-2 error against the market by the field's effective resistance
(R_eff). **Nothing is adopted** (D16), and the holdout is not opened. **Every figure is withheld (D21):** the sire
in #92's model comes from the JRA-VAN pedigree, so every result is built on JRA-VAN data.

**Status: not run yet.** The run needs the copied stores (needs-local). This report describes what is
pre-registered and how to run it. It holds no count, span, score, return, bound or sign. The run of record will add
its family's spec hash, git commit, data builds, record ids and private report path here, and its ledger row will
change from assumption to single-run.

- Registry family `japan-c2-competition-final-win` (its own family, not #92's). `run_study` freezes the spec
  (`studies.competition.build_spec`) in the run registry before `load_inputs` reads any row. A spec that differs
  from the frozen one is refused (`FamilyAlreadyFrozen`).
- Code: `countries/japan/studies/competition.py`, tested in `tests/countries/japan/test_competition_study.py` on
  synthetic data. To run it: `uv run horsetrack japan competition-study`. The command writes the private report
  (Markdown and JSON) under `$HORSETRACK_DATA_DIR/reports/` through `write_report`, appends one registry record per
  variant (failures included) with the sources `jra` and `jravan`, and regenerates the committed summary
  `docs/experiments/registry/japan/japan-c2-competition-final-win.md`, which shows every record as withheld.
- Run cost: level 1 is #92's run as is (see Method), so expect about #92's time on this Mac (about 2 h 25 min,
  peak resident memory about 1.8 GB), plus four fits per outer fold and one comparison graph per scored fold.

## Hypothesis

The frozen spec reads:

- **E33 (the C2 member, #92's family).** "A competition term added to #92's pooled member improves it standalone and
  adds to the market term at the final price (08 §3.3; opponent strength is implicit in Plackett-Luce, so a null is
  the expected result)."
- **E41's premise, 05 H12.** "The pooled member's stage-2 error against the market-only meta rises with the field's
  effective resistance (R_eff), the premise of E41's precision weighting (#98)."
- **Null**, the starting point: no term changes the member's winner log loss, alone or at the final price, and the
  residual does not vary with R_eff.

08 §4 numbers no experiment for the competition terms of its own, so the four variants carry E33, the C2 member's
id, and the R_eff record carries E41, whose precision weighting it feeds. The spec records this mapping under
`hypothesis_ids`.

What would show the null wrong: for a variant, a meeting-bootstrap one-sided 95% lower bound above 0 on its paired
per-race winner ΔLL. For R_eff, a residual that rises across the R_eff bands, with the bands' bounds apart.

## Method

- **Terms** (#94): `field_mean` and `field_max` are the mean and max of the other runners' before-the-race rating
  in the race. `field_x_rating` is `field_mean` × the runner's own rating. The rating is `studies.rating`'s
  `mu_before` at the outer fold's selected drift (#91's `DynamicRating`), so no outcome of the race enters a term.
  A runner with no other rated runner has `field_mean` and `field_max` set to 0 in the fit, because its race adds
  nothing to the Plackett-Luce likelihood (#94 sets `field_x_rating` to 0 there).
- **Level 1** is #92's code, reused as is: the reads (`ability.load_inputs`), #91's rating grid, the rating pair
  per outer fold (`select_pairs`), the shared discount grid (`level1_grid`), the discount selection
  (`select_discounts`) and the pooled member (`level1_members`). Each variant is #92's pooled `AbilityModel(k=3)` at
  the outer fold's selection (rating pair and pooled discounts), with the variant's terms added as features. It is
  fitted on the fold's training rows and predicts the fold's test quarter. The terms do not re-enter the selection,
  so a variant differs from the base only by its terms.
- **Level 2** is #92's: the store's priced starters of the price era (`studies.meta.store_frame`), and each
  member's ln p renormalised over the race's priced starters. On D9's folds, a member's stage-2 meta is
  `ability.STAGE2`'s form (u = α · m + β · ln q, unbounded, ridge 0, out of fold), and the market-only meta is
  `ability.market_only_oof` (u = β · ln q). Every D9 fold with a test row is scored.
- **Per variant:** the paired per-race winner ΔLL of the variant's member alone against the pooled member alone
  (`all_folds/delta_ll_standalone`), and of its stage-2 meta against the market-only meta
  (`all_folds/delta_ll_final_price`), each with D7's meeting-bootstrap bounds (95% one-sided, `config/scoring.yaml`,
  seed 531) and per-fold points. Its tail return at the final price (`development/return_per_100`) is N5 (#433):
  `core.eval.tail.tail_curve` over every scored runner as a flat ¥100 Win bet, with claim = the stage-2 meta's p ×
  the final odds, the bands at quantiles 0.5, 0.8, 0.9, 0.95 and 0.99 (#438's), and meeting clusters. The return is
  the top band's: the final odds for a winner (divided by m for each of m co-winners, because the store frame holds
  no dead-heat dividend) and 0 otherwise.
- **R_eff (05 H12):** for each scored fold, #94's `comparison_graph` is built from the results before 00:00
  Asia/Tokyo of the fold's first day, and `race_comparability` gives each of the fold's test races the mean
  effective resistance over its runner pairs (+inf when a pair is unconnected or a horse is unseen). The residual is
  the per-race winner log loss of the pooled member's stage-2 meta minus the market-only meta's. It is reported by
  R_eff quintile band (`np.quantile` edges, `core.eval.tail`'s band rule) and in one unconnected bucket, each with
  its mean and meeting-bootstrap bounds. The precision weighting itself is #98's E41.

## Variants tried

The variants are pre-registered, and each will be one registry record in `japan-c2-competition-final-win`. None has
run yet, so none has a figure or CI. The split for all of them is D9's walk-forward quarters on development days
(scored from 2020-Q1), and the holdout is never read.

| Variant | E | Terms added to #92's pooled member | Holds |
| --- | --- | --- | --- |
| `field_mean` | E33 | `field_mean` | ΔLL standalone and at the final price with bounds; the N5 top-band return per ¥100 with bounds; the leaderboard block |
| `field_max` | E33 | `field_max` | As above |
| `field_x_rating` | E33 | `field_x_rating` | As above |
| `field_all` | E33 | all three | As above |
| `reff_by_bucket` | E41 | none (the pooled member) | The stage-2 residual against the market-only meta by R_eff band and unconnected bucket, with bounds |

A variant whose level-1 fit fails is a failed record carrying its error, and the other variants still score. A run
that aborts in `load_inputs` or `compute_study` appends a failed record for every planned variant, writes no
report, and re-raises.

## Alternatives rejected and why

- **Reselecting the discounts and the rating pair for each variant:** this would multiply #92's level-1 grid by
  four, and the comparison with the base would then mix the term's effect with a different selection. The variants
  use the base's selection, so each paired ΔLL isolates the term.
- **Reading #92's per-runner export instead of refitting the base:** the export is a file named by an earlier run's
  spec and data builds. Refitting with #92's code on the same reads keeps the base and the variants on one data build.
- **A separate grid that fits only the pooled model:** this would roughly halve level 1, because #92's `level1_grid`
  also fits its rates member, which this study does not use. It was not built, because it would copy #92's grid; the
  reuse keeps the base member identical to #92's.
- **The real dead-heat dividends (the JRA-VAN dividends store) for the tail return:** this would add a read outside
  #92's inputs for a rare case. The spec states the odds / m approximation instead.
- **Intraday cuts of the comparison graph:** `studies.rating.read_results` stamps races at midnight JST plus race_no
  minutes, so only a cut at a JST day boundary is point-in-time safe (#94). Each fold's graph is cut at 00:00 JST of
  its first day.
- **Precision-weighting the meta by R_eff:** that is #98's E41. This study only reports the residual by bucket.

## What this rules out

Nothing yet, because the study has not run. Once it has run, the figures stay withheld (D21): the repository will
state no size or direction of any result, and the private report will hold each variant's ΔLL, return per ¥100 and
bounds. The result will be a measurement on one store build (single-run in the ledger), not a verdict: nothing is
adopted or rejected, and a weak variant ranks low.

## Leaderboard rows

Once run, the study adds these member rows (kind `member`, pool `WIN`, selection `look_ahead`, because the claim reads
the final price), each ranked by its N5 top-band return per ¥100. They are withheld rows, so the public leaderboard
shows their option ids only, with no rank or figure (D21):

- `japan-c2-competition-final-win/field_mean`
- `japan-c2-competition-final-win/field_max`
- `japan-c2-competition-final-win/field_x_rating`
- `japan-c2-competition-final-win/field_all`

`japan-c2-competition-final-win/reff_by_bucket` has no return, so it is listed unranked. A failed variant is listed
unranked too.

## Follow-ups filed

None by this PR. Proposed, for the orchestrator to file under epic #73 as `status:proposed`:

- Run `uv run horsetrack japan competition-study` on the stores (needs-local), then add the run of record to this
  report and move the ledger rows to single-run.
- Feed the R_eff export (`race_comparability` per race) and this residual table into #98's E41.
