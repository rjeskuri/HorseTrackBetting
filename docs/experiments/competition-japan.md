# Experiment report: #94's competition terms on the C2 member at the final Win price (Japan, #531)

Ticket #531 (epic #73; #94's outline 4; 08 §3.3; 05 §3.4, §4.2, H12). The study adds #94's competition terms
(`countries/japan/models/competition.py`) to #92's pooled C2 member. It measures each variant standalone and at
the final Win price, and its tail return at the final price, so that each variant becomes a ranked member row on the
leaderboard. It also splits the member's stage-2 error against the market by the field's effective resistance
(R_eff). **Nothing is adopted** (D16), and the holdout is not opened. **Every figure is withheld (D21):** the sire
in #92's model comes from the JRA-VAN pedigree, so every result is built on JRA-VAN data.

**Run of record: 2026-11-28, at git `910ff3a`.** The study ran once on the copied stores, from a clean tree at
`910ff3ae30d25199290acd9b7b7372ca9c7f1132` (branch `t/531-run-the-competition-terms-stud`; the squash merge carries
another hash), with `uv run horsetrack japan competition-study`. It appended five registry records, all scored, and
wrote its private report. This report says what ran, which records hold it and where the figures live. It holds no
count, span, score, return, bound, sign, direction or rank of any result: the figures are in the private report and
the private registry, and are withheld (D21).

- Family `japan-c2-competition-final-win`, spec sha256
  `476eebce5491970a63372420170cac4dc54714259f7621b6c72fcf2feb9f2164`, frozen in the registry when the run started
  (2026-11-28 11:09 UTC), before `load_inputs` read any row. It is the family's only freeze.
- Data builds (each record's `data_builds`): `processed/jra/races.parquet@9cce511831074b64`,
  `processed/jra/runners.parquet@6f3cec95347e0132` and `processed/jravan/horse_pedigree.parquet@891fea934fadd013`,
  #92's v2 builds, plus the run's own artifacts `reports/japan@competition-japan-476eebce5491-9a3f58d49fb3`. Every
  record carries the sources `jra` and `jravan`. The holdout was not opened: no record carries a holdout opening,
  and no line was added to `holdout_log.jsonl`.
- Records (the run ids' 16-hex prefixes):

  | Variant | E | Record | Status |
  | --- | --- | --- | --- |
  | `field_mean` | E33 | `ebc0be168433b3dd` | scored |
  | `field_max` | E33 | `bf304ee2a6ca82f7` | scored |
  | `field_x_rating` | E33 | `9bdb320739304dd7` | scored |
  | `field_all` | E33 | `93c5403a254ce72f` | scored |
  | `reff_by_bucket` | E41 | `a9889dfc1f316821` | scored |

  No variant failed, so no record carries an error.
- Private report: `reports/competition-japan-476eebce5491-9a3f58d49fb3.md` under `$HORSETRACK_DATA_DIR`, with its
  figures as JSON beside it (same stem, `.json`), written through `write_report`.
- Committed summary: the run regenerated `docs/experiments/registry/japan/japan-c2-competition-final-win.md` and its
  index line in `docs/experiments/registry/japan/README.md` through `write_summary`. It shows each record's status
  and every value as withheld (licensed source).

- Registry family `japan-c2-competition-final-win` (its own family, not #92's). `run_study` freezes the spec
  (`studies.competition.build_spec`) in the run registry before `load_inputs` reads any row. A spec that differs
  from the frozen one is refused (`FamilyAlreadyFrozen`).
- Code: `countries/japan/studies/competition.py`, tested in `tests/countries/japan/test_competition_study.py` on
  synthetic data. To run it: `uv run horsetrack japan competition-study`. The command writes the private report
  (Markdown and JSON) under `$HORSETRACK_DATA_DIR/reports/` through `write_report`, appends one registry record per
  variant (failures included) with the sources `jra` and `jravan`, and regenerates the committed summary
  `docs/experiments/registry/japan/japan-c2-competition-final-win.md`, which shows every record as withheld.
- Run cost, measured on this Mac with `/usr/bin/time -l` (the run at `nice -n 10`): 14,700 s wall (4 h 05 min,
  2026-11-28 20:09 to 2026-11-29 00:14 JST), 11,669 s user and 2,373 s sys, and a peak resident memory of about
  1.2 GB (1,207,255,040 bytes). From the log's progress lines, to about 10 min: the reads and level 1 (#92's grid)
  ran to about 23:05 JST, the variants' fits on the 48 outer folds to about 23:35, and the R_eff stage (one
  365-day comparison graph and one sparse LU per scored fold, 28 folds) to the end, about 40 min. The estimate
  before the run was about 5 to 8 h, 2 to 5 h of it the R_eff stage. The run overlapped another session's fit for
  about 15 min, under a memory guard that pauses the study when free memory falls under 15%. The guard never
  fired: free memory stayed at or above 27%. Without the window the R_eff stage does not finish: at the 2020-01-01
  cut, one factorisation of every earlier race was still running after 15 min at 3.1 GB and growing.

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
- **R_eff (05 H12):** for each scored fold, #94's `comparison_graph` is built from the results of the 365 days
  before the fold's first day (`reff_window_days`: from 00:00 Asia/Tokyo 365 days before that day, up to 00:00
  Asia/Tokyo of that day). `race_comparability` then gives each of the fold's test races the mean effective
  resistance over its runner pairs (+inf when a pair is unconnected or a horse is unseen in the window). The
  residual is the per-race winner log loss of the pooled member's stage-2 meta minus the market-only meta's. It is
  reported by R_eff quintile band (`np.quantile` edges, `core.eval.tail`'s band rule) and in one unconnected
  bucket, each with its mean and meeting-bootstrap bounds. The precision weighting itself is #98's E41.
- **The window (frozen in the spec).** The store holds 86.5k horses from 2010. A graph of every earlier race
  reaches 53k horses and 3.0M edges at the 2020-01-01 cut and 84k horses and 4.8M edges at the 2026-07-01 cut,
  and its factorisation does not finish in bounded time or memory on this Mac. A 365-day graph holds about 11k
  horses and 0.3M edges at either cut: at 2026-07-01 it has 11,443 horses, 295,427 edges and a giant component of
  11,122, and at 2026-10-01 it has 10,905 horses, 295,370 edges and a giant component of 9,339. Its LU has about
  24M nonzeros, takes 32 to 64 s and peaks at about 1.0 GB. That is below the 22.5k horses of #94's store-scale check. The
  window costs coverage: it connects 415 of the 912 races in 2020-Q1 and 476 of the 888 in 2026-Q3, against 486
  and 539 for the whole history, so about seven in eight of the races the history connects. The remaining races
  hold a horse unseen for a year and go to the unconnected bucket. These counts were measured on the store's
  race_id, race_date, race_no, status, horse_id and is_scratched only (no outcome, no price, no holdout day).
  The run of record read the same columns plus finish_pos, which #94's graph does not use.

## Variants tried

The variants were pre-registered, and each is one registry record in `japan-c2-competition-final-win`. All five
scored in the run of record (records above). Their figures and CIs are in the private report and the private
registry, and are withheld (D21). The split for all of them was D9's walk-forward quarters on development days
(scored from 2020-Q1), and the holdout was not read.

| Variant | E | Terms added to #92's pooled member | Holds |
| --- | --- | --- | --- |
| `field_mean` | E33 | `field_mean` | ΔLL standalone and at the final price with bounds; the N5 top-band return per ¥100 with bounds; the leaderboard block |
| `field_max` | E33 | `field_max` | As above |
| `field_x_rating` | E33 | `field_x_rating` | As above |
| `field_all` | E33 | all three | As above |
| `reff_by_bucket` | E41 | none (the pooled member) | The stage-2 residual against the market-only meta by R_eff band and unconnected bucket, with bounds |

No variant failed in the run of record. By design, a variant whose level-1 fit fails is a failed record carrying
its error, and the other variants still score. A failure in the R_eff stage (a `MemoryError` in a fold's factorisation, say) fails `reff_by_bucket` alone, with its
error, and the four variants still score. A run that aborts in `load_inputs` or before the R_eff stage of
`compute_study` appends a failed record for every planned variant, writes no report, and re-raises.

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
- **Every earlier race in each fold's graph:** this is the cleanest R_eff, but its factorisation does not finish
  on the store (see Method, the window).
- **A shorter window (182 or 273 days):** this is cheaper. At 182 days the factorisation took 7.5 s and the
  resistances 0.07 s a race, and at 273 days 17 s and 0.17 s a race. But the shorter windows connect fewer races:
  182 days connects 215 and 304 of the races above (about half of what the history connects), and 273 days
  connects 352 and 407. A season's layoff then makes a race unconnected, so the bands would hold mostly the
  busiest fields. A year keeps a horse's last season in its graph.
- **A 730-day window:** it connects 477 and 520 of the races above (97% of what the history connects). It has
  twice the year's edges (0.58M) and 16k horses, so its factorisation was not tried after the year's took up to
  a minute.
- **Precision-weighting the meta by R_eff:** that is #98's E41. This study only reports the residual by bucket.

## What this rules out

The figures are withheld (D21), so what the run rules out is stated in the private report only: the repository
states no size, sign or direction of any result. The private report
(`reports/competition-japan-476eebce5491-9a3f58d49fb3.md` under `$HORSETRACK_DATA_DIR`) and the private registry
records above hold each variant's ΔLL, return per ¥100 and bounds, and the R_eff residual by band. The result is a measurement on one store build (single-run in the ledger),
not a verdict: nothing is adopted or rejected, and a weak variant ranks low.

## Leaderboard rows

The run of record added these member rows (kind `member`, pool `WIN`, selection `look_ahead`, because the claim
reads the final price), each ranked by its N5 top-band return per ¥100. They are withheld rows, so the public
leaderboard shows their option ids only, with no rank or figure (D21). The private leaderboard built from the
registry after the run (`uv run horsetrack japan leaderboard`, written under `$HORSETRACK_DATA_DIR/reports/`) holds
all four as member rows; this PR does not regenerate the public `docs/leaderboard/japan.md`.

- `japan-c2-competition-final-win/field_mean`
- `japan-c2-competition-final-win/field_max`
- `japan-c2-competition-final-win/field_x_rating`
- `japan-c2-competition-final-win/field_all`

`japan-c2-competition-final-win/reff_by_bucket` has no return, so it is listed unranked, as it is in the private
leaderboard. A failed variant would be listed unranked too; none failed.

## Follow-ups filed

None by this PR. Proposed, for the orchestrator to file under epic #73 as `status:proposed`:

- Feed the R_eff export (`race_comparability` per race) and this residual table into #98's E41.
