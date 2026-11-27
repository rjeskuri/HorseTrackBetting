# Experiment report: newcomer and maiden members at the final Win price (Japan, #438)

Ticket #438 (epic #73; N7, E34, E37). Each member that could carry signal for a debut or maiden runner is fitted beside the final Win price, one at a time and all together, per class (DEBUT, MAIDEN), as a conditional logit with ln q as its offset on D9's folds. **Nothing is adopted** (D16-D19): every variant is a ranked row with its encompassing weight beside it. The holdout is not opened. **Every figure is withheld (D21)**: the members read JRA-VAN and JRDB data, so this report gives no count, span, score, rate, sign or bound; each figure is in the private report and the registry records it names.

- Registry family `japan-debut-maiden-members-v2`, spec sha256 `55bf539436caf585b84eccccc49be9b1c1f9b66aea1cbcc81e727a5bc85e44d3`, frozen before the run read any store row. The pre-registration is `docs/specs/debut-members-prereg.md`.
- Code: `countries/japan/studies/debut_members.py`. To reproduce, run `uv run horsetrack japan debut-members`.
- Private report: `$HORSETRACK_DATA_DIR/reports/debut-members-japan-<spec hash>-<run digest>.md` (the spec hash, then a digest of the run's date range and input builds, so no run overwrites another), its tables as JSON beside it, and the ranked rows in `$HORSETRACK_DATA_DIR/reports/candidate-cells/` under the same stem, all written through `write_report` with the sources `jra`, `jravan`, `jrdb`.
- Records: `docs/experiments/registry/japan/japan-debut-maiden-members-v2.md`; every one is withheld.
- Supersedes `japan-debut-maiden-members-v1`, kept in the record with its records and summary: v1 cut the late Win boards by the day file's minutes_to_post, measured to the JRA-VAN post rather than the study's JRA post, and settled a race whose winner had no WIN dividend row at the final odds in the N5 curve; v1's records are kept.

## Hypothesis

- **E34.** members that carry information the final Win price lacks for runners with thin history, concentrated in debut and maiden races (08 §4 E34 as #438 reads it: the pooled ability, workouts, body weight and the late Win path, one at a time and together)
- **E37.** a pedigree prior (sire x surface x distance band, earlier runs only) adds to the final Win price in debut and maiden races
- **Null**, the starting point: every member weight 0: the final Win price already holds what the member says about a debut or maiden runner (ΔLL 0, encompassing weight 0).

## Members

| member | ticket | status | sources | about |
| --- | --- | --- | --- | --- |
| `ability` | #92 | built | jra, jravan | #92's per-runner u and its Laplace sigma se_u (pooled jockey, trainer and sire on #91's rating), out of fold, from the v2 export named by the family's records |
| `pedigree_prior` | #438 | built | jra, jravan | the sire's offspring record in the race's surface and distance band from earlier runs only (moved here from #95) |
| `workouts_jrdb` | #137 | built | jra, jrdb | #137's workout features from JRDB's CHA and CYB (private, D21) |
| `workouts_jravan` | #229 | built | jra, jravan | #229's best final furlong against par and its trend from JRA-VAN's 坂路 and ウッド times |
| `jvdata_mining` | #406 | missing: listed, left out | jravan | the JV-Data mining predictions (#406): not landed; listed and left out, added as a counted variant when it lands |
| `body_weight` | #227 | built | jra | the race-day body weight and its change, in the views from post - 60 min |
| `late_win` | #438 | built | jra, jravan | the Win share's move to T-2 from T-20 and from T-5 on the JRA-VAN 0B41 boards (#369), each board cut at the JRA post |

## Variants tried

One registry record each, failures included. Each is scored by its per-race winner ΔLL against the final price with D7's one-sided meeting bounds, its weights, its encompassing weight and N5's curve settled at each winner's own Win dividend (a race with a winner that has no Win dividend row is left out of the curve and counted in the private report; Place after #410).

| E | variant | status | ΔLL, bounds, weight, encompassing weight, N5 |
| --- | --- | --- | --- |
| E34 | `member:ability:DEBUT` | scored | withheld (D21) |
| E37 | `member:pedigree_prior:DEBUT` | scored | withheld (D21) |
| E34 | `member:workouts_jrdb:DEBUT` | scored | withheld (D21) |
| E34 | `member:workouts_jravan:DEBUT` | scored | withheld (D21) |
| E34 | `member:body_weight:DEBUT` | scored | withheld (D21) |
| E34 | `member:late_win:DEBUT` | scored | withheld (D21) |
| E34 | `combined:DEBUT` | scored | withheld (D21) |
| E34 | `member:ability:MAIDEN` | scored | withheld (D21) |
| E37 | `member:pedigree_prior:MAIDEN` | scored | withheld (D21) |
| E34 | `member:workouts_jrdb:MAIDEN` | scored | withheld (D21) |
| E34 | `member:workouts_jravan:MAIDEN` | scored | withheld (D21) |
| E34 | `member:body_weight:MAIDEN` | scored | withheld (D21) |
| E34 | `member:late_win:MAIDEN` | scored | withheld (D21) |
| E34 | `combined:MAIDEN` | scored | withheld (D21) |

## Alternatives rejected and why

- A fit pooled over both classes: the members' meaning differs for a debut runner (no run, no weight change), so each class has its own fit.
- A member's raw values without a missing flag: a debut runner lacks some members by construction, and the flag keeps that apart from a value at the mean.
- The JV-Data mining member (#406): not landed; it is listed and enters as a counted variant when it lands.
- Settling a race without a Win dividend row at its final odds, and cutting the late Win boards by the JRA-VAN post's minutes to post: both were v1's, replaced in this family (the board cut at the study's JRA post; such a race left out of the curve).

## What this rules out

Withheld (D21): the measured gains, weights and curves are in the private report.

## Leaderboard rows

One row per variant above, by its name; built on licensed sources, so no rank or figure is shown here (D21).

## Follow-ups filed

- #406: the JV-Data mining member, a counted variant when it lands.
- #410: the Place settlement of the N5 curve.
