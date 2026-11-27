# Experiment report: #131's E2 exacta ratio-cut rules on logged boards (#281)

Scores the 16 frozen members of `japan/131-exacta-rule-v4` (E2, `exacta_study.E2_ID`; spec
[docs/specs/exacta-overload-prereg.md](../specs/exacta-overload-prereg.md) sections 5-7, frozen
hash `28929cd783cd9e25ae8c7f39f30f2b0891720b60864a38fdbb13d7dcd2a8d3bc`, the sha256 of
`exacta_study.family_specs()[E2_ID]` on #280's head `66a2338`) on the #130-logged development
meetings that carry exacta (`FORECAST`) boards. v4 supersedes v3 (owner decision 2026-11-27 on
#131); the owner's 2026-11-26 answer on #281 is to take the hash from #280's final head. Registry
hypothesis id `E43`. Code: `countries/japan/research/exacta_rules.py`, command
`uv run horsetrack japan exacta-rules`, which re-scores everything logged so far on each run
(another run of the same 16 variants, not new variants; a rerun on unchanged inputs and code adds
nothing). Nothing is adopted; no rule is staked. Leaderboard rows wait for #449.

**Small n is not refused.** The owner's reframe of 2026-11-27 (rank everything, keep every
result, no gates) holds, and v4's spec has no meeting gate: E2 is scored on whatever is logged,
and zero meetings give an n = 0 result with its records.

**The v4 real run waits for #280.** No registry family id may contain "/" (#525), so this code
records under `japan-131-exacta-rule-v4` (`E2_RULE_ID`) and checks the freeze under that id. The
Japan registry holds v4's freeze only under the slash id, so the command refuses ("not frozen")
until #280 re-freezes under the dashed id; that is by design, not a failure to fix here. The
runs below are v3 runs, kept as history.

## Hypothesis

Stated before scoring (spec section 5): at T−2, buying exacta c of category X (favourite second,
or without the favourite) when the pool's implied share is below k times the Harville probability
from the Win board, `q_c(t) / p_c(t) < k` with k in {0.6, 0.7, 0.8, 0.9}, returns more than it
stakes at the final dividend after own impact and breakage, in all races or in the owner's
headline cell (favourite at 3.0 or more at T−2 in a less popular race). The null, the starting
point: no member beats no bet (loss 0); SPA over the 16 members does not reject and StepM finds
none.

## The v4 contract (this code)

- **Ids.** Members under `E2_RULE_ID = E2_ID.replace("/", "-")` (`japan-131-exacta-rule-v4`),
  the SPA + StepM record and a failed run's record under `MAX_TEST_ID = E2_RULE_ID + "-max-test"`.
  Nothing is appended under an id with "/".
- **All 16 members are decision-time rules.** The headline scope is v4's: the favourite's Win
  odds on the board at T−2 ≥ 3.0 and the race "less popular" on its race card (race_no 1-4 or a
  DEBUT/MAIDEN race, E1's popularity segment). No pool total enters any selection, so a later
  race's capture cannot change an earlier race's membership or any metric. The v3
  `retrospective` label and `retrospective_cutoff` metric are gone. The 16-member SPA/StepM is
  the decision-time max test.
- **The day's pool tertile is a diagnostic only.** `proxy_agreement` compares the race-card proxy
  with the lower tertile of the final pool totals of each race day (v4's
  `diagnostics.less_popular_agreement`); the loader no longer computes a `less_popular` flag.
- **Floor-bound dividends.** A published ¥100 (Plus-10 not funded) or ¥110 (the minimum, possibly
  a Plus-10 top-up) does not identify the winning investment, so its impact-adjusted payout is
  unavailable (NaN; `impact_identified` false, `settle` returns `impact_available: false`). The
  published return is kept apart: `payout_published` on every settle, bet and member, and a
  `roi_published` metric on every member that bet. A member with a floor-bound hit has no
  impact-adjusted ROI (no `roi` metric, ROI and its bounds unavailable): dropping only those
  races would bias its ROI toward losses. `impact_unavailable` counts the scored races whose
  winning dividend is floor-bound. In the max test such a race enters at its published payout
  (under Plus-10 a ¥110 floor still binds after our ¥100 stake; boost-withheld races are already
  excluded), and the max-test record carries `impact_unavailable_races` as a metric.
- **D's coverage.** D counts a race only when its final exacta board prices every ordered pair of
  the final Win board's horses (n (n − 1) pairs over the non-scratched starters); a board missing
  a pair (absent or unpriced) is counted in `incomplete_boards` and left out of every category's
  mean. A missing pool total alone excludes nothing; a race with no final exacta prices at all is
  not D's and not incomplete; prices for horses off the final Win board are not required.
- **Absent post times.** A logged race with final WIN/FORECAST prices and no post time (no
  sidecar, no stored card) is loaded with its final boards and `no_decision_time`; it counts in
  D, the proxy and `logged_meetings_with_forecast`, and E2 excludes it as `no_post_time`.
- **Shrinkage status in the registry.** Each member record carries, as metrics, `n_meetings`,
  exactly one of `shrinkage_fitted`, `shrinkage_failed` or `shrinkage_not_attempted` (value 1),
  and `shrinkage_n_bets` (the rows given to the fit, 0 when not attempted). Params hold none of
  them, so a run whose fits fail and one whose fits succeed are the same 16 variants.
- **The ordering artifact is pinned.** The command reads the ordering file `current.json` names,
  takes λ2 from the bytes it hashes, and records the sha256 (first 12 hex digits) in the
  `odds_log` build id: `odds_snapshots/jra/FORECAST@<logs>+ordering-<sha12>`. A refreshed
  `current.json` naming other content is a visibly different input, and earlier records keep the
  old hash. The fit's training window is the set of race dates in the out-of-fold file its
  manifest names (`oof.path`); when its last date is on or after the earliest scored race date
  the run is refused before any member record is appended, and the refusal is a failed record
  under `MAX_TEST_ID`. On the data directory as of 2026-11-27 the current fit's out-of-fold races
  run through 2026-11-15, before both logged days (2026-11-21 and 2026-11-22).
- **Data identity.** `data_builds` keeps four keys. `odds_log` digests, per log read, the sha256
  of the bytes of the whole members consumed and of the post times resolved for its meeting
  (sidecar or stored card), plus the ordering hash above; the `odds_snapshots/jra/FORECAST@`
  prefix stays, so a refresh is another run of the same variants. `jra_races`, `jra_runners` and
  `jra_dividends` digest only the rows read from each table: the logged development races' rows
  and, for races, the cards of their meetings. Rows of holdout races and of races outside the
  loaded meetings change no build id; a race added to the card of a loaded meeting does change
  `jra_races`, logged or not (the earlier wording, "unlogged races change no build id", was too
  broad).
- **Report names.** Each run writes `e2-<sha12>-<digest12>.json` and `.md` under the reports
  directory, the digest over the run's `data_builds` and `components`, so two runs at one commit
  on different inputs (more logs, another ordering artifact) keep both reports. No latest
  pointer is written: the newest report is the newest file.

### Round-4 changes to the v4 contract (supersede the bullets above where they differ)

- **One dashed freeze id.** `exacta_run freeze` (#131's command) also freezes E2's v4 spec under
  `japan-131-exacta-rule-v4` (`exacta_run.E2_REGISTRY_ID`, which `E2_RULE_ID` now is), so the real
  freeze-then-score workflow runs without a hand-made freeze. The slashed freeze stays for #131.
- **The fit is provenance.** The command's component is the model (`ordering:
  discounted_harville`); the fit id and artifact hash sit after the "@" of the `odds_log` build:
  `odds_snapshots/jra/FORECAST@<logs>+ordering-<fit_id>-<sha12>`. A refit is another run of the
  same 16 members, and the family size (and the max test's) stays 16.
- **Out-of-fold path.** A relative `oof.path` in the ordering manifest resolves against the
  working directory, as the fitter (`models/v0/ordering.py::run_fit`) opened and saved it; one not
  found there is tried in the model dir. The fitter is unchanged.
- **No published stand-in in the own-impact max test.** When any member's hit is floor-bound
  (¥100 or ¥110), the own-impact SPA + StepM is unavailable: a failed `spa+stepm` record with the
  reason in `params["unavailable"]` and no p-value (our stake on the losers can lift a ¥110 floor
  to ¥120, so the published figure is no own-impact payout). Beside it, `spa+stepm-published`
  under `MAX_TEST_ID` is a sensitivity over the same 16 members with every hit at its published
  dividend (as `roi_published`); it adds no variant to the rule family. A floor-bound winner no
  member selected leaves the own-impact test intact.
- **D needs identified shares.** A final board quoting any pair of the final field at 1.1 or less
  is no full board: it is counted in `incomplete_boards` and left out of every category.

### Round-5 changes to the v4 contract (supersede round 4 where they differ)

- **The out-of-fold file by data dir and sha.** A relative `oof.path` resolves against the data
  dir (`HORSETRACK_DATA_DIR`), then the model dir, never the working directory, so a stale copy
  under the same name in the cwd cannot set the training window. Each candidate's bytes must be
  the manifest's `oof.sha256` (the fitter records it); a mismatch is skipped, an absolute path is
  checked the same way, and when no candidate matches the run is refused (a failed record).
- **Pre-upgrade records keep their variants.** The previous revision (`8a58d47`) wrote the fit id
  as the ordering component. In a registry holding those records, the command reuses the earliest
  one's components, so a refit or more logged days is another run of the same 16 members and the
  max test reports family size 16. The registry is not rewritten and `core.eval` is unchanged;
  the fit actually used is in the `odds_log` build's refresh and the report's `ordering_artifact`.
  A fresh registry gets `ordering: discounted_harville` as in round 4.

## Run of 2026-11-27 (git `483f790`), v3, history

- Inputs as in the run below: 2 development day folders, 2 odds logs, 0 `FORECAST` rows, 0
  holdout folders opened. n = 0 races loaded, 0 scored, 0 meetings, 0 excluded, 0 races with a
  published ¥100 winner (`impact_unavailable`).
- Registry: 16 more records under `japan/131-exacta-rule-v3` (06:19Z), status `scored`, the
  frozen `spec_hash`, metrics `n_bets` = 0 (n = 0) and `retrospective_cutoff` (1 for the 8
  headline members, 0 for the 8 "all" members, n = 0), no `roi`; family size still 16 (the label
  is a metric, not a parameter, so these are reruns of the same 16 variants). Run ids (first 12
  characters, in member order): `9885dc541e65`, `20c42cff93ff`, `a8455718928b`, `d8095052d664`,
  `48c7bd2f5f8e`, `22770dc3c8ca`, `e41f70799659`, `1ea8d7ad1701`, `9c4ee956c880`,
  `c7ed5a32ad14`, `a76baf9cac5f`, `05bd08941987`, `fdeb913266a4`, `366f3e91a62b`,
  `0e9459651379`, `8f8001b01865`.
- Data builds: the odds log digest `aa7b950687591d09` (unchanged) and, per store table, a digest
  of the rows the command consumed: with no logged race nothing is read from the store, so all
  three are the digest of no rows (`e3b0c44298fc1c14`). The earlier runs fingerprinted the whole
  parquet files, so this run's builds differ from theirs although no consumed input changed.
- No max test (no race scored); shrinkage `not_run` (no bet; `not_attempted` in the v4 code) for
  each of the 16 members.
- The records now in the family: 48, three runs of the same 16 variants at `ac74012`,
  `19f32a7` (both below) and `483f790`, all n = 0; the earlier two stay as reruns.

## Earlier run of 2026-11-27 (git `19f32a7`), v3, history

- Inputs read: 2 development day folders (2026-11-21 and 2026-11-22, Kyoto), 2 odds logs, 0
  holdout folders opened (the A2 holdout starts 2026-11-28; `LockedHoldout` not opened).
- **Logged development meetings with FORECAST boards: 0.** Neither log holds a single `FORECAST`
  row (they hold WIN, PLACE, QUINELLA and QUINELLA_PLACE only): both days predate #130's exacta
  logging, as #283's own-impact run found (ledger). So n = 0 races loaded, 0 scored, 0 meetings,
  0 races excluded.
- Registry: 16 records under `japan/131-exacta-rule-v3`, status `scored`, each with `spec_hash`
  equal to the frozen hash, metric `n_bets` = 0 with n = 0 and no `roi` metric; family size 16.
  Run ids (first 12 characters, in member order): `f51c221e8bae`, `fcbaad045303`,
  `e4126867d9b2`, `62d149109e89`, `2eea923ef220`, `d495fcf98387`, `3b23107f5f12`,
  `beaf1a8b197f`, `85ad61235918`, `5d86c134109f`, `1f690f42daa2`, `70fb0d7e23d1`,
  `9c69e6ab0aaf`, `de8c1b3df62f`, `7cd191f3950f`, `de68b71d96f0`. Ordering component
  `2b822d643cf7` (#115's full-development fit, λ2 = 0.8104). Data builds: the odds log digest
  and the digests of `processed/jra/{races,runners,dividends}.parquet`, the same four keys
  whether or not a race is logged, so a later populated run is another run of the same 16
  variants.
- An identical run a minute earlier (05:47Z, run ids `c885711c731f` … `65e61bd232de`) is recorded
  at git `ac74012`, the branch head before this code was committed (the working tree held the
  code of `19f32a7`); it stays in the registry as a run, not a variant (family size 16).
- The max test (SPA + StepM, block 5 meetings, meetings as clusters, 10,000 replicates, seed 131)
  did not run: it runs when any race is scored, and none was. No `/max-test` record; the report
  states `not_run` and why.

### The superseded v2 run

16 records under `japan/131-exacta-rule-v2` (git `7ae109e`, 2026-11-27 05:12Z, run ids
`1b511e6a65c1` … `252d21af2697`, all n = 0) were written before this ticket was retargeted to
v3. They carry v2's hash (`2d6678f5…cdb75`), whose frozen text takes the favourite from the final
Win odds, while that code took it from the board at T−2; v3 (frozen 05:03Z) is the spec that
code implemented. They stay in the registry (append-only; their status reads `scored`) and
count as a failed, superseded variant run: n = 0, so no figure is affected. Their summary is in
[registry/japan/japan--131-exacta-rule-v2.md](registry/japan/japan--131-exacta-rule-v2.md).

## Variants tried (the v3 runs above, history)

All 16 v3 members, each on the same 0 races (development days only). Under v3 the headline rows
were labelled retrospective (v3's whole-day pool tertile; v4 replaces it with the race card):

| Category | Cut k | Scope | Bets | Races | Meetings | ROI | One-sided 95% bounds | Shrinkage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fav_second | 0.6 | all | 0 | 0 | 0 | - | - | not run |
| fav_second | 0.6 | headline (retrospective) | 0 | 0 | 0 | - | - | not run |
| fav_second | 0.7 | all | 0 | 0 | 0 | - | - | not run |
| fav_second | 0.7 | headline (retrospective) | 0 | 0 | 0 | - | - | not run |
| fav_second | 0.8 | all | 0 | 0 | 0 | - | - | not run |
| fav_second | 0.8 | headline (retrospective) | 0 | 0 | 0 | - | - | not run |
| fav_second | 0.9 | all | 0 | 0 | 0 | - | - | not run |
| fav_second | 0.9 | headline (retrospective) | 0 | 0 | 0 | - | - | not run |
| no_fav | 0.6 | all | 0 | 0 | 0 | - | - | not run |
| no_fav | 0.6 | headline (retrospective) | 0 | 0 | 0 | - | - | not run |
| no_fav | 0.7 | all | 0 | 0 | 0 | - | - | not run |
| no_fav | 0.7 | headline (retrospective) | 0 | 0 | 0 | - | - | not run |
| no_fav | 0.8 | all | 0 | 0 | 0 | - | - | not run |
| no_fav | 0.8 | headline (retrospective) | 0 | 0 | 0 | - | - | not run |
| no_fav | 0.9 | all | 0 | 0 | 0 | - | - | not run |
| no_fav | 0.9 | headline (retrospective) | 0 | 0 | 0 | - | - | not run |

No member failed; none bet. ROI is total payout over total stake; its interval, once there are
bets, is a meeting bootstrap with a one-sided 95% bound in each direction (as #131's E1).

- **D, the full-board ratio** `Σ_S q / Σ_S p` on the final boards: 0 races in each of
  fav_first, fav_second and no_fav; no mean or median.
- **"Less popular" proxy against the final-pool tertile:** 0 races on 0 days; no agreement
  figure.
- **Shrinkage (`core/eval/segment_roi.py`, spec section 7): not run for any member, no bet.**

No v4 member has been scored on real data yet (see the top of this report).

## Readings of the spec, for the owner

Modelling items the frozen text leaves open, implemented as read here; none is a ruling.

- `p_c(t)`: the normalised 1/odds of the Win board at t, through discounted Harville at #115's
  full-development λ2 (the rule's text, "Harville on the Win board at t"). #114's calibrated
  `p_cal` (the spec's common `win_probabilities`) does not exist for logged days.
- **Headline scope (v4's frozen `scopes.headline`):** the favourite's Win odds on the board at
  T−2 ≥ 3.0 and `race_no` ≤ 4 or `race_class` DEBUT or MAIDEN, read from the frozen spec's
  `fav_odds_min` and `less_popular` fields. (v3's headline, the lower tertile of each logged
  day's T−2 pool totals, used later races' totals and was labelled retrospective; v4 retires it.)
- The race-card proxy is compared with the lower tertile of the *final* pool totals of each day,
  over every loaded logged race with a final FORECAST pool total, whether or not E2 can score it
  (a race with no board at T−2 or no post time counts).
- **D's sample:** every loaded logged race whose final exacta board covers the final field (see
  the contract above), whatever its pool total, its post time and its boards at T−2. The JRA
  parser writes the pool total row only when the page prints a ticket count, so a final capture
  can hold every price and no total; such a race is kept for D (and excluded from E2 as
  `no_final_board`).
- **No board at T−2:** a race whose log has FORECAST rows but nothing available at T−2 (for
  example its first capture is the final one) is loaded with empty boards at t and its final
  boards, counted by D and the proxy, and excluded from E2 as `no_board_at_t`.
- **Own impact with breakage and the minimum (D18; "takeout and breakage in"):** the winner's
  money is estimated from the published dividend D with its ¥10 rounding added back,
  `W = (1 − t) P / ((D + 5) / 100)` (the half-step convention of `implied_share`); the unrounded
  dividend after own impact is `100 (1 − t)(P + n s) / (W + s)`, and the dividend that pays is
  that rounded down to ¥10, never below ¥110. P is the race's FORECAST pool total on its final
  capture, n the member's own number of bets in the race, t = 1 − the race's payout rate.
- **JRA's Plus-10 exception and the ¥110 floor:** JRA tops a winning ticket up to ¥110 only when
  the pool can fund it; otherwise it pays ¥100. A published ¥100 leaves the winning share
  unidentified, and so does a published ¥110, which may be a top-up rather than a rounded
  dividend. Both are floor-bound: impact-adjusted payout unavailable, published return kept (see
  the contract). (The v3 code settled a ¥100 winner at the published ¥100 and treated ¥110 as an
  ordinary dividend.)
- **No final FORECAST capture:** the #130 scheduler can stop before the final board. Such a race
  is loaded with no pool total and no final exacta board, excluded as `no_final_board`, and left
  out of D; a pre-final total is never used in its place.
- **Minimum starters:** the five-starter exclusion counts the actual starters (`runners.parquet`
  runners not scratched); the T−2 board is used only for selection. A race with no runner rows in
  the store falls back to the runners priced on the final Win board (counted in the run's `read`
  block).
- Each member is a standalone rule, settled with its own bet count (a 0.9 cut buys more
  combinations than 0.6, so its own impact is larger).
- A board price of 1.1 is ¥110 (float noise snapped), so its share is unidentified and the
  combination is never bought.
- SPA and StepM run over all 16 members whenever a race is scored, members that bet nothing
  included (snooping requires the full family); with fewer than 3 meetings arch cannot run and
  the failed max test is recorded under `japan-131-exacta-rule-v4-max-test`, its reason in the
  Markdown report.
- **Shrinkage, per member:** `fit_segment_roi` on each member's own bets alone (one row per race
  × selected combination with a price on the final board, its final-board odds and outcome),
  fixed effects exacta odds bands with edges **10, 30 and 100** (< 10, 10-30, 30-100, ≥ 100 on
  the final board: roughly the favourite pairs, the middle of the board, the outsiders and the
  long shots), seed 131. The members' bets overlap (nested cuts, headline within all); an
  earlier version pooled them in one fit with crossed random effects category, cut and scope,
  where one winning ticket entered up to eight rows and the posterior was narrower than the data
  warrant. Fitted alone, a ticket is evidence once per member, and another member selecting the
  same tickets cannot change a member's fit. `fit_segment_roi` requires a segment column; a
  member's rows have no segment to shrink across, so the column is constant (`member`, the
  variant name: one level, its random effect held near 0 by the τ prior), and the member's
  `p_beats_takeout` covers all its bets. Each member reports `fitted` (bets, band effects,
  `p_beats_takeout`), `not_attempted` (no bet with a final-board price) or `failed` (the fit
  raised, for example a band whose bets all won at short odds) with the reason, and the status
  is a record metric.

## Review requests not done, by owner decision

- **Refusing E2 below 100 meetings** (v3's frozen `min_meetings`): not done. The owner's reframe
  of 2026-11-27 (rank everything, no gates) supersedes it, and v4 drops the gate.
- **Scoring `japan/131-exacta-rule-v1`, v2 or v3 instead of v4:** not done. The owner's 2026-11-26
  answer on #281 was to take the hash from #280's (#131's) final head, which now freezes v4.
- Both decisions are cited in comments in `exacta_rules.py` beside the hash check and the
  member scoring loop.

## Alternatives rejected and why

- Waiting for 100 meetings before scoring (v3's `min_meetings`): superseded by the owner's
  reframe of 2026-11-27.
- Selecting on the final boards: look-ahead (spec section 5); selection uses only the boards at t,
  with a row available at its `fetched_utc`, else its `snapshot_utc` (docs/specs/board-at-t.md).
- The final pool tertile, or a whole-day T−2 pool tertile, for the headline scope: look-ahead at
  T−2 (v3's labelled-retrospective headline is retired by v4).
- One shared bet count across members for own impact: it would charge a narrow rule for a wide
  rule's stakes.
- A pre-final pool total when the final FORECAST capture is missing: it understates P and pairs a
  pre-final pool with the official dividend.
- One shrinkage fit pooled over the 16 members: nested members hold the same tickets, so one
  outcome counted up to eight times and overstated the certainty of beating takeout.
- Settling a floor-bound winner at an impact-adjusted figure: the published ¥100 or ¥110 does not
  identify the winning investment. Dropping those races from a member's ROI: it biases the ROI
  toward losses; the ROI is unavailable instead, and the published return stays visible.
- Following `current.json` on each rescore without recording the artifact: a refit would change
  earlier selections silently. The ordering identity as a component: it would make a refit 16 new
  variants of the family; it is an input identity in `data_builds` instead.
- Store build ids over whole parquet files: they changed with rows the command never reads.
- An odds log id over file names and sizes: a corrected sidecar or a same-length rewrite kept the
  same id.
- One report per commit (`e2-<sha>.json`): a same-commit rescore overwrote the earlier report.

## What this rules out

Nothing. With 0 logged FORECAST meetings no member is measured, so E2 is neither ruled in nor
ruled out, and #131's E1 result (no favourite overload on history) stands alone.

## Follow-ups filed

- None filed by this change. The v4 real run (`uv run horsetrack japan exacta-rules`) waits for
  #280 to freeze v4 under `japan-131-exacta-rule-v4`; until then the command refuses by design.
  It then reports the members, the max test, D, the proxy agreement and the shrinkage on the
  development days logged with #130 FORECAST boards (holdout days from 2026-11-28 are never
  read).
