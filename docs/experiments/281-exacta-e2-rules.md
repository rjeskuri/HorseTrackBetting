# Experiment report: #131's E2 exacta ratio-cut rules on logged boards (#281)

Scores the 16 frozen members of `japan/131-exacta-rule-v3` (E2; spec
[docs/specs/exacta-overload-prereg.md](../specs/exacta-overload-prereg.md) sections 5-7, frozen
hash `9104e173371668a34b4594b6e4970546649b886cdccc4e726ec9749188701f0d`, the sha256 of
`exacta_study.family_specs()[E2_ID]` on #131's head; v3 takes the favourite and the category from
the Win board at T−2) on the #130-logged development meetings that carry exacta (`FORECAST`)
boards. Registry hypothesis id `E43`; every record is in
[registry/japan/japan--131-exacta-rule-v3.md](registry/japan/japan--131-exacta-rule-v3.md).
Code: `countries/japan/research/exacta_rules.py`, command `uv run horsetrack japan exacta-rules`,
which re-scores everything logged so far on each run (another run of the same 16 variants, not
new variants; a rerun on unchanged inputs and code adds nothing). Nothing is adopted; no rule is
staked. Leaderboard rows wait for #449.

**Small n is not refused.** The frozen spec says E2 is scored "only once there are min_meetings"
(`min_meetings: 100`) and is "not yet scorable" until then. The owner's reframe of 2026-11-27
(rank everything, keep every result, no gates) supersedes that line: E2 is scored on whatever is
logged, and zero meetings give an n = 0 result with its records.

## Hypothesis

Stated before scoring (spec section 5): at T−2, buying exacta c of category X (favourite second,
or without the favourite) when the pool's implied share is below k times the Harville probability
from the Win board, `q_c(t) / p_c(t) < k` with k in {0.6, 0.7, 0.8, 0.9}, returns more than it
stakes at the final dividend after own impact and breakage, in all races or in the owner's
headline cell (favourite at 3.0 or more at T−2 in a less popular race). The null, the starting
point: no member beats no bet (loss 0); SPA over the 16 members does not reject and StepM finds
none.

## Run of 2026-11-27 (git `19f32a7`)

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

## Variants tried

All 16 members, each on the same 0 races (development days only):

| Category | Cut k | Scope | Bets | Races | Meetings | ROI | One-sided 95% bounds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fav_second | 0.6 | all | 0 | 0 | 0 | - | - |
| fav_second | 0.6 | headline | 0 | 0 | 0 | - | - |
| fav_second | 0.7 | all | 0 | 0 | 0 | - | - |
| fav_second | 0.7 | headline | 0 | 0 | 0 | - | - |
| fav_second | 0.8 | all | 0 | 0 | 0 | - | - |
| fav_second | 0.8 | headline | 0 | 0 | 0 | - | - |
| fav_second | 0.9 | all | 0 | 0 | 0 | - | - |
| fav_second | 0.9 | headline | 0 | 0 | 0 | - | - |
| no_fav | 0.6 | all | 0 | 0 | 0 | - | - |
| no_fav | 0.6 | headline | 0 | 0 | 0 | - | - |
| no_fav | 0.7 | all | 0 | 0 | 0 | - | - |
| no_fav | 0.7 | headline | 0 | 0 | 0 | - | - |
| no_fav | 0.8 | all | 0 | 0 | 0 | - | - |
| no_fav | 0.8 | headline | 0 | 0 | 0 | - | - |
| no_fav | 0.9 | all | 0 | 0 | 0 | - | - |
| no_fav | 0.9 | headline | 0 | 0 | 0 | - | - |

No member failed; none bet. ROI is total payout over total stake; its interval, once there are
bets, is a meeting bootstrap with a one-sided 95% bound in each direction (as #131's E1).

- **D, the full-board ratio** `Σ_S q / Σ_S p` on the final boards: 0 races in each of
  fav_first, fav_second and no_fav; no mean or median.
- **"Less popular" proxy against the final-pool tertile:** 0 races on 0 days; no agreement
  figure.
- **Shrinkage (`core/eval/segment_roi.py`, spec section 7): not run, no bet.** The command now
  runs it on every scored run (below); with no bet it records `not_run` and the reason.

## Readings of the spec, for the owner

Modelling items the frozen text leaves open, implemented as read here; none is a ruling.

- `p_c(t)`: the normalised 1/odds of the Win board at t, through discounted Harville at #115's
  full-development λ2 (the rule's text, "Harville on the Win board at t"). #114's calibrated
  `p_cal` (the spec's common `win_probabilities`) does not exist for logged days.
- **Headline scope (v3's frozen `scopes.headline`):** the favourite's Win odds at T−2 ≥ 3.0 and
  the race in the lower tertile (at or below the 1/3 quantile, linear interpolation) of the
  FORECAST pool total *at T−2* within its logged day, all venues of the day together, over every
  logged race of the day with a total at T−2, whatever its E2 eligibility. The final pool would
  be look-ahead for a T−2 selection, so the total at T−2 is used. **Cross-race look-ahead, for
  the owner:** a day's cutoff uses every race's T−2 total, and a later race's T−2 total is not
  known at an earlier race's T−2. A strictly real-time scope would need a cutoff from earlier
  races (or earlier days) only; this is not implemented, since the frozen text says "within each
  logged day".
- The race 1-4 / newcomer-or-maiden proxy is not the scope; it is only compared with the lower
  tertile of the *final* pool totals of each day, over every loaded logged race with a final
  FORECAST capture (not only the scored races).
- **Own impact with breakage and the minimum (D18; v3's `primary_metric`, "takeout and breakage
  in"):** the winner's money is estimated from the published dividend D with its ¥10 rounding
  added back, `W = (1 − t) P / ((D + 5) / 100)` (the half-step convention of `implied_share`);
  the unrounded dividend after own impact is `100 (1 − t)(P + n s) / (W + s)`, and the dividend
  that pays is that rounded down to ¥10, never below ¥110. P is the race's FORECAST pool total
  on its final capture, n the member's own number of bets in the race, t = 1 − the race's payout
  rate.
- **No final FORECAST capture:** the #130 scheduler can stop before the final board. Such a race
  is loaded with no pool total and no final exacta board, excluded as `no_final_board`, and left
  out of D; a pre-final total is never used in its place.
- **Minimum starters:** the five-starter exclusion counts the actual starters (`runners.parquet`
  runners not scratched); the T−2 board is used only for selection. A race with no runner rows in
  the store falls back to the runners priced on the final Win board (counted in the run's `read`
  block).
- Each member is a standalone rule, settled with its own bet count (a 0.9 cut buys more
  combinations than 0.6, so its own impact is larger).
- A board price of 1.1 is ¥110 (float noise snapped), so the ¥110 floor leaves its share
  unidentified and the combination is never bought.
- SPA and StepM run over all 16 members whenever a race is scored, members that bet nothing
  included (snooping requires the full family); with fewer than 3 meetings arch cannot run and
  the failed max test is recorded under `japan/131-exacta-rule-v3/max-test`, its reason in the
  Markdown report.
- **Shrinkage:** `fit_segment_roi` on every member's bets (one row per member × race × selected
  combination with a price on the final board, its final-board odds and outcome), fixed effects
  exacta odds bands with edges **10, 30 and 100** (< 10, 10-30, 30-100, ≥ 100 on the final
  board: roughly the favourite pairs, the middle of the board, the outsiders and the long shots),
  crossed random effects category, cut and scope, seed 131. The members' bets overlap (nested
  cuts, headline within all), so one bet enters up to eight rows and the posterior is narrower
  than the data warrant; the spec's E1 shrinkage carries the same caveat. A fit that raises (for
  example a band whose bets all won) is recorded as `failed` with its reason.

## Review requests not done, by owner decision

- **Refusing E2 below 100 meetings** (the frozen `min_meetings`): not done. The owner's reframe
  of 2026-11-27 (rank everything, no gates) supersedes it; E2 is scored on any n.
- **Scoring `japan/131-exacta-rule-v1` or v2 instead of v3:** not done. The owner's 2026-11-26
  answer on #281 was to take the hash from #280's (#131's) final head, which freezes v3.

## Alternatives rejected and why

- Waiting for 100 meetings before scoring (the frozen `min_meetings`): superseded by the owner's
  reframe of 2026-11-27.
- Selecting on the final boards: look-ahead (spec section 5); selection uses only the boards at t,
  with a row available at its `fetched_utc`, else its `snapshot_utc` (docs/specs/board-at-t.md).
- The final pool tertile for the headline scope: look-ahead at T−2.
- One shared bet count across members for own impact: it would charge a narrow rule for a wide
  rule's stakes.
- A pre-final pool total when the final FORECAST capture is missing: it understates P and pairs a
  pre-final pool with the official dividend.

## What this rules out

Nothing. With 0 logged FORECAST meetings no member is measured, so E2 is neither ruled in nor
ruled out, and #131's E1 result (no favourite overload on history) stands alone.

## Follow-ups filed

- None filed by this run. Re-run `uv run horsetrack japan exacta-rules` once development days
  with #130 FORECAST boards are logged (holdout days from 2026-11-28 are never read); that run
  reports the members, the max test, D, the proxy agreement and the shrinkage.
