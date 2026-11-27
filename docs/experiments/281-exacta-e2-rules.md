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

## Run of 2026-11-27 (git `483f790`), the current code

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
  parquet files, so this run's builds differ from theirs although no consumed input changed;
  from now on a store refresh that adds only holdout or unlogged races changes no build id and
  appends nothing.
- No max test (no race scored); shrinkage `not_run` (no bet) for each of the 16 members.
- The records now in the family: 48, three runs of the same 16 variants at `ac74012`,
  `19f32a7` (both below) and `483f790`, all n = 0; the earlier two stay as reruns.

## Earlier run of 2026-11-27 (git `19f32a7`)

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

All 16 members, each on the same 0 races (development days only). Headline rows are
retrospective (the readings below say why):

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
  The command fits it per member on every run; with no bet it records `not_run` and the reason.

## Readings of the spec, for the owner

Modelling items the frozen text leaves open, implemented as read here; none is a ruling.

- `p_c(t)`: the normalised 1/odds of the Win board at t, through discounted Harville at #115's
  full-development λ2 (the rule's text, "Harville on the Win board at t"). #114's calibrated
  `p_cal` (the spec's common `win_probabilities`) does not exist for logged days.
- **Headline scope (v3's frozen `scopes.headline`):** the favourite's Win odds at T−2 ≥ 3.0 and
  the race in the lower tertile (at or below the 1/3 quantile, linear interpolation) of the
  FORECAST pool total *at T−2* within its logged day, all venues of the day together, over every
  logged race of the day with a total at T−2, whatever its E2 eligibility. The final pool would
  be look-ahead for a T−2 selection, so the total at T−2 is used. **The headline results are
  retrospective.** A day's cutoff uses every race's T−2 total, and a later race's T−2 total is
  not known at an earlier race's T−2, so an early race's eligibility can change when afternoon
  captures arrive: the headline ROI could not have been earned by a bettor acting at T−2. The
  frozen v3 definition ("within each logged day") is kept, because changing it would be a new
  family, and it is labelled: each headline member carries `retrospective: true` in the result,
  each record a `retrospective_cutoff` metric (1 for headline, 0 for "all"; a metric, never a
  parameter, so the family stays 16), and the Markdown report marks every headline row
  "retrospective". The "all" members use no cross-race information. A real-time scope (a cutoff
  from earlier races or earlier days only) would be a new pre-registered family for the owner.
- The race 1-4 / newcomer-or-maiden proxy is not the scope; it is only compared with the lower
  tertile of the *final* pool totals of each day, over every loaded logged race with a final
  FORECAST pool total, whether or not E2 can score it (a race with no board at T−2 counts).
- **D's sample:** every loaded logged race with final Win and FORECAST prices, whatever its pool
  total and its boards at T−2. The ratio needs only final prices; the JRA parser writes the pool
  total row only when the page prints a ticket count, so a final capture can hold every price
  and no total, and such a race is kept for D (and excluded from E2 as `no_final_board`).
- **No board at T−2:** a race whose log has FORECAST rows but nothing available at T−2 (for
  example its first capture is the final one) is loaded with empty boards at t and its final
  boards, counted by D and the proxy, and excluded from E2 as `no_board_at_t`.
- **Own impact with breakage and the minimum (D18; v3's `primary_metric`, "takeout and breakage
  in"):** the winner's money is estimated from the published dividend D with its ¥10 rounding
  added back, `W = (1 − t) P / ((D + 5) / 100)` (the half-step convention of `implied_share`);
  the unrounded dividend after own impact is `100 (1 − t)(P + n s) / (W + s)`, and the dividend
  that pays is that rounded down to ¥10, never below ¥110. P is the race's FORECAST pool total
  on its final capture, n the member's own number of bets in the race, t = 1 − the race's payout
  rate.
- **JRA's Plus-10 exception:** JRA tops a winning ticket up to ¥110 only when the pool can fund
  it; otherwise it pays ¥100. A published ¥100 therefore leaves the winning share unidentified
  (the payout identity cannot recover W), so such a winner is settled at the published ¥100,
  never raised to ¥110 whatever the stake, `settle` reports `impact_available: false`, and the
  run counts those scored races as `impact_unavailable` (0 in this run).
- **No final FORECAST capture:** the #130 scheduler can stop before the final board. Such a race
  is loaded with no pool total and no final exacta board, excluded as `no_final_board`, and left
  out of D; a pre-final total is never used in its place.
- **Build ids:** `data_builds` has four keys, `odds_log` (the logs read and their sizes) and
  `jra_races`, `jra_runners`, `jra_dividends`, each a digest of only the rows the command read
  from that table for the logged development races (and, for races, their meetings' cards). Rows
  of holdout or unlogged races change no build id, so a store refresh that adds only those
  appends nothing; a changed consumed row is a new build of its own table only.
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
  `p_beats_takeout`), `not_run` (no bet) or `failed` (the fit raised, for example a band whose
  bets all won at short odds) with the reason.

## Review requests not done, by owner decision

- **Refusing E2 below 100 meetings** (the frozen `min_meetings`): not done. The owner's reframe
  of 2026-11-27 (rank everything, no gates) supersedes it; E2 is scored on any n.
- **Scoring `japan/131-exacta-rule-v1` or v2 instead of v3:** not done. The owner's 2026-11-26
  answer on #281 was to take the hash from #280's (#131's) final head, which freezes v3.
- Both decisions are cited in comments in `exacta_rules.py` beside the hash check and the
  member scoring loop.

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
- One shrinkage fit pooled over the 16 members: nested members hold the same tickets, so one
  outcome counted up to eight times and overstated the certainty of beating takeout.
- Raising a published ¥100 winner to ¥110: under the Plus-10 exception ¥100 means the top-up was
  not funded, and own impact cannot be identified from it.
- Store build ids over whole parquet files: they changed with rows the command never reads.

## What this rules out

Nothing. With 0 logged FORECAST meetings no member is measured, so E2 is neither ruled in nor
ruled out, and #131's E1 result (no favourite overload on history) stands alone.

## Follow-ups filed

- None filed by this run. Re-run `uv run horsetrack japan exacta-rules` once development days
  with #130 FORECAST boards are logged (holdout days from 2026-11-28 are never read); that run
  reports the members, the max test, D, the proxy agreement and the shrinkage.
