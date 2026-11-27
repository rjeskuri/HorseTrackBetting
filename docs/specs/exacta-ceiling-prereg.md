# N2: the exacta look-ahead ceiling on final boards, pre-registration (#431)

Status: **FROZEN** before any scored read. Claude pre-registers and freezes the study and runs it
without a sign-off stop (owner reframe, 2026-11-27); the owner reviews the leaderboard. The frozen
specs are `horsetrack.countries.japan.research.exacta_ceiling.family_specs()`, plain JSON;
`uv run horsetrack japan exacta-ceiling freeze` freezes each with `Registry.freeze_family` in
`$HORSETRACK_DATA_DIR/registry/japan.jsonl` (append-only: a freeze cannot be undone). The hash is
the sha256 of each spec's sorted, compact JSON. `run_ceiling` reads no store row unless every
family is frozen with exactly its spec, so a changed design is a new family, frozen before it runs.

| Family | Role | Members (family size) | Spec sha256 |
| --- | --- | --- | --- |
| `japan-431-exacta-ceiling-v2` | the 16 members at #115's full-development λ2, p from the final Win board (the primary rows) | 2 categories × 4 cuts × 2 cells (16) | `ad9fd640840b856429af188a7c56b83820f4a89cc58b6cb86e6ddcdfb9122d4f` |
| `japan-431-exacta-ceiling-plain-harville-v2` | variant: the same 16 at plain Harville, λ2 = 1, p from the final Win board | 16 | `c5cfc892e5f71fcbbcf0579bb6d7985e0d35249180c815c010e3524ca081f90a` |
| `japan-431-exacta-ceiling-descriptive-v2` | descriptives (D, the Quinella × order split) on the v2 p, not ranked | 4 records | `33c5115be87fa9e64f5027d6e8a1b5347a920b4861bfdd8439280fea36e041f7` |
| `japan-431-exacta-ceiling` | v1, superseded by `-v2`: the 16 at full-development λ2, p from #114's `p_cal`; recorded, ranked, not executed again | 16 | `acf0cd5499bbdf30a1bae4cb072e0a84c256a01a618f3a490c93ccd4175ccfc3` |
| `japan-431-exacta-ceiling-plain-harville` | v1, superseded by `-plain-harville-v2`: the same 16 at λ2 = 1, p from #114's `p_cal`; recorded, ranked, not executed again | 16 | `5016c7045665463f4191a9f6a4c11e2c1408f7165eeb02d31c09a2dba4e44beb` |
| `japan-431-exacta-ceiling-descriptive` | v1, superseded by `-descriptive-v2`: descriptives on #114's `p_cal`; recorded, not executed again | 4 records | `5877f77e68d434d873fec240c685b9432437a6129576f8869a8affdff85348d8` |

The v2 specs are `family_specs()`, the v1 specs `superseded_specs()` (unchanged, so their hashes
still check); `exacta-ceiling freeze` freezes both (a v1 freeze again is a no-op).

**No figure from the JRA-VAN store appears here or in any public file (D21)**: counts, rates,
spans, signs, bounds and ranks live in the private report under `$HORSETRACK_DATA_DIR/reports/`
and in the private registry.

## Revision v2 (2026-11-28, Sol review round 1 of 24d080f)

v1 took p_c from #114's out-of-fold calibrated `p_cal` and scored only the races with a #114 row:
the inputs of #131's rule-v4. #131's frozen E2 is now `japan-131-exacta-rule-v5` (frozen
2026-11-28, `docs/specs/exacta-overload-prereg.md`): p_c(t) is discounted Harville at the
registered λ2 on the **normalised implied Win shares of the Win board**, scoring every day with
Win and exacta boards, and rule-v4 is superseded. N2 must score #131's frozen E2 cuts, so v2 is
rule-v5's look-ahead analogue, and changes these inputs only:

- **p_c:** discounted Harville at λ2 on the normalised implied Win shares of the **final** Win
  board (`final_odds` WIN: 1 / odds over the race's sum of 1 / odds, over the horses with a
  positive final Win odds). No fitted Win model enters; λ2 is still #115's `full_development`
  fit (and 1 for the plain-Harville variant).
- **Span:** every development day from 2014-06-07 with final Win and exacta boards; no #114 row
  is required.
- **Refunds:** a race whose exacta pool (`pool_votes` FORECAST `__pool__`) has refund votes is a
  refunded exacta and is excluded and counted (`refund`, with the diagnostic
  `exacta_pool_refund_votes`). A scratch after sales refunds the tickets on the horse while the
  JRA-VAN dividends carry no refund row (`is_refund` is always false there), so v1's
  `is_refund` test alone could score such a race; with v2 no combination on a scratched horse is
  scored as a loss or a win.
- **Seed:** the bootstrap seed and replicate count are in every record's params and in the
  private report's name.

The members, cuts, cells, λ2, favourite, stake, return, estimand, interval and ranking are v1's.
The descriptives use p, so they get a v2 family too; so does the plain-Harville variant, so that
its λ2 comparison is not confounded with the p source. **Keep everything:** v1's three families
and their records stay in the registry untouched; v1's 32 members remain ranked variant rows (the
private ranked file reads their latest scored records, labelled p source `p_cal` and their span),
and v1 is never executed again.

## Look-ahead, everywhere

Every selection here reads the **final** exacta and Win boards. The final board holds money placed
after T-2, the decision time of #131's E2 rule. A look-ahead selection is therefore not a betting
rule. It is also **not a proven upper bound** on the T-2 rule: informed late money can let a T-2
selection catch combinations before they shorten, which a final-board selection never buys. That
the look-ahead result bounds the T-2 rule is an **assumption**, a row of `docs/ledger.md`.

## Hypothesis

E43, as #131, read on the final boards: some exacta category, fav_second (the favourite second)
or no_fav (without the favourite), is underbet against the Win-derived discounted Harville
probability far enough that buying its combinations when `q_c / p_c < k` returns more than
break-even at the final dividend. The owner's idea it tests: "an exacta on a contender to beat
the favourite with the favourite second, or the reverse exacta with the favourite to win". The
null, the starting point: the exacta pool prices each category as the Win pool implies, so every
member's ρ is about 1, below break-even (takeout and breakage in).

## Members (16) and the variant (16 more)

#131's E2 members unchanged: category X ∈ {fav_second, no_fav} × cut k ∈ {0.6, 0.7, 0.8, 0.9} ×
cell ∈ {all races, headline} = **16**, family `japan-431-exacta-ceiling-v2` (v1:
`japan-431-exacta-ceiling`). fav_first is never bought (E2 does not buy it).

- **Rule (look-ahead):** buy exacta c of category X for ¥100 when `r_c = q_c / p_c < k`.
- **q_c:** `implied_share(final exacta odds × 100, rate)` = rate / ((D + 5) / 100), half a ¥10
  breakage step added back; NaN at or below ¥110 (never selected). The final exacta odds are the
  confirmed `final_odds` FORECAST rows.
- **p_c (v2):** discounted Harville (`exacta_consistency.harville_exacta`) on the normalised
  implied Win shares of the final Win board, as #131's rule-v5 on the T-2 board (the revision
  above). v1 used #114's out-of-fold calibrated `p_cal` over the starters, renormalised (rule-v4's
  source). λ2 = #115's single full-development fit (label `full_development`, owner 2026-11-26;
  partly in-sample, the fit id is in each record).
- **λ2 variant:** the same 16 members at plain Harville, λ2 = 1, family
  `japan-431-exacta-ceiling-plain-harville-v2` (v1: `japan-431-exacta-ceiling-plain-harville`;
  #131 registered both). No other variant is added: the real lower tertile of the final exacta
  pool totals as the headline's "less popular" is left out to keep the family small and
  comparable.
- **Variants searched: 64**, every member family of the study: 16 per family, v2's two (the run)
  and v1's two (superseded, recorded). Family size 16 each. Every row of the private report and
  of each run's export carries `family_size` (its own family), `variants_frozen` (the study's
  64) and `variants_searched`, the variants that run actually scored: its own 32 and each v1
  member with a scored record in the registry. A v1 member without one keeps an unavailable row
  (status and reason, ranked last) and is not counted, so no output claims more than it scored.
- **Favourite:** the lowest final Win odds (`final_odds` WIN), ties to the lowest number; the
  category is `combo_category` against it.
- **Cells:** all scored races, and #131's headline cell (`exacta_study.segment_cells`): the
  favourite's final Win odds ≥ 3.0 and a less popular race (race number 1-4, or a DEBUT or MAIDEN
  race, from the race card).

## Stake, return and the estimand

- **Stake:** ¥100 on each selected combination of a race.
- **Return:** at the final dividend D with own impact from the race's final votes, the payout
  identity: `stake / 100 × D × (P + n s) / P × W / (W + s)`, n the combinations bought, P the
  exacta pool's final votes net of refund votes × ¥100 (`pool_votes` `__pool__`), W the paying
  combination's final votes × ¥100; not rounded again.
- **ρ** = Σ returned / Σ (staked × rate) over the member's races with a bet, rate the race's exacta
  payout rate (`takeout.race_takeout(race_id, "FORECAST", race_class_raw=...)`, Premium boosts
  in). ρ = 1 returns the payout rate; **break-even is ρ = 1 / rate** (about 1.33 at rate 0.75).
- **Secondary:** the return per ¥100 staked, 100 × Σ returned / Σ staked, with its upper bound.
- **Interval:** #131's meeting bootstrap (`exacta_study.rho_interval`): whole meetings drawn with
  replacement, 10,000 replicates, the run's seed (default 431). The one-sided 95% **upper bound**
  is the CI column (the owner's 2026-11-27 reading); the one-sided 95% lower bound is reported
  beside it. A member with fewer than two meetings has no bound.

## Ranking (no verdict)

Owner reframe, 2026-11-27: no kill rule, no open/closed verdict, no sign-off stop. Every frozen
member is a leaderboard row: its look-ahead ρ, its one-sided 95% upper bound as the CI column, and
the family size and the variants-searched count. `CeilingResult.rows` holds v2's 16 members in
rank order (ρ descending; a member without a bet after every member with one); the private ranked
file ranks all 64 together, each row labelled with its p source and span. Nothing is parked,
dropped or adopted on the result.

## Descriptives (not ranked)

Family `japan-431-exacta-ceiling-descriptive-v2` (v1: `japan-431-exacta-ceiling-descriptive`),
one record per descriptive and λ2 (4):

- **D** = Σ q_c / Σ p_c over the priced combinations with an identified q_c of each category
  (fav_first, fav_second, no_fav) and cell (all, headline), over the members' scored races.
- **The Quinella × order split:** each final Quinella pair share (`implied_share` at the race's
  QUINELLA payout rate) is split into (i, j) and (j, i) by Harville's order probabilities at λ2;
  per category and cell, Σ exacta shares / Σ split Quinella shares. A combination counts when both
  its exacta share and its pair's Quinella share are identified. 1 means the exacta board orders
  each pair as Harville does on the Quinella money.

Each with a two-sided 95% meeting-bootstrap interval (10,000 replicates).

## Data and exclusions

- **Source:** the jravan store: `final_odds` (WIN, QUINELLA, FORECAST), `pool_votes` (FORECAST
  combination votes and the pool total), `dividends` (FORECAST), `races`; confirmed records only
  (データ区分 4 and 5) for `final_odds` and `pool_votes`.
- **Span:** development days only (`is_holdout_day` False, filtered inside each scan before any
  odds, votes or dividend is read; `LockedHoldout` is never opened), none before 2014-06-07 (D10),
  read one year at a time; v2: every race with final Win and exacta boards (v1: the races with
  #114 out-of-fold Win probabilities, its walk-forward test quarters). The measured span is in
  the private report only.
- **Dropped before scoring (counted), v2:** race status other than run; race class outside
  `exacta_study.CLASS_GROUPS`; no priced final Win board; takeout unknown. (v1 also dropped a
  race without #114 out-of-fold probabilities or whose priced final Win field was not #114's
  horses, and counted a race whose final Win board's implied shares differed from #114's `q` by
  more than 1e-9 as a diagnostic.)
- **#131's exclusions (`exacta_study.exclusion_reason`, counted):** a race whose boost may be
  withheld; fewer than 5 starters; no exacta row; a refunded exacta (v2: a paying row with
  `is_refund`, or refund votes on the exacta pool's `__pool__` row); more than one paying exacta
  (dead heat). Also excluded and counted: segments undefined (field outside 5-18); no priced exacta
  board; no exacta pool total; no final money on the paying combination.

## Registry and outputs

- Registry: `default_registry("japan")`, `sources=("jravan",)`, hypothesis E43, ticket 431. One
  record per member (32) and per descriptive (4), status `scored`, each with the run's `seed`
  and `n_boot` in its params; a member without a bet keeps its record with the marker
  `rho:unavailable:no combination selected`. A failed run records each member as `failed`.
- **No eligible race:** a run whose span scores no race records each member as `failed` with the
  marker `run:unavailable:no eligible race in the asked development days` and stops: no scored
  record, no private report, no ranked member file and no public report is written, so a short
  date-range check never replaces the run of record's outputs.
- Private (through `write_report`, source `jravan`): each run's report and export
  `$HORSETRACK_DATA_DIR/reports/exacta-ceiling-japan-<commit>-<build>-seed<seed>-<from>-<to>.md`
  and `.json`, named by the run's commit, data build, seed and asked days, so no run (another
  seed, a shorter span) overwrites another's. The export carries the run's identity (commit,
  build, seed, asked days, whether it is the run of record) and the first and last development
  day it scored; each row of a run's own families carries those scored days as its span, and a
  v1 row names the run it comes from. The ledger-linked ranked member file
  `$HORSETRACK_DATA_DIR/reports/candidate-cells/exacta-ceiling.json` is written only by the run
  of record: seed 431 over the asked days 2014-06-07 to 2026-11-27 (`RECORD_DAYS`); any other
  run leaves it alone. Each ranks all 64 members, each row with its p source, status, span,
  run, family size, the variants searched and the variants frozen.
- Public: `docs/experiments/exacta-ceiling-japan.md` (every figure withheld, D21), the registry
  summary under `docs/experiments/registry/japan/` (`write_summary`, never hand-edited) and the
  ledger rows.
