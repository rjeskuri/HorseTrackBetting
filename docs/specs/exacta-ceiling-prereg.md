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
| `japan-431-exacta-ceiling` | the 16 members at #115's full-development λ2 | 2 categories × 4 cuts × 2 cells (16) | `acf0cd5499bbdf30a1bae4cb072e0a84c256a01a618f3a490c93ccd4175ccfc3` |
| `japan-431-exacta-ceiling-plain-harville` | variant: the same 16 at plain Harville, λ2 = 1 | 16 | `5016c7045665463f4191a9f6a4c11e2c1408f7165eeb02d31c09a2dba4e44beb` |
| `japan-431-exacta-ceiling-descriptive` | descriptives (D, the Quinella × order split), not ranked | 4 records | `5877f77e68d434d873fec240c685b9432437a6129576f8869a8affdff85348d8` |

**No figure from the JRA-VAN store appears here or in any public file (D21)**: counts, rates,
spans, signs, bounds and ranks live in the private report under `$HORSETRACK_DATA_DIR/reports/`
and in the private registry.

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
cell ∈ {all races, headline} = **16**, family `japan-431-exacta-ceiling`. fav_first is never
bought (E2 does not buy it).

- **Rule (look-ahead):** buy exacta c of category X for ¥100 when `r_c = q_c / p_c < k`.
- **q_c:** `implied_share(final exacta odds × 100, rate)` = rate / ((D + 5) / 100), half a ¥10
  breakage step added back; NaN at or below ¥110 (never selected). The final exacta odds are the
  confirmed `final_odds` FORECAST rows.
- **p_c:** discounted Harville (`exacta_consistency.harville_exacta`) on #114's out-of-fold
  calibrated Win probabilities `p_cal` over the starters, renormalised: #131's source, so the
  ceiling is comparable with E2. λ2 = #115's single full-development fit (label
  `full_development`, owner 2026-11-26; partly in-sample, the fit id is in each record).
- **λ2 variant:** the same 16 members at plain Harville, λ2 = 1, family
  `japan-431-exacta-ceiling-plain-harville` (#131 registered both). No other variant is added: the
  real lower tertile of the final exacta pool totals as the headline's "less popular" and a
  longer span from the final Win price are left out to keep the family small and comparable.
- **Variants searched: 32** (16 per member family). Family size 16 each.
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
the family size as the variants-searched count. `CeilingResult.rows` holds the 16 members in rank
order (ρ descending; a member without a bet after every member with one); the private ranked file
ranks all 32 together. Nothing is parked, dropped or adopted on the result.

## Descriptives (not ranked)

Family `japan-431-exacta-ceiling-descriptive`, one record per descriptive and λ2 (4):

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
  read one year at a time; the races with #114 out-of-fold Win probabilities (its walk-forward
  test quarters). The measured span is in the private report only.
- **Dropped before scoring (counted):** no #114 out-of-fold probabilities; race status other than
  run; race class outside `exacta_study.CLASS_GROUPS`; a priced final Win field other than #114's
  horses; takeout unknown. A race whose final Win board's implied shares differ from #114's `q` by
  more than 1e-9 keeps its `p_cal` and is counted as a diagnostic.
- **#131's exclusions (`exacta_study.exclusion_reason`, counted):** a race whose boost may be
  withheld; fewer than 5 starters; no exacta row; a refunded exacta; more than one paying exacta
  (dead heat). Also excluded and counted: segments undefined (field outside 5-18); no priced exacta
  board; no exacta pool total; no final money on the paying combination.

## Registry and outputs

- Registry: `default_registry("japan")`, `sources=("jravan",)`, hypothesis E43, ticket 431. One
  record per member (32) and per descriptive (4), status `scored`; a member without a bet keeps
  its record with the marker `rho:unavailable:no combination selected`. A failed run records each
  member as `failed`.
- Private (through `write_report`, source `jravan`): the report
  `$HORSETRACK_DATA_DIR/reports/exacta-ceiling-japan-<commit>-<build>.md` and `.json`, and the
  ranked member file `$HORSETRACK_DATA_DIR/reports/candidate-cells/exacta-ceiling.json`.
- Public: `docs/experiments/exacta-ceiling-japan.md` (every figure withheld, D21), the registry
  summary under `docs/experiments/registry/japan/` (`write_summary`, never hand-edited) and the
  ledger rows.
