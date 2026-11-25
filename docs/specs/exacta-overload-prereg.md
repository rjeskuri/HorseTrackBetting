# Exacta favourite-overload study: pre-registration (#131)

> **PROPOSED: awaiting owner sign-off, not frozen.** Nothing here is in the run registry. No
> family has been frozen, no data has been read and no run has been scored. The target, estimand,
> metric, segments, band edges and cuts are owner decisions (CLAUDE.md, "Modeling governance").
> Every value marked *proposal* needs a yes or a replacement. Once approved, the spec is frozen with
> `default_registry("japan").freeze_family(...)` before any scored run.

Code: `src/horsetrack/countries/japan/research/exacta_consistency.py` (pure maths, no data reads).

## 1. Question and null

The owner's idea (2026-11-24): in less popular races with a vulnerable favourite, buy exactas
with the favourite second or without the favourite. The underlying claim is that the JRA exacta
pool (`FORECAST`) **overloads the favourite's combinations** compared with what the Win pool and
an ordering model imply, and that the gap survives takeout, breakage and own price impact.

Null (the starting point): the exacta pool prices every category correctly. A box of any category
then returns the race's payout rate less breakage, so it loses about 25%.

## 2. Data scope

- Development days only. A day is used only if `is_holdout_day("japan", d)` is False, and the
  filter runs before any outcome or dividend is read. A2 holdout days (from 2026-11-28) are never
  read (D14). `LockedHoldout` is not opened by this study.
- Public JRA data only: the store's `races`, `runners` and `dividends` (pool `FORECAST`), and the
  Win odds. No JRDB or JRA-VAN input or figure (D21).
- Races from 2014-06-07 onward, the start of the single takeout history. Before that date
  `race_takeout` raises `TakeoutUnknown`.
- Payout rate per race: `takeout.race_takeout(race_id, "FORECAST", race_class_raw=...).payout_rate`,
  never a constant 0.75. *Proposal:* races where `boost_may_be_withheld` is set are excluded.
- *Proposal:* also excluded are races with a dead heat for first or second (two paying exactas), a
  refunded exacta (`is_refund`), or fewer than 5 starters.
- Win probabilities `p`: #114's out-of-fold calibrated Win probabilities. Ordering model: discounted
  Harville with λ2 fitted in-fold (#115, `ordering_v0.yaml`). `harville_exacta` reuses
  `ordering._order_probs`, so the study and the v0 model share one formula.
- Precondition, checked by counting rows only (no scoring): the store holds `FORECAST` dividend
  rows for the development period. If it does not, the historical half waits on an ingest ticket.

## 3. Quantities

For race r with favourite f (`favourite`: lowest Win odds, ties to the lowest number) and an
exacta c = (first, second):

- pool share `q_c = rate_r / ((D_c + 5) / 100)` (D5 primary baseline, half a ¥10 breakage step
  added back, NaN at or below ¥110: `implied_share`);
- model probability `p_c`: discounted Harville from the Win probabilities (D5 secondary baseline:
  `harville_exacta`);
- category S ∈ {fav_first, fav_second, no_fav} (`combo_category`);
- the true finishing distribution π, seen only through the winner w.

## 4. The estimand gap (the decision this spec turns on)

The ticket's quantity is the per-race category ratio `Σ_{c∈S} q_c / Σ_{c∈S} p_c`
(`race_ratios(...).categories`). It needs `q_c` for every combination, which means the full final
exacta board. The store's `dividends` table holds **only the winning combination's dividend**, so on
history this ratio cannot be computed. Full final boards exist only on days #130 logged, a handful
so far. The options:

| | Estimator (history: winner only) | What it estimates | Verdict |
|---|---|---|---|
| A | mean of `q_w / p_w` over races whose winner is in S | `Σ_S π q/p ÷ Σ_S π`. This equals the ticket's ratio only if Harville is the truth. If the pool is exactly right (π = q) it gives `Σ q²/p ÷ Σ q`, which is above 1 whenever q/p varies. | **Reject.** Selecting on the winner makes it report an overload where none exists. |
| B1 | flat box: stake 1/\|S\| on each c in S, return `1[w∈S]·D_w/100/\|S_r\|` | `rate·Σ_S π_c/q_c ÷ \|S\|`, which is `rate` under π = q | Unbiased and realisable, but very noisy: longshot exactas dominate no_fav. |
| B2 | **Harville-weighted box**: stake `p_c / Σ_S p` on each c in S, return `1[w∈S]·(p_w/Σ_S p)·D_w/100` | `rate·Σ_S π_c p_c/q_c ÷ Σ_S p_c`. Under π = q it is `rate`. Under π = p it is rate × the p-weighted mean of p/q, roughly the inverse of the ticket's ratio. | **Recommend as primary.** |
| B3 | a flat box at Harville-fair odds: return `1[w∈S]/(p_w·\|S_r\|)` | `Σ_S π/p ÷ \|S\|`: Harville's own calibration on S | Diagnostic only, reported next to B2. |
| C | `log q_w − log p_w`, paired per race | `Σ π log(q/p)`: the log-score gain of the pool over Harville | Secondary, **totals only**. Split by category it is not proper: a forecast that overloads S gains on S. |
| D | the full-board ratio `Σ_S q / Σ_S p` (`race_ratios`) | the ticket's quantity: pool against model, no outcomes needed | Descriptive, on #130 days only. Shows consistency, not value. |

Why B2:

- It is an actual bet (stakes known from the Win board), so it is judged at the final dividends
  that pay, with breakage and the minimum payout included, as D18 requires.
- It needs only the winner's dividend. Every combination in S is staked, so selecting on the
  winner adds no bias.
- Harville weights keep its variance far below the flat box.
- Its null value is `rate` for every S, whether or not Harville is right.

The cross-pool question ("overload relative to Win plus ordering") is read from B2 and B3 together:

- B2 < rate on fav_first with B3 ≈ 1: the exacta pool overloads the favourite.
- B2 and B3 both below 1: the bias sits in the Win pool or the ordering model, not the exacta pool.

D is reported on the logged days alongside B2 for the same races, as a check that the two agree in
direction.

Estimand (*proposal*): `ρ_S = Σ_r R^{B2}_{S,r} / Σ_r rate_r`, where the sum runs over the races of
a segment.

- `ρ_S < 1` on fav_first means the pool overloads the favourite-first combinations.
- `ρ_S > 1` on fav_second or no_fav means those combinations return more than the takeout allows.
- A betting profit additionally needs `Σ R / Σ stake > 1`, that is `ρ_S > 1/rate` (about 1.33).

## 5. Families (to freeze in the registry, all variants listed before scoring)

**Family E1, category returns on history** (*proposed id* `japan/131-exacta-category-v1`).

- Members: 3 categories × 14 segment cells (section 6), using estimator B2. 42 members.
- Per-race loss: `rate_r − R_{S,r}` for fav_second and no_fav, and the sign flipped for fav_first.
  So every member's one-sided alternative is the hypothesised direction (fav_first overloaded,
  the others underloaded).
- Benchmark loss: 0.
- B1 and B3 are registered as separate reported families, not part of E1's max test. Variants: B1
  is E1′, B3 is E1″.
- C (total) is a single paired comparison.

**Family E2, ratio-cut rules** (D18 positive-edge; *proposed id* `japan/131-exacta-rule-v1`).

- Rule form: "at T−2, buy exacta c in category X ∈ {fav_second, no_fav} when
  `r_c(t) = q_c(t)/p_c(t) < k`".
  - `q_c(t)` comes from the exacta board at t: `implied_share(odds × 100, rate=...)`.
  - `p_c(t)` is Harville on the Win board at t.
  - Selection uses only the board at t. A selection variable taken from final odds is look-ahead.
- Stake (*proposal*): ¥100 per selected combination. The rule is judged at the final dividend,
  with own impact from the payout identity using that race's logged pool totals.
- Cuts k (*proposal*): 0.6, 0.7, 0.8, 0.9.
- Scope: all races, plus the "vulnerable favourite in a less popular race" cell. 2 categories ×
  4 cuts × 2 scopes = 16 members.
- Benchmark: no bet (loss 0).
- **History cannot score E2:** only the winner's `q_c` is known, so the selection set is
  unobservable. E2 is frozen now and scored only on #130-logged development days, once there are
  at least *proposal* 100 meetings. Until then it is reported as "not yet scorable".
- Buying fav_first is not in E2. *Proposal:* a loss-avoidance rule (D18: "do not buy fav_first when
  r_c(t) is above a cut") becomes a follow-up ticket.

## 6. Segments (*proposal*: marginal cells, not crossed, plus one pre-stated cross)

| Segment | Source | Candidate levels (proposal) |
|---|---|---|
| all races | | 1 |
| favourite's Win odds (final for E1, at t for E2) | Win board | ≤ 1.9 · 2.0–2.9 · 3.0–4.9 · ≥ 5.0 |
| field size | `races.field_size` (starters) | 5–10 · 11–14 · 15–18 |
| race class | `races.race_class` | newcomer and maiden · 1–3 win classes · open, listed and graded |
| pool size ("less popular") | see below | less popular · rest |
| owner's headline cell | cross | favourite odds ≥ 3.0 **and** less popular |

That is 1 + 4 + 3 + 3 + 2 + 1 = 14 cells.

The store holds no historical pool totals, so "less popular" needs a proxy. *Proposal:* races 1–4
of the day, or a newcomer or maiden race.

On #130 days the real exacta pool total exists (the odds log's `investment` row). There,
*proposal:* the lower tertile within each logged day defines "less popular", and the proxy's
agreement with it is reported.

## 7. Metric, inference and shrinkage

- Intervals: meeting (venue × day) cluster bootstrap, 95% one-sided, `n_boot` 10,000
  (`config/scoring.yaml`, D7).
- Multiple testing:
  - Hansen SPA ("does any member depart in its stated direction") and Romano-Wolf StepM ("which"),
    both over meetings (`core/eval/snooping.py`, D17).
  - StepM level 0.95. `block_size` has no default and needs an owner value (*proposal*: 5
    meetings, about one weekend's venues, for serial dependence).
  - Family sizes (42 for E1, 16 for E2) are reported with every result, failed variants included.
- Hierarchical shrinkage (D17) uses `core/eval/segment_roi.py`, which needs one row per bet with
  its odds and outcome.
  - E2 meets that on logged days: each selected combination is a bet, and the fixed effects are
    exacta odds bands.
  - E1 on history does not, because losing combinations' dividends are unknown. *Proposal:* shrink
    E1's 14 segment estimates per category with a normal-normal empirical-Bayes model on their
    bootstrap standard errors, and state that this differs from `segment_roi`.
- Own impact on history: *proposal*, a ledger assumption that ¥100 per race is negligible, checked
  against logged pool totals.
- Every run, failures included, is a registry record. Outputs go through `write_report`.

## 8. Owner questions to sign off before the freeze

1. Estimand for history: B2 (Harville-weighted box return ρ_S) as primary, with B1 and B3
   reported, C (totals only) secondary, D descriptive on logged days, and A rejected?
2. Directions: fav_first tested for ρ < 1 and fav_second and no_fav for ρ > 1 (one-sided, as in
   E1)? Or two-sided for all three?
3. Band edges: the favourite-odds, field-size and class levels in section 6, the "less popular"
   proxy (races 1–4 or newcomer and maiden), and the headline cross (favourite ≥ 3.0 and less
   popular)?
4. E2: cuts k ∈ {0.6, 0.7, 0.8, 0.9}, ¥100 per combination, at T−2, and a minimum of 100 logged
   meetings before scoring?
5. SPA and StepM `block_size` (proposal: 5 meetings).
6. E1 shrinkage: normal-normal empirical Bayes on bootstrap standard errors, since `segment_roi`
   needs losing bets' odds?
7. Exclusions: boost-withheld-possible races, dead heats for first or second, refunds, fewer than
   5 starters?
8. λ2: the in-fold fitted λ2 only, or plain Harville (λ2 = 1) also, as a registered sensitivity
   variant that adds to the family size?
