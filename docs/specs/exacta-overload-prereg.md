# Exacta favourite-overload study: pre-registration (#131)

> **Approved by the owner 2026-11-25; frozen.** The owner accepted all nine points of section 8
> as proposed (comment on #131, 2026-11-25). The families were frozen in the Japan run registry
> with `default_registry("japan").freeze_family(...)` on 2026-11-25, before any scored run. The
> frozen specs are `exacta_study.family_specs()`; the hash is the sha256 of each spec's sorted,
> compact JSON.
>
> **v2, relabelled 2026-11-27.** The owner, 2026-11-26 on #131: "Relabel λ2 honestly: it is
> #115's single full-development fit (partly in-sample), not 'in_fold'. Use a label such as
> `full_development` in the frozen spec and the report." The registry is append-only and refuses
> another spec under a frozen id, so the relabelled families were frozen on 2026-11-27 under the
> `-v2` ids below, beside the v1 records, which stay as frozen. v1 is superseded only for the
> label: each v2 spec is its v1 spec with the λ2 label `full_development` in place of `in_fold`,
> the ordering text saying the fit is partly in-sample, and two added keys (`relabel`, the
> owner's answer; `supersedes`, the v1 id). Members, family sizes, directions, segments, cells,
> cuts, block size, replicates, level, seed and exclusions are v1's. The scored runs were
> re-run under v2 on 2026-11-27 (git `5d68473`) and gave v1's values.

| Family | Role | Members (family size) | Spec hash |
|---|---|---|---|
| `japan-131-exacta-category-v2` | E1: B2 category returns, the max test | λ2 {`full_development`, 1} × 3 categories × 14 cells (84) | `eb900ab7144a6c88b5ebf4058d6aaf22570a9a5fb06d7803dbaa7a7cf89bf247` |
| `japan-131-exacta-category-b1-v2` | E1′: B1 flat box, reported | 3 × 14 (42) | `fb6844a9a189ee689f6390b6f2fde58a318370b9e73834ab5b28fbad735ec49b` |
| `japan-131-exacta-category-b3-v2` | E1″: B3 Harville calibration, reported | λ2 {`full_development`, 1} × 3 × 14 (84) | `493253276d7a803446716325353bb6c18e01409868dc6dbec53eec09a9b4912f` |
| `japan-131-exacta-logscore-v2` | C: log-score total | λ2 {`full_development`, 1} (2) | `084244a2f3cb69154da1dc97615044b58e0cf534f8077ed1cbd459a98e3650bd` |
| `japan-131-exacta-rule-v5` | E2: ratio-cut rules | 2 categories × 4 cuts × 2 scopes (16) | `37d8abb85db2bf8480f619b46440e0ea80a5ed68af992ef344b64e0b36dc2817` |
| `japan-131-exacta-rule-v4` | E2, superseded by v5: history, not executed | 2 categories × 4 cuts × 2 scopes (16) | `28929cd783cd9e25ae8c7f39f30f2b0891720b60864a38fdbb13d7dcd2a8d3bc` |
| `japan/131-exacta-rule-v3` | E2, superseded by v4: history, not executed | 2 categories × 4 cuts × 2 scopes (16) | `9104e173371668a34b4594b6e4970546649b886cdccc4e726ec9749188701f0d` |

`japan-131-exacta-rule-v5` (frozen 2026-11-28, GPT-6.2 Sol review round 5 of 2c73ca0)
supersedes `japan-131-exacta-rule-v4`. v4 inherited the E1 families' Win probabilities (#114's
out-of-fold calibrated `p_cal`, built from final Win odds) and their eligibility (races with a
#114 out-of-fold row): look-ahead for a rule decided at T−2, and it left out logged days #114
does not cover. v5 changes two inputs and nothing else:

- **`win_probabilities`:** the normalised implied Win shares on the Win board at T−2
  (1 / odds over the race's sum of 1 / odds, over the starters on that board), fed to the
  discounted Harville at the registered λ2 (`full_development`) to give `p_c(t)`. No model fit
  of final Win odds enters a selection.
- **`data`:** every #130-logged development day whose log holds the Win board and the exacta
  board at T−2, each race with both boards at T−2 scored; no #114 row is required.

`rule` restates the same decision rule with the v5 `p_c(t)`, and `revision` explains v5. The
members, cuts, scopes, selection time, favourite, category, stake, λ2, judging, metric,
bootstrap settings and exclusions are v4's. v4's spec is frozen under both
`japan/131-exacta-rule-v4` and `japan-131-exacta-rule-v4` (same hash, so v5's `supersedes`
names a freeze); it is history and is not executed. v4's text below is kept as written.

**Ids renamed 2026-11-27 (#525).** The registry refuses a `/` in a family id (it cannot name a
summary file), so the live families take the `japan-131-…` form above: the same specs, so the
same hashes, first frozen under the legacy ids `japan/131-exacta-category-v2`, `-b1-v2`,
`-b3-v2`, `japan/131-exacta-logscore-v2` and `japan/131-exacta-rule-v4`. The max test and
breakage records go to `japan-131-exacta-category-v2-max-test` and `japan-131-exacta-breakage`.
Records written before the rename stay under the legacy ids (their summary files map each `/` to
`--`), and each spec's `supersedes` names its legacy predecessor as recorded.

`japan-131-exacta-rule-v4` (frozen 2026-11-27, owner decision of 2026-11-27 on the review of
83684eb) supersedes `japan/131-exacta-rule-v3` (hash above). The v3 freeze stays in the registry
as history and is not executed. v4 changes three things and nothing else:

- **Headline "less popular":** section 6's race-card proxy, the same definition as E1's
  popularity segment and headline cell: race_no 1–4, or a DEBUT or MAIDEN race. Its
  `scopes["headline"]` names the favourite's Win odds at T−2 ≥ 3.0 (`fav_odds_min`), the proxy
  (`less_popular`: `race_no_max` 4, `race_classes` DEBUT and MAIDEN) and each input's
  `information_cutoff`: race_no and race_class from the race card, known before the day, and the
  favourite's Win odds from the board at T−2. No later race's pool total enters a race's
  selection.
- **Tertile:** the lower tertile of exacta pool totals within each logged day is reported only,
  as the proxy's agreement diagnostic (`diagnostics.less_popular_agreement`), never as a
  selection variable.
- **No meeting gate:** `min_meetings` is removed. `scored_on` is the #130-logged development days
  available at each run, reported with n and the CI and re-scored as more days are logged.

`scopes["all"]`, the members, cuts, stake, favourite, category, rule, selection time, metric,
λ2, bootstrap settings and exclusions are v3's; `revision` explains v4.

`japan/131-exacta-rule-v3` (frozen 2026-11-27) supersedes `japan/131-exacta-rule-v2` (hash
`2d6678f59291476bf3d3028b2818bc00e631dea516bc0f01276ac4e0af8cdb75`, frozen 2026-11-27, not
scored; its freeze stays in the registry): v2 took the favourite from the lowest final Win odds,
and v3 takes the favourite and the category from the Win board at T−2, as sections 5 and 6 say.
v3 adds `selection_time` (T−2), `category`, `scopes` (the headline: the favourite's Win odds at
T−2 ≥ 3.0 and less popular, by section 6's logged-day tertile), `primary_metric` (expected return
per ¥100 at the final dividend, the owner's 2026-11-27 reframe) and `revision`; nothing else
changes, `min_meetings` included.

> **rule-v3 is superseded by rule-v4 and not executed** (reviews of f4c7db6 and 83684eb,
> 2026-11-27). v3's headline scope ranked "less popular" by the lower tertile of pool totals
> within each logged day, so an early race could be classified with later races' totals, which
> are not available when it is selected. v4 resolves it: the cutoff is the race card, the same
> as E1's headline cell; the tertile is reported as an agreement diagnostic only; and there is
> no meeting gate (v3's `min_meetings` 100 is gone). Scoring E2 under v4 is #281.
>
> **rule-v4 is superseded by rule-v5 and not executed** (Sol review round 5 of 2c73ca0,
> 2026-11-28): its `p_c(t)` came from #114's final-odds calibration. #281 scores E2 under v5.

The superseded v1 families, frozen 2026-11-25 with the label `in_fold` for the same fit, and
scored that day (their records stay in the registry):

| Family (v1, superseded for the label) | Spec hash |
|---|---|
| `japan/131-exacta-category-v1` | `9ac3b740d1bbdde9645091d3fd14247442b1a210b2d109c5e58ade0983769a8f` |
| `japan/131-exacta-category-b1-v1` | `f0e915f51cc13c7b40ef478076f6c44b1b5d22b52722b321c46359e5c2bf5e8f` |
| `japan/131-exacta-category-b3-v1` | `07721b74cc11fd7181b812c42450ac523371e07435047c623cd9d159d0d3fbcb` |
| `japan/131-exacta-logscore-v1` | `0af318b97297b3085d82434921cf4c2f651864321a3e63ae03560b3e2b81c715` |
| `japan/131-exacta-rule-v1` | `b1577fd3765d5f72b0c0643235aca7cd69317d467470a0a31949da50c65a019f` |

Registry hypothesis id: `E43` (08 §4, "exotic pools are efficient relative to the Win pool and
the ordering model"). E1 and E2 are this spec's family labels, not 08's E1 and E2.

Code: `src/horsetrack/countries/japan/research/exacta_consistency.py` (pure maths),
`exacta_study.py` (the scoring maths and the frozen specs) and `exacta_run.py` (freeze, the row
count and the scored run).

## Deviations from the signed spec, accepted by the owner (2026-11-26)

Found after the scored run (2026-11-25). The owner, 2026-11-26 on #131: "accept both
deviations, documented, and land", with the λ2 relabel (the v2 note above).

- (a) λ2, owner-accepted 2026-11-26. The owner signed "λ2 in-fold fitted" (answer 8). The run
  used #115's full-development fit (λ2 = 0.8104, partly in-sample; fit `2b822d643cf7`): one fit
  over the development out-of-fold races, which saw the scored races' first two places. v1's
  specs labelled it `in_fold`; the v2 specs label it `full_development`, as the owner asked.
  Effect: B3 (E1″) and C at λ2 = 0.8104 are partly in-sample. B2's null (ρ = 1 under π = q,
  before rounding) does not depend on p, so E1's ranking and its SPA and StepM context stand. A per-fold refit for B3
  and C is #284.
- (b) Scored span, owner-accepted 2026-11-26 ("the scored span from 2020-01-05 stands as
  documented"). The signed scope runs from 2014-06-07 (section 2). The scored races start
  2020-01-05: 23,629 of the 43,058 development races, forced by #114's out-of-fold span (its
  walk-forward test quarters from 2020-Q1), since B2, B3 and C need #114's p.

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
  never a constant 0.75. Races where `boost_may_be_withheld` is set are excluded.
- Also excluded are races with a dead heat for first or second (two paying exactas), a refunded
  exacta (`is_refund`), no exacta row, or fewer than 5 starters.
- Win probabilities `p`: #114's out-of-fold calibrated Win probabilities, renormalised over the
  starters. They exist for #114's walk-forward test quarters only (2020-Q1 onward), so the scored
  races are the development races from 2020-01-01 that have them. Ordering model: discounted
  Harville at #115's full-development fit (λ2 = 0.8104, partly in-sample; registry label
  `full_development`, `in_fold` in v1; `ordering_v0.yaml`: one fit on the development
  out-of-fold races; its fit id is recorded with each run; see deviation (a)), and plain
  Harville (λ2 = 1) as a registered variant. `harville_exacta` reuses `ordering._order_probs`, so the study and the v0
  model share one formula.
- Precondition, checked by counting rows only (no scoring): the store holds `FORECAST` dividend
  rows for the development period. If it does not, the historical half waits on an ingest ticket.
  Count on 2026-11-25 (`exacta_run count`): 43,058 development races from 2014-06-07, every one
  with a `FORECAST` row (43,206 rows); #114's out-of-fold file covers 23,760 of them; the #115
  fit `2b822d643cf7` gives λ2 = 0.8104.

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
| B2 | **Harville-weighted box**: stake `p_c / Σ_S p` on each c in S, return `1[w∈S]·(p_w/Σ_S p)·D_w/100` | `rate·Σ_S π_c p_c/q_c ÷ Σ_S p_c`. Under π = q it is `rate`. Under π = p it is rate × the p-weighted mean of p/q, roughly the inverse of the ticket's ratio. | **Primary** (owner, 2026-11-25). |
| B3 | a flat box at Harville-fair odds: return `1[w∈S]/(p_w·\|S_r\|)` | `Σ_S π/p ÷ \|S\|`: Harville's own calibration on S | Diagnostic only, reported next to B2. |
| C | `log q_w − log p_w`, paired per race | `Σ π log(q/p)`: the log-score gain of the pool over Harville | Secondary, **totals only**. Split by category it is not proper: a forecast that overloads S gains on S. |
| D | the full-board ratio `Σ_S q / Σ_S p` (`race_ratios`) | the ticket's quantity: pool against model, no outcomes needed | Descriptive, on #130 days only. Shows consistency, not value. |

Why B2:

- It is an actual bet (stakes known from the Win board), so it is judged at the final dividends
  that pay, with breakage and the minimum payout included, as D18 requires.

> **Historical box results are final-Win-odds look-ahead** (Sol review round 5 of 2c73ca0,
> 2026-11-28). On history the B2 weights (#114's `p_cal`, built from final Win odds), the
> favourite and the favourite-odds bands all come from the final Win board, which is not known
> when an exacta is bought. So E1's B1, B2 and B3 results describe that historical selection
> (categories and cells defined on the final Win board), not a bet placeable before the off;
> their conclusions are limited to it. A decision-time version is E2 (rule-v5), whose inputs
> are all from the boards at T−2.
- It needs only the winner's dividend. Every combination in S is staked, so selecting on the
  winner adds no bias.
- Harville weights keep its variance far below the flat box.
- Its null value is `rate` for every S, whether or not Harville is right (with q the unrounded
  share; at the paid, rounded-down dividend it is slightly below `rate`).
- It is a p-weighted average over S: a value near `rate` can hide over- and underpriced
  combinations that offset, so it does not show that every combination is fairly priced.

The cross-pool question ("overload relative to Win plus ordering") was to be read from B2 and B3
together:

- B2 < rate on fav_first with B3 ≈ 1: the exacta pool overloads the favourite.
- B2 and B3 both below 1: the bias sits in the Win pool or the ordering model, not the exacta pool.

> **Withdrawn** (review of f4c7db6, 2026-11-27): B3 is the equal-weighted mean of `π/p` over S,
> while the cross-pool ratio `Σ_S q / Σ_S p` weights by p, so B3 can sit on the other side of 1
> from it even when outcomes follow the exacta pool. B3 is reported as an outcome-calibration
> diagnostic of Harville only; no direction of the cross-pool ratio is read from B2 and B3.

D is reported on the logged days alongside B2 for the same races, as a check that the two agree in
direction.

Estimand (owner, 2026-11-25): `ρ_S = Σ_r R^{B2}_{S,r} / Σ_r rate_r`, where the sum runs over the races of
a segment.

- `ρ_S < 1` on fav_first is a paid-return shortfall: the favourite-first box pays back less than
  the payout rate, on the Harville-weighted average. Rounding the dividend down to ¥10 alone puts
  the paid ρ slightly below 1, so only a shortfall beyond that rounding loss reads as the pool
  overloading the favourite-first combinations against the outcomes.
- `ρ_S > 1` on fav_second or no_fav means those combinations return more than the takeout allows,
  on the same average.
- A betting profit additionally needs `Σ R / Σ stake > 1`, that is `ρ_S > 1/rate` (about 1.33).

## 5. Families (to freeze in the registry, all variants listed before scoring)

**Family E1, category returns on history** (`japan-131-exacta-category-v2`).

- Members: 2 ordering variants (λ2 `full_development`, #115's full-development fit, and plain
  Harville λ2 = 1) × 3 categories × 14 segment cells (section 6), using estimator B2. **Family size 84**: SPA and StepM run over all 84.
  Plain Harville is in the family because the owner registered it as a variant (answer 8), and a
  registered variant counts toward the multiple-testing burden.
- Per-race loss: `rate_r − R_{S,r}` for fav_second and no_fav, and the sign flipped for fav_first.
  So every member's one-sided alternative is the hypothesised direction (fav_first overloaded,
  the others underloaded). A race outside a member's cell has loss 0 for that member.
- Benchmark loss: 0.
- B1 and B3 are registered as separate reported families, not part of E1's max test. Variants: B1
  is E1′ (42 members: B1 does not use p, so it has no λ2 variant), B3 is E1″ (84 members, both
  λ2). B3's estimand is the mean return per race, null 1 (Harville calibrated on S).
- C (total) is a single paired comparison per λ2 (2 members), with races paying at or below ¥110
  left out (the share is not identified there).
- SPA and StepM results are a registry record of their own,
  `japan-131-exacta-category-v2-max-test` (before #525: `japan/131-exacta-category-v2/max-test`;
  v1: `…-v1/max-test`), so they do not add a variant to E1.

**Family E2, ratio-cut rules** (D18 positive-edge; `japan-131-exacta-rule-v5`, superseding
`-rule-v4` for decision-time Win probabilities and eligibility, which superseded `-rule-v3` for
the race-card popularity cutoff, which superseded `-rule-v2` for the decision-time favourite).

- The owner's 2026-11-27 reframe on #131 ranks every member by expected return at the final
  price with its CI and drops the meeting gate; scoring E2 that way, under rule-v5, is #281.

- Rule form: "at T−2, buy exacta c in category X ∈ {fav_second, no_fav} when
  `r_c(t) = q_c(t)/p_c(t) < k`".
  - `q_c(t)` comes from the exacta board at t: `implied_share(odds × 100, rate=...)`.
  - `p_c(t)` is Harville on the Win board at t. rule-v5: discounted Harville at the registered
    λ2 on the normalised implied Win shares of the Win board at T−2 (v4: #114's `p_cal`).
  - Eligibility (rule-v5): every #130-logged development day with the Win and exacta boards
    at T−2; no #114 row is needed.
  - Selection uses only the board at t. A selection variable taken from final odds is look-ahead.
- Stake: ¥100 per selected combination. The rule is judged at the final dividend,
  with own impact from the payout identity using that race's logged pool totals.
- Cuts k: 0.6, 0.7, 0.8, 0.9.
- Scope: all races, plus the "vulnerable favourite in a less popular race" cell. 2 categories ×
  4 cuts × 2 scopes = 16 members.
- Benchmark: no bet (loss 0).
- **History cannot score E2:** only the winner's `q_c` is known, so the selection set is
  unobservable. E2 is scored only on #130-logged development days: rule-v5 (as rule-v4 before
  it) scores the eligible logged development days available at each run, reported with n and
  the CI, and is re-scored as more days are logged (the owner's 2026-11-27 reframe). rule-v3
  and rule-v4 are history and not executed: the notes under the family table.
- Buying fav_first is not in E2. A loss-avoidance rule (D18: "do not buy fav_first when
  r_c(t) is above a cut") becomes a follow-up ticket.

## 6. Segments (marginal cells, not crossed, plus one pre-stated cross)

| Segment | Source | Levels |
|---|---|---|
| all races | | 1 |
| favourite's Win odds (final for E1, at t for E2) | Win board | ≤ 1.9 · 2.0–2.9 · 3.0–4.9 · ≥ 5.0 |
| field size | starters (runners not scratched) | 5–10 · 11–14 · 15–18 |
| race class | `races.race_class` | newcomer and maiden · 1–3 win classes · open, listed and graded |
| pool size ("less popular") | see below | less popular · rest |
| owner's headline cell | cross | favourite odds ≥ 3.0 **and** less popular |

That is 1 + 4 + 3 + 3 + 2 + 1 = 14 cells.

The store holds no historical pool totals, so "less popular" needs a proxy: races 1–4
of the day, or a newcomer or maiden race.

On #130 days the real exacta pool total exists (the odds log's `investment` row). There,
the lower tertile within each logged day defines "less popular", and the proxy's
agreement with it is reported. (A whole-day tertile uses later races' totals, so it is not a
decision-time selection variable for E2: rule-v4's headline uses the race-card proxy, and the
tertile is only its reported agreement diagnostic.)

## 7. Metric, inference and shrinkage

- Intervals: meeting (venue × day) cluster bootstrap, 95% one-sided, `n_boot` 10,000
  (`config/scoring.yaml`, D7). A cell drawn from fewer than two meetings keeps its estimate and
  n but has no interval (every replicate redraws its one meeting, so `[x, x]` is no 95%
  interval): its bounds are reported as unavailable with the reason.
- Unavailable statistics (an empty cell, a 0 / 0 breakage share, a shrinkage without a positive
  SE, a one-meeting cell's bounds) are listed in the record's metrics as
  `<name>:unavailable:<reason>`, never in its params: the params are the variant's identity, so
  a change in data coverage reruns a variant and never adds one to the family. C with no
  scored race above ¥110 keeps both λ2 records with `log_score_gain` unavailable (n 0); a C
  sample of one meeting keeps its mean without bounds.
- Return per ¥100 (Sol review round 5 of 2c73ca0, 2026-11-28): each B2 and B1 member also
  reports `return_per_100` = 100 · Σ_r R_r / n, the gross yen returned per ¥100 staked on the
  box in each of the cell's n races at the final dividend, with one-sided 95% meeting-bootstrap
  bounds by the same rules. It is the governance metric (return per ¥100 at the price that
  pays) shown beside ρ; it is a metric, never a param, so family sizes are unchanged. ρ stays
  the pre-registered estimand. B3 is paid at Harville-fair odds, not the dividend, so it has
  none.
- Breakage (descriptive, review of 0c39a8d): a dividend at or below ¥110 may have been raised
  by JRA's minimum payout, so its unrounded value is unknown. Such races are counted
  (`n_floor`) and left out of `rho_unrounded` and `breakage_share`, which are unavailable when
  every race of the cell pays at the floor; the paid ρ is over every race.
- Board check: the run refuses (before recording anything) a development race whose Win board
  in the store differs from the one #114's out-of-fold artifact was built on (its normalised
  implied shares `q`), so `p_cal` is never paired with a corrected board.
- Multiple testing:
  - Hansen SPA ("does any member depart in its stated direction") and Romano-Wolf StepM ("which"),
    both over meetings (`core/eval/snooping.py`, D17).
  - StepM level 0.95. `block_size` 5 meetings (owner, answer 5), about one weekend's venues, for
    serial dependence. Seed 131.
  - Family sizes (84 for E1, 42 for E1′, 84 for E1″, 2 for C, 16 for E2) are reported with every
    result, failed variants included.
- Hierarchical shrinkage (D17) uses `core/eval/segment_roi.py`, which needs one row per bet with
  its odds and outcome.
  - E2 meets that on logged days: each selected combination is a bet, and the fixed effects are
    exacta odds bands.
  - E1 on history does not, because losing combinations' dividends are unknown. E1's 14 segment
    estimates are shrunk per category (and per λ2) with a normal-normal empirical-Bayes model on
    their bootstrap standard errors (`exacta_study.eb_shrink`, DerSimonian-Laird between-cell
    variance). This differs from `segment_roi`, and it ignores that the cells overlap (every race
    is in "all" and in one level of each marginal).
- Own impact on history: a ledger assumption that ¥100 per race is negligible, checked
  against logged pool totals.
- Every run, failures included, is a registry record. Outputs go through `write_report`.

## 8. Owner questions and answers (signed 2026-11-25)

The questions as put before the freeze, each with the owner's answer (comment on #131,
2026-11-25: "accept docs/specs/exacta-overload-prereg.md as proposed").

1. Estimand for history: B2 (Harville-weighted box return ρ_S) as primary, with B1 and B3
   reported, C (totals only) secondary, D descriptive on logged days, and A rejected?
   **Yes, as proposed.**
2. Directions: fav_first tested for ρ < 1 and fav_second and no_fav for ρ > 1 (one-sided, as in
   E1)? Or two-sided for all three? **One-sided: fav_first ρ < 1, fav_second and no_fav ρ > 1.**
3. Band edges: the favourite-odds, field-size and class levels in section 6, the "less popular"
   proxy (races 1–4 or newcomer and maiden), and the headline cross (favourite ≥ 3.0 and less
   popular)? **Yes: the 14 cells as listed.**
4. E2: cuts k ∈ {0.6, 0.7, 0.8, 0.9}, ¥100 per combination, at T−2, and a minimum of 100 logged
   meetings before scoring? **Yes: 16 members, scored only after at least 100 logged development
   meetings.**
5. SPA and StepM `block_size` (proposal: 5 meetings). **5 meetings.**
6. E1 shrinkage: normal-normal empirical Bayes on bootstrap standard errors, since `segment_roi`
   needs losing bets' odds? **Yes.**
7. Exclusions: boost-withheld-possible races, dead heats for first or second, refunds, fewer than
   5 starters? **Yes, as listed.**
8. λ2: the in-fold fitted λ2 only, or plain Harville (λ2 = 1) also, as a registered sensitivity
   variant that adds to the family size? **In-fold fitted, and plain Harville also registered as a
   variant.** It adds to the family size: E1 is 84 members (section 5). The fit used is #115's
   full-development one, labelled `full_development`: deviation (a), accepted by the owner on
   2026-11-26.
9. Own impact on history: "¥100 per race is negligible" as a ledger assumption, checked against
   #130 pool totals? **Yes** (`docs/ledger.md`).
