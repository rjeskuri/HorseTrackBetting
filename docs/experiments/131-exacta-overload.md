# Experiment report: exacta favourite overload (#131)

The pre-registered study of whether the JRA exacta pool (`FORECAST`) overloads the favourite's
combinations relative to the Win pool and an ordering model. Spec:
[docs/specs/exacta-overload-prereg.md](../specs/exacta-overload-prereg.md), approved by the owner
on 2026-11-25 and frozen in the Japan run registry before any scored run. Registry hypothesis id
`E43`. Every member's value is in [131-exacta-e1-members.md](131-exacta-e1-members.md) and every
registry record in [registry-japan.md](registry-japan.md). Nothing is adopted; no rule is staked.

## Hypothesis

Stated before scoring (spec sections 1, 4 and 5):

- The exacta pool overloads favourite-first combinations: the Harville-weighted box of the
  favourite-first exactas returns less than the race's payout rate, ρ_fav_first < 1.
- So favourite-second and no-favourite combinations are underloaded: ρ > 1 for each, most of all
  in the owner's headline cell (favourite at 3.0 or more in a less popular race).
- The null, the starting point: the pool prices every category correctly, ρ_S = 1 for every S.
- What would show it wrong: SPA over E1's 84 members fails to reject at 5%, and StepM finds no
  member. A betting edge would further need ρ_S > 1/rate (about 1.33).

## Data

- Development race days only: every day passed `is_holdout_day("japan", d)` before any outcome,
  odds or dividend was read, the readers took only the remaining race ids, and `score_races`
  drops a holdout day again before touching its exacta rows. No race day on or after 2026-11-28
  was read; `LockedHoldout` was not opened.
- Public JRA data only (registry `sources: ["jra"]`): the store's races, runners (final Win odds)
  and `FORECAST` dividends, and #114's out-of-fold Win probabilities (fit `aac501f927b0`), which
  are built on the final Win odds. No JRDB or JRA-VAN input.
- Precondition (row counts, 2026-11-25): 43,058 development races from 2014-06-07, each with a
  `FORECAST` row (43,206 rows). #114's out-of-fold file covers 23,760 of them, its walk-forward
  test quarters from 2020-Q1, so the scored span is 2020-01-05 to 2026-11-15.
- Ordering: discounted Harville with #115's λ2 = 0.8104 (fit `2b822d643cf7`), and plain Harville
  (λ2 = 1). #115's λ2 is one fit on the same development out-of-fold races, so it has seen these
  races' first two places. B2's null (ρ = 1 when the pool is right) holds whatever the ordering
  model, so this cannot manufacture an E1 rejection; B3 at the in-fold λ2 is partly in-sample.
- Payout rate per race from `takeout.race_takeout(..., "FORECAST")`.
- Exclusions: 84 races with a possibly withheld Premium boost, 46 dead heats for first or second,
  1 race with fewer than 5 starters, 0 refunds. Scored: 23,629 races, 1,980 meetings.
- Run: git `4271171` (members, C), `9910a7c` (the max test), 2026-11-25; about 70 s and 0.6 GB.

## Results

E1 (B2), the primary estimand ρ_S, with one-sided 95% meeting-bootstrap bounds in the
hypothesised direction (fav_first: upper bound; the others: lower bound):

| Cell | λ2 | fav_first ρ (upper) | fav_second ρ (lower) | no_fav ρ (lower) |
|---|---|---|---|---|
| all | in-fold | 1.024 (1.041) | 1.003 (0.980) | 0.995 (0.983) |
| all | 1 | 1.028 (1.044) | 1.004 (0.981) | 1.004 (0.991) |
| headline | in-fold | 1.037 (1.096) | 1.055 (0.983) | 0.980 (0.953) |
| headline | 1 | 1.047 (1.107) | 1.056 (0.984) | 0.986 (0.958) |
| popularity:less | in-fold | 1.026 (1.049) | 1.033 (1.000) | 0.992 (0.973) |

- Multiple testing over E1's 84 members (λ2 {in-fold, 1} × 3 categories × 14 cells; family size
  84, no failed member): **Hansen SPA p = 0.175** (consistent; lower 0.116, upper 0.187), 1,980
  meeting clusters, stationary bootstrap with mean block 5 meetings, 10,000 replicates. **Romano-
  Wolf StepM at a 5% family-wise error: no member.**
- Empirical-Bayes shrinkage pulls every cell to its category's mean: in-fold λ2, fav_first 1.025,
  fav_second 1.001 to 1.007, no_fav 0.995 to 0.997. The between-cell variance is small next to
  the cells' bootstrap SEs.
- Nominal one-sided bounds that exclude 1 in the hypothesised direction, none surviving StepM:
  fav_first class:open ρ 0.935 (upper 0.997); fav_second class:maiden 1.036 (lower 1.002);
  no_fav class:open 1.062 (lower 1.022), the same three cells at both λ2, plus fav_second
  popularity:less at its edge (lower 1.000 in-fold, 1.001 plain). With 84 members about four
  nominal exclusions are expected by chance, and the members are strongly correlated.
- C, the log-score total (pool's winning-exacta share against the model, nats per race, one-sided
  95% bounds): in-fold λ2 +0.0133 [0.0108, 0.0158]; plain Harville +0.0306 [0.0270, 0.0341]. The
  exacta pool forecasts the paying exacta better than Win plus either ordering model.
- E1′ (B1, flat box), ρ for "all": fav_first 0.958 [0.928, 0.988], fav_second 0.855
  [0.813, 0.899], no_fav 0.777 [0.752, 0.803]. Equal stakes on every combination lose more than
  the takeout, most in no_fav: longshot exactas are overbet (a favourite-longshot bias inside the
  exacta pool), the opposite of what the owner's idea needs.
- E1″ (B3, Harville's calibration, mean return, null 1), "all": in-fold λ2 fav_first 0.931
  [0.902, 0.962], fav_second 0.864 [0.808, 0.924], no_fav 1.053 [0.974, 1.148]; plain Harville
  1.191 [1.136, 1.247], 0.736 [0.688, 0.786], 1.501 [1.351, 1.686]. The ordering model is
  miscalibrated by category, plain Harville badly so; the exacta pool is not.
- Read together (spec section 4): B2 ≈ 1 in every category while B3 departs from 1, so the gap
  between the exacta pool and Win plus Harville sits in the ordering model, not in the exacta
  pool.
- E2 (ratio-cut rules): **not yet scorable.** The family needs at least 100 #130-logged
  development meetings with exacta boards; the data directory holds 0 (two logged Kyoto days,
  2026-11-21 and 22, with WIN, PLACE, QUINELLA and QUINELLA_PLACE only).
- D (full-board Σq/Σp descriptive): not run, for the same reason: no logged exacta board exists.

## Variants tried

All on the development races above, meeting bootstrap, 10,000 replicates, seed 131:

- E1 `japan/131-exacta-category-v1`: 84 members (B2 × λ2 {in-fold, 1} × 3 × 14), all scored, each
  with its one-sided bounds and its EB value. Table: [131-exacta-e1-members.md](131-exacta-e1-members.md).
- E1′ `japan/131-exacta-category-b1-v1`: 42 members (B1 has no λ2), all scored.
- E1″ `japan/131-exacta-category-b3-v1`: 84 members, all scored.
- C `japan/131-exacta-logscore-v1`: 2 members, scored (above).
- Max test `japan/131-exacta-category-v1/max-test`: two records.
  - **Failed (void):** the first run (git `4271171`) passed the spec's 95% confidence level as
    `snooping.stepm`'s `level`, which is the test size, so StepM ran at a family-wise error of
    0.95 and "rejected" 13 members. That record stays in the registry; its StepM list means
    nothing. SPA does not take a level and is identical in both records (p 0.175).
  - The re-run (git `9910a7c`, FWER 0.05): SPA p = 0.175, StepM no member.
- E2 `japan/131-exacta-rule-v1`: frozen, 16 members, not scored (above).

## Alternatives rejected and why

- A, the mean of q_w/p_w over races whose winner is in S: it selects on the winner, so it reports
  an overload when the pool is exactly right (spec section 4). Not run.
- C split by category: not a proper score per category (a forecast that overloads S gains on S).
  Only the total is reported.
- The ticket's per-race Σq/Σp ratio (D) as the historical estimand: history holds only the
  winning exacta's dividend, so it cannot be computed; it waits for logged boards.
- `core/eval/segment_roi.py` for E1's shrinkage: it needs every bet's odds and outcome, and losing
  exactas' dividends are unknown on history. Normal-normal EB on the bootstrap SEs instead.
- Two-sided tests: the owner signed one-sided directions (answer 2).
- Crossed segment cells: 14 marginal cells plus one pre-stated cross keep the family at 84.
- Historical rule tests selected on final odds: look-ahead; E2 selects only from the board at T-2.

## What this rules out

- A favourite-first overload in the JRA exacta pool, overall: ρ_fav_first is 1.024 with a
  one-sided 95% lower bound of 1.008, so any overload (ρ < 1) is excluded at that level; the
  favourite-first box returns slightly more than the pool-fair rate, not less. This direction was
  not the registered alternative, so it is an observation, not a tested claim.
- An underload of favourite-second or no-favourite combinations of more than about 2.6% and 0.8%
  of the payout (upper bounds 1.026 and 1.008 overall, in-fold λ2), and, in the owner's headline
  cell, of more than about 12.7% for favourite-second (upper bound 1.127).
- A profit from boxing a category at final dividends: it needs ρ > 1/rate ≈ 1.33, and every cell
  with more than 2,000 races has its upper bound below 1.13. Only the favourite-at-5.0-or-more
  cell (189 races) is too thin to rule it out.
- What it does not rule out: a rule that selects single combinations by their board ratio at
  T-2 (E2). A box averages over a category; a selective rule can still find mispriced
  combinations inside it, and only logged boards can test that.

## Follow-ups (proposed, not filed)

- Score E2 once 100 logged development meetings carry exacta boards (none logged yet), with the D
  descriptive on the same days and the proxy-versus-pool-tertile check for "less popular".
- The fav_first loss-avoidance rule (spec section 5): "do not buy fav_first when r_c(t) is above a
  cut", a D18 family L rule, pre-registered on logged boards.
- Check the "¥100 own impact negligible" ledger assumption against #130's exacta pool totals.
- Refit λ2 per walk-forward fold (#115 is one fit over the development races), so B3 and C at the
  in-fold λ2 are strictly out-of-sample.
