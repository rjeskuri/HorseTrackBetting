# C5 against the pools' own dividends: E43, E16, E44 (ticket #459)

Status: **pre-registered**, frozen by the run itself as its first step. Family id:
**`japan-c5-exotic-dividends`**, spec sha256
**`cdaa71d90e9888d9e9498d3e91dbd4c99a37c0b64d73ff658f6f6758dfbef76e`**. The spec is
`studies.exotic_c5.default_spec()` (`build_spec` on `config/folds.yaml`'s scheme and D7's bootstrap,
seed 459); `horsetrack japan exotic-c5` freezes it with `Registry.freeze_family` in
`$HORSETRACK_DATA_DIR/registry/japan.jsonl` before either seam reads a store or member row, and
refuses to run (`FamilyAlreadyFrozen`) under any other spec. Any change after the freeze is a new
family. Under the owner's reframe of 2026-11-27, Claude pre-registers and runs it with no sign-off;
the modeling choices below live in this frozen spec, not in rulings.

## Question

Does C5 (`models/exotic.py`, #99), an input's Win probabilities through an ordering, price each
pool's paid combinations better than the pool's own money does (08 §4 E43)? Is that so on the
super_premium race-pools (E16)? And through one ordering fitted once on the market, is the
model's gain over the market on Trio and Trifecta a multiple of its Win gain (E44, 08: 2-3x)?
The null, the starting point: the dividends are at least as good as C5 in every pool, and the
joint gain is no larger than the Win gain.

## Data and days

- The JRA store under `processed_dir("jra")` (results and dividends from netkeiba; no JRDB, no
  JRA-VAN): `races` (race_id, race_date, meeting_id, race_class_raw, field_size, status),
  `runners` (horse_no, win_odds, finish_pos, dead_heat, is_scratched) and `dividends` (pool,
  combination, dividend, is_refund), read by column and date range (`read_store_table`), the
  dividends by the development races' ids.
- Days: every day from 2014-06-07 (D10) to the run's last day for which `is_holdout_day("japan",
  d)` is False. No holdout day is asked for, and `LockedHoldout` is never opened.
- `stage2:q:form` (`load_form`): #87's member rebuilt by #87's own code (`studies.meta`): the
  store's development rows, HKTrack's `outputs/oof/m-v1-20261115-6102e309.parquet` (read-only),
  stage 2 on the form member and ln q, ridge 0, D9's folds scored from 2024-01-01. It reads JRDB:
  every record that uses it names `jrdb` among its sources, and the committed summary and the
  public report withhold its figures (D21).

## Rows

Run races with at least 3 priced starters (not scratched, a decimal Win price of at least 1.0, a
horse number); unpriced starters are left out and counted. A race-pool is scored when it pays,
holds no refund row, has no dividend under 100 yen, and each paid combination names starters of
the race. The places a Place ticket is drawn from follow jra.yaml for the field at sale: the
race's declared runners, at least `races.field_size`.

## Baseline (D5, 08 R15)

`exotic.dividend_shares` on the race-pool's dividends at the race's own payout rate
(`takeout.race_takeout`, boosts and `race_class_raw` in): R·100/(D+5) for one winning combination,
the statute's de-vig x_c = 1/((1+K)·P·(r_c − 1)) for several (owner answer 3(a) on #99).
Single-winner pools: the chance of a paid combination is its share. Place and Wide
(`place_wide_devigs: [statute]`): the chance a paid ticket pays is P × its statute share (P the
paid combinations), capped at 1. At the 110-yen minimum the share is censored
(`exotic.dividend_floor_bound`), so each Place and Wide variant is run twice (owner, #459,
2026-11-27): `drop` leaves flagged race-pools out of fitting and scoring, `bound` scores them all.

## The model and its variants

- Inputs: `normalised` (1/odds over the race's priced starters), `calibrated` (D4's spline,
  `fit_win_calibration` with 5 knots, refitted on each fold's training starters), `stage2:q:form`.
- Orderings: harville, henery, quadratic_context, discounted_harville. The fitted two are refitted
  on each fold's training races through the input's own Win probabilities (`Ordering.fit`, 04
  §2.3); a fold in which the input has no training race is not scored for them. WIN takes the
  input itself. A race whose ordering raises is left out of that variant and counted
  (`races_ordering_failed`).
- Blends (owner answer 6): `none`; `linear`, p = α·q + (1 − α)·P_C5 on each paid combination, α in
  [0, 1] minimising the mean log loss over the fold's training race-pools of that pool, input,
  ordering and floor treatment, one α per fold (a fold with more than 2,000 training races fits α
  on the 2,000 whose sha256(race_id) is smallest); `log_linear` (`exotic.pool_blend`), recorded as
  a failed variant: it needs full final boards, which only #4/#21's logs hold.
- Folds: D9's walk-forward quarters (`config/folds.yaml`) over the races' dates; every fitted
  piece sees only the fold's training races.

## Metrics

- Score: the log loss −(1/k) Σ ln p(c) over a race-pool's k paid combinations (#69).
- E43 (per variant): `delta_ll_vs_dividends` on `all_folds`, the mean over scored race-pools of
  the dividends' log loss minus the model's, with D7's one-sided 95% bounds (meeting clusters,
  10,000 replicates, seed 459); per fold, the linear blend's `alpha`.
- E16: the same per-race ΔLL (E43's fits and α) restricted to the race-pools whose payout rate
  was exactly super_premium (`super_premium_subset`; ultra_premium and premium are out; owner
  answer 5).
- E44 (per ordering, model input `calibrated` or `stage2:q:form`, market `normalised`): one
  ordering fitted per fold on the market serves both inputs. On the races both score and where
  WIN, Trio and Trifecta pay: `win_gain` and `joint_gain[TRIO]`, `joint_gain[TIERCE]` (the market's
  log loss minus the model's) each with D7 bounds, and `gain_ratio[pool]` = joint / Win, a point
  estimate (owner answer 1(b)).
- Leaderboard (`docs/specs/leaderboard.md`'s convention): each no-blend E43 and E16 variant
  carries `params['leaderboard']` (kind model, selection look_ahead: the inputs are final prices)
  and `return_per_100` (E16: `return_per_100[super_premium]`) on `development`: a flat ¥100 per
  scored race-pool on the model's most probable combination of the pool (the first in
  horse-number order on a tie), returning its paid dividend when it pays, else 0, with D7 bounds,
  n in bets. The linear blend has no most probable combination without the full board, so it has
  no return (an unmeasured row).

## Outputs

- One registry record per variant, failures included; the run is recorded whatever it finds.
- The private report and its JSON under `$HORSETRACK_DATA_DIR/reports/` through `write_report`
  (sources `jra`, `jrdb`), with every table.
- The committed summary `docs/experiments/registry/japan/japan-c5-exotic-dividends.md` (by
  `write_summary`, stage2:q:form's records withheld) and the public report
  `docs/experiments/exotic-c5-japan.md` (`render_public_report`: the publishable pair's figures
  only, every stage2:q:form variant by name and status).
- Nothing is adopted: every variant is a ranked or unmeasured leaderboard row.

## Second family: Place and Wide against the per-ticket de-vig

Status: **pre-registered** before its run, frozen by the run itself as its first step. Family id:
**`japan-c5-exotic-dividends-per-ticket`**, spec sha256
**`b559e99c14f73a307e9f075dcb405a67b713feec2640d2d31ea0be4e908c65d1`**. The spec is
`studies.exotic_c5.default_per_ticket_spec()` (`build_per_ticket_spec` on the same fold scheme and
bootstrap, seed 459); `horsetrack japan exotic-c5 --family japan-c5-exotic-dividends-per-ticket`
freezes it before either seam reads a row and refuses to run under any other spec. The first
family's spec above is unchanged (sha256 `cdaa71d9…f76e`), so its freeze and records stand.

Why a second family: the first family's Place and Wide rows rest on one baseline, P × the statute
share, which its own E43 Place result suggests understates the chance a ticket pays
(`docs/ledger.md`, the #459 Place rows). The owner's reframe on #99 (2026-11-27) registers each
candidate Place and Wide de-vig as a variant, and the first spec was frozen before the review of
PR #668 asked for a second candidate, so the second de-vig is a family of its own.

- **Baseline** (`place_wide_devigs: [per_ticket]`): the chance a paid ticket c pays is its own
  implied share, min(1, R·100/(D_c + 5)) (R the race's payout rate from `takeout.race_takeout`,
  boosts and `race_class_raw` in; D_c its dividend per 100 yen, the 5-yen mean breakage back):
  the money on c against what c pays, as in a pool with one winner. It is P × #99 question
  3(b)'s equal split R·100/(P·(D_c + 5)), which answer 3(a) did not choose as the share of the
  money; here it is a ticket chance, the reframe's second candidate. With D_c ≥ 100 the cap
  never binds (R·100/105 < 1).
- **Pools**: Place and Wide only. A pool paying one combination per result has the one share
  R·100/(D + 5) under either de-vig, so Win, Quinella, Exacta, Trio and Trifecta would repeat
  the first family's records; no E44 (`e44_pools: []`).
- **Everything else as the first family**: the same rows and days (development days only, the
  holdout never opened), the same scoreable race-pools (no refund row, no dividend under 100 yen,
  every paid combination named in the field), inputs (`normalised`, `calibrated`,
  `stage2:q:form`), orderings (all four, the fitted two refitted per fold), blends (`none`,
  `linear` with α fitted per fold on at most 2,000 hash-sampled training races against this
  baseline's q, `log_linear` recorded as not runnable), floor treatments (`drop`, `bound`), score
  (log loss at 1/k per paid combination), metrics (E43 `delta_ll_vs_dividends` with D7's bounds,
  per-fold `alpha`; E16 on super_premium race-pools) and leaderboard block on each no-blend
  variant.
- **What the comparison means**: the model's log loss and the leaderboard bet do not read the
  baseline, so on the same race-pools a variant's ΔLL here minus its statute twin's is the
  statute baseline's log loss minus the per-ticket one's, and each no-blend return equals the
  twin's. The public report sets each publishable variant beside its twin's ΔLL and bounds,
  read from the first family's registry records (`statute_twins`).
- **Outputs**: one registry record per variant; the private report under
  `$HORSETRACK_DATA_DIR/reports/`; the committed summary
  `docs/experiments/registry/japan/japan-c5-exotic-dividends-per-ticket.md` (by `write_summary`,
  stage2:q:form's records withheld) and the public report
  `docs/experiments/exotic-c5-per-ticket-japan.md` (the publishable pair's figures only). Nothing
  is adopted: every variant is a ranked or unmeasured leaderboard row.
- **The null**: the per-ticket dividends are at least as good as C5 in Place and in Wide
  (ΔLL ≤ 0).
