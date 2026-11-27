# Problem statement

Signed off by the owner on 2026-11-24 (ticket #14)

This document says what HorseTrackBetting is trying to predict, what it is judged against, how it is tested and
when a model or a betting rule may be adopted. Claude drafted it (ticket #14) and the owner signed it off.

How to read it:
- Each section first states what the brief already fixes. Those lines are not open.
- Each choice the brief left open is a line `OWNER DECISION (Dn): <question>`, followed by the options with their
  tradeoffs, kept as the record of what was considered, and an `Answer:` line with the owner's answer.
- The owner answered by number (D1 to D21) on ticket #14. The table at the end lists every decision, its answer and
  its section.

Sources: the HorseTrackBetting brief (HKTrack `docs/research/25-horsetrackbetting-brief.md`, section "Modeling
governance"), the HKTrack process review (`docs/research/24-process-review-2026-11-22.md`, sections 1 to 4 and 7),
HKTrack `docs/DECISIONS.md` #12, #26 and #27, and an internal research synthesis (not published).

Notation: race r; runner i; time to post τ; decision time t; the board at t (the odds displayed at t) `O_t`; the
final odds that pay `O_final`; the market's implied probability from the final price `q_final`; takeout τ_k of
pool k. Odds are decimal and include the stake.

## 1. Decision problem

What the brief fixes:
- A pari-mutuel bet pays at the final odds `O_final`, which are only known after betting closes. At the decision
  time t, only the board at t is visible.
- So a unit Win bet on runner i placed at t has value when

  **E[p * O_final | info_t] > 1**

  where p stands for the outcome (the runner wins) and the expectation is over the joint distribution of the
  outcome and the final price, given everything known at t.
- This is a joint outcome-and-price problem, not a probability problem with a known price. In general
  `E[p * O_final | info_t]` is not `P(win | info_t) * E[O_final | info_t]`: the two differ by
  `Cov(1{win}, O_final | info_t)`. When late money is informed, a runner that is going to win tends to shorten, so
  that covariance is negative and the product of the two expectations overstates the value.
- **Place and Wide** pay a dividend that depends on which other runners also place (the co-placers): the same
  Place ticket pays differently when it places with the favourite than with two outsiders. Their value needs the
  joint finishing distribution and the final pools, not a single marginal probability and a single price. How many
  runners place depends on the field: JRA Place pays the top three in fields of 8 or more runners and the top two
  in fields of 5 to 7, and is not sold with 4 or fewer (to confirm in the ledger against the official JRA rule).

Facts the statement relies on (each to be entered in the known-vs-assumed ledger, #12):
- **The payout identity.** For a single-winner pool (Win, Quinella, Exacta, Trio, Trifecta) a combination c with
  stake s pays `(1 - τ_k)(V_k + s)/(v_c + s)` per unit, where V_k is the final pool and v_c the final money on c,
  before breakage and the minimum payout. Own stake moves the price: negligible for Win, Place and Wide at retail
  stakes, material for long Trio and Trifecta combinations.
- **Takeout varies by pool and by race.** JRA base rates since 2014-06-07: Win and Place 20%; bracket quinella,
  Quinella and Wide 22.5%; Exacta and Trio 25%; Trifecta 27.5%; WIN5 30%. Premium and Super Premium days change the
  return rate for designated races and pools, and rates before 2014-06-07 differed. Takeout therefore comes from a
  per-race lookup (date, pool, boost), never a constant.
- **Breakage and the minimum payout.** Dividends are rounded down to ¥10 per ¥100. The JRA minimum payout is ¥110
  per ¥100 (Plus 10); it is withheld, and the dividend is ¥100, when paying it would exceed the pool. At a dividend
  of ¥110 a Place bet needs P(place) > 0.909 to have value; at ¥100 it can never have value.
- **Closing-line value in the sports-betting sense is degenerate on a tote.** Every bet is paid at the closing
  price, so the price obtained always equals the closing price and "beating the closing line" in that sense is
  identically zero. HKTrack's measure (HKTrack DECISIONS #12) is a different quantity and is not zero: the log of
  the final odds over the odds at decision, and the share of bets whose final odds stayed at or above the bet's
  minimum acceptable odds. It measures how the price moved after the decision, not a price the bet obtained. The
  brief lists closing-line value as the primary live metric among the assumptions to retest or drop; the owner
  made a measure of this kind a secondary live diagnostic, never the primary metric (D8), beside the price half's
  own score.
- **Pricing on a forecast of the final odds is biased upward.** Multiplying a probability by a point forecast of
  the final odds overstates value even when the forecast is unbiased in log terms. Under an efficient final price
  and a lognormal forecast with log-scale standard deviation σ, with p = P(win | info_t), the bias for Win at a 20%
  takeout is about `0.8(e^{σ²/2} - 1)` when the forecast is unbiased in log terms (it is the median of `O_final`):
  about +0.9 points of EV at σ = 0.15 and +2.0 at σ = 0.22. When the forecast is unbiased in level (it is
  `E[O_final | info_t]`) the bias is about `0.8(e^{σ²} - 1)`: about +1.8 points at σ = 0.15 and +4.0 at σ = 0.22.
  It adds to winner's-curse selection: picking the bets with the largest claimed value picks the largest errors.
- **Sales close before the off.** Online sales close about one minute before the off (T-1) and on-course sales
  about T-2, so the last bettable board is about T-1 (from the internal research; to be confirmed in the ledger).

### 1.1 The starting point: a null result

HKTrack's evidence is the starting point, not something to explain away (brief, "Why"):
- HKTrack's form model, blended with the JRA final odds, improved log loss over the final odds by a very small
  amount, with a confidence interval only just above zero.
- Against the board at decision time the blend looked much better, but that board is much less sharp than the
  final price, and the blend fitted on that board still trailed the final price. Most of the apparent gain was the
  board being stale, not information in the model.
- In Hong Kong the fitted blend weight on the form model was zero. That zero was the lower bound of a constraint
  that forbade negative weights, so the unconstrained value is inferred to be zero or negative; it was not
  estimated without the bound.
- No betting rule tested showed a profit that survived its uncertainty. Some Win rules had a positive point
  return at the final dividends, but every return interval reported for them spans zero (HKTrack DECISIONS #26
  and #27); the rules compared in the process review all lost money; and a large search over segments, pools and
  ticket combinations found nothing that beat the takeout.
- The Win rule HKTrack shipped claimed a clearly positive expected value; with the same probabilities at the
  actual final odds the expected value was negative. Its stage-2 combiner had been fitted on the final odds but
  was fed, live, the much less sharp board or a forecast of the final odds made from it ("fit final, fed forecast
  final"), so the weights fitted on the sharp final price over-trusted a stale input. The value was then that
  probability times the forecast of the final odds (the bias above, plus selection), and the model was judged
  against the board instead of the final price. HKTrack DECISIONS #27's fix was to fit stage 2 on the board it is
  fed.

See HKTrack `docs/research/24-process-review-2026-11-22.md` sections 1 to 4, HKTrack `docs/DECISIONS.md` #26 and
#27, and the brief's "Why". The figures themselves are not reproduced here: the models behind them used inputs from
a licensed data source whose terms forbid publishing its outputs (see D21).

The claims this null result rests on, to be cross-referenced to the known-vs-assumed ledger (#12):
1. The form model's log-loss gain over the JRA final odds is small and its interval only just excludes zero
   (one model; its test seasons were reused for model selection, so this is at best single-run).
2. The decision-time board is less sharp than the final price, and model plus board trails the final price.
3. The Hong Kong blend weight of zero is a bound-constrained estimate, not an unconstrained zero.
4. No tested betting rule showed a profit that survived: every reported return interval spans zero (some Win
   rules had positive point estimates), and the rules compared in the process review lost money at final
   dividends.
5. A large hypothesis search found nothing beating the takeout (its period is not stated in the source).
6. The shipped Win rule's claimed value reversed sign at the actual final odds.
7. The late price-move sizes quoted in the process review come from one day at one venue and from flagged tickets,
   so they are single-run and selection-conditional.
8. The JRA takeout schedule and the minimum-payout rule (public rules; measured).
9. Closing-line value in the sports-betting sense (price obtained against closing price) is zero by construction
   on a tote (arithmetic); HKTrack's decision-to-final measure (HKTrack DECISIONS #12) is not.

OWNER DECISION (D1): Which form of the estimand does the project model?

- **A. Condition on the final price and integrate over a nowcast.** Model `P(outcome | x, q_final)`, then average
  it, with the dividend, over draws of the final pools from a forecast `F_t` made at t. Handles the covariance
  exactly and trains the probability model on every year of final odds; the forecast `F_t` trains only on logged
  odds paths, which start in late 2026, so it is the weak link.
- **B. Condition on the board at t.** Model `P(outcome | x, O_t)` per time-to-post bucket and correct for the
  covariance separately. Direct and simple to explain; needs boards at every τ for training, which exist only for a
  few logged days (or a bought history for Win, Place and Quinella), and the covariance correction is a second
  model.
- **C. A residual on the final price.** Model `ln p_i = ln q_final,i + δ_i` with δ built only from information at
  t and shrunk toward zero. The Win value then depends on the unknown final price only through the normaliser, so
  it barely needs a price forecast; it can only ever claim δ, and it still needs a price model for Place, Wide and
  exotics.

Answer: A. Condition on the final price and integrate over a nowcast of the final pools made at t.

OWNER DECISION (D2): What is the decision time t, and which other horizons are reported?

- **The last bettable board (about T-1).** Closest to the final price, so the smallest price uncertainty; leaves no
  time for a person to act, so it only suits automated placement.
- **A fixed earlier time (T-2, T-5 or similar).** Leaves time to act; more price uncertainty and more late money
  still to come, which the price half must model.
- **A grid (for example T-60, T-30, T-16, T-5, T-2, T-1).** Shows how value and price uncertainty change toward the
  off and supports a time-varying blend; costs more logging and more scoring, and multiplies the number of
  comparisons (D17).

Answer: A grid (T-60, T-30, T-16, T-5, T-2, T-1) with T-2 as the primary decision time.

OWNER DECISION (D3): Which pools does this statement cover first?

- **Win only.** The cleanest estimand and the most history; leaves out the pools where the co-placer structure
  might matter.
- **Win and Place.** Adds the pool where the co-placer dividend matters most; needs the joint top-three
  distribution and the Place pool's own baseline (D5).
- **Win, Place and Wide.** Covers the pools HKTrack bet; Wide price paths exist only from the project's own logs.
- **All pools including exotics.** Most opportunities and the highest takeouts; exotic boards before the decision
  exist only if logged going forward, and own price impact matters for long combinations.

Answer: Win, Place and Wide. (Wide is quinella place, its own pool; final Wide dividends exist for all history, but
pre-race Wide odds paths exist only from our own logs.)

Amended 2026-11-24 (ticket #127): every JRA pool is in research scope in two priority tiers. Tier 1: Win, Place and
Wide (quinella place). Tier 2, lower priority: Quinella, Exacta, Trio, Trifecta, Bracket quinella and WIN5. The
combination methods (box, wheel with multi, formation) are studied for coverage and cost against single tickets.

## 2. Baseline

What the brief fixes:
- **The baseline is the final price**: the market's probability implied by `O_final`, the price that pays.
- **The board at t is a secondary diagnostic only.** It is reported to show how much of the gap to the final price
  is timing, and it never supports a claim of edge. HKTrack used the decision-time board as the baseline; because
  that board is staler than the final price, beating it was easy and said nothing about value (section 1.1).
- A baseline with fitted parameters (D4, D5) is fitted on training folds only, like any model.

OWNER DECISION (D4): How does the final Win price become a probability?

- **Normalised 1/odds**, `q_i = (1/O_i) / Σ_j (1/O_j)`. No parameters, nothing to fit, no leakage. Keeps the
  favourite-longshot bias, so a model can "beat" it by learning that public, well-known bias, which is not
  information the market lacks.
- **Power**, `q_i ∝ (1/O_i)^β`, β fitted per period. One parameter that absorbs the favourite-longshot bias, so a
  model has to beat a calibrated market; β drifts over time and must be refitted inside each fold.
- **Shin.** Models a share of insider money and spreads the margin unevenly toward longshots. Has a structural
  story; one parameter per race, and it can behave poorly in large fields and on tote pools, where it was not
  designed.
- **A calibrated market** (the final price passed through a fitted calibration curve, such as a spline in
  `ln q`). The hardest and most honest baseline; the most flexible to fit, so the most care is needed to keep it
  out of fold.

Answer: A calibrated market: the final Win price through a fitted calibration curve (spline in ln q), fitted in-fold
only.

OWNER DECISION (D5): What is the baseline for Place, Wide and the exotic pools?

- **Derived from the Win baseline** through an ordering model (Harville, Henery, Stern or Plackett-Luce). Needs only
  Win prices; mixes the ordering model's error into the baseline, so a model can "beat" a weak ordering instead of
  the pool.
- **The pool's own final prices.** For single-winner exotic pools the implied share is `(1 - τ_k)/D_c` from the
  dividend, with breakage added back. For Place and Wide, an owner-chosen de-vig of the pool's own final odds or,
  with pool totals and per-runner money, the payout rule itself. The fair test of a model in that pool; needs the
  pool's final data for every race, and Place and Wide need a de-vig rule that the co-placer payout makes
  non-standard.
- **Both, with the pool's own price as the primary.** Shows separately how much a model gains from ordering and
  how much against the pool; two baselines to maintain.

Answer: Both, with the pool's own final prices as the primary baseline and the Win-derived ordering baseline as
secondary.

## 3. Metric

What the brief fixes:
- The loss is **log loss**, and the reported gain is the **log-loss (pseudo-R²) gain over the final-price
  baseline**, per race and paired: the same races, the same folds.
- Every gain carries a **paired bootstrap confidence interval**: the model and the baseline are resampled with the
  same indices, so the interval is on the difference.

Scoring the price half separately is not fixed by the brief; it follows from the owner's answers to D1 and D8.
Because D1 (A) integrates the outcome model over a nowcast of the final pools, two halves are scored: the **outcome
half** (the probability of the result, by the metric above) and the **price half** (the forecast of `O_final` from
info_t, scored as D8 sets). A bet's value needs both (section 1).

OWNER DECISION (D6): Which outcome is scored, and against what is pseudo-R² defined?

- **The winner only** (conditional-logit log loss per race). Matches the Win pool and HKTrack's numbers; uses one
  bit of each race.
- **The top-three order** (a discounted Plackett-Luce likelihood). Uses more of each race and matches Place and
  Wide; mixes in an ordering model's error.
- **Both, with one named primary.** Covers both pools; two primary results invite choosing the better one.
- Pseudo-R² reference, for any of the above: relative to equal chances for every runner (comparable with the
  literature) or relative to the final-price baseline (reads directly as the gain). The second makes small gains
  look large; the first makes them look tiny.

Answer: Pool-matched primary: a Win claim is judged on winner log loss vs the Win baseline; a Place or Wide claim on
top-3 order log loss (and the placed-set score) vs that pool's own baseline. Both reported for every model.
Pseudo-R² reported vs equal chances, with the gain over the final price in nats per race alongside.

OWNER DECISION (D7): What is the bootstrap's resampling unit, and what is the confidence level?

- **Unit: race.** Most resamples and the narrowest intervals; understates the interval when errors are correlated
  within a meeting or a day (shared going, shared track bias), so it is a lower bound.
- **Unit: meeting (venue and day).** Keeps within-meeting correlation; wider intervals, and the correlation has
  not yet been measured for JRA.
- **Unit: race day (all venues).** Also keeps correlation across venues on the same day (weather, national
  betting patterns); the fewest units and the widest intervals.
- **Block or stationary bootstrap over days** (block length to set). Keeps serial correlation across weeks; adds a
  tuning choice.
- Level: 90% or 95%, and one-sided (a gain claim is one-sided) or two-sided. A higher level and a two-sided test
  need more data for the same claim.

Answer: Resample meetings (venue x day); 95%, one-sided.

OWNER DECISION (D8): How is the price half (forecasting `O_final` from info_t) scored, and is a closing-line-style
measure used as a live diagnostic?

- **Log predictive density of the final log share** per runner (and per combination). Proper and uses the whole
  forecast distribution, which is what the value calculation integrates over; needs a density, and is sensitive to
  tail misspecification.
- **CRPS of the final log share.** Proper and robust to tail misspecification; harder to aggregate across
  combinations and less familiar.
- **Point error (MAE of the log final odds).** Simple, and HKTrack's move model was scored this way; ignores the
  forecast's spread, which drives the bias in section 1.
- **Calibration diagnostics in any case:** PIT histograms and interval coverage per τ bucket. Necessary, but they
  do not rank forecasts.
- **Baseline for the price half:** "no move" (`q_final = q_t`), or a fitted average drift by odds band and τ. The
  first is parameter-free; the second is a fairer benchmark.
- **Joint check in shadow:** regress the realised return at the final dividend on the claimed value; a calibrated
  value has slope near 1. It is the only score of the two halves together, and it needs many bets before it says
  anything.
- **Closing-line-style live diagnostic (HKTrack DECISIONS #12): primary, secondary or not used.** Per bet, the log
  of `O_final` over the odds at decision, and the share of bets still good at the final price. Cheap, available
  from the first live day and read on real bets; but it is not a proper score of the forecast, it is conditional
  on the bets chosen, and the brief lists it as an assumption to retest or drop. As the primary live metric it
  replaces profit with a faster signal; as a secondary diagnostic it sits beside the joint check; not used, the
  price half is judged only by the scores above.

Answer: Primary: log predictive density of the final log share, with PIT and interval coverage per horizon; price
baseline: a fitted average drift by odds band and horizon; the shadow calibration slope as the joint check. A
closing-line-style measure (HKTrack DECISIONS #12) is a secondary live diagnostic, never the primary metric.

## 4. Splits

What the brief fixes:
- **Walk-forward splits by date.** Every prediction for a race comes from models fitted only on data available
  before that race's fold. No random splits.
- Features are joined as of the decision time: a row is usable at t only if it was available at t. The final odds
  are a label or a conditioner (D1), never an input value at decision time.

OWNER DECISION (D9): What is the fold scheme: the unit, the refit cadence, the window and the minimum training
window?

- **Fold unit:** season, calendar year, quarter or race day. Longer folds give fewer, more stable estimates; shorter
  folds track drift and give more paired differences but share seasonal regimes across folds.
- **Refit cadence:** yearly, quarterly or monthly. Frequent refits follow drift (the market's bias moves over
  time) and multiply compute and the number of fitted versions.
- **Window:** expanding (all history so far) or rolling (the last N seasons). Expanding uses more data; rolling
  adapts to regime change and discards evidence.
- **Minimum training window:** for example 3, 5 or 8 seasons before the first scored fold. Longer windows give
  better early models and fewer scored folds.
- **Stacked models:** each level fitted on the out-of-fold output of the level below (nested walk-forward), or a
  simpler single level. Nested folds prevent the meta-model from learning in-sample fits; they cost one extra fold
  layer of data.
- **Embargo:** none, or a gap (for example one week) between training and scoring. A gap guards against leakage
  through slowly published data; it wastes a little data.

Answer: Fold unit quarter, quarterly refit, expanding window, 5 seasons minimum before the first scored fold, nested
out-of-fold for stacked models, 1-week embargo.

OWNER DECISION (D10): How are races before 2014-06-07 (the earlier, race-varying takeout regime) treated?

- **Include with an era term.** More history for models that do not depend on the price; the market's behaviour
  differs across the reform, and the era term must be fitted.
- **Exclude.** One takeout regime throughout; loses about four seasons.
- **Include only for components that do not touch the price** (for example a horse-ability model), and exclude
  them wherever the market or a dividend enters. Keeps the history where it is harmless; two training windows to
  maintain.

Answer: Races before 2014-06-07 are used only for components that do not touch the price (ability, speed/par,
lineage); excluded wherever the market or a dividend enters.

## 5. Holdout

What the brief fixes:
- **A locked final holdout per country.** It is cut before any model trains, is not used for development, model
  selection or tuning, and is opened only by owner decision.
- **Every use of it is logged**: the date, the ticket, the hypothesis, the code version and the result (ticket #71
  builds the lock and the log).
- The owner's answer to D14 (No) is consistent with this: holdout days are fully locked, so their price paths do
  not train the price half either, and no exception to "not used for development" is needed.

### 5.1 Exposure so far (disclosure)

Before the owner chose the holdout (D11 to D15), these were the periods already seen:

- **HKTrack (Japan).** Models were trained on seasons from 2010/11 through races of 2026-11-15. Test seasons from
  2023/24 on were scored repeatedly and used for model selection, so HKTrack reused its holdout. HKTrack research
  also fitted and scored on 2024-01 to 2026-11 (board-based blending), 2025-01 to 2026-11-15 (finishing order),
  out-of-fold races from 2020 to 2025 (segment and pool searches) and every race of 2024 to 2026 (ticket
  packages). One large hypothesis search does not state its period. HKTrack's live scoring covered the races of
  2026-11-22.
- **HKTrack (Hong Kong).** All Hong Kong history was used for training and testing.
- **Internal research, read on 2026-11-24.**
  - Pattern analysis (an in-sample search over segments, pools, odds bands and price drift) used JRA races and Hong
    Kong races before 2024-01-01 only (2010-01 to 2023-12).
  - JRA results and dividend aggregates for 2014-06-07 to 2026-11-15 were read, with breakdowns by year and era,
    plus a sanity check on 2010 to 2014-05.
  - The JRA odds snapshots logged at Kyoto on 2026-11-21 and 2026-11-22 were read (descriptive move statistics; no
    outcomes modelled), and public JRA pages were read, including the WIN5 page of 2026-11-23.

Consequence for Japan:

| Period | Touched by | Exposure |
|---|---|---|
| 2010-01 to 2023-12 | HKTrack training and test seasons; the pattern search; the aggregates | Touched by everything above; cannot be a clean holdout |
| 2024-01-01 to 2026-11-15 | HKTrack test seasons and research; the aggregates | Not touched by the pattern search, so a fair test for hypotheses it generated; not fair for anything HKTrack or the aggregates measured |
| 2026-11-16 to 2026-11-23 | The Kyoto snapshots; HKTrack live scoring of 2026-11-22; the WIN5 page | Touched |
| After 2026-11-23 | Nothing | Untouched |

No historical period of either country is untouched. Only race days after 2026-11-23 are untouched in Japan.

OWNER DECISION (D11): What kind of holdout does Japan use?

- **A1. Prospective, contiguous.** Every JRA race from a start date on. Logged days before the start are
  development for the price half. Untouched by anyone, and every holdout race has a logged board at each decision
  time, which is the only data on which a betting rule at those times can be judged without look-ahead. Slow to
  reach power: on the order of a year of all-venue racing to detect a gain of a few thousandths of a nat per race.
  The price half starts with few logged days, and the holdout starts later.
- **A2. Prospective, interleaved.** From a start date, a seeded, pre-registered random half of race days goes to
  the holdout and the other half to development. Both sides grow at the same rate and the price half gets data at
  once; seasonal regimes are shared between them and the discipline must be strict.
- **B. A historical block** (the candidate is 2024-01-01 to 2026-11-15, about three seasons). Available now and fair
  for hypotheses from the pattern search; contaminated for anything HKTrack or the aggregates measured; usable
  once. Its only pre-race board is a single licensed tote snapshot taken well before the off, on which HKTrack
  already fitted and scored for 2024 to 2026 (HKTrack DECISIONS #27); it has no boards at the near-off decision
  times, so it can test probability claims, and betting rules only at that snapshot's time.
- **C. Hybrid.** Use B once as a confirmation set for a pre-registered list of probability claims from the pattern
  search, which may demote a hypothesis but never adopt one; use A1 or A2 as the only holdout for adoption and for
  every betting rule. An early, honest read without spending the prospective holdout; two governance objects and a
  rule for which hypotheses may touch B.

Answer: A2. Prospective, interleaved: from the start date a seeded, pre-registered random half of race days is the
holdout and the other half development.

OWNER DECISION (D12): What are the Japan holdout's dates and size?

- The start date (for A1 or A2), or the block's start and end (for B).
- Its size before the first opening: a fixed end date, a number of races, or a power target (the races needed to
  detect a stated gain at the D7 level).
- Whether a spent holdout rolls forward (a fresh block after each opening) or stays spent.
- A fixed date is simple and verifiable; a power target opens the holdout only when it can answer, but its end is
  not known in advance.

Answer: Start 2026-11-28 (the next JRA race day). Opened only when it reaches a power target (the races needed to
detect a stated gain at the D7 level). After an opening, a fresh block rolls forward from that date.

OWNER DECISION (D13): What is Hong Kong's holdout?

- **Prospective from resumption.** All Hong Kong history is already touched, so the only clean holdout begins when
  Hong Kong work resumes; nothing can be adopted for Hong Kong until it has accumulated.
- **Prospective from now**, with the data ported but no model work. Accumulates races while Hong Kong is paused;
  requires the Hong Kong ingest to keep running.
- **Defer the choice** until Hong Kong resumes. No cost now; Hong Kong history keeps being the only data.

Answer: Prospective from now: the Hong Kong data keeps being ingested while Hong Kong model work is paused, so clean
races accumulate.

OWNER DECISION (D14): May holdout-period price paths train the price half?

- **No.** The holdout is fully locked, outcomes and prices alike. Cleanest; the price model trains only on
  development days, which are few at first.
- **Yes, paths only, never outcomes.** The price model can learn from every logged day because the final price is
  not an outcome; the outcome half is still judged cleanly. The risk is that path features correlate with outcomes
  (informed late money), so a price model fitted on holdout paths partly sees the holdout's results.
- **Yes, with a lag.** Holdout paths become training data only after a set delay or after the holdout is spent.
  A compromise; more bookkeeping.

Answer: No. Holdout days are fully locked, outcomes and prices alike.

OWNER DECISION (D15): Who may open a holdout, how often, and what does a use mean?

- **Once per adoption decision**, by the owner, for a pre-registered claim. Keeps the holdout's evidential value;
  slow.
- **Once per pre-registered family** (several claims scored together, with a multiple-testing correction).
  Spends the holdout on fewer openings; a family's correction is wider.
- **Counted uses with a budget** (for example a fixed number of openings, each logged, after which the holdout is
  spent). Flexible; every opening weakens the next.
- In every option, a use is logged whether or not the result is reported, and a failed claim counts as a use.

Answer: Once per pre-registered family, opened by the owner, scored together with the D17 correction; every opening
is logged and a failed claim counts as a use.

Amended 2026-11-27 (epic #447, owner sign-off): opening stays the owner's, through `LockedHoldout.open`, once per
pre-registered family at its power target (D12 unchanged). A use checks whether the leaderboard's ranking, built on
development days, holds up on the block; it is not a pass/fail bar and adopts or rejects nothing. Every opening is
logged and counts as a use whatever it shows.

## 6. Adoption rule

What the brief fixes:
- **Model adoption is an owner decision**, taken per adoption on the evidence the statement prescribes.
  - Amended 2026-11-27: nothing is adopted or rejected (D16 as amended). Every option, model and member is a row on
    the country's leaderboard, ranked by its return at the final price; the owner reviews the leaderboard, and the
    top-ranked options are shadowed and played (D19 as amended).
- **Every modeling PR carries an experiment report**: the hypothesis; every variant tried, including failures, with
  confidence intervals; the alternatives rejected and why; what the result rules out; and the follow-ups, filed as
  tickets.
- **Each country keeps a model card** describing the current pipeline end to end.
- A betting rule is judged at the final dividends, after takeout, breakage, the minimum payout and own price
  impact, with selection made only from information at t. Any selection variable taken from the final odds is
  look-ahead.

OWNER DECISION (D16): What evidence adopts a model?

- **Holdout gain with an interval above zero.** The lower bound of the log-loss gain over the final price on the
  holdout, at the D7 level, is above zero. Simple and the brief's metric; a gain just above zero can be too small
  to bet on.
- **Holdout gain above a minimum effect size** (a stated gain per race, or the minimum detectable effect). Adopts
  only gains that could matter; the threshold is a judgement.
- **Development consistency as well:** a gain in most development folds, not only on average, and no recent
  decay. Guards against one good season; costs power.
- **Encompassing as well:** in a combination of the model and the final price, the model's weight is positive with
  an interval above zero. Directly asks "does it add anything the price lacks"; a second test on the same data.
- **For a component of a stack:** the gain of the full stack over the stack without it, on the same bootstrap
  indices. Avoids adopting components that only duplicate others; costs a refit per component.

Answer: On the holdout: the lower bound of the log-loss gain over the final price is above 0 at the D7 level, AND in
a model + final-price combination the model's weight is positive with its interval above 0; a stack component must
also show the full-stack gain over the stack without it.

Amended 2026-11-27 (epic #447, owner sign-off): nothing is adopted or rejected. Every option (race type × pool ×
combination × rule) and every model and member is a row on the country's leaderboard, ranked by its expected return
per ¥100 staked at the final price (walk-forward, takeout and breakage included), shrunk within its pool, with its CI
and the number of variants searched beside it. A model or member ranks by the return of the bets it selects and is
listed unranked until one is measured. An option whose selection is read from the final price (look-ahead) ranks in
the same list, labelled by its selection (decision-time or look-ahead). The log-loss gain over the final price, the
encompassing weight and the full-stack gain stay reported diagnostics, not thresholds. The top-ranked options are
shadowed and played (D19).

OWNER DECISION (D17): How are many candidate rules and variants tested without mining noise?

- **One pre-registered primary per experiment.** The simplest valid inference; everything else is exploratory and
  cannot adopt.
- **Pre-registered families with a max-statistic test:** Hansen's SPA ("does any rule beat the benchmark") and
  Romano-Wolf StepM ("which ones"), with the bootstrap over meetings or days (D7). Controls for the family size;
  needs every variant registered before scoring, including the ones dropped.
- **Hierarchical shrinkage of segment returns**, with the favourite-longshot bias as fixed effects by odds band.
  Gives the probability that a segment beats the takeout; pulls most segment returns back to the average, which is
  the point.
- **False-discovery screens (Benjamini-Hochberg or Benjamini-Yekutieli)** as a screen only, not a decision tool.
- Any option: the number of rules tried is reported with every result.

Answer: Pre-registered families tested with Hansen SPA and Romano-Wolf StepM, bootstrapped over meetings; segment
returns shrunk hierarchically with odds-band fixed effects; FDR only as a screen; the number of rules tried reported
with every result.

Amended 2026-11-27 (epic #447, owner sign-off): multiple testing is handled by shrinkage of the ranking (#78's
hierarchical model within a study, the leaderboard's per-pool shrinkage across studies) and by the count of variants
searched beside every row. SPA, StepM and FDR stay reported diagnostics and gate nothing. Every variant is still
registered before scoring, failures included.

OWNER DECISION (D18): Are betting rules split into two families with different evidence standards?

- **Yes: loss-avoidance rules and positive-edge rules.** A loss-avoidance rule ("never bet at the minimum payout",
  "use the Trio instead of six Trifectas when short") is compared with a stated alternative on the same races, by
  the paired difference in realised return, one-sided; it only has to lose less and never claims profit. A
  positive-edge rule must show a realised return above zero at final dividends on the locked holdout, on top of
  the log-loss evidence. Paired comparisons have far more power, so useful loss-avoidance rules can be adopted
  years before any edge is shown; the cost is two standards to keep apart.
- **No: one standard for every rule** (positive realised return on the holdout). Simple; no loss-avoidance rule can
  be adopted until an edge exists, which may be never.

Answer: Yes. Loss-avoidance rules are adopted on a one-sided paired difference vs a stated alternative on the same
races and never claim profit; positive-edge rules need a realised return above 0 at final dividends on the holdout,
on top of the log-loss evidence.

Amended 2026-11-27 (epic #447, owner sign-off): one leaderboard, not two evidence standards. A loss-avoidance rule and
its stated alternative are each a row, with the paired difference reported beside them; no rule needs a holdout
return before it is played.

OWNER DECISION (D19): What are the staking rungs, and what evidence moves a rule up a rung?

- **Rungs proposed for choice:** avoid (the default for every race and pool); shadow (paper bets, logged with the
  claimed value, the board at t and the result); flat minimum stake; flat small stake under a cap; shrunk
  fractional Kelly under a cap.
- **The entry evidence for each rung** (to be set): for example, flat minimum needs the D16 evidence for the
  rule's probability claim and a shadow calibration slope whose interval includes 1; flat small adds a positive
  realised return on the holdout; Kelly adds a further prospective period and an edge large enough that the shrunk
  stake is materially above zero.
- **The staking method at the top rung:** fractional Kelly (a fixed fraction) or shrunk Kelly (the fraction falls
  with the estimate's uncertainty). At today's edge-to-noise ratio both give about zero, so the choice only matters
  once an edge exists; overestimating the edge by more than a factor of two makes Kelly growth negative.
- **Caps:** per ticket, per race and per day.

Answer: Five rungs: avoid (default), shadow, flat minimum, flat small under a cap, shrunk fractional Kelly under a
cap. Flat minimum needs the D16 evidence and a shadow calibration slope whose interval includes 1; flat small adds a
positive realised holdout return; Kelly adds a further prospective period. Caps per ticket, race and day are set
later on a ticket.

Amended 2026-11-27 (epic #447, owner sign-off): three rungs. Avoid (every option not played); shadow (any option, on
paper); played: the top-ranked options get small flat stakes placed by hand by the owner, at most ¥30,000 per race day
over all venues, every bet logged in the manual bet log and joined to the final dividends. No Kelly or bankroll-scaled
stake; no code places a bet. The flat stake is the owner's, in `countries/japan/config/betting.yaml`: ¥1,000 per bet
(owner, 2026-11-27, ticket #448). A placed bet is a fact: a bet past the cap or off the flat stake is recorded and
flagged. A bet on a holdout day is logged and counts toward the cap, but is settled only when the owner opens the
holdout (D14 unchanged). A look-ahead option may be played through its board-at-t proxy, labelled as such.

## 7. Out of scope

What the brief fixes:
- **Japan is active.** Hong Kong has its data ported and is otherwise paused. Other countries come later, each with
  its own data-access research ticket.
- **Not ported from HKTrack:** features, models, betting rules and the phone page. Each comes back only through a
  ticket that justifies it, and no HKTrack assumption is carried over without a ticket that tests it.
- **Not in this statement:** the choice of any model, feature or betting rule; real-money staking before a rule is
  adopted; private information, which reaches the pool only as money and is read, if at all, from price moves.
  - Amended 2026-11-27: no rule is adopted any more (D16 as amended), and real-money staking is in scope only as D19
    as amended sets it: small flat stakes on the top-ranked options, placed by hand by the owner, at most ¥30,000 per
    race day, every bet logged. No code places a bet.

OWNER DECISION (D20): Are NAR (local government) tracks in scope for Japan?

- **No, JRA only.** One operator, one set of rules and the existing data; leaves out many races.
- **Yes, as a later country-like extension** with its own data-access ticket. Keeps this statement JRA-only while
  leaving the door open; NAR pools are smaller, so own price impact and data quality differ.
- **Yes, now.** More races for the holdout and for power; new sources, rules and takeouts before any JRA result.

Answer: Later, as an extension: this statement is JRA-only; NAR comes in with its own data-access ticket, takeouts
and holdout.

## 8. Governance

What the brief fixes:
- **Owner decisions, never Rulings:** target and estimand, loss and metric, baseline, splits, holdout use, feature
  inclusion, model adoption, staking and betting rules. They are taken through `needs-input` on the ticket or a
  pre-registered spec the owner approves.
  - Amended 2026-11-27: the decisions that stay the owner's, taken through `needs-input` and never a Ruling, are this
    statement's decisions and their amendments, every holdout opening, staking and caps, and any D21 publication
    question. D21 bars licensed-source figures from the public repository: no `needs-input` answer or ticket
    approval permits one, and changing that takes an amendment of D21 itself, signed off under D21's own amendment
    rule and logged in the change log. There is no adoption to decide (D16 as amended). Claude pre-registers each study,
    its spec frozen before any scored read, logs every variant in the registry and runs the study without waiting
    for sign-off; the study's modeling choices live in that frozen spec, never in a `Ruling:` line.
- **Sign-off level:** this statement as a whole, plus each adoption decision.
  - Amended 2026-11-27: sign-off is on this statement and its amendments. There are no adoption decisions (D16 as
    amended) and studies do not stop for sign-off; the owner reviews the leaderboard.
- **A `Ruling:` is only for a reversible implementation choice** (code layout, formats, tooling). A Ruling never
  settles any item in the list above.
- **A known-vs-assumed ledger** (#12) labels every claim as measured (with its source), single-run or assumption.
  The claims of section 1.1 enter it.

Added 2026-11-27 (epic #447, owner sign-off): **keep everything.** No dataset, feature, member or result, nulls and
failed variants included, is deleted or excluded from the record. A kill rule becomes "ranks low": the option stays a
row on the leaderboard. Parking a ticket sets the order of work only.

The owner-decision list in this statement: D1 to D21, in the table below, all answered at sign-off on 2026-11-24.
The evaluation-harness tickets that depend on them (scoring #69, folds #70, holdout #71) now have the decisions
they waited for.

OWNER DECISION (D21): Two governance points: may figures derived from the licensed data source appear in this
public repository, and how is this statement amended after sign-off?

- **Licensed-source figures, option 1: never.** Public documents cite them by internal document and decision
  number only, as section 1.1 does. Safe under the licence; public readers cannot check the numbers.
- **Licensed-source figures, option 2: aggregate metrics only** (a log-loss gain or an interval, never a
  coefficient, a row or a per-runner value). More informative; needs a reading of the licence terms that permits
  it.
- **Amendments, option 1:** any change to a signed section needs a new owner sign-off on a ticket, with a dated
  change log at the end of this file. Strict; slows small corrections.
- **Amendments, option 2:** factual corrections (a takeout rate, a typo) need no sign-off but are logged; any
  change to a decision needs sign-off. Faster; the line between "factual" and "decision" must be drawn.

Answer: Licensed-source figures never appear in the public repository (cite by internal document and decision number
only). Amendments: factual corrections need no sign-off but are logged in a dated change log at the end of the file;
any change to a decision needs owner sign-off on a ticket.

## Decisions summary

| ID | Question | Decision | Section |
|---|---|---|---|
| D1 | Estimand form: condition on the final price and integrate over a nowcast, condition on the board at t, or a residual on the final price | A: condition on the final price, integrate over a nowcast of the final pools made at t | 1. Decision problem |
| D2 | Decision time t and the horizons reported | Grid T-60, T-30, T-16, T-5, T-2, T-1; T-2 primary | 1. Decision problem |
| D3 | Pools covered first | Win, Place and Wide first (tier 1); all other JRA pools in research scope at lower priority (tier 2), amended 2026-11-24 (#127) | 1. Decision problem |
| D4 | How the final Win price becomes a probability (normalised 1/odds, power, Shin, calibrated) | Calibrated market (spline in ln q), fitted in-fold only | 2. Baseline |
| D5 | Baseline for Place, Wide and the exotic pools | Both: pool's own final prices primary, Win-derived ordering secondary | 2. Baseline |
| D6 | The outcome scored and the pseudo-R² reference | Pool-matched primary (winner for Win, top-3 order for Place and Wide), both reported; pseudo-R² vs equal chances plus nats per race over the final price | 3. Metric |
| D7 | Bootstrap resampling unit (race, meeting, day, block) and confidence level | Meetings (venue x day); 95%, one-sided | 3. Metric |
| D8 | How the price half (forecasting O_final from info_t) is scored, its baseline, and whether a closing-line-style measure is a live diagnostic | Log predictive density with PIT and coverage; fitted drift baseline; shadow slope as joint check; closing-line-style measure secondary only | 3. Metric |
| D9 | Fold scheme: unit, refit cadence, window, minimum training window, nesting, embargo | Quarter folds, quarterly refit, expanding, 5 seasons minimum, nested OOF, 1-week embargo | 4. Splits |
| D10 | Treatment of races before 2014-06-07 | Only for components that do not touch the price | 4. Splits |
| D11 | Japan holdout design (prospective contiguous, prospective interleaved, historical block, hybrid) | A2: prospective, interleaved (seeded random half of race days) | 5. Holdout |
| D12 | Japan holdout dates and size | Start 2026-11-28; opened at a power target; rolls forward after an opening | 5. Holdout |
| D13 | Hong Kong holdout | Prospective from now; ingest keeps running while model work is paused | 5. Holdout |
| D14 | Whether holdout-period price paths may train the price half | No: fully locked, outcomes and prices | 5. Holdout |
| D15 | Who opens a holdout, how often, and what counts as a use | Once per pre-registered family, by the owner, through `LockedHoldout.open` at its power target; a use checks whether the leaderboard's ranking holds up and adopts or rejects nothing; every opening logged and counted (amended 2026-11-27, epic #447) | 5. Holdout |
| D16 | Evidence that adopts a model (the adoption threshold) | Nothing adopted or rejected: every option, model and member is a leaderboard row ranked by its shrunk return per ¥100 at the final price, with its CI and the variants searched; log-loss gain, encompassing weight and full-stack gain are reported diagnostics (amended 2026-11-27, epic #447) | 6. Adoption rule |
| D17 | Multiple-testing policy | Shrinkage of the ranking (#78 within a study, per pool across studies) and the variants searched beside every row; SPA, StepM and FDR reported diagnostics only; every variant registered before scoring (amended 2026-11-27, epic #447) | 6. Adoption rule |
| D18 | Two families of betting rules with separate evidence standards, or one | One leaderboard: a loss-avoidance rule and its stated alternative are each a row, the paired difference beside them; no holdout return needed before play (amended 2026-11-27, epic #447) | 6. Adoption rule |
| D19 | Staking rungs, their entry evidence, method and caps | Three rungs: avoid, shadow, played (top-ranked options, flat stakes by hand, at most ¥30,000 per race day, every bet logged and joined to the final dividends); no Kelly; no code places a bet (amended 2026-11-27, epic #447) | 6. Adoption rule |
| D20 | NAR tracks in or out of scope | Later, as an extension with its own ticket; this statement is JRA-only | 7. Out of scope |
| D21 | Licensed-source figures in the public repo; amending this statement | Never; factual corrections logged without sign-off, decision changes need owner sign-off | 8. Governance |

## Change log

- 2026-11-24: signed off; D1-D21 answered on ticket #14; seven review corrections applied (listed):
  1. Section 1.1: the Hong Kong blend weight's unconstrained value is stated as an inference (zero or negative),
     not as a measured slightly negative estimate.
  2. Section 1: the forecast-of-final-odds bias states both conditions: `0.8(e^{σ²/2} - 1)` for a forecast
     unbiased in log terms (median), `0.8(e^{σ²} - 1)` for one unbiased in level.
  3. Section 1: the minimum-payout break-even uses P(place), and the Place text notes that the places paid depend
     on field size (to confirm in the ledger).
  4. Section 3: scoring the price half separately is moved out of "What the brief fixes" and tied to D1 and D8.
  5. Section 5: the "not used for development" line is stated as consistent with D14 (No).
  6. Section 5.1: the exposure table describes exposure instead of assigning a holdout use.
  7. D11 option B: the historical block's tradeoff notes the licensed pre-race tote snapshot HKTrack fitted and
     scored on (HKTrack DECISIONS #27).
- 2026-11-24: D3 amended by the owner (ticket #127): all pools in scope, Win/Place/Wide first.
- 2026-11-27: D15, D16, D17, D18 and D19 amended by the owner (epic #447, owner sign-off; ticket #448): every option
  is ranked on a leaderboard by its return at the final price instead of adopted or rejected, every result is kept,
  and the top-ranked options are played with small flat stakes by hand under a ¥30,000 race-day cap. Sections 6, 7
  and 8 carry dated notes to match, and section 8 gains the keep-everything rule. The owner's words: "I don't want
  to throw away any data even if it doesn't cross some threshold. All of it is useful. I fully expect most if not
  all findings to not show consistent profitability... We are trying to find the BEST options among the worst. This
  is a fun data experiment not a full time job."
