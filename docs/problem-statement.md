# Problem statement

Status: draft, awaiting owner sign-off

This document says what HorseTrackBetting is trying to predict, what it is judged against, how it is tested and
when a model or a betting rule may be adopted. Claude drafted it (ticket #14). Nothing in it is adopted until the
owner signs it off.

How to read it:
- Each section first states what the brief already fixes. Those lines are not open.
- Each open choice is a line `OWNER DECISION (Dn): <question>`, followed by the options with their tradeoffs and an
  empty `Answer:` line. The options are listed in no order of preference, and none is chosen here.
- The owner answers by number (D1 to D21). The table at the end lists every decision and its section.

Sources: the HorseTrackBetting brief (HKTrack `docs/research/25-horsetrackbetting-brief.md`, section "Modeling
governance"), the HKTrack process review (`docs/research/24-process-review-2026-11-22.md`, sections 1 to 4 and 7),
HKTrack `docs/DECISIONS.md` #26 and #27, and an internal research synthesis (not published).

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
  joint finishing distribution and the final pools, not a single marginal probability and a single price.

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
  of ¥110 a Place bet needs P(top 3) > 0.909 to have value; at ¥100 it can never have value.
- **Closing-line value is degenerate on a tote.** Every bet is paid at the closing price, so "beating the closing
  line" is identically zero and cannot be the live metric. The price half needs its own score (D8).
- **Pricing on a forecast of the final odds is biased upward.** Multiplying a probability by a point forecast of
  the final odds overstates value even when the forecast is unbiased in log terms. Under an efficient final price
  and a lognormal forecast with log-scale standard deviation σ, the bias for Win at a 20% takeout is about
  `0.8(e^{σ²} - 1)`: about +1.8 points of EV at σ = 0.15 and +4.0 at σ = 0.22. It adds to winner's-curse
  selection: picking the bets with the largest claimed value picks the largest errors.
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
  that forbade negative weights; without the bound the estimate was slightly negative.
- Every betting rule tested lost money at the final dividends, and a large search over segments, pools and ticket
  combinations found nothing that beat the takeout.
- The Win rule HKTrack shipped claimed a clearly positive expected value; with the same probabilities at the
  actual final odds the expected value was negative. The rule priced bets as a probability fitted on the board
  times a forecast of the final odds (the bias above, plus selection), and it was judged against the board instead
  of the final price.

See HKTrack `docs/research/24-process-review-2026-11-22.md` sections 1 to 4, HKTrack `docs/DECISIONS.md` #26 and
#27, and the brief's "Why". The figures themselves are not reproduced here: the models behind them used inputs from
a licensed data source whose terms forbid publishing its outputs (see D21).

The claims this null result rests on, to be cross-referenced to the known-vs-assumed ledger (#12):
1. The form model's log-loss gain over the JRA final odds is small and its interval only just excludes zero
   (one model; its test seasons were reused for model selection, so this is at best single-run).
2. The decision-time board is less sharp than the final price, and model plus board trails the final price.
3. The Hong Kong blend weight of zero is a bound-constrained estimate, not an unconstrained zero.
4. Every betting rule tested lost money at final dividends.
5. A large hypothesis search found nothing beating the takeout (its period is not stated in the source).
6. The shipped Win rule's claimed value reversed sign at the actual final odds.
7. The late price-move sizes quoted in the process review come from one day at one venue and from flagged tickets,
   so they are single-run and selection-conditional.
8. The JRA takeout schedule and the minimum-payout rule (public rules; measured).
9. Closing-line value is zero by construction on a tote (arithmetic).

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

Answer:

OWNER DECISION (D2): What is the decision time t, and which other horizons are reported?

- **The last bettable board (about T-1).** Closest to the final price, so the smallest price uncertainty; leaves no
  time for a person to act, so it only suits automated placement.
- **A fixed earlier time (T-2, T-5 or similar).** Leaves time to act; more price uncertainty and more late money
  still to come, which the price half must model.
- **A grid (for example T-60, T-30, T-16, T-5, T-2, T-1).** Shows how value and price uncertainty change toward the
  off and supports a time-varying blend; costs more logging and more scoring, and multiplies the number of
  comparisons (D17).

Answer:

OWNER DECISION (D3): Which pools does this statement cover first?

- **Win only.** The cleanest estimand and the most history; leaves out the pools where the co-placer structure
  might matter.
- **Win and Place.** Adds the pool where the co-placer dividend matters most; needs the joint top-three
  distribution and the Place pool's own baseline (D5).
- **Win, Place and Wide.** Covers the pools HKTrack bet; Wide price paths exist only from the project's own logs.
- **All pools including exotics.** Most opportunities and the highest takeouts; exotic boards before the decision
  exist only if logged going forward, and own price impact matters for long combinations.

Answer:

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

Answer:

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

Answer:

## 3. Metric

What the brief fixes:
- The loss is **log loss**, and the reported gain is the **log-loss (pseudo-R²) gain over the final-price
  baseline**, per race and paired: the same races, the same folds.
- Every gain carries a **paired bootstrap confidence interval**: the model and the baseline are resampled with the
  same indices, so the interval is on the difference.
- Two halves are scored: the **outcome half** (the probability of the result) and the **price half** (the forecast
  of `O_final` from info_t). A bet's value needs both (section 1).

OWNER DECISION (D6): Which outcome is scored, and against what is pseudo-R² defined?

- **The winner only** (conditional-logit log loss per race). Matches the Win pool and HKTrack's numbers; uses one
  bit of each race.
- **The top-three order** (a discounted Plackett-Luce likelihood). Uses more of each race and matches Place and
  Wide; mixes in an ordering model's error.
- **Both, with one named primary.** Covers both pools; two primary results invite choosing the better one.
- Pseudo-R² reference, for any of the above: relative to equal chances for every runner (comparable with the
  literature) or relative to the final-price baseline (reads directly as the gain). The second makes small gains
  look large; the first makes them look tiny.

Answer:

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

Answer:

OWNER DECISION (D8): How is the price half (forecasting `O_final` from info_t) scored?

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

Answer:

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

Answer:

OWNER DECISION (D10): How are races before 2014-06-07 (the earlier, race-varying takeout regime) treated?

- **Include with an era term.** More history for models that do not depend on the price; the market's behaviour
  differs across the reform, and the era term must be fitted.
- **Exclude.** One takeout regime throughout; loses about four seasons.
- **Include only for components that do not touch the price** (for example a horse-ability model), and exclude
  them wherever the market or a dividend enters. Keeps the history where it is harmless; two training windows to
  maintain.

Answer:

## 5. Holdout

What the brief fixes:
- **A locked final holdout per country.** It is cut before any model trains, is not used for development, model
  selection or tuning, and is opened only by owner decision.
- **Every use of it is logged**: the date, the ticket, the hypothesis, the code version and the result (ticket #71
  builds the lock and the log).

### 5.1 Exposure so far (disclosure)

The owner has not chosen a holdout. Before choosing, these are the periods already seen:

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

| Period | Touched by | Status |
|---|---|---|
| 2010-01 to 2023-12 | HKTrack training and test seasons; the pattern search; the aggregates | Development only |
| 2024-01-01 to 2026-11-15 | HKTrack test seasons and research; the aggregates | Not touched by the pattern search, so a fair test for hypotheses it generated; not fair for anything HKTrack or the aggregates measured |
| 2026-11-16 to 2026-11-23 | The Kyoto snapshots; HKTrack live scoring of 2026-11-22; the WIN5 page | Touched |
| After 2026-11-23 | Nothing | Untouched |

No historical period of either country is untouched. Only race days after 2026-11-23 are untouched in Japan.

OWNER DECISION (D11): What kind of holdout does Japan use?

- **A1. Prospective, contiguous.** Every JRA race from a start date on. Logged days before the start are
  development for the price half. Untouched by anyone, and every holdout race has a logged decision-time board,
  which is the only data on which a betting rule can be judged without look-ahead. Slow to reach power: on the
  order of a year of all-venue racing to detect a gain of a few thousandths of a nat per race. The price half
  starts with few logged days, and the holdout starts later.
- **A2. Prospective, interleaved.** From a start date, a seeded, pre-registered random half of race days goes to
  the holdout and the other half to development. Both sides grow at the same rate and the price half gets data at
  once; seasonal regimes are shared between them and the discipline must be strict.
- **B. A historical block** (the candidate is 2024-01-01 to 2026-11-15, about three seasons). Available now and fair
  for hypotheses from the pattern search; contaminated for anything HKTrack or the aggregates measured; usable
  once; and it has no decision-time boards, so it can test probability claims but not betting rules.
- **C. Hybrid.** Use B once as a confirmation set for a pre-registered list of probability claims from the pattern
  search, which may demote a hypothesis but never adopt one; use A1 or A2 as the only holdout for adoption and for
  every betting rule. An early, honest read without spending the prospective holdout; two governance objects and a
  rule for which hypotheses may touch B.

Answer:

OWNER DECISION (D12): What are the Japan holdout's dates and size?

- The start date (for A1 or A2), or the block's start and end (for B).
- Its size before the first opening: a fixed end date, a number of races, or a power target (the races needed to
  detect a stated gain at the D7 level).
- Whether a spent holdout rolls forward (a fresh block after each opening) or stays spent.
- A fixed date is simple and verifiable; a power target opens the holdout only when it can answer, but its end is
  not known in advance.

Answer:

OWNER DECISION (D13): What is Hong Kong's holdout?

- **Prospective from resumption.** All Hong Kong history is already touched, so the only clean holdout begins when
  Hong Kong work resumes; nothing can be adopted for Hong Kong until it has accumulated.
- **Prospective from now**, with the data ported but no model work. Accumulates races while Hong Kong is paused;
  requires the Hong Kong ingest to keep running.
- **Defer the choice** until Hong Kong resumes. No cost now; Hong Kong history keeps being the only data.

Answer:

OWNER DECISION (D14): May holdout-period price paths train the price half?

- **No.** The holdout is fully locked, outcomes and prices alike. Cleanest; the price model trains only on
  development days, which are few at first.
- **Yes, paths only, never outcomes.** The price model can learn from every logged day because the final price is
  not an outcome; the outcome half is still judged cleanly. The risk is that path features correlate with outcomes
  (informed late money), so a price model fitted on holdout paths partly sees the holdout's results.
- **Yes, with a lag.** Holdout paths become training data only after a set delay or after the holdout is spent.
  A compromise; more bookkeeping.

Answer:

OWNER DECISION (D15): Who may open a holdout, how often, and what does a use mean?

- **Once per adoption decision**, by the owner, for a pre-registered claim. Keeps the holdout's evidential value;
  slow.
- **Once per pre-registered family** (several claims scored together, with a multiple-testing correction).
  Spends the holdout on fewer openings; a family's correction is wider.
- **Counted uses with a budget** (for example a fixed number of openings, each logged, after which the holdout is
  spent). Flexible; every opening weakens the next.
- In every option, a use is logged whether or not the result is reported, and a failed claim counts as a use.

Answer:

## 6. Adoption rule

What the brief fixes:
- **Model adoption is an owner decision**, taken per adoption on the evidence the statement prescribes.
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

Answer:

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

Answer:

OWNER DECISION (D18): Are betting rules split into two families with different evidence standards?

- **Yes: loss-avoidance rules and positive-edge rules.** A loss-avoidance rule ("never bet at the minimum payout",
  "use the Trio instead of six Trifectas when short") is compared with a stated alternative on the same races, by
  the paired difference in realised return, one-sided; it only has to lose less and never claims profit. A
  positive-edge rule must show a realised return above zero at final dividends on the locked holdout, on top of
  the log-loss evidence. Paired comparisons have far more power, so useful loss-avoidance rules can be adopted
  years before any edge is shown; the cost is two standards to keep apart.
- **No: one standard for every rule** (positive realised return on the holdout). Simple; no loss-avoidance rule can
  be adopted until an edge exists, which may be never.

Answer:

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

Answer:

## 7. Out of scope

What the brief fixes:
- **Japan is active.** Hong Kong has its data ported and is otherwise paused. Other countries come later, each with
  its own data-access research ticket.
- **Not ported from HKTrack:** features, models, betting rules and the phone page. Each comes back only through a
  ticket that justifies it, and no HKTrack assumption is carried over without a ticket that tests it.
- **Not in this statement:** the choice of any model, feature or betting rule; real-money staking before a rule is
  adopted; private information, which reaches the pool only as money and is read, if at all, from price moves.

OWNER DECISION (D20): Are NAR (local government) tracks in scope for Japan?

- **No, JRA only.** One operator, one set of rules and the existing data; leaves out many races.
- **Yes, as a later country-like extension** with its own data-access ticket. Keeps this statement JRA-only while
  leaving the door open; NAR pools are smaller, so own price impact and data quality differ.
- **Yes, now.** More races for the holdout and for power; new sources, rules and takeouts before any JRA result.

Answer:

## 8. Governance

What the brief fixes:
- **Owner decisions, never Rulings:** target and estimand, loss and metric, baseline, splits, holdout use, feature
  inclusion, model adoption, staking and betting rules. They are taken through `needs-input` on the ticket or a
  pre-registered spec the owner approves.
- **Sign-off level:** this statement as a whole, plus each adoption decision.
- **A `Ruling:` is only for a reversible implementation choice** (code layout, formats, tooling). A Ruling never
  settles any item in the list above.
- **A known-vs-assumed ledger** (#12) labels every claim as measured (with its source), single-run or assumption.
  The claims of section 1.1 enter it.

The owner-decision list in this statement: D1 to D21, in the table below. Sections 2 to 7 cannot be implemented
until their decisions are answered; the evaluation-harness tickets that depend on them (scoring #69, folds #70,
holdout #71) stay blocked until sign-off.

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

Answer:

## Decisions summary

| ID | Decision | Section |
|---|---|---|
| D1 | Estimand form: condition on the final price and integrate over a nowcast, condition on the board at t, or a residual on the final price | 1. Decision problem |
| D2 | Decision time t and the horizons reported | 1. Decision problem |
| D3 | Pools covered first | 1. Decision problem |
| D4 | How the final Win price becomes a probability (normalised 1/odds, power, Shin, calibrated) | 2. Baseline |
| D5 | Baseline for Place, Wide and the exotic pools | 2. Baseline |
| D6 | The outcome scored and the pseudo-R² reference | 3. Metric |
| D7 | Bootstrap resampling unit (race, meeting, day, block) and confidence level | 3. Metric |
| D8 | How the price half (forecasting O_final from info_t) is scored, and its baseline | 3. Metric |
| D9 | Fold scheme: unit, refit cadence, window, minimum training window, nesting, embargo | 4. Splits |
| D10 | Treatment of races before 2014-06-07 | 4. Splits |
| D11 | Japan holdout design (prospective contiguous, prospective interleaved, historical block, hybrid) | 5. Holdout |
| D12 | Japan holdout dates and size | 5. Holdout |
| D13 | Hong Kong holdout | 5. Holdout |
| D14 | Whether holdout-period price paths may train the price half | 5. Holdout |
| D15 | Who opens a holdout, how often, and what counts as a use | 5. Holdout |
| D16 | Evidence that adopts a model (the adoption threshold) | 6. Adoption rule |
| D17 | Multiple-testing policy | 6. Adoption rule |
| D18 | Two families of betting rules with separate evidence standards, or one | 6. Adoption rule |
| D19 | Staking rungs, their entry evidence, method and caps | 6. Adoption rule |
| D20 | NAR tracks in or out of scope | 7. Out of scope |
| D21 | Licensed-source figures in the public repo; amending this statement | 8. Governance |
