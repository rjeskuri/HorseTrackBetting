# Model card: Japan

The current pipeline for this country, end to end, in data-science terms. This is a skeleton: nothing below is decided until the problem statement and the adoption history say so.

Diagrams of the modeling (the decision, the C0-C6 stack, the meta-model's members, Place/Wide/exotics, adoption, the race-day timeline): [docs/modeling.md](../../../../docs/modeling.md).

## Status

Status: no model adopted

## Decision problem

Not decided here: see [the problem statement](../../../../docs/problem-statement.md).

## Data and as-of contract

Sources, their licences, and the as-of rule (what may be known at prediction time): to be filled in. Not decided: see [the problem statement](../../../../docs/problem-statement.md).

## Target and estimand

Not decided: see [the problem statement](../../../../docs/problem-statement.md).

## Baseline

Not decided: see [the problem statement](../../../../docs/problem-statement.md).

## Features

None adopted.

- Workout feature set (#137, `handicap/workouts.py`): best final-furlong time against the course par, its trend over the last three workouts, days since the last hard workout, and `course_kinds_since_last_race` (the sum of CYB's per-course presence flags: the course kinds used since the last race, not a workout count), from JRDB CHA and CYB as of decision time. It exists as an opt-in JRDB feature set behind `HORSETRACK_JRDB_FEATURES` and is not adopted. JRDB's terms allow private use only, so its development-data log-loss report stays under the data dir.

## Model

None adopted.

### C1: final-price nowcast (#85; not adopted)

- C1 structural nowcast (#85, `models/nowcast.py`): a forecast of each pool's final shares from the board at t = post − τ, read from rows in `logger.compact.SCHEMA` (a row counts from its `snapshot_utc`; each combination takes its latest row at or before t). Shares are money over the pool: per-combination money when the rows carry it, else 1/O normalised for Win and, for the stake-back-share pools (Place, Wide), the figures read by JRA's rule (`core.pricing.stake_back_share_dividend`): a Place board with high figures by least squares over both figures (`place_money_fit`, #410), else the low figures inverted in closed form; a Place board's places paid P come from the field at the start of sales, the runners priced on the race's earliest Place board, so a runner scratched after sales open leaves P as it was (#410). A joint fit whose figures above the floor hold some runners' money only as a sum (an SVD rank test of its equations) keeps its minimum-norm shares and is flagged `undetermined` by `place_shares`; the rank is computed only where the read is named (`place_shares`, each case's closing board), not in `pool_shares`, `race_view` or the draws. The model is s_final = φ_t s_t + (1 − φ_t) s_late. φ_t = V_t / V_final comes from a pool-growth profile, ln(V_final / V_m) = h(m) exp(θ·x): h is piecewise linear in the board's minutes to post m, and x holds venue, race number and grade (ridge). The late composition is logistic-normal, s_late = softmax(ln s_t + β·f + σ ε). Its features f are the share velocities over [t−5, t] and [t−15, t] and ln of the favourite rank. β comes from least squares within race on the observed late-money ratio. σ, and the random walk's σ, each maximise the training log score of the final ln-shares. There is one (β, σ) cell per pool and τ of the D2 grid. Draws are share vectors with shares ≥ φ_t s_t, so their spread scales with 1 − φ_t, and V_final = V_t / φ_t. The random-walk baseline is softmax(ln s_t + σ_rw ε). The model is pool-generic and fits Wide wherever Wide paths exist. `Nowcast.draws(view, pool=, n=, seed=)` draws its own shocks; `Nowcast.draws_from_shocks(view, pool=, shocks=)`, the interface the decision layer calls, takes per-runner shocks (a Wide pair's (ε_a + ε_b)/√2) so every pool is drawn from one set. `ShadowNowcast` puts a fit behind #116's draws interface, reading a Place board's high figures as `race_view` does, and a pointer file naming a `nowcast_c1` fit switches the shadow values to it (`models/v0/loader.py`). `current.json` still names #116's drift. Fits refuse holdout days, and the D9 fold fits (`fit_folds`) see only their training days. A late scratch follows the owner's #69 rule: a combination priced at t and not at the close is removed, every forecaster's drawn shares are renormalised over the combinations priced at the close, and the case is scored; the fits train on the cases whose board at t is the final board. It is not adopted.
- Its run (`studies/nowcast.py`, `uv run horsetrack japan nowcast-study`, `docs/experiments/nowcast-japan.md`) trains on the JRA-VAN time-series paths in the store, Win and Place, on race days from 2014-06-07 (D10) on D9's folds. It scores the nowcast against the random walk and against #116's drift (D8's price baseline, fitted on the same boards) per pool and τ, by log score and CRPS of the final ln-shares with race-day and meeting bootstrap bounds, PIT and 50%/90% coverage. JRA-VAN is a licensed source, so every figure is withheld (D21). In v2 only Win was sound: its Place inversion used a wrong rule. #410's v3 (family `japan-c1-nowcast-final-shares-v3`) reads Place by JRA's rule. v5 (`japan-c1-nowcast-final-shares-v5`, spec sha256 `a07a5be8…`, git `0f0df42`) measures that rule on the closing boards of the scored Place cases, read as scored, against the final Place votes, and keeps coverage and agreement apart: a determined read stands as a result only when its shares are within a stated tolerance on at least `PLACE_MIN_WITHIN_TOLERANCE` of its validated boards. Its measurement is record `86795d4a880bbd20`; whether each read stands as a result, and so which Place cases the gains hold, is withheld (D21), see the private report. The figures are in the private report `nowcast-japan-a07a5be8273d.md`. v4's verdict (record `8810ca538e68f8e4`) took coverage for agreement and v3's measurement validated other boards; both are superseded. The JRA-VAN series has no Wide pool, and the two logged Wide days carry no post times, so Wide is not fitted.

### C2: dynamic rating (#91; not adopted)

- C2 dynamic rating (#91, `models/rating.py`): each horse's ability is a Gaussian random walk, N(prior_mean, prior_var) at its debut, with drift_var_per_day times the days off added between races and spell_var added after more than spell_days off. A race updates every runner by Weng and Lin's Bradley-Terry full-pair rule (performance variance perf_var, a dead heat at 1/2), in race order, so the state before a race (mu_before, sigma_before = σ_h(t)) reads earlier races only; no smoothed (TTT) value is used. `rating_member` gives the within-race log-probability softmax((mu_before − race mean) / sqrt(scale² + sigma_before²)). It is not adopted. Its run (`studies/rating.py`, `docs/experiments/rating-japan.md`) reads only placings from the JRA results store, picks drift and scale per D9 fold by inner-fold log loss (drift 0, scale 1.5 in every fold), and exports mu_before, sigma_before and the member's logp for every priced starter at each selected pair, with the fold each row is out of fold for (`oof_fold`), under the data dir for the meld (#98). On development folds the member correlates 0.72 with ln q_final, has pseudo-R² 0.12 against the market's 0.26, and takes a negative stage-2 weight beside ln q (α −0.045, meeting-clustered bounds −0.064 and −0.026).

### C2: top-3 Plackett-Luce ability with pooled jockey, trainer and sire (#92; not adopted)

- C2 ability (#92, `models/ability.py`): the finishing order's top 3 as three successive choices, stage s a softmax of u over the runners not placed before s, weighted by discounts (1, w2, w3); runners placed 4th or later, and starters without a placing, stay in every choice set and are never chosen (positions 4+ unused, eased runners). A dead heat counts each co-placed runner at 1/m in the shared stage; at k = 1 the likelihood is the conditional logit. u = β · rating + a jockey, a trainer and a sire effect, each effect N(0, τ²) with τ² by empirical Bayes (MacKay's evidence fixed point), fitted as a penalised MAP by Newton's method. Each runner gets u, its Laplace standard error σ_i (an unseen jockey, trainer or sire at effect 0 with its prior variance) and p_ability = softmax(u) within the race. It is not adopted.
- Its run (`studies/ability.py`, `uv run horsetrack japan ability-study`, `docs/experiments/ability-japan.md`) follows the owner's #92 design (2026-11-26): the rating is #91's member at the pair #91's rule picks per fold; the sire is the JRA-VAN pedigree's; level 1 trains on results only from 2010 and is out of fold from 2015-Q1 on D9's quarters, with the discounts and the rating pair re-selected per fold on inner folds from 2011-Q1; level 2 is #87's stage-2 meta at the final Win price against D4's calibrated market, and the member's C4 marginal is its stage-2 meta against the market-only stage-2 meta (u = β · ln q on the same rows and folds). E33 compares it with the same model trained on the winner only, E34 with #129's as-of connection rates (race days in [day − 365, day − 1]) in place of the pooled effects, on races with a runner whose jockey, trainer or sire has fewer than 30 prior starts. The sire is JRA-VAN data, so every figure is withheld (D21); the per-runner export (u, σ_i, p_ability) is under the data dir for the meld (#98), named by the spec and the data builds.

### C2: lineage prior (#95; built, E37 not yet run)

- C2 lineage (#95, `models/lineage.py`): `relationship_matrix` is the additive relationship A among a set of horses by the tabular method over their ancestors only (inbreeding counted, a founder or an unknown parent at 0). `LineagePrior` fits, on the runs strictly before a time t, the same discounted top-3 Plackett-Luce likelihood as #92 with u = g(sire, surface, band) + ¼ · g(damsire, surface, band), where g(s, cell) = g_sire[s] + d[s, cell], g_sire ~ N(0, τ_sire²) with τ_sire² by empirical Bayes and d ~ N(0, τ_cell²) at the caller's τ_cell, as a penalised MAP. One table serves the sire and the damsire roles. Only stallions with a progeny run before t, and their cells (`sire_cells`), have parameters: an unseen cell takes g_sire, and an unseen or unknown stallion takes 0. The debut prior mean of a horse's ability in a cell is the trainer's effect plus that lineage term (`prior_mean`, `prior_means`); `cell_effects_` is the fitted table per stallion × surface × band, the prior #93's aptitudes can start from. Surfaces and distance bands are the caller's labels, and the bands' cut points belong to E37's frozen spec. The model has no per-horse effect.
- E37's scoring (`studies/lineage.py`): `run_segments` flags each run as debut, first time on the surface or first time in the distance band from that horse's earlier runs only, and `segment_delta_ll` gives each segment's paired per-race winner ΔLL with D7's meeting bounds, standalone (the ability member with the prior against it without) and at the final price (the stage-2 metas). The E37 run on the stores, and the feed of the prior into #92's member for debut and thin-history runners, are not yet built.

### C3: race-level finish speed (#96, scored by #305; not adopted)

C3 forecasts a race's top-3 time at the decision time.
- **Code:** the model is `countries/japan/models/speed.py` (#96). The run is `countries/japan/studies/speed_run.py` (#305).
- **Data:** the store's JRA races and runners (netkeiba results), column-scoped. No JRDB or JRA-VAN input is read.
  Holdout days are dropped before anything else, and jump races are excluded.
- **Target:** `top3_time_s`, the mean finish time of the runners placed 1–3 (owner, #96).
  - It is read per runner: each runner placed 1 or 2 counts once, and the runners placed 3rd share one slot at their
    mean.
  - A race with no 3rd-placed runner uses the runners placed 1–2. There are 110 such races.
- **Model:** a linear mixed model, `top3_time = par(cell) + v(day, track, surface) + ε`.
  - The cell is track × surface × distance × layout × rail × class. Class is a category with no scale imposed (the
    class scale is the owner's), and a missing key is the level `unknown`.
  - The day effect v is a random effect, N(0, τ²). ε is Student-t (5 df in #305's primary run, df not signed) or
    normal.
  - The model is fitted by maximum likelihood (conditional maximisation: Student-t reweighting, then σ² and τ²
    profiled with v integrated out) on each fold's training races only.
- **Prediction at t:** E[top-3 time] = par + the posterior mean of the day's v.
  - The posterior uses the residuals, against the frozen par, of the same (day, track, surface) races whose result is
    available strictly before t.
  - A result is available at post_utc + 30 min, the owner's same-day results rule (#80, #96). The as-of views carry
    the same rule for the finish columns of races and runners since #305 (`config/availability.yaml`); the study
    applies it itself.
  - With no result by t, v is its prior, so the forecast is the par. No variant prior is injected yet; the cushion and
    moisture prior is E29 (#90, #230).
- **Evaluation:** #70's quarterly expanding walk-forward folds (D9). The decision time is T-2 (D2). The metric is the
  time RMSE against the par-only baseline on the same races, with meeting-clustered paired one-sided 95% bounds (D7).
- **Result (#305):** RMSE 1.009 s against the par's 1.242 s, a gain of +0.233 s (lower bound +0.224). That is 37,047
  development races, 2015-Q4 to 2026-11-23 ([docs/experiments/speed-japan.md](../../../../docs/experiments/speed-japan.md)).
  The gain is 0 on the 26% of races with no same-day result by t.
- **Limitations:**
  - Rail is unknown on every race in the store and layout on 88%, so the 'course' part of the owner's request is not
    resolved yet (#90, #329). The run predates #253, which now fills `rail_setting` in the store build from the
    cached JRA going pages; historical rail still needs #90, and the rerun by rail and layout is #329.
  - The 30-minute publication rule is an assumption in the ledger.
  - C3 has no test against the final price yet (E39 marginal, #97).

### C4: meta-model (#87; not adopted)

- C4 meta-model (#87, `models/meta.py` over `models/condlogit.py`): a conditional logit within race at the final Win price. Stage 2 is u_i = Σ_k α_k(z) m_ki + β ln q_i. The offset form is u_i = ln q_i + Σ_k α_k(z) m_ki, where the offset can be ln q or the D4 calibrated market. α and β are unbounded maximum likelihood (08 R13), and dead heats count at 1/k. An optional ridge shrinks only the member weights, toward the owner's target α = 0 (the market alone). Segment weights enter as interaction columns, α_k(z) = α_k0 + Σ_j α_kj z_j, and an optional (ln q)² term is available. The members m_k are out-of-fold level-1 log-probabilities. The meta is itself out of fold on D9's folds (`oof_predictions`, which refuses leaky or overlapping folds), and `predict(frame, q)` takes any price vector, so the decision layer can pass nowcast draws. `ceiling_table` gives E9's ln(π/q) table. Coefficient standard errors are the inverse information (races independent) and, given a cluster column, the meeting-clustered sandwich that D7's unit calls for. It is not adopted. Its first run (`studies/meta.py`, `docs/experiments/meta-japan.md`) used HKTrack's research-24 members. These read JRDB, or are not confirmed free of it, so their results stay under the data dir. In that run the meta trains on member rows from 2023-01-05 only, so D9's 5-season minimum is not met at level 2 on any scored fold (0.98 to 3.73 years), and level 1 is HKTrack's season walk-forward, refit once a season, not quarterly. By the owner's answer (b) on #87 (2026-11-25), that run is a documented failure to meet D9 and supports no adoption claim; nothing is adopted, and the D9-compliant run comes with #91's and #98's members.

### C5: exotic layer (#99, scored by #459; not adopted)

- C5 exotic layer (#99, `models/exotic.py`): one race's Win probabilities p (renormalised over the starters) to every pool's combination probabilities through an ordering of the first three, each stage a softmax over the runners left: Harville; Henery (independent unit-variance normal running times, means solved so the Win probabilities hold, the trifecta by quadrature); quadratic context (HKTrack research 16/19: slope 1 + a + h (mean ln p placed − h0) + e (H − e0), curvature b, per stage, by maximum likelihood); discounted Harville (#115's λ2, λ3). Pool marginals (`pool_array`) give Place by the places of the field at sale, Wide, Quinella, Exacta, Trio and Trifecta. The market baseline (D5, 08 R15) is the pool's own money: `dividend_shares`, R·100/(D + 5) for one winning combination and the statute's de-vig x_c = 1/((1 + K)·P·(r_c − 1)) for several, at the race's own payout rate (boosts in); `dividend_floor_bound` flags race-pools at the 110-yen minimum, where the share is censored.
- #459's run (`studies/exotic_c5.py`, `horsetrack japan exotic-c5`; spec `docs/specs/exotic-c5-prereg.md`, family `japan-c5-exotic-dividends`): on D9's walk-forward quarters from 2020-Q1, three inputs (normalised 1/odds, D4's calibrated market refitted per fold, #87's JRDB-fed `stage2:q:form`) × the four orderings (the fitted two refitted on each fold's training races through the input) × blend (none; α·q + (1 − α)·P_C5 on the paid combination, α per pool and fold on the training races; the log-linear `pool_blend`, not runnable without full boards), scored per pool by the log loss at 1/k per paid combination against the dividends (E43), on super_premium race-pools (E16), and E44's joint against Win gain through one market-fitted ordering. Place and Wide take P × the statute share as the ticket's chance, with floor-bound race-pools dropped or kept. Each no-blend variant is a leaderboard row: a flat ¥100 on its most probable combination at the paid dividend (look-ahead: final Win prices). Results: `docs/experiments/exotic-c5-japan.md`; the `stage2:q:form` figures are withheld (D21).

### C6: shadow decision layer (#100; not adopted)

- C6 shadow decider (#100, `models/decision.py`): the EV per yen of a flat stake on every combination of the Win, Place and Wide boards at t, as the mean over C1's nowcast draws of P × D, never P̄ × D̄ (08 R5). P comes from C4's P(win) on Win draw m, and, for Place and Wide, C5's ordered top three from it; D is the dividend in draw m of the pool's own draws with our stake inside it, the money on a combination being the draw's share × V_final with each read as the decimal it prints as, broken to 10 yen, at least 110 and 100 when the Plus 10 is withheld. Place and Wide enumerate every ordered top three and pay the ticket by the stake-back-share rule with its co-placers; Place pays the places of the field at sale (`decide(view, field_at_sale=)`, by default the Win board's runners). The cross-pool coupling is an assumption: Win and Place/Wide draws share per-runner shocks (one iid N(0, 1) matrix per race and τ from a stable hash, handed to C1's `Nowcast.draws_from_shocks` for every pool, a Wide pair's shock (ε_a + ε_b)/√2), so the EV keeps the covariance of P and D within a draw; no other cross-pool dependence is modelled. One EV per refit of C4/C5 is reported beside each row. The decider calls `meta[r].predict(q, view)`, which C4's `MetaFit.predict(frame, q)` does not match: it needs an adapter, which #101 provides. `ShadowDecider` returns only `ShadowDecision` records (race, pool, combination, τ, the board at t, the EV per refit and config/betting.yaml's flat stake, within the race-day cap); it selects nothing, applies no threshold, refuses any mode but "shadow", and has no code path that places a bet. It has run on synthetic tests only; #101 runs it on logged days. It is not adopted.

## Evaluation

Splits, metric and confidence interval: not decided, see [the problem statement](../../../../docs/problem-statement.md). Holdout use is recorded in [the holdout log](../../../../docs/holdout-log.md).

## Results

None adopted.

- C3 time test (#305): the same-day variant lowers the time RMSE of E[top-3 time] by +0.233 s against the par alone (one-sided 95% lower bound +0.224). This is a time result only, not a result at the final price. See the C3 entry under Model.
- C1 nowcast (#85): scored against the random walk and #116's drift on D9's folds of the JRA-VAN paths, in registry family `japan-c1-nowcast-final-shares-v2` (late scratches scored by #69). It supersedes `japan-c1-nowcast-final-shares`, whose spec dropped late-scratch cases. Figures withheld (D21). See `docs/experiments/nowcast-japan.md` for the registry records, the trial runs and the private report.

- C2 ability (#92): E33 (top-3 against winner-only training) and E34 (pooling against #129's as-of rates) scored after stage 2 on D9's folds from 2020-Q1, in registry family `japan-c2-ability-final-win-v2`, with the member's correlation with ln q_final, its standalone and stage-2 results, its C4 marginal against the market-only stage-2 meta and a by-class table. It supersedes `japan-c2-ability-final-win`, which had no market-only comparator and ended the rates' window on the last race day before the race. Figures withheld (D21). See `docs/experiments/ability-japan.md` for the records and the private report.

## Known limitations

- C1 nowcast (#85, #410): Place is read by JRA's stake-back-share rule. v5's measurement (record `86795d4a880bbd20`) gives, on the closing boards of the scored Place cases, by read used and P, the share of validated boards whose figures the rule reproduces from the votes and the share whose shares fall within 7.5% relative and 0.01 absolute per runner of the vote shares. Which reads stand as a result, and the rates, are withheld (D21) and are in the private report; the card does not assert the tolerance on every board. v5 read P from the priced runners; the reads and #116's drift baseline now take it from the field at the start of sales, a changed spec frozen as family `japan-c1-nowcast-final-shares-v6` (v5's freeze and records kept as history), and no run under that rule is made yet; `ShadowNowcast`'s draws still take P from the priced runners. Boards whose figures do not determine the money are read `undetermined` and are a diagnostic whatever their measurement. Those boards, and the boards the rule does not reproduce, are open follow-ups. The low figures alone fail where the favourite is at the floor on the tests' synthetic boards and are only the fallback for boards without high figures. Wide's inversion uses the same rule, unmeasured against Wide money. The JRA-VAN paths are 5-6 minutes apart, so the board at T-2 and T-1 can be minutes old; its age at t is measured in the private report. φ_t is a point forecast, with no V_final uncertainty in the draws. A `RaceBoard` carries no lag boards, so through `ShadowNowcast` the velocity features are 0.
- C2 ability (#92): σ_i is the inverse Hessian of the discounted objective, which is conservative when w2 or w3 is below 1. The sire is the JRA-VAN pedigree's current snapshot (`static`), and the jockey and trainer are the store's keys, so a jockey's or trainer's effect pools every ride or runner in the training window, with no drift in time. The only horse-level feature is #91's rating; card features (weight, draw, age, sex, layoff) are left to #93-#95 (owner, 2026-11-26).
- C2 lineage (#95): the store's netkeiba `horse_pedigree` links the sire, dam and damsire by name only (its id columns are empty) and covers mainly the foals of 2018 on, so a lineage prior from it reaches few runs before 2020. The prior has no per-horse effect, so a horse's own runs inform its sire's and damsire's effects as if independent.

## Adoption history

| Date | Model id | Ticket | Owner sign-off |
| ---- | -------- | ------ | -------------- |
| none adopted | | | |
