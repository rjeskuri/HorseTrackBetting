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

## Evaluation

Splits, metric and confidence interval: not decided, see [the problem statement](../../../../docs/problem-statement.md). Holdout use is recorded in [the holdout log](../../../../docs/holdout-log.md).

## Results

None adopted.

- C3 time test (#305): the same-day variant lowers the time RMSE of E[top-3 time] by +0.233 s against the par alone (one-sided 95% lower bound +0.224). This is a time result only, not a result at the final price. See the C3 entry under Model.

## Known limitations

None recorded yet.

## Adoption history

| Date | Model id | Ticket | Owner sign-off |
| ---- | -------- | ------ | -------------- |
| none adopted | | | |
