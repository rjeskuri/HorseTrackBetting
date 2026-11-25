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

- C4 meta-model (#87, `models/meta.py` over `models/condlogit.py`): a conditional logit within race at the final Win price. Stage 2 is u_i = Σ_k α_k(z) m_ki + β ln q_i. The offset form is u_i = ln q_i + Σ_k α_k(z) m_ki, where the offset can be ln q or the D4 calibrated market. α and β are unbounded maximum likelihood (08 R13), and dead heats count at 1/k. An optional ridge shrinks only the member weights, toward the owner's target α = 0 (the market alone). Segment weights enter as interaction columns, α_k(z) = α_k0 + Σ_j α_kj z_j, and an optional (ln q)² term is available. The members m_k are out-of-fold level-1 log-probabilities. The meta is itself out of fold on D9's folds (`oof_predictions`, which refuses leaky or overlapping folds), and `predict(frame, q)` takes any price vector, so the decision layer can pass nowcast draws. `ceiling_table` gives E9's ln(π/q) table. Coefficient standard errors are the inverse information (races independent) and, given a cluster column, the meeting-clustered sandwich that D7's unit calls for. It is not adopted. Its first run (`studies/meta.py`, `docs/experiments/meta-japan.md`) used HKTrack's research-24 members. These read JRDB, or are not confirmed free of it, so their results stay under the data dir. In that run the meta trains on member rows from 2023-01-05 only, so D9's 5-season minimum is not met at level 2 on any scored fold (0.98 to 3.73 years), and level 1 is HKTrack's season walk-forward, refit once a season, not quarterly. By the owner's answer (b) on #87 (2026-11-25), that run is a documented failure to meet D9 and supports no adoption claim; nothing is adopted, and the D9-compliant run comes with #91's and #98's members.

## Evaluation

Splits, metric and confidence interval: not decided, see [the problem statement](../../../../docs/problem-statement.md). Holdout use is recorded in [the holdout log](../../../../docs/holdout-log.md).

## Results

None yet.

## Known limitations

None recorded yet.

## Adoption history

| Date | Model id | Ticket | Owner sign-off |
| ---- | -------- | ------ | -------------- |
| none adopted | | | |
