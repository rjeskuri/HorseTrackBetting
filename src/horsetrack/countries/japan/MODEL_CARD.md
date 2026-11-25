# Model card: Japan

The current pipeline for this country, end to end, in data-science terms. This is a skeleton: nothing below is decided until the problem statement and the adoption history say so.

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

- Workout feature set (#137, `handicap/workouts.py`): best final-furlong time against the course par, its trend over the last three workouts, days since the last hard workout, and CYB's per-course workout fields since the last race, from JRDB CHA and CYB as of decision time. It exists as an opt-in JRDB feature set behind `HORSETRACK_JRDB_FEATURES` and is not adopted. JRDB's terms allow private use only, so its development-data log-loss report stays under the data dir.

## Model

None adopted.

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
