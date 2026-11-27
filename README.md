# HorseTrackBetting

Research and tooling for pari-mutuel horse racing. A bet pays at the final odds, so the work ranks every option (race type, pool, combination, rule, model) by its return per ¥100 at the final price and keeps every result, nulls and failed variants included, on one leaderboard per country. It measures honestly: walk-forward splits, a paired bootstrap, and a locked holdout per country that checks whether the ranking holds up. A null result is an acceptable answer: the aim is to find the best of the options, played small and by hand, with every bet logged as data.

## Layout
- `src/horsetrack/core`: only HTTP/ingest, storage, pari-mutuel pricing maths and the evaluation harness.
- `src/horsetrack/countries/japan`: active. Sources, odds logger, features, models, model card.
- `src/horsetrack/countries/hongkong`: data only, otherwise paused.
- `docs/`: problem statement, known-vs-assumed ledger, experiment-report template.

Code moves into `core` only when two countries use it identically.

How the pieces fit (sources, machines, race-day flow, store, evaluation and workflow): [docs/architecture.md](docs/architecture.md).

## Data
No data lives in this repository. Set `HORSETRACK_DATA_DIR` to a directory outside the checkout. Raw caches, stores, models, reports and real page fixtures go there. Committed test fixtures are synthetic.

The repository carries no secrets, scraped or licensed content, or personal data. No code logs in to a betting account or places a bet.

## Develop
- `scripts/verify.sh`: the whole suite plus lint.
- `uv run pytest -q tests`: the tests.
