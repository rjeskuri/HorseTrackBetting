# HorseTrackBetting

Research and tooling for pari-mutuel horse racing. A bet pays at the final odds, so the work asks whether a model adds anything over the final price, and measures it honestly: walk-forward splits, a paired bootstrap, a locked holdout per country. A null result is an acceptable answer.

## Layout
- `src/horsetrack/core`: only HTTP/ingest, storage, pari-mutuel pricing maths and the evaluation harness.
- `src/horsetrack/countries/japan`: active. Sources, odds logger, features, models, model card.
- `src/horsetrack/countries/hongkong`: data only, otherwise paused.
- `docs/`: problem statement, known-vs-assumed ledger, experiment-report template.

Code moves into `core` only when two countries use it identically.

## Data
No data lives in this repository. Set `HORSETRACK_DATA_DIR` to a directory outside the checkout. Raw caches, stores, models, reports and real page fixtures go there. Committed test fixtures are synthetic.

The repository carries no secrets, scraped or licensed content, or personal data. No code logs in to a betting account or places a bet.

## Develop
- `scripts/verify.sh`: the whole suite plus lint.
- `uv run pytest -q tests`: the tests.
