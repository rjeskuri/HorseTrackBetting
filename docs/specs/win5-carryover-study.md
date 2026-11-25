# The WIN5 carryover study (ticket #135)

Status: **signed by the owner on #135, 2026-11-25.** Family id: **`win5-carryover-v1`**. The spec
is committed at `src/horsetrack/countries/japan/config/win5_study.yaml`; `horsetrack japan
win5-study freeze` freezes it, as loaded, with `Registry.freeze_family` in
`$HORSETRACK_DATA_DIR/registry/japan.jsonl` (append-only: a freeze cannot be undone). The freeze
happens before any scoring: `horsetrack japan win5-study score` and `research.win5.score_study`
refuse a spec that is not the one frozen for the family (`NotPreRegistered`), and any change to the
spec after the freeze is a new family, frozen again before it is scored.

## Question

D18 positive-edge family: on WIN5 days with a carryover, is the return of buying WIN5 coverage, at
the final dividend, above 1 after the takeout, and how does it depend on carryover size?

## Data and days

- The WIN5 history table (`processed_dir("jra")/win5.parquet`, docs/specs/store.md), one row per
  WIN5 day since 2011-04-24, with `payout_yen`, the yen paid over every winning combination (a
  dead-heat day lists several).
- Development days only (D14): every day with `is_holdout_day("japan", d)` is dropped before any
  day is scored, and the count dropped is reported.
- Only 17 carryover days exist in the history (backfill of 2026-11-25: 910 WIN5 days, 17 with a
  carryover in). Every bin is small; the report states the count per bin, and a wide interval is
  the expected outcome.

## The owner's decisions (answered on #135, 2026-11-25)

1. **Coverage**: the pool-level return, the pool's own return to its bettors: a day's value is
   `day_value(row) = payout_yen / pool_yen` (dividend x winning tickets / pool, every combination
   of a dead heat counted), 0 on a no-hit day. A live ticket rule (top-k per leg) is a later
   follow-up once WIN5 legs are logged (#130).
2. **Metric and comparator**: the mean per-day value of a bin against 1.0 (`comparator: 1.0`).
3. **Carryover bins**: multiples of the break-even carryover `0.30 x pool`. A day's ratio is
   `carryover_in_yen / (0.30 x pool_yen)`, with 0.30 on every day (`break_even_takeout`); edges
   `[0, 0.5, 1, 2]` give the bins (0, 0.5), [0.5, 1), [1, 2) and [2, inf). A day with no carryover
   is in no bin.
4. **Estimand**: the 6億円 per 100-yen payout cap is modelled (`payout_cap_yen: 600000000`): the
   value is the realised payout, which the cap holds, and the excess a capped day carries over
   counts on the next day (`research.win5.capped_payout`). Boosted-takeout days count at their own
   rate from `config/takeout_boosts.yaml` (`research.win5.win5_takeout`; 2024-09-16 and
   2025-05-04, ultra premium on every pool, take out 15%): their realised payout already carries
   it, and `payout_mismatches` checks every day's money at its own rate.
5. **Interval and decision rule**: a day-block bootstrap (the WIN5 day is the unit) of each bin's
   mean, `n_boot: 10000` resamples under `seed: 20261125`; the lower bound is the 5th percentile of
   the bootstrap means (`one_sided_level: 0.95`). A positive edge is a bin whose one-sided 95%
   lower bound is above 1.0, stated here before scoring.

## Output and recording

`score_study` returns, per bin, `n_days`, `mean_value`, `lower_bound` and `positive_edge` (an
empty bin: no mean, no bound, no edge), with the family id, the spec hash and the holdout days
excluded. `horsetrack japan win5-study score` appends one RunRecord (status `scored`, ticket 135,
the frozen spec as params, one metric per non-empty bin with its n and lower bound); a scoring run
that fails after the freeze is recorded as `failed`. The PR that scores the study carries an
experiment report (docs/experiment-report.md) that notes the 17 carryover days.
