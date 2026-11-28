# The WIN5 rollover study (ticket #323)

Status: pre-registered, to be frozen before any scored read. Family id: **`win5-rollover-v1`**. The
spec is committed at `src/horsetrack/countries/japan/config/win5_rollover.yaml`; `horsetrack japan
win5-rollover freeze` freezes it, as loaded, with `Registry.freeze_family` in
`$HORSETRACK_DATA_DIR/registry/japan.jsonl` (append-only). `horsetrack japan win5-rollover score` and
`research.win5_rollover.score_rollover` refuse a spec that is not the one frozen for the family
(`NotPreRegistered`); a change after the freeze is a new family, frozen again before it is scored.

## Question

Follow-up of #135 (docs/specs/win5-carryover-study.md, docs/experiments/win5-carryover-japan.md): its
[1, 2) carryover bin scored a mean day value of 1.059 on 4 days, none of them a no-hit day, and its
day bootstrap cannot see a rollover that never occurred in the sample. How often does a WIN5 day
have no winning ticket, with what interval, and what is a carryover bin's expected value net of
that risk?

## Null

The starting point: a carryover day's expected value is below 1 once rollover risk is counted. The
study is shown wrong for a bin only when its adjusted EV stays above 1 at the upper bound of the
no-hit interval.

## Data and days

- The WIN5 history table (`research.win5.read_history`), one row per WIN5 day since 2011-04-24.
- Development days only (D14): every day with `is_holdout_day("japan", d)` is dropped before any
  day is scored, and the count dropped is reported. The holdout is never opened.

## Estimand

- A **no-hit day** is a day with no winning ticket (`winning_tickets == 0`, `is_no_hit`). A day that
  paid and passed the excess over the 6億円 cap on is a hit.
- The **no-hit rate** `q` is the share of days that are no-hit days: overall (all development days,
  with or without carryover) and per carryover bin.
- A no-hit day pays 0 and passes the money on, so for a bin with `won_value`, the mean
  `win5.day_value` (payout over the net pool of #135) of its won days:
  - **rollover-adjusted EV** = `(1 - q) x won_value`;
  - **break-even no-hit rate** = `1 - 1 / won_value` (0 when `won_value <= 1`: no rate helps).
- With `q` the bin's own rate, the adjusted EV equals #135's mean day value of the bin (no-hit days
  counted as 0): the adjustment only matters at an `q` the sample did not show, which is why the
  interval's upper bound is scored.

## Interval

The exact two-sided Clopper-Pearson interval of the no-hit count, `confidence: 0.95` (`interval:
clopper-pearson`), for the overall rate and for each bin. At 0 no-hit days the lower bound is 0 and
the upper bound is `1 - 0.025^(1/n)`: the rollover that never occurred in the sample still leaves
an upper bound.

## Bins

As #135: the ratio is `carryover_in / (break_even_takeout x net pool)` with `break_even_takeout:
0.30`, `ratio_edges: [0, 0.5, 1, 2]`, bins (0, 0.5), [0.5, 1), [1, 2), [2, inf); a day with no
carryover is in no bin (it counts in the overall rate). The pool is net of refunds (`pool:
net-of-refunds`, `research.win5.net_pool`); `payout_cap_yen: 600000000`.

## Headline, pre-registered

The **headline** is the **pooled** adjusted EV: `(1 - q) x won_value` with `q` the overall
no-hit rate over all development days, and its stress figure at the upper bound of the overall
interval (`adjusted_ev_pooled`, `adjusted_ev_pooled_at_upper`). The bin's own rate is the secondary
figure (`adjusted_ev`, `adjusted_ev_at_upper`): on a handful of days it carries almost no
information (0 of 4 days has an upper bound near 60%), and it reproduces #135's mean. Both are
reported for every bin, with the bin's break-even no-hit rate beside the interval, so the report can
say whether the break-even rate lies inside it. The pooled rate averages days whose pools differ
and ignores that a no-hit day is what creates a carryover day; the report states that limit.

## Output and recording

`score_rollover` returns the family id, the spec hash, the holdout days excluded, `overall`
(`n_days`, `n_no_hit`, `no_hit_rate`, `no_hit_lower`, `no_hit_upper`) and per bin the same fields
with `won_mean_value`, `break_even_no_hit_rate`, `adjusted_ev`, `adjusted_ev_at_upper`,
`adjusted_ev_pooled` and `adjusted_ev_pooled_at_upper`. A bin with no day has `None` for the rest;
a bin whose days are all no-hit has a rate and interval and `None` for the won-value figures.
`horsetrack japan win5-rollover score` prints aggregates only and appends one RunRecord (ticket
323, status `scored`, the frozen spec as params): one `no_hit_rate` metric for all development
days and one per non-empty bin, `lo` and `hi` the interval, `n` the days. A scoring run that fails
after the freeze is recorded as `failed`.
