# DRAFT: the WIN5 carryover study (ticket #135)

Status: **DRAFT, not signed, not frozen.** Nothing here is pre-registered until the owner signs
the open decisions below and the spec is frozen with `Registry.freeze_family` in
`$HORSETRACK_DATA_DIR/registry/japan.jsonl` (append-only: a freeze cannot be undone). No day is
scored before that; `research.win5.score_study` refuses an unfrozen spec (`NotPreRegistered`).

## Question

D18 positive-edge family: on WIN5 days with a carryover, is the return of buying WIN5 coverage,
at the final dividend, above 1 after the 30% takeout, and how does it depend on carryover size?

## What is fixed by the code (ticket #135)

- Data: the WIN5 history table (`processed_dir("jra")/win5.parquet`, docs/specs/store.md), one row
  per WIN5 day since 2011-04-24.
- Development days only (D14): every day with `is_holdout_day("japan", d)` is dropped before any
  day is scored, and the count dropped is reported.
- Bins: the spec's `carryover_edges_yen` e0 < ... < en on `carryover_in_yen`; bins [e_i, e_i+1)
  and a last open bin [en, inf); a day below e0 is in no bin. Output per bin: `n_days`, the mean
  of the per-day value.

## Open decisions (owner)

1. **Coverage rule** (what ticket set is "bought" on a day): e.g. every combination (the pool's
   own return), the top-k favourites per leg at a time before the close, or a fixed budget spread
   by win probability. Each needs its odds as of the decision time, and WIN5 leg odds are only
   logged live (#130), so a historical rule may be limited to the pool-level figure.
2. **Metric**: the per-day value `day_value(row)` (for the pool-level figure: dividend x winning
   tickets / pool; on a no-hit day 0), and the comparator (1.0, or the same rule on non-carryover
   days).
3. **Carryover-size bins**: the edges (e.g. a first edge just above 0 and edges at multiples of the
   break-even carryover `0.30 x pool`, which depends on the day's pool).
4. **Estimand details**: whether the payout cap (6億円 per 100 yen, the excess carrying over even
   on a hit) is modelled, and whether boosted-takeout days (`config/takeout_boosts.yaml`: the
   all-pool entries, e.g. 2024-09-16 and 2025-05-04, include WIN5) use their own rate or are excluded. The code uses 0.30 on
   every day.
5. **Uncertainty and decision rule**: the interval (e.g. a day-block bootstrap) and what result
   counts as a positive edge, to be stated before scoring.

## Recording

Each scoring run, failures included, is a registry record under the frozen family, and the PR that
scores it carries an experiment report (docs/experiment-report.md).
